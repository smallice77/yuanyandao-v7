# -*- coding: utf-8 -*-
"""四种输出格式渲染：Markdown / JSON / CSV / 逗号关键词。"""

from __future__ import annotations

import csv
import io
import json
from typing import Dict, List

__all__ = ["render", "to_markdown", "to_csv", "to_keywords", "to_json"]

_FORMATS = ("markdown", "json", "csv", "keywords", "keyword_comma")


def to_json(result: Dict[str, object]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2, sort_keys=False)


def to_keywords(result: Dict[str, object]) -> str:
    data = result.get("data") or {}
    if isinstance(data, dict) and data.get("keywords_text"):
        return str(data["keywords_text"])
    words = (result.get("meta") or {}).get("hit_keywords") or []
    return ", ".join(dict.fromkeys(str(w) for w in words))


def _table(headers: List[str], rows: List[List[object]]) -> str:
    if not rows:
        return "_（无数据）_"
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join("" if cell is None else str(cell) for cell in row) + " |")
    return "\n".join(out)


def to_markdown(result: Dict[str, object]) -> str:
    meta = result.get("meta") or {}
    intent = result.get("intent", "qa")
    data = result.get("data") or {}
    status = result.get("status", "ok")
    icon = {"ok": "✅", "denied": "⛔", "not_found": "🟡", "error": "❌", "degraded": "⚠️"}.get(status, "•")
    lines: List[str] = []
    lines.append("# 元衍道枢 V7.0 · 调度结果")
    lines.append("")
    lines.append(
        "> %s %s ｜ 意图 `%s` ｜ 用户 `%s` ｜ 权限 L%s ｜ 操作编号 `%s` ｜ 耗时 %sms"
        % (
            icon,
            status,
            intent,
            meta.get("user_id", "-"),
            meta.get("permission_level", "-"),
            meta.get("op_code", "-"),
            meta.get("latency_ms", "-"),
        )
    )
    lines.append("")

    answer = str(result.get("answer") or "").strip()
    if answer:
        lines.append(answer)
        lines.append("")

    if intent == "sandbox" and isinstance(data, dict) and data.get("risk_table"):
        inner = data.get("sandbox") or {}
        dims = inner.get("dimensions") or []
        headers = ["分支", "方案"] + [str(d.get("name")) for d in dims] + ["加权风险", "状态"]
        rows = []
        for row in data["risk_table"]:
            rows.append(
                [row.get("branch_id"), row.get("title")]
                + [row.get(dim.get("id")) for dim in dims]
                + [row.get("weighted_score"), row.get("status")]
            )
        lines.append("### 风险权重表")
        lines.append("")
        lines.append(_table(headers, rows))
        lines.append("")
        conflicts = inner.get("conflicts") or []
        if conflicts:
            lines.append("### 冲突分支")
            lines.append("")
            lines.append(
                _table(
                    ["冲突", "分支", "冲突变量", "严重度"],
                    [
                        [
                            c.get("conflict_id"),
                            " / ".join(c.get("branches", [])),
                            "、".join("%s(%s↔%s)" % (v.get("variable"), v.get("left"), v.get("right")) for v in c.get("variables", [])),
                            c.get("severity"),
                        ]
                        for c in conflicts
                    ],
                )
            )
            lines.append("")
        convergence = inner.get("convergence") or {}
        if convergence:
            lines.append("### 收敛结论")
            lines.append("")
            lines.append(
                "- 状态：**%s**，选择 **%s %s**（加权风险 %.2f）"
                % (
                    convergence.get("status"),
                    convergence.get("selected") or "-",
                    convergence.get("selected_title") or "",
                    float(convergence.get("score") or 0),
                )
            )
            lines.append("- 依据：%s" % convergence.get("rationale", ""))
            if convergence.get("alternatives"):
                lines.append(
                    "- 备选："
                    + "；".join(
                        "%s %s(%.2f)" % (a["branch_id"], a["title"], float(a["weighted_score"]))
                        for a in convergence["alternatives"]
                    )
                )
            lines.append("")

    if intent == "train" and isinstance(data, dict) and data.get("train_plan"):
        plan = data["train_plan"]
        lines.append("### 训练配置（qlora_config.yaml）")
        lines.append("")
        lines.append("```yaml")
        lines.append(str(plan.get("yaml", "")).rstrip())
        lines.append("```")
        lines.append("")
        vram = plan.get("vram") or {}
        lines.append("### 显存估算")
        lines.append("")
        lines.append(
            _table(
                ["项目", "数值"],
                [
                    ["4bit 权重", "%s GB" % vram.get("weights_gb")],
                    ["LoRA/优化器", "%s GB" % vram.get("adapters_gb")],
                    ["激活", "%s GB" % vram.get("activation_gb")],
                    ["logits", "%s GB" % vram.get("logits_gb")],
                    ["预计峰值", "%s GB" % vram.get("estimated_peak_gb")],
                    ["上限", "%s GB" % vram.get("limit_gb")],
                    ["余量", "%s GB" % vram.get("headroom_gb")],
                ],
            )
        )
        lines.append("")
        dataset = plan.get("dataset") or {}
        if dataset.get("steps"):
            lines.append("### 数据集清洗方案")
            lines.append("")
            for step in dataset["steps"]:
                lines.append("%s. **%s**：%s" % (step.get("step"), step.get("name"), step.get("detail")))
            lines.append("")
        files = plan.get("files") or {}
        if files:
            lines.append("### 已生成文件")
            lines.append("")
            for key, value in files.items():
                lines.append("- %s：`%s`" % (key, value))
            lines.append("")

    if intent in ("reload", "maintenance") and isinstance(data, dict):
        stats = data.get("index") or {}
        if stats:
            lines.append("### 词库状态")
            lines.append("")
            lines.append(
                _table(
                    ["分类", "等级", "文档数", "关键词数"],
                    [
                        [c.get("name"), c.get("level"), c.get("docs"), c.get("keywords")]
                        for c in stats.get("category_detail", [])
                    ],
                )
            )
            lines.append("")

    citations = result.get("citations") or []
    if citations:
        lines.append("### 引用来源")
        lines.append("")
        lines.append(
            _table(
                ["#", "文档", "编号", "分类", "相关度"],
                [[c.get("index"), c.get("title"), c.get("doc_id"), c.get("category"), c.get("score")] for c in citations],
            )
        )
        lines.append("")

    warnings = result.get("warnings") or []
    if warnings:
        lines.append("### 提示")
        lines.append("")
        for item in warnings:
            lines.append("- %s" % item)
        lines.append("")

    lines.append("---")
    lines.append(
        "_锚点：%s ｜ 角色：%s ｜ 请求：%s_"
        % (result.get("anchor"), result.get("role_name") or result.get("role"), meta.get("request_id", "-"))
    )
    return "\n".join(lines).rstrip() + "\n"


def to_csv(result: Dict[str, object]) -> str:
    intent = result.get("intent", "qa")
    data = result.get("data") or {}
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")

    if intent == "sandbox" and isinstance(data, dict) and data.get("risk_table"):
        inner = data.get("sandbox") or {}
        dims = inner.get("dimensions") or []
        writer.writerow(["branch_id", "title", "status"] + [str(d.get("id")) for d in dims] + ["weighted_score"])
        for row in data["risk_table"]:
            writer.writerow(
                [row.get("branch_id"), row.get("title"), row.get("status")]
                + [row.get(dim.get("id")) for dim in dims]
                + [row.get("weighted_score")]
            )
        return buffer.getvalue()

    if intent == "train" and isinstance(data, dict) and data.get("train_plan"):
        plan = data["train_plan"]
        writer.writerow(["section", "key", "value"])
        for section, values in (plan.get("config") or {}).items():
            if isinstance(values, dict):
                for key, value in values.items():
                    writer.writerow([section, key, json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value])
            else:
                writer.writerow(["root", section, values])
        return buffer.getvalue()

    if intent in ("export_keywords", "keywords") and isinstance(data, dict) and data.get("keywords_list"):
        writer.writerow(["category", "keyword"])
        for category, words in (data.get("keywords_by_category") or {}).items():
            for word in words:
                writer.writerow([category, word])
        return buffer.getvalue()

    citations = result.get("citations") or []
    if citations:
        writer.writerow(["index", "doc_id", "title", "category", "score"])
        for item in citations:
            writer.writerow([item.get("index"), item.get("doc_id"), item.get("title"), item.get("category"), item.get("score")])
        return buffer.getvalue()

    writer.writerow(["field", "value"])
    meta = result.get("meta") or {}
    for key, value in meta.items():
        writer.writerow([key, json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value])
    writer.writerow(["status", result.get("status")])
    writer.writerow(["intent", intent])
    return buffer.getvalue()


def render(result: Dict[str, object], fmt: str = "markdown") -> str:
    fmt = (fmt or "markdown").lower()
    if fmt in ("json",):
        return to_json(result)
    if fmt in ("csv",):
        return to_csv(result)
    if fmt in ("keywords", "keyword_comma", "keyword"):
        return to_keywords(result)
    return to_markdown(result)
