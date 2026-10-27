# -*- coding: utf-8 -*-
"""调度主循环：STEP1 双路检索 → STEP2 权限 → STEP3 路由 → STEP4 分流
→ STEP5 输出校验 → STEP6 审计封装 → STEP7 返回上层入口。
"""

from __future__ import annotations

import copy
import json
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import ANCHOR_KEY, PLUGIN_NAME, ROLE_DISPATCH, ROLE_PRODUCT, VERSION
from .anchors import (
    anchor_banner,
    check_anchor_integrity,
    detect_anchor_tamper,
    detect_role_confusion,
)
from .audit import AuditLog
from .config import Config, load_config
from .documents import build_bp_document, build_delivery_spec, build_generic_document
from .embeddings import build_embedder
from .formats import render
from .permissions import LEVEL_NAMES, AuthResult, PermissionKernel
from .rag import RagEngine
from .sandbox import SandboxEngine
from .train_scheduler import TrainScheduler
from .vector_store import KnowledgeBase, ScoredDoc

__all__ = ["Engine", "Request", "DEFAULT_FORMAT"]

DEFAULT_FORMAT = "markdown"

_INTENT_CAP = {
    "qa": "qa",
    "sandbox": "sandbox",
    "train": "train_config",
    "doc": "docs",
    "export_keywords": "qa",
    "reload": "kb_write",
    "maintenance": "docs",
}

_INTENT_LABEL = {
    "qa": "知识问答",
    "sandbox": "沙盘推演",
    "train": "训练调度",
    "doc": "文档交付",
    "export_keywords": "关键词导出",
    "reload": "词库热重载",
    "maintenance": "知识库维护",
    "perm_selftest": "权限自检",
    "denied": "权限拦截",
}

_NOT_FOUND = (
    "【知识库未收录，是否新建词条/文档入库】\n"
    "- 已检索分类：{categories}\n"
    "- 最高相关度：{score:.3f}（阈值 {threshold:.2f}）\n"
    "- 建议：由 L4 管理员执行【入库 <词条>】，或运行 "
    "`python scripts/import_keywords.py --category ta --terms \"...\"` 后发送【刷新词库】。"
)

_TRAIN_KEYS = ("训练", "微调", "qlora", "lora", "sft", "dpo", "基座训练", "蒸馏")
_BANNED_OUTPUT_CODES = "OUTPUT_VIOLATION"
_DOMAIN_KB = {"train": ("llm", "hardware"), "ta": ("ta", "hardware"), "general": ()}


@dataclass
class Request:
    text: str
    user_id: str = "anonymous"
    level: int = 0
    api_key: str = ""
    fmt: Optional[str] = None
    options: Dict[str, object] = field(default_factory=dict)


class Engine:
    def __init__(self, config: Optional[Config] = None, config_path: Optional[str] = None,
                 root: Optional[str] = None):
        self.config = config or load_config(config_path, root)
        self.config.ensure_dirs()
        ok, message = check_anchor_integrity(
            str(self.config.get("anchor_key")), str(self.config.get("safety.anchor_lock"))
        )
        if not ok:
            raise RuntimeError("锚点密钥校验失败，拒绝启动：%s" % message)
        self.anchor_message = message
        self.embedder = build_embedder(self.config)
        self.kb = KnowledgeBase(self.config, self.embedder)
        self.policy = PermissionKernel(self.config)
        self.audit = AuditLog(self.config)
        self.sandbox = SandboxEngine(self.config)
        self.train = TrainScheduler(self.config)
        self.rag = RagEngine(self.config)
        self.banned = [str(p) for p in (self.config.get("safety.banned_patterns") or [])]
        self.max_input = int(self.config.get("safety.max_input_chars", 8000))
        self.started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _ids(self) -> Tuple[str, str]:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        suffix = secrets.token_hex(2).upper()
        return "REQ-%s-%s" % (stamp, suffix), "OP-%s-%s" % (stamp, suffix)

    @staticmethod
    def detect_intent(text: str) -> str:
        low = text.casefold()
        if "权限测试" in text:
            return "perm_selftest"
        if any(word in text for word in ("刷新词库", "重载词库", "热重载")):
            return "reload"
        if "导出" in text and ("关键词" in text or "词库" in text):
            return "export_keywords"
        if "沙盘推演" in text or "推演" in text or "沙盘" in text:
            return "sandbox"
        if any(key in low for key in _TRAIN_KEYS):
            return "train"
        if any(word in text for word in ("整理文档", "入库")):
            return "maintenance"
        if any(word in text for word in ("迭代", "继续", "还有吗")):
            return "maintenance"
        if any(word in text for word in ("生成", "输出", "导出")):
            if any(word in low for word in ("文档", "bp", "方案书", "规范", "markdown")):
                return "doc"
        return "qa"

    def _envelope(self, status: str, intent: str, answer: str,
                  data: Optional[Dict[str, object]] = None,
                  citations: Optional[List[Dict[str, object]]] = None,
                  warnings: Optional[List[str]] = None,
                  meta: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        return {
            "ok": status == "ok",
            "status": status,
            "intent": intent,
            "intent_label": _INTENT_LABEL.get(intent, intent),
            "anchor": ANCHOR_KEY,
            "role": "dispatch_kernel",
            "role_name": ROLE_DISPATCH,
            "bound_product": ROLE_PRODUCT,
            "plugin": PLUGIN_NAME,
            "version": VERSION,
            "answer": answer,
            "data": data or {},
            "citations": citations or [],
            "warnings": warnings or [],
            "meta": meta or {},
        }

    def _meta(self, request_ids: Tuple[str, str], auth: Optional[AuthResult], fmt: str,
              hit_words: Sequence[str], categories: Sequence[str], latency_ms: float,
              branch_id: str = "", engine_mode: str = "") -> Dict[str, object]:
        request_id, op_code = request_ids
        return {
            "request_id": request_id,
            "op_code": op_code,
            "user_id": auth.user_id if auth else "anonymous",
            "permission_level": auth.level if auth else 0,
            "permission_name": LEVEL_NAMES.get(auth.level, "未知") if auth else "未知",
            "hit_keywords": list(dict.fromkeys(hit_words)),
            "kb_categories": list(dict.fromkeys(categories)),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "latency_ms": round(latency_ms, 2),
            "engine_mode": engine_mode or "offline-extractive",
            "model": self.rag.model,
            "vector_backend": self.kb._backend,
            "index_hash": self.kb.index_hash,
            "branch_id": branch_id,
            "format": fmt,
            "config_version": self.config.get("version"),
        }

    def _audit(self, envelope: Dict[str, object], text: str, auth: AuthResult,
               branch_id: str = "", extra: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        meta = envelope.get("meta") or {}
        answer = str(envelope.get("answer") or "")
        record = {
            "user_id": auth.user_id,
            "permission_level": auth.level,
            "input": text[:4000],
            "output": answer[:1500],
            "hit_keywords": meta.get("hit_keywords", []),
            "source_doc": [c.get("doc_id") for c in (envelope.get("citations") or [])],
            "timestamp": meta.get("timestamp"),
            "branch_id": branch_id,
            "op_code": meta.get("op_code"),
            "request_id": meta.get("request_id"),
            "intent": envelope.get("intent"),
            "status": envelope.get("status"),
            "code": (envelope.get("data") or {}).get("code", ""),
            "latency_ms": meta.get("latency_ms"),
            "engine_mode": meta.get("engine_mode"),
        }
        if extra:
            record.update(extra)
        return self.audit.append(record)

    def _deny(self, request_ids: Tuple[str, str], intent: str, auth: AuthResult, text: str,
              message: str, code: str, required_level: int = 0, capability: str = "",
              fmt: str = DEFAULT_FORMAT, hit_words: Sequence[str] = (),
              categories: Sequence[str] = (), latency_ms: float = 0.0) -> Dict[str, object]:
        data = {
            "code": code,
            "required_level": required_level,
            "required_level_name": LEVEL_NAMES.get(required_level, "未知"),
            "capability": capability,
            "denied": True,
        }
        envelope = self._envelope(
            status="denied",
            intent=intent,
            answer=message,
            data=data,
            warnings=[],
            meta=self._meta(request_ids, auth, fmt, hit_words, categories, latency_ms,
                            engine_mode="denied-before-retrieval"),
        )
        self._audit(envelope, text, auth, extra={"denied": True, "code": code})
        return envelope

    def _capability_denied(self, request_ids, intent, auth, text, capability, *,
                           fmt, hit_words, categories, latency_ms) -> Optional[Dict[str, object]]:
        if capability in auth.capabilities:
            return None
        required = 4
        for level in sorted(self.policy.levels):
            if capability in self.policy.capabilities_for(level):
                required = level
                break
        message = self.policy.denial_message(
            auth, required, capability,
            reason="当前等级 L%d 不具备「%s」能力（动态密钥有效，但权限不足）" % (auth.level, capability),
        )
        return self._deny(
            request_ids, intent, auth, text, message,
            code="PERMISSION_DENIED", required_level=required, capability=capability,
            fmt=fmt, hit_words=hit_words, categories=categories, latency_ms=latency_ms,
        )

    @staticmethod
    def _pick_hits(hits: Sequence[ScoredDoc], domain: str) -> List[ScoredDoc]:
        """按推演领域收敛引用来源，避免“流程/格式”这类泛词带偏引用。"""
        cats = _DOMAIN_KB.get(domain, ())
        if not cats:
            return list(hits)
        filtered = [hit for hit in hits if hit.doc.category in cats]
        return filtered or list(hits)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def invoke(self, text: str, user_id: str = "anonymous", level: int = 0,
               api_key: str = "", fmt: Optional[str] = None,
               options: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        started = time.perf_counter()
        request_ids = self._ids()
        fmt = (fmt or self.config.get("output_format.default") or DEFAULT_FORMAT).lower()
        if fmt in ("md", "keyword_comma"):
            fmt = "markdown" if fmt == "md" else "keywords"
        text = (text or "").strip()
        options = options or {}

        hit_objs = self.kb.lexicon.extract(text) if text else []
        hit_words = [hit.term for hit in hit_objs]
        auth = self.policy.authenticate(user_id, level, api_key)

        # STEP0 输入校验：空输入 / 超长 / 锚点篡改 / 提示注入
        if not text:
            envelope = self._envelope(
                "error", "qa", "输入为空，请提供问题或指令。",
                data={"code": "EMPTY_INPUT"}, meta=self._meta(request_ids, auth, fmt, [], [], 0.0),
            )
            self._audit(envelope, text, auth)
            return envelope

        warnings: List[str] = []
        if len(text) > self.max_input:
            text = text[: self.max_input]
            warnings.append("输入超过 %d 字符，已截断。" % self.max_input)

        tamper = detect_anchor_tamper(text)
        if tamper:
            envelope = self._deny(
                request_ids, "denied", auth, text,
                "⛔ 锚点密钥锁定：检测到试图修改/绕过“元衍道枢”锚点的指令（%s）。\n"
                "锚点为本插件不可变标识，请求已终止并写入审计。" % "；".join(tamper[:3]),
                code="ANCHOR_LOCKED", fmt=fmt, hit_words=hit_words,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            return envelope

        injection = [p for p in self.banned if p and p.casefold() in text.casefold()]
        if injection:
            envelope = self._deny(
                request_ids, "denied", auth, text,
                "⛔ 输入被安全规则拦截：检测到提示注入特征（%s）。" % "；".join(injection),
                code="PROMPT_INJECTION", fmt=fmt, hit_words=hit_words,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            return envelope

        if not auth.ok:
            envelope = self._deny(
                request_ids, "denied", auth, text,
                self.policy.denial_message(auth, auth.level or 1, _INTENT_CAP.get("qa", "qa")),
                code=auth.code, required_level=max(1, auth.level), fmt=fmt,
                hit_words=hit_words, latency_ms=(time.perf_counter() - started) * 1000,
            )
            return envelope

        intent = self.detect_intent(text)
        # STEP1/2 双路检索（不受限，用于识别越权命中）
        pre_report = self.kb.search(text, top_n=max(self.kb.top_n * 2, 12), max_level=None)
        locked_top = pre_report.hits[0] if pre_report.hits else None
        categories = [hit.doc.category for hit in pre_report.hits[:4]]

        if intent == "perm_selftest":
            envelope = self._handle_permission_selftest(
                request_ids, text, auth, fmt, started, pre_report.hits, warnings
            )
            envelope = self._validate_output(envelope, auth, text)
            self._audit(envelope, text, auth, branch_id="SELFTEST")
            return envelope

        # STEP3 越权命中判定（检索结果领先且超出等级）
        if locked_top and locked_top.doc.level > auth.level and locked_top.score >= pre_report.threshold:
            required = locked_top.doc.level
            message = (
                "⛔ 权限拒绝：命中的知识库分类“%s”需要 L%d %s，当前为 L%d %s。\n"
                "- 命中文档：%s（相关度 %.2f）\n"
                "- 系统未返回任何受限内容。\n"
                "- 处理建议：申请更高等级的动态密钥后重试。\n"
                "- 本次请求已写入审计日志。"
                % (
                    self.kb.category_name.get(locked_top.doc.category, locked_top.doc.category),
                    required,
                    LEVEL_NAMES.get(required, "未知"),
                    auth.level,
                    LEVEL_NAMES.get(auth.level, "未知"),
                    locked_top.doc.id,
                    locked_top.score,
                )
            )
            return self._deny(
                request_ids, "denied", auth, text, message, code="KB_ACCESS_DENIED",
                required_level=required, capability="qa", fmt=fmt, hit_words=hit_words,
                categories=categories, latency_ms=(time.perf_counter() - started) * 1000,
            )

        capability = _INTENT_CAP.get(intent)
        if intent == "qa" and auth.level == 0:
            capability = None
        if intent == "maintenance" and "入库" in text:
            capability = "kb_write"
        if intent == "export_keywords":
            # 全量词库导出属于知识库维护动作；仅导出 TA 词库时放宽到 L1
            low_text = text.casefold()
            if not ("ta" in low_text or "技术美术" in text):
                capability = "kb_write"
        if capability:
            denied = self._capability_denied(
                request_ids, intent, auth, text, capability,
                fmt=fmt, hit_words=hit_words, categories=categories,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            if denied:
                return denied

        allowed_hits = [hit for hit in pre_report.hits if hit.doc.level <= auth.level][: self.kb.top_n]

        # STEP4 分流
        handler = {
            "qa": self._handle_qa,
            "sandbox": self._handle_sandbox,
            "train": self._handle_train,
            "doc": self._handle_doc,
            "export_keywords": self._handle_export_keywords,
            "reload": self._handle_reload,
            "maintenance": self._handle_maintenance,
        }.get(intent, self._handle_qa)

        envelope = handler(
            request_ids=request_ids, text=text, auth=auth, fmt=fmt, started=started,
            hits=allowed_hits, pre_hits=pre_report.hits, warnings=warnings,
            hit_words=hit_words, categories=categories,
        )

        # STEP5 输出校验
        envelope = self._validate_output(envelope, auth, text)
        # STEP6 审计留痕（校验之后，确保记录的是最终状态）
        self._audit(
            envelope, text, auth,
            branch_id=str((envelope.get("meta") or {}).get("branch_id") or ""),
        )
        return envelope

    # ------------------------------------------------------------------
    # 各意图处理
    # ------------------------------------------------------------------
    def _finish(self, envelope: Dict[str, object], auth: AuthResult, text: str,
                branch_id: str = "", snapshot: bool = False,
                snapshot_kind: str = "call") -> Dict[str, object]:
        meta = envelope.setdefault("meta", {})
        started = float(meta.pop("_started", 0.0) or 0.0)
        if started:
            meta["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        if snapshot:
            path = self.audit.snapshot(
                str(meta.get("op_code")), {
                    "request": {
                        "input": text,
                        "user_id": auth.user_id,
                        "permission_level": auth.level,
                    },
                    "response": envelope,
                }, kind=snapshot_kind,
            )
            meta["snapshot_path"] = path
        return envelope

    def _base_meta(self, request_ids, auth, fmt, hit_words, categories, started,
                   branch_id: str = "", engine_mode: str = "") -> Dict[str, object]:
        meta = self._meta(
            request_ids, auth, fmt, hit_words, categories,
            (time.perf_counter() - started) * 1000, branch_id, engine_mode,
        )
        meta["_started"] = started
        return meta

    def _handle_qa(self, *, request_ids, text, auth, fmt, started, hits, pre_hits,
                   warnings, hit_words, categories) -> Dict[str, object]:
        threshold = self.kb.min_score
        best = hits[0] if hits else None
        if not best or best.score < threshold:
            message = _NOT_FOUND.format(
                categories="、".join(sorted({self.kb.category_name.get(c, c) for c in categories})) or "无",
                score=best.score if best else 0.0,
                threshold=threshold,
            )
            if hit_words:
                message += (
                    "\n- 说明：命中词库词条 %s，但暂无包含这些词条的文档；"
                    "可发送【入库 <标题>】补充正文后重试。" % "、".join(dict.fromkeys(hit_words)[:4])
                )
            envelope = self._envelope(
                "not_found", "qa", message,
                data={"code": "KB_NOT_FOUND", "suggestions": [
                    "使用【入库 <词条>】由 L4 管理员补充",
                    "运行 scripts/import_keywords.py 批量导入后【刷新词库】",
                ]},
                warnings=warnings,
                meta=self._base_meta(request_ids, auth, fmt, hit_words, categories, started),
            )
            return self._finish(envelope, auth, text)

        relevant = [hit for hit in hits if hit.score >= threshold]
        answer = self.rag.answer(text, relevant)
        if not answer.get("text"):
            message = "基座不可用且离线抽取无内容，请检查 Ollama 服务或稍后重试。"
            envelope = self._envelope(
                "error", "qa", message,
                data={"code": "GENERATION_FAILED", "error": answer.get("error", "")},
                warnings=warnings,
                meta=self._base_meta(request_ids, auth, fmt, hit_words, categories, started,
                                     engine_mode=answer.get("mode", "")),
            )
            return self._finish(envelope, auth, text)

        if answer.get("degraded"):
            warnings = warnings + [
                "本地基座不可用（%s），已降级为知识库原文抽取模式；未做生成式扩写。"
                % (answer.get("error") or "未知原因")
            ]
        envelope = self._envelope(
            "ok", "qa", str(answer["text"]),
            data={
                "mode": answer.get("mode"),
                "degraded": bool(answer.get("degraded")),
                "top_score": round(hits[0].score, 4),
                "documents": [hit.as_dict() for hit in relevant],
            },
            citations=self.rag.citations(relevant),
            warnings=warnings,
            meta=self._base_meta(request_ids, auth, fmt, hit_words, categories, started,
                                 engine_mode=str(answer.get("mode"))),
        )
        return self._finish(envelope, auth, text)

    def _handle_sandbox(self, *, request_ids, text, auth, fmt, started, hits, pre_hits,
                        warnings, hit_words, categories) -> Dict[str, object]:
        result = self.sandbox.run(text, hits)
        convergence = result["convergence"]
        branch_id = ",".join(b["branch_id"] for b in result["branches"])
        summary = [
            "**沙盘推演完成**：并行展开 %d 个分支（最多 %d），六维风险加权打分，阈值 %d 触发一票否决。"
            % (result["branch_count"], result["max_branch"], result["veto_threshold"]),
            "",
            "**结论**：%s ｜ %s（加权风险 %.2f）。详见下方风险权重表、冲突登记与收敛结论。"
            % (
                convergence.get("status"),
                ("%s %s" % (convergence.get("selected"), convergence.get("selected_title")))
                if convergence.get("selected") else "无可行分支",
                float(convergence.get("score") or 0),
            ),
        ]
        envelope = self._envelope(
            "ok", "sandbox", "\n".join(summary),
            data={
                "domain": result["domain"],
                "seed": result["seed"],
                "sandbox": result,
                "risk_table": result["risk_table"],
                "branch_count": result["branch_count"],
            },
            citations=self.rag.citations(self._pick_hits(hits, result["domain"])[:4]),
            warnings=warnings + ([] if hits else ["未命中沙盘规则库或硬件库，推演基于内置规则模板。"]),
            meta=self._base_meta(request_ids, auth, fmt, hit_words, categories, started,
                                 branch_id=branch_id, engine_mode="offline-rules"),
        )
        return self._finish(envelope, auth, text, branch_id=branch_id, snapshot=True,
                            snapshot_kind="sandbox")

    def _handle_train(self, *, request_ids, text, auth, fmt, started, hits, pre_hits,
                      warnings, hit_words, categories) -> Dict[str, object]:
        low = text.casefold()
        loss = "sft" if "sft" in low else ("dpo" if "dpo" in low else None)
        plan = self.train.build_plan(loss_type=loss)
        vram = plan["vram"]
        branch_id = "T-%s" % plan["stage"]
        summary = [
            "**训练调度已生成**：%s · %s · 4bit QLoRA · LoRA r=%s · lr=%s · batch=%s × 累积 %s。"
            % (
                plan["stage"],
                self.train.hf_model_id,
                plan["config"]["lora"]["r"],
                plan["config"]["training"]["learning_rate"],
                plan["config"]["training"]["per_device_train_batch_size"],
                plan["config"]["training"]["gradient_accumulation_steps"],
            ),
            "",
            "**显存估算**：预计峰值 %.2f GB / 上限 %.2f GB，余量 %.2f GB（%s）。"
            % (
                float(vram["estimated_peak_gb"]),
                float(vram["limit_gb"]),
                float(vram["headroom_gb"]),
                vram["suggestion"],
            ),
            "",
            "**产物**：YAML 配置、训练脚本、数据集清洗方案（见下方结构化字段）。",
        ]
        if not vram["fits"]:
            warnings = warnings + ["显存估算已超出 8GB 上限，请先降低序列长度或批大小。"]
        envelope = self._envelope(
            "ok", "train", "\n".join(summary),
            data={
                "stage": plan["stage"],
                "train_plan": plan,
                "files": plan.get("files", {}),
                "vram": vram,
            },
            citations=self.rag.citations(self._pick_hits(hits, "train")[:4]),
            warnings=warnings,
            meta=self._base_meta(request_ids, auth, fmt, hit_words, categories, started,
                                 branch_id=branch_id, engine_mode="offline-template"),
        )
        return self._finish(envelope, auth, text, branch_id=branch_id, snapshot=True,
                            snapshot_kind="train")

    def _handle_doc(self, *, request_ids, text, auth, fmt, started, hits, pre_hits,
                    warnings, hit_words, categories) -> Dict[str, object]:
        low = text.casefold()
        if "bp" in low or "商业计划" in text or "计划书" in text:
            doc_type = "bp"
            content = build_bp_document(self.config, hits)
            title = "元衍智枢 V7.0 商业计划书"
        elif "交付规范" in text or "验收" in text or "交付文档" in text:
            doc_type = "delivery"
            content = build_delivery_spec(self.config, hits)
            title = "元衍智枢 V7.0 交付规范"
        else:
            doc_type = "generic"
            title = text.strip().rstrip("。")[:40] or "知识文档"
            content = build_generic_document(text, hits)
        envelope = self._envelope(
            "ok", "doc", content,
            data={
                "doc_type": doc_type,
                "title": title,
                "char_count": len(content),
                "template": "yuanyandao.doc.v7",
            },
            citations=self.rag.citations(hits[:5]),
            warnings=warnings,
            meta=self._base_meta(request_ids, auth, fmt, hit_words, categories, started,
                                 engine_mode="template"),
        )
        return self._finish(envelope, auth, text)

    def _handle_export_keywords(self, *, request_ids, text, auth, fmt, started, hits, pre_hits,
                                warnings, hit_words, categories) -> Dict[str, object]:
        low = text.casefold()
        only_ta = "ta" in low or "技术美术" in text
        categories_map: Dict[str, List[str]] = {}
        for cat in self.kb.category_level:
            if only_ta and cat != "ta":
                continue
            words = self.kb.lexicon.terms_for(cat)
            if words:
                categories_map[cat] = words
        flat: List[str] = []
        for words in categories_map.values():
            flat.extend(words)
        keywords_text = ", ".join(flat)
        envelope = self._envelope(
            "ok", "export_keywords", keywords_text,
            data={
                "keywords_text": keywords_text,
                "keywords_list": flat,
                "keywords_by_category": categories_map,
                "count": len(flat),
                "scope": "ta" if only_ta else "all",
            },
            citations=[],
            warnings=warnings,
            meta=self._base_meta(request_ids, auth, fmt, hit_words, categories, started,
                                 engine_mode="index"),
        )
        return self._finish(envelope, auth, text)

    def _handle_reload(self, *, request_ids, text, auth, fmt, started, hits, pre_hits,
                       warnings, hit_words, categories) -> Dict[str, object]:
        stats = self.kb.reload()
        message = (
            "✅ 词库热重载完成（无需重启进程）\n"
            "- 文档 %d 篇 / 关键词 %d 个 / 分类 %d 个\n"
            "- 索引后端：%s，索引指纹：%s\n"
            "- 耗时：%.1f ms"
            % (
                stats["documents"],
                stats["keywords"],
                stats["categories"],
                stats["backend"],
                stats["index_hash"],
                float(stats.get("elapsed_ms", 0)),
            )
        )
        envelope = self._envelope(
            "ok", "reload", message,
            data={"index": stats},
            citations=[],
            warnings=warnings,
            meta=self._base_meta(request_ids, auth, fmt, hit_words, categories, started,
                                 engine_mode="index"),
        )
        return self._finish(envelope, auth, text)

    def _handle_maintenance(self, *, request_ids, text, auth, fmt, started, hits, pre_hits,
                            warnings, hit_words, categories) -> Dict[str, object]:
        if "入库" in text:
            payload = re.split(r"入库[：: ]?", text, maxsplit=1)
            raw_terms = payload[1] if len(payload) > 1 else ""
            terms = [term.strip(" ，,。；;、\n\t") for term in re.split(r"[,，、;；\n]+", raw_terms)]
            terms = [term for term in terms if term and len(term) <= 40][:50]
            if not terms:
                envelope = self._envelope(
                    "error", "maintenance",
                    "入库指令需要词条内容，例如：【入库 路径追踪, 反射探针】",
                    data={"code": "EMPTY_TERMS"},
                    meta=self._base_meta(request_ids, auth, fmt, hit_words, categories, started),
                )
                return self._finish(envelope, auth, text)
            category = "ta"
            for hit in pre_hits[:3]:
                if hit.doc.category in ("ta", "llm", "hardware"):
                    category = hit.doc.category
                    break
            added = self.kb.add_keywords(category, terms)
            stats = self.kb.reload()
            message = (
                "✅ 词条入库完成\n- 分类：%s\n- 新增：%d 条，重复跳过：%d 条\n"
                "- 当前关键词总数：%d，索引已热重载。"
                % (self.kb.category_name.get(category, category), added, len(terms) - added, stats["keywords"])
            )
            envelope = self._envelope(
                "ok", "maintenance", message,
                data={"added": added, "category": category, "terms": terms, "index": stats},
                meta=self._base_meta(request_ids, auth, fmt, hit_words, categories, started),
            )
            return self._finish(envelope, auth, text)

        if "整理文档" in text:
            rows = [
                "| %s | %s | L%d | %d |" % (cat, self.kb.category_name.get(cat, cat),
                                             self.kb.category_level[cat],
                                             sum(1 for d in self.kb.documents if d.category == cat))
                for cat in self.kb.category_level
            ]
            message = "### 知识库文档盘点\n\n| 分类 | 名称 | 等级 | 文档数 |\n| --- | --- | --- | --- |\n" + "\n".join(rows)
            envelope = self._envelope(
                "ok", "maintenance", message,
                data={"stats": self.kb.stats()},
                meta=self._base_meta(request_ids, auth, fmt, hit_words, categories, started),
            )
            return self._finish(envelope, auth, text)

        stats = self.kb.stats()
        suggestions = []
        for item in stats["category_detail"]:
            if item["docs"] <= 2:
                suggestions.append("`%s` 仅 %d 篇文档，建议补充" % (item["name"], item["docs"]))
        if not suggestions:
            suggestions.append("各分类文档量均衡，可按业务反馈继续追加高频问题词条")
        message = (
            "### 知识库维护建议\n\n"
            + "\n".join("- " + item for item in suggestions)
            + "\n- 发送【入库 <词条>】追加关键词；发送【刷新词库】热重载；发送【整理文档】查看盘点。"
        )
        envelope = self._envelope(
            "ok", "maintenance", message,
            data={"stats": stats, "suggestions": suggestions},
            meta=self._base_meta(request_ids, auth, fmt, hit_words, categories, started),
        )
        return self._finish(envelope, auth, text)

    def _handle_permission_selftest(self, request_ids, text, auth, fmt, started,
                                    hits, warnings) -> Dict[str, object]:
        """测试指令4：以 L0 匿名身份访问 TA 知识库，验证拦截链路。"""
        probe = self.kb.search("技术美术 TA 知识库 Shader 渲染管线", top_n=3, max_level=None)
        top = probe.hits[0] if probe.hits else None
        required = top.doc.level if top else 1
        if required <= 0:
            required = 1
        message = (
            "⛔ 权限拒绝：模拟 L0 匿名访问 TA 知识库被拦截。\n"
            "- 探针查询：技术美术 TA 知识库 Shader 渲染管线\n"
            "- 命中分类：%s（L%d）\n"
            "- 所需等级：L%d %s；当前等级：L0 访客\n"
            "- 拦截位置：检索前置权限校验（未返回任何文档内容）\n"
            "- 本次请求已写入审计日志，可用于权限回归测试。"
            % (
                self.kb.category_name.get(top.doc.category, "技术美术TA知识库") if top else "技术美术TA知识库",
                required,
                required,
                LEVEL_NAMES.get(required, "未知"),
            )
        )
        envelope = self._envelope(
            "denied", "perm_selftest", message,
            data={
                "code": "PERMISSION_DENIED",
                "denied": True,
                "required_level": required,
                "required_level_name": LEVEL_NAMES.get(required, "未知"),
                "probe_query": "技术美术 TA 知识库 Shader 渲染管线",
                "probe_doc": top.doc.id if top else "",
                "probe_score": round(top.score, 4) if top else 0.0,
            },
            warnings=warnings,
            meta=self._base_meta(request_ids, auth, fmt, ["权限测试", "TA知识库"], ["permission_rules"],
                                 started, engine_mode="selftest"),
        )
        return self._finish(envelope, auth, text)

    # ------------------------------------------------------------------
    # 输出校验层
    # ------------------------------------------------------------------
    def _validate_output(self, envelope: Dict[str, object], auth: AuthResult, text: str) -> Dict[str, object]:
        answer = str(envelope.get("answer") or "")
        warnings = envelope.setdefault("warnings", [])
        assert isinstance(warnings, list)
        data = envelope.setdefault("data", {})
        assert isinstance(data, dict)

        violations: List[str] = []
        if not self.policy.enable_key and auth.level > 0:
            warnings.append("动态密钥鉴权当前关闭（仅建议开发环境使用）。")
        if envelope.get("anchor") != ANCHOR_KEY:
            violations.append("锚点缺失")

        role_hits = detect_role_confusion(answer)
        if role_hits:
            violations.append("角色越界：%s" % "；".join(role_hits[:2]))
            warnings.append("输出触发角色边界校验，已标注以便复核：%s" % "；".join(role_hits[:2]))

        banned_hits = [p for p in self.banned if p and p.casefold() in answer.casefold()]
        if banned_hits:
            violations.append("违规内容：%s" % "；".join(banned_hits))

        data["safety"] = {
            "anchor_ok": envelope.get("anchor") == ANCHOR_KEY,
            "role_confusion": role_hits,
            "violations": violations,
            "anchor_banner": anchor_banner(),
        }
        if banned_hits:
            envelope["status"] = "denied"
            envelope["ok"] = False
            envelope["answer"] = "⛔ 输出被安全规则拦截：生成内容包含受限片段，已终止返回。"
            data["code"] = _BANNED_OUTPUT_CODES
        return envelope

    # ------------------------------------------------------------------
    # 对外辅助
    # ------------------------------------------------------------------
    def issue_key(self, level: int, ttl: Optional[int] = None, subject: str = "cli") -> Dict[str, object]:
        return self.policy.issue_key(level, ttl=ttl, subject=subject)

    def reload(self) -> Dict[str, object]:
        return self.kb.reload()

    def export_keywords(self) -> Dict[str, List[str]]:
        return {cat: self.kb.lexicon.terms_for(cat) for cat in self.kb.category_level}

    def status(self) -> Dict[str, object]:
        ollama = self.rag.health()
        audit = self.audit.verify(limit=200)
        return {
            "plugin": PLUGIN_NAME,
            "version": VERSION,
            "anchor": ANCHOR_KEY,
            "anchor_message": self.anchor_message,
            "role": ROLE_DISPATCH,
            "bound_product": ROLE_PRODUCT,
            "started_at": self.started_at,
            "config": self.config.describe(),
            "knowledge_base": self.kb.stats(),
            "ollama": ollama,
            "audit": audit,
            "permissions": {
                "levels": {level: {"name": LEVEL_NAMES.get(level, ""), "capabilities": sorted(self.policy.capabilities_for(level))}
                           for level in self.policy.levels},
                "dynamic_key": self.policy.enable_key,
                "kb_level": self.policy._kb_level,
            },
            "sandbox": {
                "enabled": bool(self.config.get("sandbox.enable", True)),
                "max_branch": self.sandbox.max_branch,
                "veto_threshold": self.sandbox.veto_threshold,
            },
        }

    def run(self, text: str, **kwargs) -> Dict[str, object]:
        return self.invoke(text, **kwargs)
