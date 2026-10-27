# -*- coding: utf-8 -*-
"""文档/资产交付模板（BP、交付规范、通用知识文档）。"""

from __future__ import annotations

import time
from typing import Dict, Optional, Sequence

from .vector_store import ScoredDoc

__all__ = ["build_bp_document", "build_delivery_spec", "build_generic_document"]

_TEMPLATE_NOTE = (
    "> 本文件由元衍道枢 V7.0 预设交付模板生成；其中商业数字与市场结论为占位项，"
    "需由项目负责人核定后对外使用。"
)


def build_bp_document(config, hits: Sequence[ScoredDoc] = ()) -> str:
    version = config.get("version", "7.0")
    model = config.get("hardware.base_model", "qwen2.5-7b-base")
    vram = config.get("hardware.vram_limit_gb", 8)
    quant = config.get("hardware.quant_bit", 4)
    engine = config.get("vector_store.engine", "memory")
    top_n = config.get("vector_store.top_n", 6)
    today = time.strftime("%Y-%m-%d")
    refs = "\n".join(
        "- 《%s》 `%s`（%s）" % (hit.doc.title, hit.doc.id, hit.doc.category) for hit in hits[:5]
    ) or "- 暂无引用（本次未命中项目资产库文档）"

    return """# 元衍智枢 V{version} 商业计划书（BP）

> 文档编号：BP-YYZS-V{version}
> 生成日期：{today}
> 生成方：元衍道枢 V{version} 调度内核 · 文档交付模板
{note}

---

## 1. 项目摘要

元衍智枢是一款**轻量化私有 AI 仿真中台**：在单台 RTX5060 8G 工作站上完成知识问答、
方案沙盘推演、交付资产生成与模型微调调度的全链路闭环，所有数据留在内网，
每一次调用都可审计、可回溯。

一句话定位：**把“大模型能力”装进一台可控、可审计、可交付的机器里。**

## 2. 问题与机会

| 现状痛点 | 具体表现 | 元衍智枢的应对 |
| --- | --- | --- |
| 数据不能出内网 | 云端 API 无法用于涉密与内部资产 | 全本地部署，断网可用 |
| 通用模型答非所问 | 缺少领域知识，容易编造 | RAG 双路检索 + 未收录固定话术 |
| 方案讨论不可追溯 | 结论散落在聊天记录 | 沙盘分支快照 + 哈希链审计 |
| 部署成本高 | 动辄多卡服务器 | 单卡 {vram}G 4bit QLoRA 闭环 |
| 权限形同虚设 | 一套账号看全部知识 | 五级权限 + 动态密钥 + 知识库隔离 |

## 3. 产品定义与边界

- **元衍智枢**：面向终端用户的业务中台，提供问答、推演、交付、训练四个入口。
- **元衍道枢**：其内部的调度内核，负责检索、鉴权、路由、校验与留痕，不直接面向终端。
- 二者是“产品 / 内核”关系，任何场景不得混用。

明确不做：公网多租户 SaaS、通用开放域闲聊、未经审核的自动对外发布。

## 4. 目标用户与典型场景

| 用户 | 场景 | 价值 |
| --- | --- | --- |
| 技术美术 / 内容团队 | 渲染管线、资产规范、性能预算问答 | 减少查文档与反复问人 |
| 算法 / 平台工程师 | 8G 单卡微调方案与排错 | 少走显存与超参弯路 |
| 项目负责人 | 技术路线沙盘推演与风险打分 | 决策有依据、可复盘 |
| 交付 / 售前 | BP、交付规范、验收清单 | 文档产出标准化 |

## 5. 解决方案架构

```text
用户 / 上层业务
      │
      ▼
元衍智枢（业务中台：入口、会话、呈现）
      │  invoke
      ▼
元衍道枢（调度内核）
  ① 双路检索（关键词 + 向量）  ② 五级权限校验
  ③ 知识库路由 Top-{top_n}     ④ 意图分流
  ⑤ 输出校验（锚点/角色/幻觉） ⑥ 审计与快照
      │
      ├── 本地基座：{model}（4bit QLoRA）
      ├── 向量库：{engine}
      └── 审计：JSONL 哈希链 + 分支快照
```

## 6. 核心能力

1. **双路 RAG**：关键词精确命中 + 语义向量召回，支持词库热重载。
2. **多分支沙盘**：最多 5 路并行推演，六维风险加权打分，冲突自动收敛。
3. **训练调度**：内置 {vram}G 单卡 {quant}bit QLoRA 模板，产出 YAML、脚本与数据清洗方案。
4. **五级权限**：访客到内核管理员，知识库按等级隔离，动态密钥带签名与有效期。
5. **全链路审计**：输入、输出、命中词条、引用文档、分支快照、操作编号全留痕。

## 7. 技术实现要点

- 基座：`{model}`，NF4 双重量化 + LoRA（r=8）在 {vram}GB 显存内完成微调。
- 检索：{engine} 索引 + BM25，Top-N={top_n}，低于阈值直接判定“未收录”。
- 推理：Ollama 本地服务，不可用时自动降级为知识库原文抽取，不编造。
- 部署：单机 Docker / 原生 Python 双路径，一条命令启动。

## 8. 交付形态

| 形态 | 内容 | 适用 |
| --- | --- | --- |
| 单机版 | 安装包 + 知识库 + 模型权重 | 个人与小团队 |
| 内网版 | 服务端 + 权限矩阵 + 审计对接 | 企业内网 |
| 定制版 | 专属词库 + 交付模板 + 培训 | 行业客户 |

## 9. 商业模式（占位，需核定）

- 私有化授权：按节点 / 年授权。
- 知识与模板服务：词库建设、交付文档定制。
- 训练与调优服务：领域数据清洗、QLoRA 微调、效果评测。

## 10. 里程碑

| 阶段 | 目标 | 验收标准 |
| --- | --- | --- |
| M1（2 周） | 内核跑通 | 六条内置指令全部通过 |
| M2（4 周） | 中台可用 | 问答 / 推演 / 交付三链路联通 |
| M3（6 周） | 训练闭环 | 8G 单卡完成一次领域微调 |
| M4（8 周） | 交付就绪 | 部署文档 + 测试报告 + 权限审计证据 |

## 11. 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| 8G 显存边界 | 训练参数受限 | 4bit + r=8 + 短序列，先跑通再扩量 |
| 知识库覆盖不足 | 答“未收录” | 建立入库流程与词库评审 |
| 基座能力上限 | 复杂推理弱 | 限定场景 + 引用约束 + 人工复核 |
| 权限配置错误 | 越权访问 | 默认最小权限 + 审计告警 |

## 12. 成功指标

- 知识问答命中率与引用可追溯率
- 沙盘推演结论被采纳比例
- 单卡训练一次跑通率
- 审计链路完整率（应为 100%）

## 13. 引用与依据

{refs}

---
_Template: yuanyandao.bp.v{version} · 生成于 {today}_
""".format(
        version=version,
        today=today,
        note=_TEMPLATE_NOTE,
        vram=vram,
        model=model,
        quant=quant,
        engine=engine,
        top_n=top_n,
        refs=refs,
    )


def build_delivery_spec(config, hits: Sequence[ScoredDoc] = ()) -> str:
    version = config.get("version", "7.0")
    today = time.strftime("%Y-%m-%d")
    return """# 元衍智枢 V{version} 交付规范

> 生成日期：{today}　生成方：元衍道枢交付模板
{note}

## 1. 交付物清单

| 编号 | 交付物 | 形式 | 验收人 |
| --- | --- | --- | --- |
| D1 | 插件主程序与配置 | 代码仓库 / 压缩包 | 技术负责人 |
| D2 | 知识库与词库 | kb_seed.json + 增量词条 | 内容负责人 |
| D3 | 权限矩阵 | config.yaml + 密钥签发记录 | 安全负责人 |
| D4 | 部署文档 | README + 一键启动脚本 | 运维 |
| D5 | 测试报告 | test_cases_v7.json + 执行结果 | 测试 |
| D6 | 审计样例 | audit/log.jsonl + 快照 | 合规 |

## 2. 验收标准

1. 六条内置调试指令全部返回预期结果。
2. 五级权限拦截有可复现用例与日志证据。
3. 审计日志哈希链校验通过。
4. 8G 单卡完成一次 dry-run 训练。
5. 断网环境下问答仍可用（抽取模式）。

## 3. 部署步骤

```bash
python -m yuanyandao --health
python -m yuanyandao serve --port 18080
python -m yuanyandao "导出全部TA关键词" --level 4 --key <KEY>
```

## 4. 交付后支持

- 词库增量：`python scripts/import_keywords.py --category ta --file new_terms.txt`
- 索引热重载：发送“刷新词库”指令
- 审计导出：读取 audit/log.jsonl 与 audit/snapshot/

---
_Template: yuanyandao.delivery.v{version}_
""".format(version=version, today=today, note=_TEMPLATE_NOTE)


def build_generic_document(query: str, hits: Sequence[ScoredDoc] = ()) -> str:
    today = time.strftime("%Y-%m-%d")
    lines = [
        "# %s" % query.strip().rstrip("。"),
        "",
        "> 生成日期：%s　生成方：元衍道枢 V7.0 文档交付模板" % today,
        "",
        "## 1. 背景与目标",
        "",
        "本文件根据知识库中检索到的资料整理，用于沉淀结论与引用依据。",
        "",
        "## 2. 要点整理",
        "",
    ]
    if not hits:
        lines.append("- 知识库未收录相关内容，请先新建词条或补充文档。")
    for idx, hit in enumerate(hits[:6], start=1):
        lines.append("%d. **%s**（%s）" % (idx, hit.doc.title, hit.doc.id))
        lines.append("   %s" % hit.doc.content)
        lines.append("")
    lines.extend(
        [
            "## 3. 结论与下一步",
            "",
            "- 本文所有要点均可回溯到上方引用文档；未覆盖部分标注为待补充。",
            "- 待补充：请由业务方确认口径后写入知识库，再重新生成。",
            "",
            "---",
            "_Template: yuanyandao.doc.v7_",
        ]
    )
    return "\n".join(lines)
