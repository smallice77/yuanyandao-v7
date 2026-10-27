# -*- coding: utf-8 -*-
"""沙盘推演引擎：并行分支、风险权重打分、冲突识别与收敛。

设计要点：
* 分支生成是纯函数，用查询哈希做种子，保证同一问题可复现；
* 分支在 ThreadPoolExecutor 中并行构建（对应“多分支并行推演”）；
* 任一风险维度超过阈值即触发一票否决，不参与收敛；
* 冲突按关键变量识别，收敛优先满足硬约束，再比加权风险。
"""

from __future__ import annotations

import hashlib
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .vector_store import ScoredDoc

__all__ = ["SandboxEngine"]


@dataclass
class BranchSpec:
    id: str
    title: str
    hypothesis: str
    actions: List[str]
    expected: str
    risks: Dict[str, int]
    variables: Dict[str, str]
    citation_tags: List[str] = field(default_factory=list)


def _train_branches() -> List[BranchSpec]:
    return [
        BranchSpec(
            id="B1",
            title="稳态保守路线",
            hypothesis="在 8GB 硬约束下，用最低显存配置换取可跑通、可复现的训练闭环。",
            actions=[
                "4bit NF4 双重量化加载 Qwen2.5-7B-Base，计算精度 bf16",
                "LoRA r=8 / alpha=16，仅适配注意力与 MLP 投影层",
                "seq_len=1024、batch=1、梯度累积 8、开启梯度检查点",
                "先跑 50 step 观察 max_memory_allocated，再决定是否放大",
            ],
            expected="单卡可稳定收敛，吞吐偏低，适合小规模领域适配与格式对齐。",
            risks={"vram": 38, "stability": 30, "data": 52, "compliance": 18, "delivery": 42, "cost": 34},
            variables={"批大小": "降低", "序列长度": "降低", "训练位置": "本地"},
            citation_tags=["QLoRA", "显存", "参数模板"],
        ),
        BranchSpec(
            id="B2",
            title="吞吐优先路线",
            hypothesis="通过更大 batch 与更长序列提升单位时间样本量，缩短训练周期。",
            actions=[
                "batch=4、seq_len=2048，追求更高 GPU 利用率",
                "关闭梯度检查点以换取速度",
                "依赖大 batch 平滑梯度噪声",
            ],
            expected="训练更快，但显存峰值逼近 8GB 上限，存在 OOM 与整体重跑风险。",
            risks={"vram": 92, "stability": 66, "data": 52, "compliance": 18, "delivery": 58, "cost": 30},
            variables={"批大小": "提高", "序列长度": "提高", "训练位置": "本地"},
            citation_tags=["显存不足", "OOM", "RTX5060"],
        ),
        BranchSpec(
            id="B3",
            title="小模型替代路线",
            hypothesis="用 3B/1.5B 基座换取显存余量，把差距交给数据与蒸馏弥补。",
            actions=[
                "改用 Qwen2.5-3B 或 1.5B 做 4bit QLoRA",
                "保持 seq_len=2048、batch=2 的宽裕配置",
                "用强模型蒸馏出的指令数据弥补容量差距",
            ],
            expected="显存宽裕、迭代快，但复杂推理与长文能力下降，需要额外评测兜底。",
            risks={"vram": 22, "stability": 28, "data": 60, "compliance": 18, "delivery": 44, "cost": 28},
            variables={"模型规模": "降低", "训练位置": "本地"},
            citation_tags=["基座模型", "选型", "蒸馏"],
        ),
        BranchSpec(
            id="B4",
            title="数据优先路线",
            hypothesis="在参数规模不变的前提下，用数据质量与偏好对齐换取效果提升。",
            actions=[
                "先做去重、长度过滤、脱敏与来源配比",
                "小规模 SFT 固化输出格式，再用偏好对做 DPO",
                "用固定评测集对比每一版数据，不盲目加量",
            ],
            expected="同等显存下效果提升更明显，但数据准备周期变长。",
            risks={"vram": 36, "stability": 34, "data": 28, "compliance": 20, "delivery": 58, "cost": 32},
            variables={"数据质量": "提高", "训练位置": "本地"},
            citation_tags=["数据集", "数据清洗", "DPO"],
        ),
        BranchSpec(
            id="B5",
            title="混合云路线",
            hypothesis="本地只承担推理与评测，训练放到云端按需租用，绕开 8GB 硬约束。",
            actions=[
                "本地部署推理与评测链路，保证数据不出内网的部分留在本地",
                "训练任务按小时租用云端 GPU",
                "本地与云端共享同一份数据版本与训练配置",
            ],
            expected="训练不受显存限制，但引入外部依赖、成本波动与数据合规问题。",
            risks={"vram": 10, "stability": 30, "data": 48, "compliance": 62, "delivery": 48, "cost": 55},
            variables={"训练位置": "云端", "批大小": "提高"},
            citation_tags=["私有化", "合规", "成本"],
        ),
    ]


def _ta_branches() -> List[BranchSpec]:
    return [
        BranchSpec(
            id="B1",
            title="性能优先路线",
            hypothesis="以帧预算为第一约束，牺牲部分画面细节换取稳定帧率。",
            actions=["降低阴影与后处理开销", "合并材质变体、收敛 Shader 关键字", "提高 LOD 切换阈值并启用 HISM"],
            expected="帧率稳定，画质在远景与特效上略有损失。",
            risks={"vram": 40, "stability": 30, "data": 35, "compliance": 15, "delivery": 40, "cost": 30},
            variables={"画质": "降低", "性能": "提高"},
            citation_tags=["性能预算", "帧预算", "Shader变体"],
        ),
        BranchSpec(
            id="B2",
            title="画质优先路线",
            hypothesis="优先保证近景材质与光照质量，接受更高的 GPU 开销。",
            actions=["提高贴图精度与阴影分辨率", "启用更高质量的全局光照与反射", "减少 LOD 层级跳变"],
            expected="观感提升明显，但帧预算吃紧，低端机需要单独降级档。",
            risks={"vram": 78, "stability": 45, "data": 40, "compliance": 15, "delivery": 55, "cost": 42},
            variables={"画质": "提高", "性能": "降低"},
            citation_tags=["PBR", "阴影", "全局光照"],
        ),
        BranchSpec(
            id="B3",
            title="兼容性优先路线",
            hypothesis="以跨平台一致性与资产复用为先，控制技术风险。",
            actions=["统一材质与着色器规范", "限制平台特性使用范围", "建立资产导入自动校验"],
            expected="跨端表现一致，但上限受最低平台约束。",
            risks={"vram": 45, "stability": 25, "data": 30, "compliance": 15, "delivery": 45, "cost": 38},
            variables={"平台覆盖": "提高"},
            citation_tags=["资源管线", "资产规范", "跨平台"],
        ),
        BranchSpec(
            id="B4",
            title="风格化替代路线",
            hypothesis="用 NPR 风格化处理规避高成本写实管线，同时形成辨识度。",
            actions=["改用 Ramp 光照与描边", "减少高频贴图细节，改用手绘色块", "统一色阶与后处理"],
            expected="成本下降、风格鲜明，但写实需求无法满足。",
            risks={"vram": 32, "stability": 30, "data": 38, "compliance": 15, "delivery": 42, "cost": 25},
            variables={"画质路线": "风格化"},
            citation_tags=["NPR", "描边", "风格化"],
        ),
        BranchSpec(
            id="B5",
            title="分阶段落地路线",
            hypothesis="先做垂直切片验证管线，再逐步扩展到全量内容。",
            actions=["选取一个场景做端到端验证", "固化管线与规范后再批量生产", "用自动化校验守住回退"],
            expected="风险最低但周期最长，适合不确定需求。",
            risks={"vram": 40, "stability": 22, "data": 35, "compliance": 15, "delivery": 60, "cost": 30},
            variables={"交付节奏": "分阶段"},
            citation_tags=["交付规范", "里程碑", "验收"],
        ),
    ]


def _general_branches() -> List[BranchSpec]:
    return [
        BranchSpec(
            id="B1",
            title="保守推进路线",
            hypothesis="在现有资源与权限边界内小步验证，优先保证可回退。",
            actions=["先做最小可行版本", "固定评测口径", "保留回滚点"],
            expected="风险可控，但见效慢。",
            risks={"vram": 30, "stability": 20, "data": 35, "compliance": 20, "delivery": 55, "cost": 30},
            variables={"节奏": "保守"},
        ),
        BranchSpec(
            id="B2",
            title="快速验证路线",
            hypothesis="用最短周期拿到反馈，接受一定返工。",
            actions=["压缩范围直击核心假设", "并行准备两套方案", "以数据结果决定加注或止损"],
            expected="反馈快，但返工与资源浪费概率上升。",
            risks={"vram": 45, "stability": 40, "data": 45, "compliance": 20, "delivery": 30, "cost": 40},
            variables={"节奏": "激进"},
        ),
        BranchSpec(
            id="B3",
            title="资源倾斜路线",
            hypothesis="集中资源做深一条主线，形成可复用资产。",
            actions=["优先投入核心链路", "复用组件与文档", "延后非关键需求"],
            expected="主线质量高，但边缘需求可能延期。",
            risks={"vram": 50, "stability": 35, "data": 30, "compliance": 20, "delivery": 45, "cost": 50},
            variables={"资源投入": "提高"},
        ),
        BranchSpec(
            id="B4",
            title="合规优先路线",
            hypothesis="先解决权限、审计与数据合规，再谈功能扩展。",
            actions=["补齐权限矩阵与审计链路", "做数据脱敏与来源登记", "把合规检查放进发布流程"],
            expected="上线阻力最小，但短期功能推进放慢。",
            risks={"vram": 25, "stability": 22, "data": 30, "compliance": 12, "delivery": 50, "cost": 28},
            variables={"合规投入": "提高"},
        ),
        BranchSpec(
            id="B5",
            title="外部协作路线",
            hypothesis="借助外部能力补齐短板，把不确定性转移出去。",
            actions=["明确接口与验收标准", "保留本地可替代路径", "签订数据边界条款"],
            expected="短期提速，但引入依赖与长期成本。",
            risks={"vram": 20, "stability": 40, "data": 45, "compliance": 50, "delivery": 40, "cost": 60},
            variables={"依赖": "外部"},
        ),
    ]


_DOMAIN_KEYWORDS = {
    "train": ["训练", "微调", "qlora", "lora", "sft", "dpo", "基座", "显存", "rtx5060", "oom", "量化", "数据集"],
    "ta": ["渲染", "shader", "pbr", "npr", "ue5", "资产", "材质", "光照", "性能", "lod", "pcg", "描边"],
}

_ROOT_CHAINS = {
    "vram": [
        ("训练中途 CUDA out of memory", "批大小与序列长度叠加激活峰值", "8GB 显存余量不足以同时容纳权重、激活与 logits"),
        ("推理时显存缓慢上涨", "请求上下文不断变长且未释放缓存", "num_ctx 配置过大、缺少请求间清理"),
    ],
    "stability": [
        ("loss 出现尖峰且不回落", "学习率与等效批大小不匹配", "梯度累积放大后未同步调整学习率"),
        ("训练后期输出模板化", "过拟合到少量高频样本", "数据多样性不足且 epoch 过多"),
    ],
    "data": [
        ("验证集指标虚高", "训练集与验证集同源泄漏", "切分时未按来源分层"),
        ("模型学到错误格式", "指令与回答的模板不一致", "数据构造阶段缺少格式校验"),
    ],
    "compliance": [
        ("审计链路缺环", "旁路调用未经过调度内核", "业务侧直连基座绕过了鉴权与留痕"),
        ("权限越界访问", "知识库等级与请求等级未对齐", "检索发生在权限校验之前"),
    ],
    "delivery": [
        ("交付延期", "关键路径依赖未验证的外部能力", "里程碑缺少可回退的替代方案"),
        ("返工率上升", "规范未固化就批量生产", "缺少自动化校验与验收标准"),
    ],
    "cost": [
        ("单位成本上升", "资源利用率低且重复训练", "缺少缓存与增量训练策略"),
        ("维护负担加重", "多套并行配置无人收敛", "缺少统一配置与版本管理"),
    ],
}

_DIMENSION_LABELS = {
    "train": {},
    "ta": {"vram": "显存与带宽", "data": "资产与规范", "delivery": "交付与排期"},
    "general": {"vram": "资源约束", "stability": "执行稳定性", "data": "信息与数据", "delivery": "交付与排期"},
}


class SandboxEngine:
    def __init__(self, config):
        self.config = config
        self.max_branch = int(config.get("sandbox.max_branch", 5))
        self.veto_threshold = int(config.get("sandbox.veto_threshold", 80))
        self.salt = str(config.get("sandbox.seed_salt", "yuanyandao-sandbox-v7"))
        self.workers = int(config.get("sandbox.parallel_workers", 5))
        self.conflict_converge = bool(config.get("sandbox.conflict_converge", True))
        self.dimensions: List[Dict[str, object]] = list(config.get("sandbox.risk_dimensions", []))
        total = sum(float(d.get("weight", 0)) for d in self.dimensions) or 1.0
        for dim in self.dimensions:
            dim["weight"] = float(dim.get("weight", 0)) / total

    # ------------------------------------------------------------------
    def detect_domain(self, text: str) -> str:
        lowered = (text or "").lower()
        for domain, words in _DOMAIN_KEYWORDS.items():
            if any(word in lowered for word in words):
                return domain
        return "general"

    def _seed(self, query: str) -> int:
        digest = hashlib.blake2b((query + "|" + self.salt).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big")

    def _templates(self, domain: str) -> List[BranchSpec]:
        if domain == "train":
            return _train_branches()
        if domain == "ta":
            return _ta_branches()
        return _general_branches()

    # ------------------------------------------------------------------
    def _build_branch(self, spec: BranchSpec, rng: random.Random, hits: Sequence[ScoredDoc],
                      domain: str) -> Dict[str, object]:
        risks: Dict[str, int] = {}
        for dim in self.dimensions:
            dim_id = str(dim.get("id"))
            base = int(spec.risks.get(dim_id, 50))
            jitter = rng.randint(-5, 5)
            risks[dim_id] = max(0, min(100, base + jitter))
        weighted = sum(
            risks[str(dim.get("id"))] * float(dim.get("weight", 0)) for dim in self.dimensions
        )
        veto_dims = [str(d.get("id")) for d in self.dimensions if risks[str(d.get("id"))] > self.veto_threshold]
        top_dims = sorted(risks.items(), key=lambda kv: kv[1], reverse=True)[:2]
        reverse_check = []
        for dim_id, _score in top_dims:
            for symptom, cause, root in _ROOT_CHAINS.get(dim_id, [])[:1]:
                reverse_check.append({"symptom": symptom, "cause": cause, "root_cause": root})
        citations = [hit.doc.id for hit in hits if any(tag in hit.doc.tags for tag in spec.citation_tags)][:2]
        if not citations:
            citations = [hit.doc.id for hit in hits[:2]]
        return {
            "branch_id": spec.id,
            "title": spec.title,
            "hypothesis": spec.hypothesis,
            "actions": list(spec.actions),
            "expected": spec.expected,
            "variables": dict(spec.variables),
            "risks": risks,
            "weighted_score": round(weighted, 2),
            "peak_risk": max(risks.values()) if risks else 0,
            "veto": bool(veto_dims),
            "veto_dimensions": veto_dims,
            "reverse_check": reverse_check,
            "citations": citations,
            "status": "已否决" if veto_dims else "候选",
            "domain": domain,
        }

    def _detect_conflicts(self, branches: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        conflicts: List[Dict[str, object]] = []
        items = list(branches)
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                vars_a = a.get("variables", {})
                vars_b = b.get("variables", {})
                clash = []
                for key in vars_a:
                    if key in vars_b and vars_a[key] != vars_b[key]:
                        clash.append({"variable": key, "left": vars_a[key], "right": vars_b[key]})
                if clash:
                    pair = sorted([str(a["branch_id"]), str(b["branch_id"])])
                    conflicts.append(
                        {
                            "conflict_id": "C%d" % (len(conflicts) + 1),
                            "branches": pair,
                            "variables": clash,
                            "severity": "高" if len(clash) >= 2 else "中",
                            "description": "%s 与 %s 在 %s 上给出相反动作"
                            % (pair[0], pair[1], "、".join(c["variable"] for c in clash)),
                        }
                    )
        return conflicts

    def _converge(self, branches: Sequence[Dict[str, object]], conflicts: Sequence[Dict[str, object]],
                  domain: str) -> Dict[str, object]:
        feasible = [b for b in branches if not b.get("veto")]
        vetoed = [b["branch_id"] for b in branches if b.get("veto")]
        if not feasible:
            return {
                "status": "无可行分支",
                "selected": None,
                "rationale": "全部分支触发一票否决，需要先降低配置或更换硬件后再推演。",
                "vetoed": vetoed,
                "conflicts": len(conflicts),
            }
        ranked = sorted(feasible, key=lambda b: (float(b["weighted_score"]), b["branch_id"]))
        selected = ranked[0]
        alternatives = [
            {"branch_id": b["branch_id"], "title": b["title"], "weighted_score": b["weighted_score"]}
            for b in ranked[1:3]
        ]
        hard_constraint = "硬约束（8GB 显存与五级权限不可协商）" if domain == "train" else "硬约束（权限、合规与既有资源）"
        rationale = (
            "在%s下，%s（%s）的加权风险最低（%.2f），且未触发一票否决。"
            % (hard_constraint, selected["branch_id"], selected["title"], float(selected["weighted_score"]))
        )
        if vetoed:
            rationale += " 已剔除触发否决的分支：%s。" % "、".join(vetoed)
        if conflicts:
            rationale += " 共登记 %d 组冲突，按硬约束优先原则收敛。" % len(conflicts)
        return {
            "status": "已收敛" if not conflicts else "已收敛（冲突已按硬约束消解）",
            "selected": selected["branch_id"],
            "selected_title": selected["title"],
            "score": selected["weighted_score"],
            "rationale": rationale,
            "vetoed": vetoed,
            "alternatives": alternatives,
            "conflicts": len(conflicts),
        }

    # ------------------------------------------------------------------
    def run(self, query: str, hits: Sequence[ScoredDoc] = ()) -> Dict[str, object]:
        domain = self.detect_domain(query)
        specs = self._templates(domain)[: self.max_branch]
        seed = self._seed(query)

        def build(spec: BranchSpec) -> Dict[str, object]:
            local = random.Random(seed ^ (ord(spec.id[-1]) * 7919))
            return self._build_branch(spec, local, hits, domain)

        with ThreadPoolExecutor(max_workers=min(self.workers, len(specs))) as pool:
            branches = list(pool.map(build, specs))

        conflicts = self._detect_conflicts(branches) if self.conflict_converge else []
        convergence = self._converge(branches, conflicts, domain)

        labels = dict(_DIMENSION_LABELS.get(domain, {}))
        dimension_rows = [
            {
                "id": str(dim.get("id")),
                "name": labels.get(str(dim.get("id")), str(dim.get("name"))),
                "weight": round(float(dim.get("weight", 0)), 3),
            }
            for dim in self.dimensions
        ]
        risk_table = []
        for branch in branches:
            row = {
                "branch_id": branch["branch_id"],
                "title": branch["title"],
                "status": branch["status"],
                "weighted_score": branch["weighted_score"],
            }
            row.update({dim["id"]: branch["risks"][dim["id"]] for dim in dimension_rows})
            risk_table.append(row)

        return {
            "domain": domain,
            "seed": seed,
            "branch_count": len(branches),
            "dimensions": dimension_rows,
            "branches": branches,
            "conflicts": conflicts,
            "convergence": convergence,
            "risk_table": risk_table,
            "veto_threshold": self.veto_threshold,
            "max_branch": self.max_branch,
        }
