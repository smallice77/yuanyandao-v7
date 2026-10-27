# -*- coding: utf-8 -*-
"""文本向量化：默认零依赖哈希向量器，可选 Ollama 真嵌入模型。

哈希向量器把字符 n-gram、词典词、同义词概念一起哈希到固定维度，
签名为伪随机 ±1，做 L2 归一化后用余弦相似度检索。它不是语义模型，
但对同义改写、术语变体有稳定召回，且完全离线、可复现。
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

from .keywords import normalize, tokenize

__all__ = ["BaseEmbedder", "HashingEmbedder", "OllamaEmbedder", "SYNONYM_GROUPS", "build_embedder"]

SYNONYM_GROUPS: List[Sequence[str]] = [
    ("显存", "vram", "gpu内存", "显存占用", "video memory"),
    ("显存不足", "oom", "out of memory", "爆显存", "显存溢出"),
    ("微调", "finetune", "fine-tune", "微调训练"),
    ("量化", "quantization", "quant", "4bit", "nf4"),
    ("沙盘推演", "推演", "仿真", "simulation", "what-if"),
    ("检索", "retrieval", "召回", "rag", "向量检索"),
    ("权限", "permission", "鉴权", "auth", "权限校验"),
    ("审计", "audit", "日志", "留痕", "审计日志"),
    ("训练", "train", "training", "训练调度"),
    ("渲染", "rendering", "render"),
    ("材质", "material", "pbr"),
    ("基座", "base model", "基座模型", "foundation model"),
    ("数据集", "dataset", "语料", "数据"),
    ("推理", "inference", "推理服务"),
    ("收敛", "convergence", "收敛结论"),
    ("风险", "risk", "风险打分"),
    ("降级", "fallback", "兜底"),
    ("向量", "vector", "embedding", "嵌入"),
]

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def _feature_vector(text: str, dim: int) -> List[float]:
    norm = normalize(text)
    features: Counter = Counter()

    for token in tokenize(text):
        features[token] += 1.0

    # 中文三字组补充细粒度特征
    for run in _CJK_RE.findall(norm):
        for i in range(len(run) - 2):
            features["3g:" + run[i:i + 3]] += 0.5

    # 同义词概念桥接
    for group in SYNONYM_GROUPS:
        canonical = group[0]
        if any(variant in norm for variant in group):
            features["syn:" + canonical] += 1.5

    vector = [0.0] * dim
    for feature, count in features.items():
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % dim
        sign = -1.0 if (value >> 63) & 1 else 1.0
        vector[index] += sign * (1.0 + math.log(count))
    return vector


def _l2_normalize(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 1e-12:
        return vector
    return [v / norm for v in vector]


class BaseEmbedder:
    name = "base"
    dim = 512
    degraded = False

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError

    def describe(self) -> Dict[str, object]:
        return {"name": self.name, "dim": self.dim, "degraded": self.degraded}


class HashingEmbedder(BaseEmbedder):
    name = "hash-ngram-v1"

    def __init__(self, dim: int = 512):
        self.dim = int(dim)
        self.degraded = False

    def embed(self, text: str) -> List[float]:
        return _l2_normalize(_feature_vector(text or "", self.dim))


class OllamaEmbedder(BaseEmbedder):
    """调用本地 Ollama 生成语义向量；失败自动回落到哈希向量器。"""

    name = "ollama"

    def __init__(self, endpoint: str, model: str, timeout: float = 10.0, fallback_dim: int = 512):
        from .net import HttpError, post_json

        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._post_json = post_json
        self._http_error = HttpError
        self._fallback = HashingEmbedder(fallback_dim)
        self.dim = fallback_dim
        self.degraded = True
        self._probe()

    def _probe(self) -> None:
        vector = self._call("ping")
        if vector:
            self.dim = len(vector)
            self.degraded = False

    def _call(self, text: str) -> Optional[List[float]]:
        try:
            data = self._post_json(
                self.endpoint + "/api/embeddings",
                {"model": self.model, "prompt": text},
                timeout=self.timeout,
            )
            vector = data.get("embedding")
            if isinstance(vector, list) and vector:
                return [float(x) for x in vector]
        except Exception:
            pass
        try:
            data = self._post_json(
                self.endpoint + "/api/embed",
                {"model": self.model, "input": text},
                timeout=self.timeout,
            )
            vectors = data.get("embeddings")
            if isinstance(vectors, list) and vectors and isinstance(vectors[0], list):
                return [float(x) for x in vectors[0]]
        except Exception:
            pass
        return None

    def embed(self, text: str) -> List[float]:
        vector = self._call(text or "")
        if vector is None:
            self.degraded = True
            return self._fallback.embed(text)
        self.degraded = False
        return _l2_normalize(vector)

    def describe(self) -> Dict[str, object]:
        info = super().describe()
        info.update({"model": self.model, "endpoint": self.endpoint})
        return info


def build_embedder(config) -> BaseEmbedder:
    provider = str(config.get("vector_store.embed_provider", "hash")).lower()
    dim = int(config.get("vector_store.embed_dim", 512))
    if provider == "ollama":
        return OllamaEmbedder(
            endpoint=str(config.get("hardware.ollama_endpoint", "http://127.0.0.1:11434")),
            model=str(config.get("hardware.embed_model", "nomic-embed-text")),
            timeout=float(config.get("hardware.health_timeout_s", 3)),
            fallback_dim=dim,
        )
    return HashingEmbedder(dim)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))
