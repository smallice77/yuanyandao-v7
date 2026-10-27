# -*- coding: utf-8 -*-
"""知识库与双路检索：BM25 关键词召回 + 向量语义召回 + 词典精确命中加权。

后端支持 memory（零依赖）、faiss（安装后自动启用）与 chroma（可选）。
索引落盘到 vector_db/，便于审计排查与热重载比对。
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .embeddings import BaseEmbedder, cosine
from .keywords import KeywordHit, Lexicon, normalize, tokenize

__all__ = ["Document", "ScoredDoc", "KnowledgeBase", "SearchReport"]


@dataclass
class Document:
    id: str
    title: str
    content: str
    category: str
    level: int
    tags: List[str] = field(default_factory=list)
    source: str = "seed"

    def as_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "category": self.category,
            "level": self.level,
            "tags": list(self.tags),
            "source": self.source,
        }

    @property
    def searchable(self) -> str:
        return " ".join([self.title, self.content, " ".join(self.tags)])


@dataclass
class ScoredDoc:
    doc: Document
    score: float
    bm25: float
    vector: float
    exact: float
    coverage: float = 0.0
    hits: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "doc_id": self.doc.id,
            "title": self.doc.title,
            "category": self.doc.category,
            "level": self.doc.level,
            "score": round(self.score, 4),
            "bm25": round(self.bm25, 4),
            "vector": round(self.vector, 4),
            "coverage": round(self.coverage, 3),
            "matched_terms": self.hits[:8],
        }


@dataclass
class SearchReport:
    query: str
    hits: List[ScoredDoc]
    blocked: Optional[ScoredDoc] = None   # 命中但越权的最高分文档
    threshold: float = 0.0
    elapsed_ms: float = 0.0

    @property
    def best_score(self) -> float:
        return self.hits[0].score if self.hits else 0.0


class KnowledgeBase:
    """知识库 + 混合索引。线程安全（读写锁粒度较粗，索引小、重建快）。"""

    def __init__(self, config, embedder: BaseEmbedder, data_path: Optional[Path] = None,
                 state_path: Optional[Path] = None):
        self.config = config
        self.embedder = embedder
        self.data_path = Path(data_path) if data_path else config.path_for("runtime.data_dir") / "kb_seed.json"
        self.state_path = (
            Path(state_path) if state_path else config.path_for("runtime.state_dir") / "kb_extra.json"
        )
        self.engine = str(config.get("vector_store.engine", "memory")).lower()
        self.k1 = float(config.get("vector_store.bm25_k1", 1.5))
        self.b = float(config.get("vector_store.bm25_b", 0.75))
        self.top_n = int(config.get("vector_store.top_n", 6))
        self.min_score = float(config.get("vector_store.min_score", 0.18))
        self.kw_weight = float(config.get("vector_store.keyword_weight", 0.55))
        self.vec_weight = float(config.get("vector_store.vector_weight", 0.45))
        self.dual = bool(config.get("vector_store.dual_retrieval", True))

        self.documents: List[Document] = []
        self.lexicon = Lexicon()
        self.category_level: Dict[str, int] = {}
        self.category_name: Dict[str, str] = {}
        self._extras: Dict[str, object] = {"keywords": {}, "docs": []}
        self._lock = threading.RLock()
        self._vectors: List[List[float]] = []
        self._faiss = None
        self._backend = "memory"
        self._tf: List[Counter] = []
        self._token_sets: List[set] = []
        self._doc_len: List[int] = []
        self._df: Counter = Counter()
        self._avgdl = 0.0
        self.built_at = 0.0
        self.index_hash = ""
        self.stats_counters = {"searches": 0, "reloads": 0}

        self.load()

    # ------------------------------------------------------------------
    # 装载与构建
    # ------------------------------------------------------------------
    def _read_seed(self) -> Dict[str, object]:
        if not self.data_path.exists():
            raise FileNotFoundError("种子知识库不存在: %s" % self.data_path)
        with self.data_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _read_extras(self) -> Dict[str, object]:
        if not self.state_path.exists():
            return {"keywords": {}, "docs": []}
        try:
            with self.state_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return {"keywords": {}, "docs": []}
        if not isinstance(data, dict):
            return {"keywords": {}, "docs": []}
        data.setdefault("keywords", {})
        data.setdefault("docs", [])
        return data

    def load(self) -> Dict[str, object]:
        started = time.time()
        seed = self._read_seed()
        extras = self._read_extras()
        docs: List[Document] = []
        lexicon = Lexicon()
        category_level: Dict[str, int] = {}
        category_name: Dict[str, str] = {}

        for category in seed.get("categories", []):
            cat_id = str(category.get("id"))
            level = int(category.get("level", 0))
            category_level[cat_id] = level
            category_name[cat_id] = str(category.get("name", cat_id))
            for term in category.get("keywords", []):
                lexicon.add(str(term), cat_id)
            for raw in category.get("docs", []):
                docs.append(
                    Document(
                        id=str(raw.get("id")),
                        title=str(raw.get("title", "")),
                        content=str(raw.get("content", "")),
                        category=cat_id,
                        level=level,
                        tags=[str(t) for t in raw.get("tags", [])],
                        source="seed",
                    )
                )

        keyword_extra = extras.get("keywords", {}) or {}
        for cat_id, terms in keyword_extra.items():
            if cat_id not in category_level:
                continue
            for term in terms:
                lexicon.add(str(term), str(cat_id))

        for raw in extras.get("docs", []) or []:
            cat_id = str(raw.get("category", "public"))
            docs.append(
                Document(
                    id=str(raw.get("id") or "usr-%d" % (len(docs) + 1)),
                    title=str(raw.get("title", "未命名词条")),
                    content=str(raw.get("content", "")),
                    category=cat_id,
                    level=int(raw.get("level", category_level.get(cat_id, 3))),
                    tags=[str(t) for t in raw.get("tags", [])],
                    source=str(raw.get("source", "user")),
                )
            )

        lexicon.finalize()
        self.documents = docs
        self.lexicon = lexicon
        self.category_level = category_level
        self.category_name = category_name
        self._extras = extras

        self._build_bm25(docs)
        self._build_vectors(docs)
        self.built_at = time.time()
        self.index_hash = hashlib.sha256(
            "|".join(d.id + ":" + hashlib.sha1(d.content.encode("utf-8")).hexdigest() for d in docs).encode()
        ).hexdigest()[:16]
        self._persist(seed_version=str(seed.get("version", "")))
        self.stats_counters["reloads"] += 1
        return self.stats(elapsed_ms=(time.time() - started) * 1000)

    def reload(self) -> Dict[str, object]:
        with self._lock:
            if hasattr(self, "_vectors"):
                self._vectors = []
            return self.load()

    # -- BM25 ----------------------------------------------------------
    def _build_bm25(self, docs: Sequence[Document]) -> None:
        self._tf = []
        self._doc_len = []
        self._token_sets = []
        self._df = Counter()
        for doc in docs:
            tokens = tokenize(doc.searchable)
            counts = Counter(tokens)
            self._tf.append(counts)
            self._token_sets.append(set(counts))
            self._doc_len.append(len(tokens))
            for token in counts:
                self._df[token] += 1
        total = sum(self._doc_len)
        self._avgdl = (total / len(docs)) if docs else 0.0

    def _bm25(self, query_tokens: Sequence[str], index: int) -> float:
        if not query_tokens:
            return 0.0
        tf = self._tf[index]
        dl = self._doc_len[index] or 1
        total = len(self.documents)
        score = 0.0
        for token in set(query_tokens):
            freq = tf.get(token, 0)
            if not freq:
                continue
            df = self._df.get(token, 0)
            idf = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
            denom = freq + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1.0))
            score += idf * freq * (self.k1 + 1) / (denom or 1.0)
        return score

    # -- 向量 ----------------------------------------------------------
    def _build_vectors(self, docs: Sequence[Document]) -> None:
        self._vectors = [self.embedder.embed(doc.searchable) for doc in docs]
        self._backend = "memory"
        if self.engine == "faiss":
            try:
                import numpy as np  # type: ignore
                import faiss  # type: ignore

                matrix = np.array(self._vectors, dtype="float32")
                faiss.normalize_L2(matrix)
                index = faiss.IndexFlatIP(matrix.shape[1])
                index.add(matrix)
                self._faiss = index
                self._backend = "faiss"
            except Exception:
                self._faiss = None
                self._backend = "memory"
        elif self.engine == "chroma":
            try:
                import chromadb  # type: ignore

                del chromadb  # 仅探测可用性，检索走内存向量以保证维度一致
                self._backend = "chroma-compatible"
            except Exception:
                self._backend = "memory"

    def _vector_search(self, vector: Sequence[float], limit: int) -> List[tuple]:
        if self._faiss is not None:
            try:
                import numpy as np

                query = np.array([list(vector)], dtype="float32")
                scores, indices = self._faiss.search(query, min(limit, len(self.documents)))
                return [(int(i), float(s)) for s, i in zip(scores[0], indices[0]) if i >= 0]
            except Exception:
                pass
        scored = [(i, cosine(vector, vec)) for i, vec in enumerate(self._vectors)]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def search(self, query: str, top_n: Optional[int] = None, max_level: Optional[int] = None,
               apply_dual: bool = True) -> SearchReport:
        started = time.time()
        top_n = int(top_n or self.top_n)
        with self._lock:
            docs = self.documents
            if not docs:
                return SearchReport(query=query, hits=[], threshold=self.min_score)
            query_tokens = tokenize(query)
            qvec = self.embedder.embed(query)
            pool = min(len(docs), max(top_n * 4, 20))
            vec_pairs = self._vector_search(qvec, pool) if apply_dual else []
            vector_map = {idx: max(0.0, score) for idx, score in vec_pairs}

            keyword_hits: List[KeywordHit] = self.lexicon.extract(query)
            query_terms = {normalize(hit.term) for hit in keyword_hits}

            candidates = set(vector_map.keys())
            for idx in range(len(docs)):
                if self._bm25(query_tokens, idx) > 0:
                    candidates.add(idx)
            if not candidates:
                candidates = set(range(len(docs)))

            scored: List[ScoredDoc] = []
            for idx in candidates:
                doc = docs[idx]
                if max_level is not None and doc.level > max_level:
                    continue
                bm25_raw = self._bm25(query_tokens, idx)
                bm25n = bm25_raw / (bm25_raw + 2.0) if bm25_raw else 0.0
                unique_query_tokens = set(query_tokens)
                if unique_query_tokens and idx < len(self._token_sets):
                    coverage = len(unique_query_tokens & self._token_sets[idx]) / len(unique_query_tokens)
                else:
                    coverage = 0.0
                # 部分命中不应被 BM25 放大：用查询覆盖率做阻尼
                bm25n *= math.sqrt(coverage)
                vec = vector_map.get(idx, 0.0)
                doc_norm = normalize(doc.searchable)
                matched = [hit.term for hit in keyword_hits if normalize(hit.term) in doc_norm]
                exact = (len(set(matched)) / len(query_terms)) if query_terms else 0.0

                score = self.kw_weight * bm25n + self.vec_weight * vec
                if self.dual:
                    score += 0.12 * exact
                if exact == 0.0 and vec < 0.06:
                    score = min(score, 0.05)
                score = min(1.0, score)
                scored.append(
                    ScoredDoc(
                        doc=doc,
                        score=score,
                        bm25=bm25_raw,
                        vector=vec,
                        exact=exact,
                        coverage=coverage,
                        hits=sorted(set(matched), key=len, reverse=True),
                    )
                )

            scored.sort(key=lambda item: (-item.score, item.doc.id))
            self.stats_counters["searches"] += 1
            return SearchReport(
                query=query,
                hits=scored[:top_n],
                threshold=self.min_score,
                elapsed_ms=(time.time() - started) * 1000,
            )

    def detect_blocked(self, report: SearchReport, user_level: int) -> Optional[ScoredDoc]:
        """判断未受限检索的最高分结果是否越权。"""
        if not report.hits:
            return None
        best = report.hits[0]
        if best.doc.level > user_level and best.score >= report.threshold:
            return best
        return None

    def unrestricted_top(self, query: str, top_n: Optional[int] = None) -> Optional[ScoredDoc]:
        report = self.search(query, top_n=top_n, max_level=None)
        if report.hits and report.hits[0].score >= report.threshold:
            return report.hits[0]
        return None

    # ------------------------------------------------------------------
    # 增量维护
    # ------------------------------------------------------------------
    def _save_extras(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self._extras, fh, ensure_ascii=False, indent=2)
        tmp.replace(self.state_path)

    def add_keywords(self, category: str, terms: Iterable[str]) -> int:
        with self._lock:
            if category not in self.category_level:
                raise KeyError("未知知识库分类: %s" % category)
            bucket = self._extras.setdefault("keywords", {})  # type: ignore[union-attr]
            existing = list(bucket.get(category, []))
            known = set(existing) | set(self.lexicon.terms_for(category))
            added = 0
            for term in terms:
                term = str(term).strip()
                if not term or term in known:
                    continue
                existing.append(term)
                known.add(term)
                added += 1
            bucket[category] = existing
            self._save_extras()
            return added

    def add_document(self, title: str, content: str, category: str = "public",
                     tags: Optional[Sequence[str]] = None) -> Document:
        with self._lock:
            if category not in self.category_level:
                raise KeyError("未知知识库分类: %s" % category)
            docs = self._extras.setdefault("docs", [])  # type: ignore[union-attr]
            doc_id = "usr-%03d" % (len(docs) + 1)
            raw = {
                "id": doc_id,
                "title": title.strip() or "未命名词条",
                "content": content.strip(),
                "category": category,
                "level": self.category_level.get(category, 3),
                "tags": list(tags or []),
                "source": "user",
            }
            docs.append(raw)
            self._save_extras()
            return Document(**raw)

    # ------------------------------------------------------------------
    # 落盘与统计
    # ------------------------------------------------------------------
    def _persist(self, seed_version: str) -> None:
        meta_path = self.config.path_for("vector_store.metadata_path")
        index_path = self.config.path_for("vector_store.index_path")
        try:
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            metadata = {
                "schema": "yuanyandao.index.v7",
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "engine_requested": self.engine,
                "backend": self._backend,
                "seed_version": seed_version,
                "embedder": self.embedder.describe(),
                "documents": [doc.as_dict() for doc in self.documents],
                "categories": {
                    cat: {"level": self.category_level[cat], "name": self.category_name.get(cat, cat)}
                    for cat in self.category_level
                },
                "index_hash": self.index_hash,
                "counts": {
                    "documents": len(self.documents),
                    "keywords": len(self.lexicon.all_terms()),
                    "categories": len(self.category_level),
                },
            }
            tmp = meta_path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(metadata, fh, ensure_ascii=False, indent=2)
            tmp.replace(meta_path)

            if self._backend == "faiss":
                index_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    import faiss  # type: ignore

                    faiss.write_index(self._faiss, str(index_path))
                except Exception:
                    pass
            else:
                # memory 后端落盘为 JSON，避免用 .faiss 扩展名造成误解
                index_path = index_path.with_suffix(".json")
                index_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "dim": len(self._vectors[0]) if self._vectors else int(self.config.get("vector_store.embed_dim", 512)),
                    "backend": self._backend,
                    "embedder": self.embedder.describe(),
                    "doc_ids": [doc.id for doc in self.documents],
                    "vectors": [[round(v, 6) for v in vec] for vec in self._vectors],
                }
                tmp_idx = index_path.with_suffix(".tmp")
                with tmp_idx.open("w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False)
                tmp_idx.replace(index_path)
        except OSError:
            # 落盘失败（只读介质 / 权限问题）不应阻断检索
            pass

    def stats(self, elapsed_ms: float = 0.0) -> Dict[str, object]:
        return {
            "documents": len(self.documents),
            "keywords": len(self.lexicon.all_terms()),
            "categories": len(self.category_level),
            "category_detail": [
                {
                    "id": cat,
                    "name": self.category_name.get(cat, cat),
                    "level": self.category_level[cat],
                    "docs": sum(1 for d in self.documents if d.category == cat),
                    "keywords": len(self.lexicon.terms_for(cat)),
                }
                for cat in self.category_level
            ],
            "engine_requested": self.engine,
            "backend": self._backend,
            "embedder": self.embedder.describe(),
            "index_hash": self.index_hash,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.built_at)),
            "elapsed_ms": round(elapsed_ms, 2),
            "searches": self.stats_counters["searches"],
            "reloads": self.stats_counters["reloads"],
        }
