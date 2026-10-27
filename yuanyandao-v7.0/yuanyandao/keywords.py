# -*- coding: utf-8 -*-
"""关键词词库：归一化、词典最长匹配抽取、BM25 分词。

不依赖 jieba 等分词库：中文使用字符二元组（bigram）作为检索词，
拉丁文按词切分并拆出子词，保证在零依赖环境下仍有稳定的召回。
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

__all__ = [
    "KeywordHit",
    "Lexicon",
    "normalize",
    "tokenize",
    "STOPWORDS",
]

_CJK = r"\u4e00-\u9fff\u3400-\u4dbf"
_TOKEN_RE = re.compile(r"[%s]+|[a-z0-9][a-z0-9_+#.\-]*" % _CJK)
_LATIN_SPLIT_RE = re.compile(r"[-_/.+#]")

STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "及", "或", "对", "把", "被", "吗",
    "呢", "吧", "啊", "我", "你", "他", "她", "它", "这", "那", "请", "帮",
    "如何", "怎么", "什么", "一个", "一下", "可以", "需要", "进行", "the",
    "a", "an", "of", "to", "and", "or", "is", "are", "how", "what", "why",
}

_DASHES = {"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\uff0d": "-"}


def normalize(text: str) -> str:
    """统一全半角、大小写、连字符与空白，便于词典匹配。"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _DASHES.items():
        text = text.replace(src, dst)
    text = text.casefold()
    text = re.sub(r"[ \t\u3000]+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """检索分词：中文 bigram + 拉丁词/子词。"""
    tokens: List[str] = []
    for match in _TOKEN_RE.finditer(normalize(text)):
        chunk = match.group(0)
        if re.match(r"^[%s]+$" % _CJK, chunk):
            if len(chunk) == 1:
                tokens.append(chunk)
            else:
                tokens.extend(chunk[i:i + 2] for i in range(len(chunk) - 1))
        else:
            tokens.append(chunk)
            for part in _LATIN_SPLIT_RE.split(chunk):
                if part and part != chunk and len(part) >= 2:
                    tokens.append(part)
    return [t for t in tokens if t and t not in STOPWORDS]


@dataclass(frozen=True)
class KeywordHit:
    term: str
    category: str
    weight: float
    start: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "term": self.term,
            "category": self.category,
            "weight": round(self.weight, 3),
        }


class Lexicon:
    """分类关键词词典，支持最长匹配与增量扩充。"""

    def __init__(self) -> None:
        self._terms: Dict[str, List[Tuple[str, str]]] = {}
        self._by_category: Dict[str, List[str]] = {}
        self._canonical: Dict[str, str] = {}   # 归一化词 -> 显示词

    # -- 构建 ---------------------------------------------------------
    def add(self, term: str, category: str) -> None:
        display = term.strip()
        key = normalize(display)
        if len(key) < 1:
            return
        bucket = self._terms.setdefault(key[0], [])
        if any(existing == key for existing, _ in bucket):
            return
        bucket.append((key, category))
        self._canonical.setdefault(key, display)
        self._by_category.setdefault(category, [])
        if display not in self._by_category[category]:
            self._by_category[category].append(display)

    def extend(self, terms: Iterable[str], category: str) -> int:
        added = 0
        for term in terms:
            before = len(self._by_category.get(category, []))
            self.add(term, category)
            if len(self._by_category.get(category, [])) > before:
                added += 1
        return added

    def finalize(self) -> None:
        for bucket in self._terms.values():
            bucket.sort(key=lambda item: len(item[0]), reverse=True)

    # -- 查询 ---------------------------------------------------------
    def extract(self, text: str, limit: int = 64) -> List[KeywordHit]:
        norm = normalize(text)
        hits: List[KeywordHit] = []
        i = 0
        n = len(norm)
        while i < n:
            bucket = self._terms.get(norm[i])
            matched = None
            if bucket:
                for term, category in bucket:
                    if norm.startswith(term, i):
                        matched = (term, category)
                        break
            if matched:
                term, category = matched
                specificity = 1.0 + min(1.0, 0.18 * (len(term) - 2)) if len(term) > 2 else 1.0
                hits.append(KeywordHit(self._canonical.get(term, term), category, specificity, i))
                i += len(term)
            else:
                i += 1
            if len(hits) >= limit:
                break
        return hits

    def terms_for(self, category: str) -> List[str]:
        return list(self._by_category.get(category, []))

    def categories(self) -> List[str]:
        return list(self._by_category.keys())

    def all_terms(self) -> List[str]:
        out: List[str] = []
        for terms in self._by_category.values():
            out.extend(terms)
        return out

    def term_index(self) -> Dict[str, List[str]]:
        """显示词 -> 所属分类列表（供索引侧精确命中加权）。"""
        index: Dict[str, List[str]] = {}
        for category, terms in self._by_category.items():
            for term in terms:
                index.setdefault(term, []).append(category)
        return index


def idf(total_docs: int, doc_freq: int) -> float:
    """BM25 的 IDF，外加平滑避免负值。"""
    return math.log(1.0 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
