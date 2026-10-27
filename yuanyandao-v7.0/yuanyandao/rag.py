# -*- coding: utf-8 -*-
"""RAG 生成层：优先调用本地 Ollama 基座，不可用时降级为知识库原文抽取。

降级不是“让模型随便答”，而是只回显检索到的原文并标注来源，
从机制上避免离线状态下编造内容。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .net import HttpError, get_json, post_json
from .vector_store import ScoredDoc

__all__ = ["RagEngine", "EXTRACT_NOTE"]

EXTRACT_NOTE = (
    "当前未连接本地基座（Ollama），以上内容为知识库原文摘录，未做生成式扩写，"
    "以避免幻觉；如需生成式回答，请先启动 ollama serve。"
)


def _trim(text: str, limit: int = 260) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


_ANSWER_PREFIXES = ("### 回答", "## 回答", "# 回答", "**回答**", "回答：", "回答:", "答：", "答:")


def _clean_answer(text: str) -> str:
    """去掉模型复述的“### 回答”之类小标题，避免正文出现重复标题。"""
    text = (text or "").strip()
    changed = True
    while changed and text:
        changed = False
        for prefix in _ANSWER_PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip(" ：:\n\t")
                changed = True
    return text.strip()


class RagEngine:
    def __init__(self, config):
        self.config = config
        hw = config.section("hardware")
        self.endpoint = str(hw.get("ollama_endpoint", "http://127.0.0.1:11434")).rstrip("/")
        self.model = str(hw.get("ollama_model", "qwen2.5:7b"))
        self.mode = str(hw.get("model_mode", "base")).lower()
        self.temperature = float(hw.get("temperature", 0.2))
        self.top_p = float(hw.get("top_p", 0.9))
        self.max_tokens = int(hw.get("max_new_tokens", 1024))
        self.num_ctx = int(hw.get("num_ctx", 8192))
        self.health_timeout = float(hw.get("health_timeout_s", 3))
        self.generate_timeout = float(hw.get("generate_timeout_s", 90))
        self.offline_fallback = bool(config.get("runtime.offline_fallback", True))
        self._health_cache: Optional[Dict[str, object]] = None
        self._health_at = 0.0
        self._cache_ttl = 30.0
        self.system_prompt = self._load_system_prompt()

    # ------------------------------------------------------------------
    @staticmethod
    def _load_system_prompt() -> str:
        path = Path(__file__).resolve().parent.parent / "system_prompt.md"
        if path.exists():
            try:
                return path.read_text(encoding="utf-8").strip()
            except OSError:
                pass
        return (
            "你是元衍道枢（调度内核）的生成层，只根据给定资料回答。"
            "不得虚构资料外的事实；资料不足时明确说明未收录。"
            "不得声称自己是元衍智枢业务中台。"
        )

    # ------------------------------------------------------------------
    def health(self, force: bool = False) -> Dict[str, object]:
        now = time.time()
        if not force and self._health_cache is not None and now - self._health_at < self._cache_ttl:
            return dict(self._health_cache)
        result: Dict[str, object] = {
            "ok": False,
            "endpoint": self.endpoint,
            "model": self.model,
            "models": [],
            "error": "",
        }
        try:
            data = get_json(self.endpoint + "/api/tags", timeout=self.health_timeout)
            models = [str(m.get("name", "")) for m in data.get("models", [])]
            result["models"] = models
            base = self.model.split(":")[0]
            result["ok"] = any(m == self.model or m.split(":")[0] == base for m in models)
            if not result["ok"] and models:
                result["error"] = "服务可用，但未找到模型 %s（本地已有 %s）" % (self.model, ", ".join(models[:5]))
            elif not models:
                result["error"] = "服务可用，但尚未拉取任何模型"
        except HttpError as exc:
            result["error"] = str(exc)
        except Exception as exc:  # 网络层异常一律视为不可用
            result["error"] = "%s: %s" % (type(exc).__name__, exc)
        self._health_cache = result
        self._health_at = now
        return dict(result)

    # ------------------------------------------------------------------
    def build_prompt(self, query: str, hits: Sequence[ScoredDoc]) -> str:
        context_lines = []
        for idx, hit in enumerate(hits, start=1):
            context_lines.append(
                "[%d] 《%s》（%s / %s）\n%s"
                % (idx, hit.doc.title, hit.doc.id, hit.doc.category, _trim(hit.doc.content, 420))
            )
        context = "\n\n".join(context_lines)
        if self.mode == "instruct":
            return (
                "%s\n\n<|im_start|>user\n参考资料：\n%s\n\n问题：%s<|im_end|>\n<|im_start|>assistant\n"
                % (self.system_prompt, context, query)
            )
        return (
            "%s\n\n### 参考资料\n%s\n\n### 要求\n"
            "只依据参考资料回答，逐条标注引用编号；资料不足时直接说明“知识库未收录”。\n\n"
            "### 问题\n%s\n\n### 回答\n" % (self.system_prompt, context, query)
        )

    def generate(self, query: str, hits: Sequence[ScoredDoc]) -> Dict[str, object]:
        started = time.time()
        health = self.health()
        if not health.get("ok"):
            return {
                "ok": False,
                "mode": "offline-extractive",
                "text": "",
                "error": str(health.get("error") or "Ollama 不可用"),
                "elapsed_ms": round((time.time() - started) * 1000, 2),
            }
        payload = {
            "model": self.model,
            "prompt": self.build_prompt(query, hits),
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_tokens,
                "num_ctx": self.num_ctx,
            },
        }
        try:
            data = post_json(self.endpoint + "/api/generate", payload, timeout=self.generate_timeout)
            text = _clean_answer(str(data.get("response", "")))
            return {
                "ok": bool(text),
                "mode": "ollama:%s" % self.model,
                "text": text,
                "error": "" if text else "基座返回空内容",
                "elapsed_ms": round((time.time() - started) * 1000, 2),
                "eval_count": data.get("eval_count"),
            }
        except HttpError as exc:
            self._health_cache = None
            return {
                "ok": False,
                "mode": "offline-extractive",
                "text": "",
                "error": str(exc),
                "elapsed_ms": round((time.time() - started) * 1000, 2),
            }

    # ------------------------------------------------------------------
    def extractive_answer(self, hits: Sequence[ScoredDoc]) -> str:
        if not hits:
            return ""
        lines = ["**知识库检索结果（抽取模式）**", ""]
        for idx, hit in enumerate(hits, start=1):
            lines.append("%d. 《%s》 `%s` · %s（相关度 %.2f）" % (idx, hit.doc.title, hit.doc.id, hit.doc.category, hit.score))
            lines.append("   %s" % _trim(hit.doc.content, 240))
            lines.append("")
        lines.append("> " + EXTRACT_NOTE)
        return "\n".join(lines).strip()

    def answer(self, query: str, hits: Sequence[ScoredDoc]) -> Dict[str, object]:
        generated = self.generate(query, hits)
        if generated.get("ok"):
            return {
                "text": generated["text"],
                "mode": generated["mode"],
                "degraded": False,
                "error": "",
                "elapsed_ms": generated.get("elapsed_ms", 0),
            }
        if not self.offline_fallback:
            return {
                "text": "",
                "mode": "unavailable",
                "degraded": True,
                "error": generated.get("error", "基座不可用"),
                "elapsed_ms": generated.get("elapsed_ms", 0),
            }
        return {
            "text": self.extractive_answer(hits),
            "mode": "offline-extractive",
            "degraded": True,
            "error": generated.get("error", ""),
            "elapsed_ms": generated.get("elapsed_ms", 0),
        }

    def citations(self, hits: Sequence[ScoredDoc]) -> List[Dict[str, object]]:
        return [
            {
                "index": idx,
                "doc_id": hit.doc.id,
                "title": hit.doc.title,
                "category": hit.doc.category,
                "level": hit.doc.level,
                "score": round(hit.score, 4),
                "matched_terms": hit.hits[:6],
            }
            for idx, hit in enumerate(hits, start=1)
        ]
