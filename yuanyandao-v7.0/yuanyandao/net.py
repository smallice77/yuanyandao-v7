# -*- coding: utf-8 -*-
"""标准库 HTTP 客户端（避免引入 requests 依赖）。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

__all__ = ["HttpError", "get_json", "post_json"]


class HttpError(RuntimeError):
    def __init__(self, message: str, status: Optional[int] = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


def _request(url: str, data: Optional[bytes], timeout: float, headers: Dict[str, str]) -> Any:
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:  # 4xx/5xx
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise HttpError("HTTP %s: %s" % (exc.code, url), status=exc.code, body=body) from exc
    except urllib.error.URLError as exc:
        raise HttpError("无法连接 %s: %s" % (url, exc.reason)) from exc
    except TimeoutError as exc:
        raise HttpError("请求超时: %s" % url) from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HttpError("响应不是合法 JSON: %s" % url, body=raw[:500]) from exc


def get_json(url: str, timeout: float = 3.0, headers: Optional[Dict[str, str]] = None) -> Any:
    merged = {"Accept": "application/json"}
    merged.update(headers or {})
    return _request(url, None, timeout, merged)


def post_json(url: str, payload: Dict[str, Any], timeout: float = 10.0,
              headers: Optional[Dict[str, str]] = None) -> Any:
    merged = {"Content-Type": "application/json", "Accept": "application/json"}
    merged.update(headers or {})
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return _request(url, body, timeout, merged)
