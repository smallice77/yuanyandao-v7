# -*- coding: utf-8 -*-
"""HTTP 调度服务（标准库实现），供 DeepSeek Harness / 上层业务调用。

接口：
    GET  /health           健康检查
    GET  /v1/status        完整状态（知识库、Ollama、审计链）
    GET  /v1/schema        工具 schema（供 DSH 注册）
    GET  /v1/keywords      导出关键词
    POST /v1/invoke        {text, user_id, permission_level, api_key, format}
    POST /v1/reload        {api_key, permission_level}  → 热重载（需 L4）
    POST /v1/key           {level, ttl}                 → 签发动态密钥（管理员）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from . import ANCHOR_KEY, PLUGIN_NAME, VERSION
from .dispatcher import Engine
from .formats import render

__all__ = ["serve", "main", "make_server"]


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "YuanYanDaoShu/%s" % VERSION
    engine: Engine = None          # type: ignore[assignment]
    config: Any = None
    max_body = 1048576

    # -- 基础 ---------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        sys.stderr.write("[http] %s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, payload: Any, content_type: str = "application/json; charset=utf-8",
              raw: Optional[bytes] = None) -> None:
        body = raw if raw is not None else json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Admin-Token")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, message: str) -> None:
        self._send(code, {"ok": False, "error": message})

    def _read_json(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None, "Content-Length 非法"
        if length <= 0:
            return {}, None
        if length > self.max_body:
            return None, "请求体超过 %d 字节上限" % self.max_body
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, "请求体必须是 UTF-8 JSON"
        if not isinstance(data, dict):
            return None, "请求体必须是 JSON 对象"
        return data, None

    def _is_local(self) -> bool:
        return self.client_address[0] in ("127.0.0.1", "::1")

    # -- 路由 ---------------------------------------------------------
    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send(204, {})

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._dispatch_get()
        except Exception as exc:  # 保底：任何异常都返回 JSON，而不是断开连接
            self._error(500, "服务内部错误：%s: %s" % (type(exc).__name__, exc))

    def _dispatch_get(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/health":
            self._send(200, {"ok": True, "plugin": PLUGIN_NAME, "version": VERSION, "anchor": ANCHOR_KEY})
            return
        if path == "/v1/status":
            self._send(200, {"ok": True, "status": self.engine.status()})
            return
        if path == "/v1/schema":
            self._send(200, {"ok": True, "schema": tool_schema()})
            return
        if path == "/v1/keywords":
            query = parse_qs(parsed.query)
            category = (query.get("category") or [""])[0]
            data = self.engine.export_keywords()
            if category:
                data = {category: data.get(category, [])}
            self._send(200, {"ok": True, "keywords": data})
            return
        if path == "/":
            self._send(
                200,
                {
                    "plugin": PLUGIN_NAME,
                    "version": VERSION,
                    "anchor": ANCHOR_KEY,
                    "endpoints": ["/health", "/v1/status", "/v1/schema", "/v1/invoke", "/v1/keywords"],
                },
            )
            return
        self._error(404, "未找到路径: %s" % path)

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._dispatch_post()
        except Exception as exc:  # 保底：任何异常都返回 JSON，而不是断开连接
            self._error(500, "服务内部错误：%s: %s" % (type(exc).__name__, exc))

    def _dispatch_post(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        data, error = self._read_json()
        if error:
            self._error(400, error)
            return
        assert data is not None

        if path == "/v1/invoke":
            text = str(data.get("text") or data.get("input") or "").strip()
            if not text:
                self._error(400, "字段 text 不能为空")
                return
            fmt = str(data.get("format") or "json").lower()
            level = int(data.get("permission_level") or data.get("level") or 0)
            payload = self.engine.invoke(
                text,
                user_id=str(data.get("user_id") or "http-user"),
                level=level,
                api_key=str(data.get("api_key") or data.get("key") or ""),
                fmt=fmt,
            )
            rendered = render(payload, fmt)
            self._send(
                200,
                {
                    "tool": "yuanyandao.invoke",
                    "ok": bool(payload.get("ok")),
                    "status": payload.get("status"),
                    "result": payload,
                    "rendered": rendered,
                },
            )
            return

        if path == "/v1/reload":
            payload = self.engine.invoke(
                "刷新词库",
                user_id=str(data.get("user_id") or "http-admin"),
                level=int(data.get("permission_level") or 4),
                api_key=str(data.get("api_key") or ""),
                fmt=str(data.get("format") or "json"),
            )
            self._send(200, {"ok": bool(payload.get("ok")), "result": payload})
            return

        if path == "/v1/key":
            token = os.environ.get("YYDS_ADMIN_TOKEN", "")
            provided = self.headers.get("X-Admin-Token", "")
            if token:
                if provided != token:
                    self._error(403, "管理员令牌校验失败")
                    return
            elif not self._is_local():
                self._error(403, "未配置 YYDS_ADMIN_TOKEN 时仅允许本机签发密钥")
                return
            level = int(data.get("level") or 0)
            ttl = int(data.get("ttl") or 0) or None
            try:
                issued = self.engine.issue_key(level, ttl=ttl, subject=str(data.get("subject") or "http"))
            except ValueError as exc:
                self._error(400, str(exc))
                return
            self._send(200, {"ok": True, "key": issued})
            return

        self._error(404, "未找到路径: %s" % path)


def tool_schema() -> Dict[str, Any]:
    """DeepSeek Harness 工具注册用的 schema。"""
    return {
        "name": "yuanyandao_invoke",
        "description": (
            "元衍道枢 V7.0 调度内核：检索私有知识库、执行沙盘推演、生成训练配置或交付文档。"
            "锚点：元衍道枢。权限不足时返回标准化拒绝。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "问题或指令，例如：沙盘推演 RTX5060 4bit QLoRA训练基座风险"},
                "user_id": {"type": "string", "description": "用户标识"},
                "permission_level": {"type": "integer", "minimum": 0, "maximum": 4, "description": "五级权限等级"},
                "api_key": {"type": "string", "description": "动态密钥（L1 及以上必填）"},
                "format": {"type": "string", "enum": ["markdown", "json", "csv", "keywords"]},
            },
            "required": ["text"],
        },
        "endpoint": "POST /v1/invoke",
        "anchor": ANCHOR_KEY,
    }


def make_server(engine: Engine, host: str = "127.0.0.1", port: int = 18080,
                max_body: int = 1048576) -> ThreadingHTTPServer:
    handler = type("BoundApiHandler", (ApiHandler,), {"engine": engine, "max_body": max_body})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    return httpd


def serve(engine: Engine, host: str = "127.0.0.1", port: int = 18080) -> int:
    max_body = int(engine.config.get("server.max_body_bytes", 1048576))
    httpd = make_server(engine, host, port, max_body)
    print("元衍道枢 V%s HTTP 服务已启动：http://%s:%d" % (VERSION, host, port), file=sys.stderr)
    print("健康检查：http://%s:%d/health" % (host, port), file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n收到中断，正在关闭服务…", file=sys.stderr)
    finally:
        httpd.server_close()
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="yuanyandao serve")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("-c", "--config", default=None)
    args = parser.parse_args(argv)
    engine = Engine(config_path=args.config)
    host = args.host or str(engine.config.get("server.host", "127.0.0.1"))
    port = int(args.port or engine.config.get("server.port", 18080))
    return serve(engine, host, port)


if __name__ == "__main__":
    raise SystemExit(main())
