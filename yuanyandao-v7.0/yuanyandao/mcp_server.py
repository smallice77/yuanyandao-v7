# -*- coding: utf-8 -*-
"""MCP stdio 服务：把元衍道枢包装成 DeepSeek Harness 可直接挂载的工具。

协议：JSON-RPC 2.0，换行分隔，stdout 只输出协议消息，日志一律走 stderr。

工具名采用短名 `invoke` / `reload` / `status`，在 DSH 中呈现为
`mcp__<serverName>__invoke`。兼容旧长名（`yuanyandao_invoke` 等）作为别名。

环境变量（供 DSH bundle 注入）：
    YYDS_MCP_LEVEL    默认权限等级，0-4
    YYDS_MCP_KEY      默认动态密钥（可选）
    YYDS_MCP_TRUSTED  1=本地受信模式，启动时就地签发默认等级密钥
    YYDS_MCP_USER     审计中的用户标识
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from . import ANCHOR_KEY, VERSION
from .dispatcher import Engine
from .formats import render

__all__ = ["main", "MCPServer", "TOOLS"]

PROTOCOL_VERSION = "2024-11-05"
TRUSTED_KEY_TTL_S = 30 * 24 * 3600

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "invoke",
        "description": (
            "元衍道枢 V7.0 调度内核（锚点：元衍道枢）。接收中文问题或指令，执行双路检索、"
            "五级权限校验与意图分流，返回结构化结果：知识问答 / 沙盘推演 / 训练配置 / "
            "交付文档 / 词库维护。需要成文文档、风险对比或训练配置时优先调用。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "问题或指令，例如：沙盘推演 RTX5060 4bit QLoRA训练基座风险",
                },
                "format": {
                    "type": "string",
                    "enum": ["markdown", "json", "csv", "keywords"],
                    "description": "输出格式，默认 markdown",
                },
                "user_id": {"type": "string", "description": "审计用用户标识"},
                "permission_level": {"type": "integer", "minimum": 0, "maximum": 4},
                "api_key": {"type": "string", "description": "动态密钥；省略时使用服务默认等级"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "reload",
        "description": "热重载元衍道枢关键词与向量索引（词条入库后调用；需要 L4 权限）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "permission_level": {"type": "integer", "minimum": 0, "maximum": 4},
                "api_key": {"type": "string"},
                "user_id": {"type": "string"},
            },
        },
    },
    {
        "name": "status",
        "description": "查看元衍道枢运行状态：知识库规模、索引后端、Ollama 可用性、审计链校验结果。",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

_TOOL_ALIASES = {
    "yuanyandao_invoke": "invoke",
    "yuanyandao_reload": "reload",
    "yuanyandao_status": "status",
}


class MCPServer:
    def __init__(
        self,
        engine: Engine,
        default_level: Optional[int] = None,
        default_key: str = "",
        trusted: Optional[bool] = None,
        default_user: str = "dsh-mcp",
    ):
        self.engine = engine
        env = os.environ
        raw_level = env.get("YYDS_MCP_LEVEL", "0").strip() or "0"
        self.default_user = default_user or env.get("YYDS_MCP_USER") or "dsh-mcp"
        try:
            self.default_level = int(default_level if default_level is not None else raw_level)
        except ValueError:
            self.default_level = 0
        self.default_key = default_key or env.get("YYDS_MCP_KEY", "")
        if trusted is None:
            trusted = env.get("YYDS_MCP_TRUSTED", "").strip().lower() in ("1", "true", "yes", "on")
        self.trusted = bool(trusted)
        if self.trusted and self.default_level >= 1 and not self.default_key:
            issued = self.engine.issue_key(
                self.default_level, ttl=TRUSTED_KEY_TTL_S, subject="dsh-mcp"
            )
            self.default_key = str(issued["key"])
            sys.stderr.write(
                "[yyds-mcp] 本地受信模式：已签发 L%d 密钥（至 %s）\n"
                % (self.default_level, issued["expires_at"])
            )
            sys.stderr.flush()
        self._shutdown = False

    # ------------------------------------------------------------------
    def _response(self, msg_id: Any, result: Any) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def _error(self, msg_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}

    def _auth(self, args: Dict[str, Any]) -> Tuple[int, str, str]:
        raw_level = args.get("permission_level")
        if raw_level is None or raw_level == "":
            level = self.default_level
        else:
            try:
                level = int(raw_level)
            except (TypeError, ValueError):
                level = self.default_level
        key = str(args.get("api_key") or self.default_key or "")
        user = str(args.get("user_id") or self.default_user)
        return level, key, user

    # ------------------------------------------------------------------
    def handle(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = str(request.get("method") or "")
        msg_id = request.get("id")
        params = request.get("params") or {}

        if method == "initialize":
            return self._response(
                msg_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "yuanyandao", "version": VERSION, "anchor": ANCHOR_KEY},
                    "instructions": (
                        "元衍道枢 V7.0 是本地 RAG 调度内核（锚点：元衍道枢）。"
                        "用 invoke 提问私有知识库、做沙盘推演、生成训练配置或交付文档；"
                        "返回内容为中文，包含引用来源与调用元数据。"
                    ),
                },
            )

        if method in ("notifications/initialized", "initialized"):
            return None

        if method == "ping":
            return self._response(msg_id, {})

        if method == "tools/list":
            return self._response(msg_id, {"tools": TOOLS})

        if method == "resources/list":
            return self._response(msg_id, {"resources": []})

        if method == "prompts/list":
            return self._response(msg_id, {"prompts": []})

        if method == "tools/call":
            name = str(params.get("name") or "")
            args = params.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            try:
                return self._call_tool(msg_id, name, args)
            except Exception as exc:  # 工具内异常不应打断协议
                return self._response(
                    msg_id,
                    {
                        "content": [{"type": "text", "text": "工具执行失败：%s: %s" % (type(exc).__name__, exc)}],
                        "isError": True,
                    },
                )

        if method == "shutdown":
            self._shutdown = True
            return self._response(msg_id, None)

        if msg_id is None:
            return None
        return self._error(msg_id, -32601, "未知方法: %s" % method)

    # ------------------------------------------------------------------
    def _call_tool(self, msg_id: Any, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        tool = _TOOL_ALIASES.get(name, name)

        if tool == "invoke":
            text = str(args.get("text") or "").strip()
            if not text:
                return self._response(
                    msg_id,
                    {"content": [{"type": "text", "text": "参数 text 不能为空"}], "isError": True},
                )
            fmt = str(args.get("format") or "markdown")
            if fmt not in ("markdown", "json", "csv", "keywords"):
                fmt = "markdown"
            level, key, user = self._auth(args)
            payload = self.engine.invoke(text, user_id=user, level=level, api_key=key, fmt=fmt)
            return self._response(
                msg_id,
                {
                    "content": [{"type": "text", "text": render(payload, fmt)}],
                    "structuredContent": payload,
                    "isError": False,
                },
            )

        if tool == "reload":
            level, key, user = self._auth(args)
            payload = self.engine.invoke(
                "刷新词库", user_id=user, level=level or 4, api_key=key, fmt="json"
            )
            return self._response(
                msg_id,
                {
                    "content": [{"type": "text", "text": render(payload, "markdown")}],
                    "structuredContent": payload,
                    "isError": False,
                },
            )

        if tool == "status":
            status = self.engine.status()
            return self._response(
                msg_id,
                {
                    "content": [{"type": "text", "text": json.dumps(status, ensure_ascii=False, indent=2)}],
                    "structuredContent": status,
                    "isError": False,
                },
            )

        return self._response(
            msg_id,
            {"content": [{"type": "text", "text": "未知工具: %s" % name}], "isError": True},
        )

    # ------------------------------------------------------------------
    def run(self) -> int:
        stdin = sys.stdin.buffer
        stdout = sys.stdout.buffer
        while not self._shutdown:
            line = stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                response = self._error(None, -32700, "JSON 解析失败: %s" % exc)
            else:
                if not isinstance(request, dict):
                    response = self._error(None, -32600, "请求必须是 JSON 对象")
                else:
                    response = self.handle(request)
            if response is not None:
                stdout.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
                stdout.flush()
        return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="yuanyandao mcp")
    parser.add_argument("-c", "--config", default=None)
    parser.add_argument("--level", type=int, default=None, help="默认权限等级")
    parser.add_argument("--key", default="", help="默认动态密钥")
    parser.add_argument("--trusted", action="store_true", help="本地受信模式：就地签发默认等级密钥")
    args = parser.parse_args(argv)
    engine = Engine(config_path=args.config)
    server = MCPServer(
        engine,
        default_level=args.level,
        default_key=args.key,
        trusted=True if args.trusted else None,
    )
    return server.run()


if __name__ == "__main__":
    raise SystemExit(main())
