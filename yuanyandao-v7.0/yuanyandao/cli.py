# -*- coding: utf-8 -*-
"""命令行入口：

    python -m yuanyandao "导出全部TA关键词" --level 4 --auto-key
    python -m yuanyandao serve            # 启动 HTTP 调度服务
    python -m yuanyandao mcp              # 以 MCP stdio 方式接入 DeepSeek Harness
    python -m yuanyandao health
    python -m yuanyandao issue-key --level 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from . import ANCHOR_KEY, PLUGIN_NAME, VERSION
from .dispatcher import Engine
from .formats import render

_SUBCOMMANDS = {"serve", "mcp", "health", "issue-key", "reload", "status", "keywords", "query"}


def _reconfigure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def _build_engine(args) -> Engine:
    return Engine(config_path=getattr(args, "config", None), root=getattr(args, "root", None))


def _resolve_key(engine: Engine, args) -> str:
    key = getattr(args, "key", "") or os.environ.get("YYDS_KEY", "")
    if key:
        return key
    if getattr(args, "level", 0) >= 1:
        if getattr(args, "auto_key", False) or os.environ.get("YYDS_AUTO_KEY", "").strip() in ("1", "true", "yes"):
            issued = engine.issue_key(int(args.level), subject=args.user)
            print("[auto-key] 已签发 L%d 动态密钥：%s" % (args.level, issued["key"]), file=sys.stderr)
            return str(issued["key"])
    return ""


def _print_result(payload: dict, fmt: str, out: Optional[str]) -> int:
    text = render(payload, fmt)
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print("已写入 %s" % out)
    else:
        print(text)
    if payload.get("status") == "denied":
        return 1
    if payload.get("status") in ("error",):
        return 2
    return 0


def _query_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yuanyandao",
        description="%s V%s · 私有 RAG 沙盘推演调度内核" % (PLUGIN_NAME, VERSION),
    )
    parser.add_argument("text", nargs="+", help="问题或指令，例如：导出全部TA关键词")
    parser.add_argument("-f", "--format", default="markdown",
                        choices=["markdown", "md", "json", "csv", "keywords"])
    parser.add_argument("-l", "--level", type=int, default=0, help="权限等级 0-4")
    parser.add_argument("-u", "--user", default="cli-user", help="用户标识")
    parser.add_argument("-k", "--key", default="", help="动态密钥")
    parser.add_argument("--auto-key", action="store_true", help="本地按相同密钥签发临时密钥（仅开发用）")
    parser.add_argument("-c", "--config", default=None, help="配置文件路径")
    parser.add_argument("--root", default=None, help="项目根目录")
    parser.add_argument("-o", "--out", default=None, help="输出到文件")
    parser.add_argument("--save-state", action="store_true", help="兼容参数：保留状态目录")
    return parser


def _run_query(argv: List[str]) -> int:
    args = _query_parser().parse_args(argv)
    engine = _build_engine(args)
    key = _resolve_key(engine, args)
    payload = engine.invoke(
        " ".join(args.text), user_id=args.user, level=args.level, api_key=key, fmt=args.format
    )
    return _print_result(payload, args.format, args.out)


def _run_health() -> int:
    engine = Engine()
    print(json.dumps(engine.status(), ensure_ascii=False, indent=2))
    return 0


def _run_key(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="yuanyandao issue-key")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--ttl", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("-c", "--config", default=None)
    args = parser.parse_args(argv)
    engine = _build_engine(args)
    issued = engine.issue_key(args.level, ttl=args.ttl or None, subject="cli")
    if args.json:
        print(json.dumps(issued, ensure_ascii=False))
    else:
        print("权限等级：L%d %s" % (issued["level"], issued["level_name"]))
        print("有效期至：%s" % issued["expires_at"])
        print("动态密钥：%s" % issued["key"])
    return 0


def _run_reload(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="yuanyandao reload")
    parser.add_argument("-c", "--config", default=None)
    args = parser.parse_args(argv)
    engine = _build_engine(args)
    stats = engine.reload()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def _run_keywords(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="yuanyandao keywords")
    parser.add_argument("--category", default="")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("-c", "--config", default=None)
    args = parser.parse_args(argv)
    engine = _build_engine(args)
    data = engine.export_keywords()
    if args.category:
        data = {args.category: data.get(args.category, [])}
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        for cat, words in data.items():
            print("%s (%d): %s" % (cat, len(words), ", ".join(words)))
    return 0


def _run_serve(argv: List[str]) -> int:
    from .server import main as server_main

    return server_main(argv)


def _run_mcp(argv: List[str]) -> int:
    from .mcp_server import main as mcp_main

    return mcp_main(argv)


def main(argv: Optional[List[str]] = None) -> int:
    _reconfigure_stdio()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print("%s V%s ｜ 锚点：%s" % (PLUGIN_NAME, VERSION, ANCHOR_KEY))
        print("用法：python -m yuanyandao \"<指令>\" [-l 0-4] [-k KEY] [--auto-key] [-f markdown|json|csv|keywords]")
        print("子命令：serve | mcp | health | status | reload | keywords | issue-key")
        return 0
    command = argv[0]
    if command in ("serve",):
        return _run_serve(argv[1:])
    if command in ("mcp",):
        return _run_mcp(argv[1:])
    if command in ("health", "status"):
        return _run_health()
    if command == "issue-key":
        return _run_key(argv[1:])
    if command == "reload":
        return _run_reload(argv[1:])
    if command == "keywords":
        return _run_keywords(argv[1:])
    if command == "query":
        argv = argv[1:]
    return _run_query(argv)


if __name__ == "__main__":
    raise SystemExit(main())
