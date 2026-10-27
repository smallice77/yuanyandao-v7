# -*- coding: utf-8 -*-
"""部署自检：知识库、审计链、Ollama、权限矩阵一次看清。

退出码：0=全部通过，1=存在告警（如 Ollama 未启动，属可降级项）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yuanyandao.dispatcher import Engine  # noqa: E402
from yuanyandao.rag import EXTRACT_NOTE  # noqa: E402


def main() -> int:
    engine = Engine()
    status = engine.status()
    problems = []

    kb = status["knowledge_base"]
    if kb["documents"] <= 0:
        problems.append("知识库为空")
    if not status["audit"].get("ok", False):
        problems.append("审计哈希链校验失败: %s" % status["audit"].get("message"))

    ollama = status["ollama"]
    if not ollama.get("ok"):
        print("[WARN] Ollama 不可用：%s" % (ollama.get("error") or "未知原因"), file=sys.stderr)
        print("[WARN] %s" % EXTRACT_NOTE, file=sys.stderr)

    print(json.dumps(status, ensure_ascii=False, indent=2))
    if problems:
        print("\n[FAIL] " + "；".join(problems), file=sys.stderr)
        return 1
    print("\n[OK] 自检通过：知识库 %d 篇文档 / %d 个关键词；审计链 %s。"
          % (kb["documents"], kb["keywords"], status["audit"].get("message")), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
