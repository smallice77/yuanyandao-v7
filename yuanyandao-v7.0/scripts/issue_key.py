# -*- coding: utf-8 -*-
"""签发五级动态密钥。

    python scripts/issue_key.py --level 4
    python scripts/issue_key.py --level 2 --ttl 3600 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yuanyandao.dispatcher import Engine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="元衍道枢动态密钥签发")
    parser.add_argument("--level", type=int, required=True, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--ttl", type=int, default=0, help="有效期（秒），默认取配置 key_ttl_s")
    parser.add_argument("--subject", default="cli")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    engine = Engine(config_path=args.config)
    issued = engine.issue_key(args.level, ttl=args.ttl or None, subject=args.subject)
    if args.json:
        print(json.dumps(issued, ensure_ascii=False))
    else:
        print("权限等级：L%d %s" % (issued["level"], issued["level_name"]))
        print("有效期至：%s（%d 秒）" % (issued["expires_at"], issued["ttl_s"]))
        print("动态密钥：%s" % issued["key"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
