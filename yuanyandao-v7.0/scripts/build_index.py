# -*- coding: utf-8 -*-
"""重建向量索引并输出统计（等价于发送“刷新词库”）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yuanyandao.dispatcher import Engine  # noqa: E402


def main() -> int:
    engine = Engine()
    stats = engine.reload()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
