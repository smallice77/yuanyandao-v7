# -*- coding: utf-8 -*-
"""关键词批量入库脚本：支持逗号分隔、逐行导入与增量更新。

示例：
    python scripts/import_keywords.py --category ta --terms "路径追踪,体积云,次表面散射"
    python scripts/import_keywords.py --category llm --file data/new_terms.txt --reload
    python scripts/import_keywords.py --category ta --terms "..." --doc-title "路径追踪规范" \
        --doc-content "..."
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yuanyandao.dispatcher import Engine  # noqa: E402

SPLIT_RE = re.compile(r"[,，、;；\n\r\t]+")


def parse_terms(raw: str) -> list:
    return [item.strip() for item in SPLIT_RE.split(raw) if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="元衍道枢关键词批量入库（增量）")
    parser.add_argument("--category", required=False, help="目标知识库分类，如 ta / llm / hardware")
    parser.add_argument("--terms", default="", help="逗号分隔的关键词串")
    parser.add_argument("--file", default="", help="关键词文件（逗号或换行分隔，UTF-8）")
    parser.add_argument("--doc-title", default="", help="同时入库一段文档的标题")
    parser.add_argument("--doc-content", default="", help="同时入库的文档正文")
    parser.add_argument("--doc-tags", default="", help="文档标签，逗号分隔")
    parser.add_argument("--no-reload", action="store_true", help="只写入不重建索引")
    parser.add_argument("--dry-run", action="store_true", help="只预览，不落盘")
    parser.add_argument("--list-categories", action="store_true", help="列出可用分类")
    parser.add_argument("--config", default=None, help="配置文件路径")
    args = parser.parse_args()

    engine = Engine(config_path=args.config)

    if args.list_categories:
        print(json.dumps(
            [
                {"id": cat, "name": engine.kb.category_name.get(cat, cat), "level": level}
                for cat, level in engine.kb.category_level.items()
            ],
            ensure_ascii=False,
            indent=2,
        ))
        return 0

    if not args.category:
        parser.error("--category 必填（可用 --list-categories 查看）")
    if args.category not in engine.kb.category_level:
        print("未知分类: %s" % args.category, file=sys.stderr)
        return 2

    terms = parse_terms(args.terms)
    if args.file:
        path = Path(args.file)
        if not path.exists():
            print("文件不存在: %s" % path, file=sys.stderr)
            return 2
        terms.extend(parse_terms(path.read_text(encoding="utf-8")))
    terms = list(dict.fromkeys(terms))

    if args.dry_run:
        print(json.dumps({"category": args.category, "terms": terms, "count": len(terms)}, ensure_ascii=False, indent=2))
        return 0

    added = engine.kb.add_keywords(args.category, terms) if terms else 0
    doc_added = None
    if args.doc_title and args.doc_content:
        doc = engine.kb.add_document(
            args.doc_title,
            args.doc_content,
            category=args.category,
            tags=parse_terms(args.doc_tags),
        )
        doc_added = doc.id

    result = {
        "category": args.category,
        "submitted": len(terms),
        "added": added,
        "duplicated": len(terms) - added,
        "document_added": doc_added,
    }
    if not args.no_reload:
        stats = engine.kb.reload()
        result["index"] = {
            "documents": stats["documents"],
            "keywords": stats["keywords"],
            "index_hash": stats["index_hash"],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
