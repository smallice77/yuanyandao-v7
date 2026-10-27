# -*- coding: utf-8 -*-
"""极简 YAML 子集解析器（PyYAML 缺失时的兜底）。

仅支持本项目配置所需的语法子集：
  * 缩进映射 / 缩进列表
  * 行内列表 ``[0, 1, 2]``
  * 引号字符串、整数、浮点、布尔、null
  * 单行注释（``#`` 在引号外时生效）
  * 列表项内嵌映射（``- id: x`` 后跟同缩进的键值对）

不做锚点、多文档、块标量等高级特性；遇到无法识别的内容会抛出
``MiniYamlError``，避免“静默解析错误”这种更危险的情况。
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

__all__ = ["loads", "load", "MiniYamlError"]


class MiniYamlError(ValueError):
    """YAML 子集解析失败。"""


_NUM_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$")


# ----------------------------------------------------------------------
# 预处理：去注释、去空行、计算缩进
# ----------------------------------------------------------------------
def _strip_comment(line: str) -> str:
    quote: Optional[str] = None
    bracket = 0
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "[{":
            bracket += 1
        elif ch in "]}":
            bracket = max(0, bracket - 1)
        elif ch == "#" and bracket == 0:
            # YAML 注释要求 # 前为空白或行首
            if i == 0 or line[i - 1] in " \t":
                return line[:i]
    return line


def _prepare(text: str) -> List[Tuple[int, str]]:
    rows: List[Tuple[int, str]] = []
    for raw in text.replace("\t", "    ").splitlines():
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        rows.append((indent, line.strip()))
    return rows


# ----------------------------------------------------------------------
# 标量解析
# ----------------------------------------------------------------------
def _split_kv(text: str) -> Optional[Tuple[str, str]]:
    """把 ``key: value`` 拆开；不是键值行时返回 None。"""
    quote: Optional[str] = None
    depth = 0
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == ":" and depth == 0:
            if i + 1 == len(text) or text[i + 1] in " \t":
                key = text[:i].strip().strip("\"'")
                if not key:
                    return None
                return key, text[i + 1:].strip()
    return None


def _parse_inline_list(text: str) -> List[Any]:
    body = text.strip()[1:-1].strip()
    if not body:
        return []
    items: List[str] = []
    buf = ""
    quote: Optional[str] = None
    for ch in body:
        if quote:
            buf += ch
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            buf += ch
        elif ch == ",":
            items.append(buf.strip())
            buf = ""
        else:
            buf += ch
    if buf.strip():
        items.append(buf.strip())
    return [_scalar(x) for x in items]


def _scalar(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    if text[0] == "[" and text[-1] == "]":
        return _parse_inline_list(text)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        inner = text[1:-1]
        if text[0] == '"':
            inner = (
                inner.replace('\\"', '"')
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\\\", "\\")
            )
        return inner
    low = text.lower()
    if low in ("null", "~", "none"):
        return None
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if _NUM_RE.match(text):
        if re.match(r"^[+-]?\d+$", text):
            return int(text)
        try:
            return float(text)
        except ValueError:
            return text
    return text


# ----------------------------------------------------------------------
# 块解析
# ----------------------------------------------------------------------
def _parse_block(rows: List[Tuple[int, str]], i: int, indent: int) -> Tuple[Any, int]:
    if i >= len(rows):
        return {}, i
    if rows[i][1].startswith("- ") or rows[i][1] == "-":
        return _parse_list(rows, i, indent)
    return _parse_map(rows, i, indent)


def _parse_map(rows: List[Tuple[int, str]], i: int, indent: int) -> Tuple[Any, int]:
    out: dict = {}
    n = len(rows)
    while i < n:
        cur_indent, text = rows[i]
        if cur_indent < indent:
            break
        if cur_indent > indent:
            raise MiniYamlError("非法缩进: %r" % text)
        if text.startswith("- "):
            break
        kv = _split_kv(text)
        if kv is None:
            raise MiniYamlError("期望 'key: value'，实际: %r" % text)
        key, rest = kv
        i += 1
        if rest:
            out[key] = _scalar(rest)
            continue
        if i < n and rows[i][0] > indent:
            value, i = _parse_block(rows, i, rows[i][0])
        elif i < n and rows[i][0] == indent and rows[i][1].startswith("- "):
            value, i = _parse_list(rows, i, indent)
        else:
            value = None
        out[key] = value
    return out, i


def _parse_list(rows: List[Tuple[int, str]], i: int, indent: int) -> Tuple[Any, int]:
    out: list = []
    n = len(rows)
    while i < n:
        cur_indent, text = rows[i]
        if cur_indent != indent or not (text.startswith("- ") or text == "-"):
            break
        content = text[1:].strip()
        i += 1
        if not content:
            if i < n and rows[i][0] > indent:
                value, i = _parse_block(rows, i, rows[i][0])
            else:
                value = None
            out.append(value)
            continue
        kv = _split_kv(content)
        if kv is not None and not content.startswith(("[", "'", '"')):
            key, rest = kv
            item: dict = {}
            item[key] = _scalar(rest) if rest else None
            if i < n and rows[i][0] > indent:
                body, i = _parse_block(rows, i, rows[i][0])
                if isinstance(body, dict):
                    item.update(body)
                elif item[key] is None:
                    item[key] = body
            out.append(item)
            continue
        out.append(_scalar(content))
    return out, i


def loads(text: str) -> Any:
    """解析 YAML 文本。"""
    rows = _prepare(text)
    if not rows:
        return {}
    value, i = _parse_block(rows, 0, rows[0][0])
    if i != len(rows):
        raise MiniYamlError("存在未消费的行: %r" % (rows[i][1],))
    return value


def load(path: str) -> Any:
    """从文件读取并解析 YAML。"""
    with open(path, "r", encoding="utf-8") as fh:
        return loads(fh.read())
