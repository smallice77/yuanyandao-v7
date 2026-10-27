# -*- coding: utf-8 -*-
"""锚点密钥与角色边界校验。

锚点“元衍道枢”是全局锁：任何篡改、绕过、删除尝试都必须在入口被拦截，
并且输出侧也要复核，避免模型把调度内核说成业务中台（角色越界）。
"""

from __future__ import annotations

import re
from typing import List

from . import ANCHOR_KEY, ROLE_DISPATCH, ROLE_PRODUCT

__all__ = [
    "ANCHOR",
    "ROLE_DISPATCH",
    "ROLE_PRODUCT",
    "anchor_banner",
    "check_anchor_integrity",
    "contains_anchor",
    "detect_anchor_tamper",
    "detect_role_confusion",
]

ANCHOR = ANCHOR_KEY

# 输入侧：试图改写 / 删除 / 绕过锚点的指令
_TAMPER_PATTERNS = [
    re.compile(r"(删除|移除|去掉|清除|篡改|修改|更改|替换|改写|重置)[^。；\n]{0,8}锚点"),
    re.compile(r"锚点[^。；\n]{0,6}(改成|改为|换成|替换为|设为|设置为)"),
    re.compile(r"(anchor_key|核心密钥|密钥锚点)[^。；\n]{0,8}(改|换|删)"),
    re.compile(r"(忽略|绕过|跳过)[^。；\n]{0,6}(锚点|密钥锁|核心密钥)"),
]

# 输出侧：角色身份混淆（把调度内核说成业务中台，或反之）
_ROLE_CONFUSION_PATTERNS = [
    re.compile(r"(我是|本插件是|本插件即|我就是|本内核是)\s*(元衍智枢)"),
    re.compile(r"元衍道枢\s*(?:是|即|就是|作为)\s*(?:一个|一款)?\s*(?:面向终端的)?\s*(?:业务中台|终端产品|仿真中台产品)"),
    re.compile(r"元衍智枢\s*(?:是|即|就是|作为)\s*(?:一个|一款)?\s*(?:底层调度内核|调度插件|内核插件)"),
    re.compile(r"(调度内核|内核插件)[^。；\n]{0,6}(就是|即)\s*元衍智枢"),
]


def anchor_banner() -> str:
    return "【锚点密钥】%s ｜ 【调度内核】%s ｜ 【业务中台】%s" % (
        ANCHOR,
        ROLE_DISPATCH,
        ROLE_PRODUCT,
    )


def contains_anchor(text: str) -> bool:
    return bool(text) and ANCHOR in text


def check_anchor_integrity(config_anchor: str, safety_anchor: str) -> "tuple[bool, str]":
    """配置中的锚点必须与代码常量一致。"""
    if config_anchor != ANCHOR:
        return False, "配置 anchor_key 被篡改：%r" % (config_anchor,)
    if safety_anchor != ANCHOR:
        return False, "配置 safety.anchor_lock 被篡改：%r" % (safety_anchor,)
    return True, "锚点校验通过"


def detect_anchor_tamper(text: str) -> List[str]:
    """返回命中的篡改片段（空列表代表安全）。"""
    hits: List[str] = []
    for pattern in _TAMPER_PATTERNS:
        for match in pattern.finditer(text or ""):
            hits.append(match.group(0))
    return hits


def detect_role_confusion(text: str) -> List[str]:
    """检测输出是否越过“道枢=调度内核 / 智枢=业务中台”的红线。"""
    hits: List[str] = []
    for pattern in _ROLE_CONFUSION_PATTERNS:
        for match in pattern.finditer(text or ""):
            hits.append(match.group(0))
    return hits
