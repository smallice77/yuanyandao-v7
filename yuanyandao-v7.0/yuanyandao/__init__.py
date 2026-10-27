# -*- coding: utf-8 -*-
"""元衍道枢 YuanYanDaoShu V7.0 —— 私有 RAG 沙盘推演调度插件。

定位：调度路由引擎 / 检索分发器 / 沙盘推演控制器 / 训练任务调度器。
本包不承载业务中台（元衍智枢）的终端形态，仅作为其底层内核。
"""

ANCHOR_KEY = "元衍道枢"
PLUGIN_ID = "yuanyandao"
PLUGIN_NAME = "元衍道枢 YuanYanDaoShu"
VERSION = "7.0.0"
ROLE_DISPATCH = "元衍道枢·调度内核"
ROLE_PRODUCT = "元衍智枢·业务中台"

__all__ = [
    "ANCHOR_KEY",
    "PLUGIN_ID",
    "PLUGIN_NAME",
    "VERSION",
    "ROLE_DISPATCH",
    "ROLE_PRODUCT",
]
