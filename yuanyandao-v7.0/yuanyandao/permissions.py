# -*- coding: utf-8 -*-
"""五级权限内核与动态密钥鉴权。

权限判定发生在检索之前：先按等级确定可见知识库范围，再决定是否放行，
避免“先检索后过滤”造成的信息泄漏。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

__all__ = ["AuthResult", "PermissionKernel", "LEVEL_NAMES"]

LEVEL_NAMES = {
    0: "访客",
    1: "基础用户",
    2: "高级用户",
    3: "开发操作员",
    4: "内核管理员",
}

_KEY_PREFIX = "YYDS"
_KEY_ALGO = "sha256"


@dataclass
class AuthResult:
    ok: bool
    level: int
    user_id: str
    capabilities: Set[str] = field(default_factory=set)
    code: str = "OK"
    reason: str = ""
    key_expires_at: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "level": self.level,
            "level_name": LEVEL_NAMES.get(self.level, "未知"),
            "user_id": self.user_id,
            "capabilities": sorted(self.capabilities),
            "code": self.code,
            "reason": self.reason,
            "key_expires_at": self.key_expires_at,
        }


class PermissionKernel:
    def __init__(self, config):
        self.config = config
        self.levels = [int(x) for x in config.get("permission.level_list", [0, 1, 2, 3, 4])]
        self.enable_key = bool(config.get("permission.enable_dynamic_key", True))
        self.ttl = int(config.get("permission.key_ttl_s", 7200))
        self.single_use = bool(config.get("permission.key_single_use", False))
        self._capabilities: Dict[int, Set[str]] = {}
        raw_caps = config.get("permission.capabilities", {}) or {}
        for level, caps in raw_caps.items():
            self._capabilities[int(level)] = {str(c) for c in caps}
        self._kb_level: Dict[str, int] = {
            str(k): int(v) for k, v in (config.get("permission.kb_level", {}) or {}).items()
        }
        self.level_desc: Dict[int, str] = {}
        for level, desc in (config.get("permission.level_desc", {}) or {}).items():
            self.level_desc[int(level)] = str(desc)
        secret_env = str(config.get("permission.key_secret_env", "YYDS_SECRET"))
        self.secret = (os.environ.get(secret_env) or str(config.get("permission.key_secret", ""))).encode("utf-8")
        if not self.secret:
            raise ValueError("动态密钥服务端密钥为空：请设置 %s 或 permission.key_secret" % secret_env)
        self._state_path = Path(config.path_for("runtime.state_dir")) / "used_nonces.json"
        self._used: Dict[str, int] = self._load_used()

    # ------------------------------------------------------------------
    # 能力
    # ------------------------------------------------------------------
    def capabilities_for(self, level: int) -> Set[str]:
        return set(self._capabilities.get(level, set()))

    def can(self, level: int, capability: str) -> bool:
        return capability in self.capabilities_for(level)

    def kb_min_level(self, category: str) -> int:
        return int(self._kb_level.get(category, 4))

    def visible_categories(self, level: int) -> List[str]:
        return [cat for cat, need in self._kb_level.items() if need <= level]

    def level_name(self, level: int) -> str:
        return self.level_desc.get(level) or LEVEL_NAMES.get(level, "未知")

    # ------------------------------------------------------------------
    # 密钥
    # ------------------------------------------------------------------
    def _sign(self, level: int, expires: int, nonce: str) -> str:
        message = "%d:%d:%s" % (level, expires, nonce)
        return hmac.new(self.secret, message.encode("utf-8"), getattr(hashlib, _KEY_ALGO)).hexdigest()[:24]

    def issue_key(self, level: int, ttl: Optional[int] = None, subject: str = "issued") -> Dict[str, object]:
        if level not in self.levels:
            raise ValueError("非法权限等级: %s" % level)
        ttl = int(ttl or self.ttl)
        expires = int(time.time()) + ttl
        nonce = secrets.token_hex(6)
        key = "%s.%d.%d.%s.%s" % (_KEY_PREFIX, level, expires, nonce, self._sign(level, expires, nonce))
        return {
            "key": key,
            "level": level,
            "level_name": LEVEL_NAMES.get(level, "未知"),
            "subject": subject,
            "issued_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "expires_at": datetime.fromtimestamp(expires, tz=timezone.utc).isoformat(timespec="seconds"),
            "ttl_s": ttl,
        }

    def _load_used(self) -> Dict[str, int]:
        if not self._state_path.exists():
            return {}
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            return {str(k): int(v) for k, v in data.items() if int(v) > time.time()}
        except Exception:
            return {}

    def _mark_used(self, nonce: str, expires: int) -> bool:
        if nonce in self._used:
            return False
        self._used[nonce] = expires
        now = int(time.time())
        self._used = {k: v for k, v in self._used.items() if v > now}
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(self._used), encoding="utf-8")
        except OSError:
            pass
        return True

    def verify_key(self, key: str, expected_level: int) -> Dict[str, object]:
        if not key:
            return {"ok": False, "code": "MISSING_KEY", "reason": "缺少动态密钥"}
        parts = str(key).strip().split(".")
        if len(parts) != 5 or parts[0] != _KEY_PREFIX:
            return {"ok": False, "code": "BAD_KEY_FORMAT", "reason": "动态密钥格式非法"}
        try:
            level = int(parts[1])
            expires = int(parts[2])
        except ValueError:
            return {"ok": False, "code": "BAD_KEY_FORMAT", "reason": "动态密钥字段非法"}
        nonce, signature = parts[3], parts[4]
        if level != expected_level:
            return {
                "ok": False,
                "code": "LEVEL_MISMATCH",
                "reason": "密钥等级 L%d 与声明等级 L%d 不一致" % (level, expected_level),
            }
        if not hmac.compare_digest(signature, self._sign(level, expires, nonce)):
            return {"ok": False, "code": "BAD_SIGNATURE", "reason": "动态密钥签名校验失败"}
        if expires <= int(time.time()):
            return {"ok": False, "code": "KEY_EXPIRED", "reason": "动态密钥已过期"}
        if self.single_use and not self._mark_used(nonce, expires):
            return {"ok": False, "code": "KEY_REPLAY", "reason": "动态密钥已被使用（防重放）"}
        return {
            "ok": True,
            "code": "OK",
            "level": level,
            "expires_at": datetime.fromtimestamp(expires, tz=timezone.utc).isoformat(timespec="seconds"),
        }

    # ------------------------------------------------------------------
    # 鉴权入口
    # ------------------------------------------------------------------
    def authenticate(self, user_id: str, level: int, api_key: str = "") -> AuthResult:
        user_id = (user_id or "anonymous").strip()[:64] or "anonymous"
        try:
            level = int(level)
        except (TypeError, ValueError):
            return AuthResult(False, 0, user_id, set(), "BAD_LEVEL", "权限等级必须是 0-4 的整数")
        if level not in self.levels:
            return AuthResult(False, 0, user_id, set(), "BAD_LEVEL", "权限等级超出 0-4 范围")
        if level == 0:
            return AuthResult(
                True, 0, user_id, self.capabilities_for(0), "OK", "访客模式：无需动态密钥"
            )
        if not self.enable_key:
            return AuthResult(True, level, user_id, self.capabilities_for(level), "OK", "动态密钥鉴权已关闭")
        result = self.verify_key(api_key, level)
        if not result.get("ok"):
            return AuthResult(
                False,
                level,
                user_id,
                set(),
                str(result.get("code")),
                str(result.get("reason")),
            )
        return AuthResult(
            True,
            level,
            user_id,
            self.capabilities_for(level),
            "OK",
            "动态密钥校验通过",
            str(result.get("expires_at")),
        )

    # ------------------------------------------------------------------
    # 提示文案
    # ------------------------------------------------------------------
    def denial_message(self, auth: AuthResult, required_level: int, capability: str,
                       reason: str = "") -> str:
        need = self.level_name(required_level)
        current = self.level_name(auth.level)
        reason = reason or auth.reason or "当前等级不满足该操作要求"
        return (
            "⛔ 权限拒绝：{reason}\n"
            "- 请求等级：L{cur} {cur_name}\n"
            "- 所需等级：L{need_level} {need}\n"
            "- 受控能力：{capability}\n"
            "- 处理建议：向管理员申请 L{need_level} 动态密钥后重试，或改用公开知识库问答。\n"
            "- 本次请求已写入审计日志。"
        ).format(
            reason=reason,
            cur=auth.level,
            cur_name=current,
            need_level=required_level,
            need=need,
            capability=capability,
        )
