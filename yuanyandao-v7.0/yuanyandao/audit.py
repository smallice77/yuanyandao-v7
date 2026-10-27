# -*- coding: utf-8 -*-
"""审计日志（JSONL + 哈希链）与快照归档。

每条记录串上前一条的哈希，任何一条被改动都会让后续链校验失败，
满足“可溯源调用链”的要求；沙盘与训练类操作额外落盘完整快照。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

__all__ = ["AuditLog"]

_GENESIS = "0" * 64


class AuditLog:
    def __init__(self, config):
        self.config = config
        self.path: Path = Path(config.path_for("audit.log_path"))
        self.snapshot_dir: Path = Path(config.path_for("audit.snapshot_dir"))
        self.snapshot_enabled = bool(config.get("audit.snapshot_save", True))
        self.hash_chain = bool(config.get("audit.hash_chain", True))
        self.fields: List[str] = list(config.get("audit.log_fields", []))
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.snapshot_enabled:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._last_hash, self._last_seq = self._scan_tail()

    # ------------------------------------------------------------------
    def _scan_tail(self) -> "tuple[str, int]":
        if not self.path.exists():
            return _GENESIS, 0
        last_hash, last_seq = _GENESIS, 0
        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    last_hash = str(record.get("record_hash") or last_hash)
                    last_seq = int(record.get("seq") or last_seq)
        except OSError:
            pass
        return last_hash, last_seq

    @staticmethod
    def _canonical(record: Dict[str, object]) -> str:
        payload = {
            k: v
            for k, v in record.items()
            if k not in ("record_hash", "prev_hash")
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _hash(self, record: Dict[str, object]) -> str:
        return hashlib.sha256(self._canonical(record).encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    def append(self, record: Dict[str, object]) -> Dict[str, object]:
        """写入一条审计记录，返回带 seq/hash 的完整记录。"""
        with self._lock:
            entry: Dict[str, object] = {}
            for field in self.fields:
                entry[field] = record.get(field)
            for key, value in record.items():
                entry.setdefault(key, value)
            entry["seq"] = self._last_seq + 1
            entry["prev_hash"] = self._last_hash if self.hash_chain else _GENESIS
            entry["record_hash"] = self._hash(entry)
            line = json.dumps(entry, ensure_ascii=False, sort_keys=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
            self._last_seq = int(entry["seq"])
            self._last_hash = str(entry["record_hash"])
            return entry

    # ------------------------------------------------------------------
    def snapshot(self, op_code: str, payload: Dict[str, object], kind: str = "call") -> Optional[str]:
        if not self.snapshot_enabled:
            return None
        safe_op = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in op_code)[:48]
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = self.snapshot_dir / ("%s_%s_%s.json" % (kind, safe_op, stamp))
        counter = 0
        while target.exists():
            counter += 1
            target = self.snapshot_dir / ("%s_%s_%s_%02d.json" % (kind, safe_op, stamp, counter))
        body = {
            "op_code": op_code,
            "kind": kind,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "payload": payload,
        }
        try:
            target.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
            return str(target)
        except OSError:
            return None

    # ------------------------------------------------------------------
    def verify(self, limit: int = 0) -> Dict[str, object]:
        """重算哈希链，返回校验结果（limit>0 时只校验最后 N 条）。"""
        if not self.path.exists():
            return {"ok": True, "checked": 0, "message": "暂无审计日志"}
        records: List[Dict[str, object]] = []
        with self.path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    return {"ok": False, "checked": len(records), "message": "存在非法 JSON 行"}
        if limit > 0:
            records = records[-limit:]
        prev = _GENESIS if limit == 0 else str(records[0].get("prev_hash", _GENESIS))
        checked = 0
        for record in records:
            digest = self._hash(record)
            if digest != record.get("record_hash"):
                return {
                    "ok": False,
                    "checked": checked,
                    "message": "第 %s 条记录哈希不匹配，日志可能被篡改" % record.get("seq"),
                    "seq": record.get("seq"),
                }
            if self.hash_chain and record.get("prev_hash") != prev and limit == 0:
                return {
                    "ok": False,
                    "checked": checked,
                    "message": "第 %s 条记录前向哈希断裂" % record.get("seq"),
                    "seq": record.get("seq"),
                }
            prev = str(record.get("record_hash"))
            checked += 1
        return {"ok": True, "checked": checked, "message": "哈希链校验通过", "tail_seq": records[-1].get("seq") if records else 0}

    def tail(self, n: int = 5) -> List[Dict[str, object]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        out: List[Dict[str, object]] = []
        for line in lines[-n:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
