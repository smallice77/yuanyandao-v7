# -*- coding: utf-8 -*-
"""配置装载：PyYAML 优先，缺失时回退到内置 miniyaml；统一解析相对路径。"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["Config", "load_config", "DEFAULTS", "PROJECT_ROOT", "DEFAULT_CONFIG_PATH"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def _load_yaml_text(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if os.environ.get("YYDS_FORCE_MINI_YAML", "").strip() not in ("", "0", "false"):
        from . import miniyaml

        return miniyaml.loads(text) or {}
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return data or {}
    except Exception:
        from . import miniyaml

        return miniyaml.loads(text) or {}


DEFAULTS: Dict[str, Any] = {
    "version": "7.0",
    "plugin_name": "元衍道枢",
    "plugin_id": "yuanyandao",
    "bind_upper": "元衍智枢 V7.0 私有AI仿真中台",
    "anchor_key": "元衍道枢",
    "runtime": {
        "project_root": ".",
        "data_dir": "./data",
        "state_dir": "./state",
        "default_format": "markdown",
        "log_level": "INFO",
        "request_timeout_s": 90,
        "offline_fallback": True,
    },
    "hardware": {
        "device": "cuda",
        "vram_limit_gb": 8,
        "quant_bit": 4,
        "quant_method": "bnb_4bit",
        "quant_type": "nf4",
        "double_quant": True,
        "base_model": "qwen2.5-7b-base",
        "hf_model_id": "Qwen/Qwen2.5-7B",
        "ollama_endpoint": "http://127.0.0.1:11434",
        "ollama_model": "qwen2.5:7b-instruct",
        "embed_model": "qwen3-embedding:0.6b",
        "model_mode": "instruct",
        "max_new_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.9,
        "num_ctx": 4096,
        "health_timeout_s": 3,
        "generate_timeout_s": 90,
    },
    "vector_store": {
        "engine": "memory",
        "index_path": "./vector_db/index.faiss",
        "metadata_path": "./vector_db/metadata.json",
        "top_n": 6,
        "dual_retrieval": True,
        "hot_reload": True,
        "embed_provider": "hash",
        "keyword_weight": 0.55,
        "vector_weight": 0.45,
        "embed_dim": 512,
        "min_score": 0.18,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "chroma_collection": "yuanyandao_v7",
    },
    "permission": {
        "level_list": [0, 1, 2, 3, 4],
        "enable_dynamic_key": True,
        "key_ttl_s": 7200,
        "key_single_use": False,
        "key_secret_env": "YYDS_SECRET",
        "key_secret": "yuanyandao-v7-dev-secret-change-me",
        "level_desc": {},
        "capabilities": {},
        "kb_level": {},
    },
    "sandbox": {
        "enable": True,
        "max_branch": 5,
        "risk_score_enable": True,
        "conflict_converge": True,
        "save_branch_snapshot": True,
        "parallel_workers": 5,
        "veto_threshold": 80,
        "seed_salt": "yuanyandao-sandbox-v7",
        "risk_dimensions": [],
    },
    "train_scheduler": {
        "enable": True,
        "default_lora_rank": 8,
        "default_lr": 2e-4,
        "train_batch_size": 2,
        "gradient_accumulation": 4,
        "loss_type": "dpo",
        "output_dir": "./train_output",
        "dataset_path": "./datasets",
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "max_seq_len": 1024,
        "warmup_ratio": 0.03,
        "num_epochs": 2,
        "lr_scheduler": "cosine",
        "optim": "paged_adamw_8bit",
        "bf16": True,
        "gradient_checkpointing": True,
        "dpo_beta": 0.1,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    },
    "audit": {
        "log_path": "./audit/log.jsonl",
        "snapshot_save": True,
        "snapshot_dir": "./audit/snapshot/",
        "hash_chain": True,
        "log_fields": [
            "user_id",
            "permission_level",
            "input",
            "output",
            "hit_keywords",
            "source_doc",
            "timestamp",
            "branch_id",
            "op_code",
        ],
    },
    "output_format": {
        "markdown": True,
        "json": True,
        "csv": True,
        "keyword_comma": True,
        "default": "markdown",
    },
    "safety": {
        "anchor_lock": "元衍道枢",
        "forbid_hallucinate": True,
        "role_isolate": True,
        "max_input_chars": 8000,
        "min_answer_chars": 8,
        "banned_patterns": [],
    },
    "server": {
        "host": "127.0.0.1",
        "port": 18080,
        "max_body_bytes": 1048576,
        "token": "",
    },
}

_PATH_KEYS = (
    "runtime.data_dir",
    "runtime.state_dir",
    "vector_store.index_path",
    "vector_store.metadata_path",
    "audit.log_path",
    "audit.snapshot_dir",
    "train_scheduler.output_dir",
    "train_scheduler.dataset_path",
)


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Config:
    """只读配置视图 + 路径解析。"""

    def __init__(self, raw: Dict[str, Any], path: Optional[Path] = None, root: Optional[Path] = None):
        self.raw = raw
        self.path = Path(path) if path else DEFAULT_CONFIG_PATH
        if root is not None:
            self.root = Path(root).resolve()
        elif path is not None:
            self.root = Path(path).resolve().parent.parent
        else:
            self.root = PROJECT_ROOT
        self._resolved: Dict[str, Path] = {}
        for key in _PATH_KEYS:
            value = self.get(key)
            if isinstance(value, str) and value:
                self._resolved[key] = (self.root / value).resolve()

    # -- 访问 ---------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> Dict[str, Any]:
        value = self.raw.get(name)
        return value if isinstance(value, dict) else {}

    def path_for(self, dotted: str, create: bool = False) -> Path:
        if dotted in self._resolved:
            path = self._resolved[dotted]
        else:
            value = self.get(dotted)
            if not isinstance(value, str):
                raise KeyError("未配置路径: %s" % dotted)
            path = (self.root / value).resolve()
            self._resolved[dotted] = path
        if create:
            if path.suffix:
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                path.mkdir(parents=True, exist_ok=True)
        return path

    # -- 校验 ---------------------------------------------------------
    def validate(self) -> None:
        anchor = self.get("anchor_key")
        if anchor != "元衍道枢" or self.get("safety.anchor_lock") != "元衍道枢":
            raise ValueError("锚点密钥校验失败：anchor_key/safety.anchor_lock 必须为“元衍道枢”")
        levels = self.get("permission.level_list", [])
        if list(levels) != [0, 1, 2, 3, 4]:
            raise ValueError("permission.level_list 必须为 [0,1,2,3,4]")
        engine = str(self.get("vector_store.engine", "memory")).lower()
        if engine not in ("memory", "faiss", "chroma"):
            raise ValueError("vector_store.engine 仅支持 memory / faiss / chroma")
        if not self.get("sandbox.risk_dimensions"):
            raise ValueError("sandbox.risk_dimensions 不能为空")
        total = sum(float(item.get("weight", 0)) for item in self.get("sandbox.risk_dimensions"))
        if abs(total - 1.0) > 0.02:
            raise ValueError("风险维度权重之和应为 1.0，当前为 %.3f" % total)

    def ensure_dirs(self) -> None:
        for key in (
            "runtime.data_dir",
            "runtime.state_dir",
            "audit.log_path",
            "audit.snapshot_dir",
            "train_scheduler.output_dir",
        ):
            self.path_for(key, create=True)

    def describe(self) -> Dict[str, Any]:
        return {
            "plugin": self.get("plugin_name"),
            "version": self.get("version"),
            "anchor": self.get("anchor_key"),
            "base_model": self.get("hardware.base_model"),
            "engine": self.get("vector_store.engine"),
            "top_n": self.get("vector_store.top_n"),
            "project_root": str(self.root),
        }


def _load_dotenv(path: Path) -> None:
    """读取项目根目录的 .env（仅补缺，不覆盖已有环境变量）。"""
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def load_config(path: Optional[str] = None, root: Optional[str] = None) -> Config:
    """装载配置；``path`` 为空时使用仓库内 config/config.yaml。"""
    cfg_path = Path(path).resolve() if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError("配置文件不存在: %s" % cfg_path)
    _load_dotenv(cfg_path.parent.parent / ".env")
    raw = _load_yaml_text(cfg_path)
    merged = _deep_merge(DEFAULTS, raw)
    config = Config(merged, path=cfg_path, root=Path(root).resolve() if root else None)
    config.validate()
    return config
