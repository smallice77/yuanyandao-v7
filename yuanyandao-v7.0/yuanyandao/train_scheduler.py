# -*- coding: utf-8 -*-
"""训练调度子单元：RTX5060 8G 专属 QLoRA 模板（SFT / DPO）。

输出三件套：可直接运行的 YAML 配置、训练脚本、数据集清洗与格式化方案，
并附带显存估算，避免“参数看起来对、真跑就 OOM”。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

__all__ = ["TrainScheduler"]


def _fmt_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value != 0 and abs(value) < 1e-3:
            return repr(value)
        return str(value)
    text = str(value)
    if text == "":
        return '""'
    safe = all(ch.isalnum() or ch in "._-/" for ch in text) and not text[0].isdigit()
    if safe:
        return text
    return json.dumps(text, ensure_ascii=False)


def dump_yaml(data: object, indent: int = 0) -> str:
    """确定性 YAML 输出（不依赖 PyYAML，避免环境差异导致格式漂移）。"""
    pad = " " * indent
    lines: List[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            key_text = str(key)
            if not all(ch.isalnum() or ch in "_-" for ch in key_text):
                key_text = json.dumps(key_text, ensure_ascii=False)
            if isinstance(value, dict):
                lines.append("%s%s:" % (pad, key_text))
                lines.append(dump_yaml(value, indent + 2).rstrip("\n"))
            elif isinstance(value, list):
                lines.append("%s%s:" % (pad, key_text))
                for item in value:
                    if isinstance(item, dict):
                        body = dump_yaml(item, indent + 4).rstrip("\n").splitlines()
                        first = body[0].lstrip() if body else ""
                        lines.append("%s  - %s" % (pad, first))
                        lines.extend(body[1:])
                    else:
                        lines.append("%s  - %s" % (pad, _fmt_scalar(item)))
            else:
                lines.append("%s%s: %s" % (pad, key_text, _fmt_scalar(value)))
    elif isinstance(data, list):
        for item in data:
            lines.append("%s- %s" % (pad, _fmt_scalar(item)))
    else:
        lines.append("%s%s" % (pad, _fmt_scalar(data)))
    return "\n".join(lines) + "\n"


_TRAIN_SCRIPT = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RTX5060 8G · Qwen2.5-7B-Base 4bit QLoRA 训练脚本（由元衍道枢 V7.0 生成）。

用法：
    python train.py --config ./train_output/<run_name>/qlora_config.yaml
    python train.py --config ... --dry-run     # 只加载模型并跑 2 个 step
"""

import argparse
import json
import os

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="qlora_config.yaml 路径")
    parser.add_argument("--dry-run", action="store_true", help="只验证加载与单步前向")
    parser.add_argument("--max-steps", type=int, default=-1)
    return parser.parse_args()


def build_model(cfg):
    quant = BitsAndBytesConfig(
        load_in_4bit=cfg["model"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["model"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["model"]["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base_model"],
        quantization_config=quant,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"]
    )
    lora = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def build_dataset(cfg, tokenizer):
    data = load_dataset("json", data_files=cfg["dataset"]["path"], split="train")
    print("样本数:", len(data), "字段:", data.column_names)
    return data


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    os.makedirs(cfg["training"]["output_dir"], exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = build_model(cfg)
    dataset = build_dataset(cfg, tokenizer)

    stage = cfg["training"]["stage"]
    if args.dry_run:
        batch = tokenizer(dataset[0]["prompt"], return_tensors="pt", truncation=True,
                          max_length=cfg["training"]["max_length"]).to("cuda")
        with torch.no_grad():
            model(**batch)
        print("dry-run OK, peak VRAM: %.2f GB" % (torch.cuda.max_memory_allocated() / 1024**3))
        return

    common = dict(
        output_dir=cfg["training"]["output_dir"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        optim=cfg["training"]["optim"],
        bf16=cfg["training"]["bf16"],
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to=[],
        max_steps=args.max_steps,
        seed=cfg["training"]["seed"],
    )

    if stage == "dpo":
        from trl import DPOConfig, DPOTrainer

        train_args = DPOConfig(beta=cfg["dpo"]["beta"], max_length=cfg["training"]["max_length"], **common)
        trainer = DPOTrainer(
            model=model,
            args=train_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
    else:
        from trl import SFTConfig, SFTTrainer

        train_args = SFTConfig(max_length=cfg["training"]["max_length"], **common)
        trainer = SFTTrainer(
            model=model,
            args=train_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

    trainer.train()
    trainer.save_model(cfg["training"]["output_dir"])
    tokenizer.save_pretrained(cfg["training"]["output_dir"])
    print("训练完成，适配器已保存到", cfg["training"]["output_dir"])


if __name__ == "__main__":
    main()
'''


class TrainScheduler:
    def __init__(self, config):
        self.config = config
        ts = config.section("train_scheduler")
        hw = config.section("hardware")
        self.cfg = ts
        self.base_model = str(hw.get("base_model", "qwen2.5-7b-base"))
        self.hf_model_id = str(hw.get("hf_model_id", "Qwen/Qwen2.5-7B"))
        self.vram_limit = float(hw.get("vram_limit_gb", 8))

    # ------------------------------------------------------------------
    def estimate_vram(self, params_b: float = 7.6, seq_len: int = 1024, batch: int = 1,
                      lora_r: int = 8, grad_ckpt: bool = True) -> Dict[str, object]:
        """粗粒度显存估算：权重 + 适配器 + 激活 + logits，用于训练前预警。"""
        weights = params_b * 0.5 * 1.06
        adapters = 0.03 + 0.002 * lora_r
        activation = 0.28 * (params_b / 7.6) * (seq_len / 1024.0) * batch * (2.4 if grad_ckpt else 6.0)
        logits = 152064 * seq_len * batch * 2 / (1024 ** 3)
        total = weights + adapters + activation + logits
        return {
            "formula": "权重(4bit) + LoRA/优化器 + 激活(含梯度检查点系数) + logits",
            "weights_gb": round(weights, 2),
            "adapters_gb": round(adapters, 2),
            "activation_gb": round(activation, 2),
            "logits_gb": round(logits, 2),
            "estimated_peak_gb": round(total, 2),
            "limit_gb": self.vram_limit,
            "headroom_gb": round(self.vram_limit - total, 2),
            "fits": total <= self.vram_limit - 0.4,
            "suggestion": (
                "余量充足，可尝试把序列长度提到 2048"
                if total <= self.vram_limit - 1.8
                else "余量偏紧，保持 seq_len=1024、batch=1，并开启梯度检查点"
                if total <= self.vram_limit - 0.4
                else "预计超出 8GB，请降低序列长度或批大小"
            ),
        }

    # ------------------------------------------------------------------
    def build_config(self, loss_type: Optional[str] = None, run_name: Optional[str] = None,
                     seq_len: Optional[int] = None, batch: Optional[int] = None,
                     lora_r: Optional[int] = None) -> Dict[str, object]:
        ts = self.cfg
        loss = (loss_type or str(ts.get("loss_type", "dpo"))).lower()
        if loss not in ("sft", "dpo"):
            loss = "dpo"
        run = run_name or "qwen25-7b-qlora-%s" % time.strftime("%Y%m%d-%H%M")
        seq = int(seq_len or ts.get("max_seq_len", 1024))
        bs = int(batch or ts.get("train_batch_size", 2))
        rank = int(lora_r or ts.get("default_lora_rank", 8))
        output_dir = (Path(str(ts.get("output_dir", "./train_output"))) / run).as_posix()
        stage_needs = (
            {"prompt": "string", "chosen": "string", "rejected": "string"}
            if loss == "dpo"
            else {"prompt": "string", "response": "string"}
        )
        return {
            "run_name": run,
            "model": {
                "base_model": self.hf_model_id,
                "load_in_4bit": True,
                "bnb_4bit_quant_type": str(self.config.get("hardware.quant_type", "nf4")),
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": "bfloat16",
                "trust_remote_code": True,
            },
            "lora": {
                "r": rank,
                "alpha": int(ts.get("lora_alpha", 16)),
                "dropout": float(ts.get("lora_dropout", 0.05)),
                "target_modules": list(ts.get("target_modules", [])),
            },
            "training": {
                "stage": loss,
                "per_device_train_batch_size": bs,
                "gradient_accumulation_steps": int(ts.get("gradient_accumulation", 4)),
                "learning_rate": float(ts.get("default_lr", 2e-4)),
                "num_train_epochs": int(ts.get("num_epochs", 2)),
                "lr_scheduler_type": str(ts.get("lr_scheduler", "cosine")),
                "warmup_ratio": float(ts.get("warmup_ratio", 0.03)),
                "optim": str(ts.get("optim", "paged_adamw_8bit")),
                "bf16": bool(ts.get("bf16", True)),
                "gradient_checkpointing": bool(ts.get("gradient_checkpointing", True)),
                "max_length": seq,
                "output_dir": output_dir,
                "seed": 42,
            },
            "dpo": {"beta": float(ts.get("dpo_beta", 0.1)), "loss_type": "sigmoid"},
            "dataset": {
                "path": (Path(str(ts.get("dataset_path", "./datasets"))) / "train.jsonl").as_posix(),
                "format": "jsonl",
                "fields": stage_needs,
            },
            "vram": self.estimate_vram(seq_len=seq, batch=bs, lora_r=rank,
                                       grad_ckpt=bool(ts.get("gradient_checkpointing", True))),
        }

    # ------------------------------------------------------------------
    def dataset_plan(self, loss_type: str = "dpo") -> Dict[str, object]:
        return {
            "input_dir": str(self.cfg.get("dataset_path", "./datasets")),
            "target_files": ["train.jsonl", "valid.jsonl"],
            "record_format": (
                '{"prompt": "...", "chosen": "...", "rejected": "..."}'
                if loss_type == "dpo"
                else '{"prompt": "...", "response": "..."}'
            ),
            "steps": [
                {"step": 1, "name": "格式统一", "detail": "全部转为 UTF-8 JSONL，一行一条；字段名与模板严格一致"},
                {"step": 2, "name": "去重", "detail": "精确去重后做 MinHash/SimHash 近似去重，阈值 0.9，防止评测集泄漏"},
                {"step": 3, "name": "长度过滤", "detail": "按 tokenizer 实际 token 数过滤，保留 16~%d token 区间" % int(self.cfg.get("max_seq_len", 1024))},
                {"step": 4, "name": "质量过滤", "detail": "剔除空回答、重复率过高、指令与回答语言不一致的样本"},
                {"step": 5, "name": "脱敏", "detail": "正则替换手机号、邮箱、身份证、内网路径与密钥串"},
                {"step": 6, "name": "偏好对校验", "detail": "DPO 需保证 chosen 与 rejected 针对同一 prompt，且长度差不超过 3 倍"} if loss_type == "dpo" else {"step": 6, "name": "回答校验", "detail": "SFT 需保证回答完整、无截断、无占位符"},
                {"step": 7, "name": "分层切分", "detail": "按来源分层切分 train/valid（9:1），避免同源泄漏"},
                {"step": 8, "name": "统计留档", "detail": "输出样本数、token 分布、去重率与过滤原因分布"},
            ],
            "commands": [
                "python -c \"import json;print(sum(1 for _ in open('datasets/train.jsonl',encoding='utf-8')))\"",
                "python -m json.tool --json-lines datasets/train.jsonl > /dev/null",
            ],
            "notes": [
                "8G 显存下建议先用 500~2000 条高质量样本跑通再扩量",
                "验证集必须与训练集不同源，否则指标不可信",
            ],
        }

    # ------------------------------------------------------------------
    def materialize(self, plan: Dict[str, object]) -> Dict[str, str]:
        """把 yaml / 脚本 / 数据方案写入 output_dir，返回文件路径。"""
        train_cfg = plan["config"]
        run_name = str(train_cfg["run_name"])
        out_dir = Path(str(train_cfg["training"]["output_dir"]))
        if not out_dir.is_absolute():
            out_dir = Path(self.config.root) / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = out_dir / "qlora_config.yaml"
        script_path = out_dir / "train.py"
        data_path = out_dir / "dataset_plan.json"
        yaml_path.write_text(dump_yaml(train_cfg), encoding="utf-8")
        script_path.write_text(_TRAIN_SCRIPT, encoding="utf-8")
        data_path.write_text(json.dumps(plan["dataset"], ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "config_path": str(yaml_path),
            "script_path": str(script_path),
            "dataset_plan_path": str(data_path),
        }

    # ------------------------------------------------------------------
    def build_plan(self, loss_type: Optional[str] = None, run_name: Optional[str] = None) -> Dict[str, object]:
        cfg = self.build_config(loss_type=loss_type, run_name=run_name)
        yaml_text = dump_yaml(cfg)
        stage = str(cfg["training"]["stage"]).upper()
        plan = {
            "stage": stage,
            "config": cfg,
            "yaml": yaml_text,
            "script": _TRAIN_SCRIPT,
            "dataset": self.dataset_plan(str(cfg["training"]["stage"])),
            "vram": cfg["vram"],
            "checklist": [
                "先用 --dry-run 验证模型加载与单步前向",
                "确认 tokenizer 的 pad_token 与 padding_side",
                "确认数据集字段名与 dataset.fields 一致",
                "训练前记录 nvidia-smi 基线占用",
                "每 200 step 存一次检查点，避免 OOM 后全部重跑",
            ],
        }
        try:
            plan["files"] = self.materialize(plan)
        except OSError as exc:
            plan["files"] = {}
            plan["file_error"] = str(exc)
        return plan
