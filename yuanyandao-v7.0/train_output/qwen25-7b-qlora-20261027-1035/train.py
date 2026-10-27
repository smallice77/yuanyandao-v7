#!/usr/bin/env python
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
