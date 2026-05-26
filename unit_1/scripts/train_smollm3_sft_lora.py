"""Unit 1 — SFT SmolLM3-3B-Base + LoRA on smoltalk2_everyday_convs_think.

Follows the course's `hands-on` recipe:
  - Base model:  HuggingFaceTB/SmolLM3-3B-Base
  - Dataset:     HuggingFaceTB/smoltalk2_everyday_convs_think  (single 'text' column,
                 already SmolLM3-chat-templated by the dataset authors)
  - Method:      TRL SFTTrainer + PEFT LoRA + 4-bit quantization
                 (so it fits on the local 16 GB GPU)
  - Hyperparams: per_device_train_batch_size=2, gradient_accumulation_steps=8
                 (effective batch 16), max_steps=1000, lr=5e-5, warmup=100,
                 max_length=2048, bf16=True
  - Push:        only after training completes (no checkpoint-push)

Run:
  HF_TOKEN=hf_... python unit_1/scripts/train_smollm3_sft_lora.py
"""
from __future__ import annotations

import multiprocessing
import os
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = os.getenv("SMOL_MODEL", "HuggingFaceTB/SmolLM3-3B-Base")
DATASET_ID = os.getenv(
    "SMOL_DATASET", "HuggingFaceTB/smoltalk2_everyday_convs_think"
)
OUT_DIR = os.getenv("SMOL_OUT", "runs/smollm3-3b-sft-lora")
HUB_REPO = os.getenv(
    "SMOL_HUB_REPO", "VoicesColeby/smollm3-3b-sft-lora-smoltalk-everyday"
)
MAX_STEPS = int(os.getenv("SMOL_MAX_STEPS", "1000"))
MAX_LEN = int(os.getenv("SMOL_MAX_LEN", "2048"))


def main() -> int:
    print(f"[cfg] model={MODEL_ID}", flush=True)
    print(f"[cfg] dataset={DATASET_ID}", flush=True)
    print(f"[cfg] out={OUT_DIR}  hub={HUB_REPO}", flush=True)
    print(
        f"[cfg] max_steps={MAX_STEPS}  max_length={MAX_LEN}  "
        f"cuda={torch.cuda.is_available()}",
        flush=True,
    )

    # --- tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 4-bit quantized base model + bf16 compute
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print("[model] loading base in 4-bit (this can take a few minutes on first run)…", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.cache_implementation = "static"

    # --- dataset (single 'text' field already chat-templated)
    print(f"[data] loading {DATASET_ID}…", flush=True)
    ds = load_dataset(DATASET_ID, split="train")
    print(f"[data] rows={len(ds)}  columns={ds.column_names}", flush=True)

    # --- LoRA
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- SFT config
    args = SFTConfig(
        output_dir=OUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,            # effective batch 16
        max_steps=MAX_STEPS,
        learning_rate=5e-5,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        max_length=MAX_LEN,
        bf16=True,
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        report_to="none",
        push_to_hub=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        peft_config=lora,
        processing_class=tokenizer,
    )
    print("[train] starting…", flush=True)
    trainer.train()
    print("[train] complete.", flush=True)

    print(f"[push] saving + pushing adapter to {HUB_REPO}…", flush=True)
    trainer.push_to_hub(
        commit_message="SFT + LoRA on smoltalk2_everyday_convs_think",
    )
    print("[push] done.", flush=True)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
