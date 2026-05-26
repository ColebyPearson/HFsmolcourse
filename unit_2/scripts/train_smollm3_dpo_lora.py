"""Unit 2 — DPO SmolLM3-3B + LoRA on a preference dataset.

Follows Unit 2's recipe:
  - Reference / starting policy:  HuggingFaceTB/SmolLM3-3B (already-SFT'd
                                   instruction-tuned model; the course's
                                   default starting point for DPO)
  - Preference dataset:           trl-lib/ultrafeedback_binarized
                                   (canonical TRL preference format —
                                   prompt + chosen + rejected as message lists;
                                   chat template applied automatically)
  - Method:                       TRL DPOTrainer + PEFT LoRA + 4-bit
                                   bitsandbytes quant. When LoRA is used,
                                   DPOTrainer reuses the base weights as
                                   pi_ref by disabling the adapter during
                                   the reference forward pass — no separate
                                   ref_model needed.
  - Hyperparams:                  beta=0.1, lr=5e-7 (course recipe, much
                                   lower than SFT), max_steps=200 (trimmed
                                   from the chapter's 1000 to fit the local
                                   16 GB GPU's wall-clock budget; cosine LR.
                                   batch=1 x grad-accum=8 = effective 8
                                   (DPO sees chosen+rejected per step, so
                                   memory is ~2x SFT).
                                   max_prompt_length=512, max_length=1024.
  - Push:                         only after training succeeds. hub_model_id
                                   set in DPOConfig (the lesson learned in
                                   Unit 1 — otherwise push goes to
                                   output_dir basename).

Run:
  PYTHONUTF8=1 python unit_2/scripts/train_smollm3_dpo_lora.py
"""
from __future__ import annotations

import multiprocessing
import os
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

MODEL_ID = os.getenv("SMOL_MODEL", "HuggingFaceTB/SmolLM3-3B")
DATASET_ID = os.getenv("SMOL_DATASET", "trl-lib/ultrafeedback_binarized")
OUT_DIR = os.getenv("SMOL_OUT", "runs/smollm3-3b-dpo-lora")
HUB_REPO = os.getenv("SMOL_HUB_REPO", "VoicesColeby/smollm3-3b-dpo-lora")
MAX_STEPS = int(os.getenv("SMOL_MAX_STEPS", "200"))
MAX_LEN = int(os.getenv("SMOL_MAX_LEN", "1024"))
MAX_PROMPT_LEN = int(os.getenv("SMOL_MAX_PROMPT_LEN", "512"))
BETA = float(os.getenv("SMOL_BETA", "0.1"))


def main() -> int:
    print(f"[cfg] model={MODEL_ID}", flush=True)
    print(f"[cfg] dataset={DATASET_ID}", flush=True)
    print(f"[cfg] out={OUT_DIR}  hub={HUB_REPO}", flush=True)
    print(
        f"[cfg] max_steps={MAX_STEPS}  max_length={MAX_LEN}  "
        f"max_prompt_length={MAX_PROMPT_LEN}  beta={BETA}  "
        f"cuda={torch.cuda.is_available()}",
        flush=True,
    )

    # --- tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 4-bit quantized base + bf16 compute (cached from Unit 1 if same base)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print("[model] loading base in 4-bit…", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # --- dataset: prompt / chosen / rejected as message lists
    print(f"[data] loading {DATASET_ID}…", flush=True)
    ds = load_dataset(DATASET_ID, split="train")
    print(f"[data] rows={len(ds)}  columns={ds.column_names}", flush=True)

    # --- LoRA — same SmolLM3 target modules as Unit 1
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # --- DPO config
    args = DPOConfig(
        output_dir=OUT_DIR,
        hub_model_id=HUB_REPO,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,            # effective batch 8
        max_steps=MAX_STEPS,
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        beta=BETA,
        max_prompt_length=MAX_PROMPT_LEN,
        max_length=MAX_LEN,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        report_to="none",
        push_to_hub=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
    )

    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        peft_config=lora,
        processing_class=tokenizer,
    )
    print("[train] starting…", flush=True)
    trainer.train()
    print("[train] complete.", flush=True)

    print(f"[push] pushing adapter to {HUB_REPO}…", flush=True)
    trainer.push_to_hub(
        commit_message="DPO + LoRA on ultrafeedback_binarized (beta=0.1, 200 steps)",
    )
    print("[push] done.", flush=True)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
