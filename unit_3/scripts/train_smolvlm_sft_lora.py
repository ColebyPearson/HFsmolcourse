"""Unit 3 — SFT SmolVLM2-2.2B-Instruct + LoRA on ChartQA.

Follows Unit 3's recipe:
  - Base model:  HuggingFaceTB/SmolVLM2-2.2B-Instruct (Vision-Language Model)
  - Dataset:     HuggingFaceM4/ChartQA (chart visual question answering;
                 we use 10% slices of train + val for the course-recommended
                 budget)
  - Method:      TRL SFTTrainer + PEFT LoRA. VLM-specific:
                   * AutoProcessor (not AutoTokenizer) — handles text + images
                   * AutoModelForImageTextToText (not AutoModelForCausalLM)
                   * dataset must have an `images` column AND a `messages`
                     column with content lists holding both image dicts and
                     text dicts
                   * MUST set `max_length=None` on SFTConfig (otherwise
                     truncation removes image tokens)
                   * narrower LoRA target_modules — just q_proj + v_proj
  - Hyperparams: lr=1e-4, batch=1 x grad-accum=4 = effective 4, max_steps=150
                 (course recipe uses 1 epoch ~ 450 steps on 10% slice; we
                 cap at 150 to fit a ~30-45 min wall-clock on the local
                 16 GB GPU. cosine LR, bf16 (no 4-bit — 2.2B params fit
                 comfortably and 4-bit + vision-tower has been historically
                 flaky).
  - Push:        only after training succeeds. hub_model_id set in
                 SFTConfig (the lesson from Unit 1).

Run:
  PYTHONUTF8=1 python unit_3/scripts/train_smolvlm_sft_lora.py
"""
from __future__ import annotations

import multiprocessing
import os
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTConfig, SFTTrainer

MODEL_ID = os.getenv("SMOL_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
DATASET_ID = os.getenv("SMOL_DATASET", "HuggingFaceM4/ChartQA")
OUT_DIR = os.getenv("SMOL_OUT", "runs/smolvlm2-2.2b-sft-lora-chartqa")
HUB_REPO = os.getenv("SMOL_HUB_REPO", "VoicesColeby/smolvlm2-2.2b-sft-lora-chartqa")
MAX_STEPS = int(os.getenv("SMOL_MAX_STEPS", "150"))

SYSTEM_MESSAGE = (
    "You are a Vision Language Model specialised in interpreting visual data "
    "from chart images. Analyse the provided chart image and respond to "
    "queries with concise answers — usually a single word, number, or short "
    "phrase. The charts include line, bar, and pie types with colours, "
    "labels, and text. Be accurate and succinct; no extra explanation unless "
    "absolutely necessary."
)


def format_example(sample):
    """Map ChartQA's {image, query, label} into TRL's VLM SFT format.

    TRL's SFT collator (`prepare_multimodal_messages`) expects `content` to be
    a *string* containing `<image>` placeholders that match the number of
    entries in `images`. The chapter's typed-parts format
    (`content: [{"type":"image",...},{"type":"text",...}]`) predates that
    collator and isn't compatible with current TRL — it raises
    `ValueError: Number of images provided (1) does not match number of image
    placeholders (0)`.
    """
    return {
        "images": [sample["image"]],
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": sample["query"]},  # TRL collator auto-inserts <image> placeholders
            {"role": "assistant", "content": sample["label"][0]},
        ],
    }


def main() -> int:
    print(f"[cfg] model={MODEL_ID}", flush=True)
    print(f"[cfg] dataset={DATASET_ID} (using 10% slice of train)", flush=True)
    print(f"[cfg] out={OUT_DIR}  hub={HUB_REPO}", flush=True)
    print(f"[cfg] max_steps={MAX_STEPS}  cuda={torch.cuda.is_available()}", flush=True)

    # --- processor + model (bf16, no 4-bit — vision tower is finicky with bnb)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("[model] loading SmolVLM2 in bf16…", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto",
    )
    model.config.use_cache = False

    # --- dataset (10% slice; the course uses the same scope)
    print(f"[data] loading {DATASET_ID}…", flush=True)
    raw_train = load_dataset(DATASET_ID, split="train[:10%]")
    print(f"[data] raw rows={len(raw_train)}  columns={raw_train.column_names}", flush=True)
    train_ds = [format_example(s) for s in raw_train]
    print(f"[data] formatted rows={len(train_ds)}", flush=True)

    # --- LoRA (VLM-typical: just q/v_proj; alpha=16, r=8 per course recipe)
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )

    # --- SFT config — CRITICAL: max_length=None so image tokens aren't
    # truncated; the chapter explicitly warns about this.
    args = SFTConfig(
        output_dir=OUT_DIR,
        hub_model_id=HUB_REPO,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,            # effective batch 4
        max_steps=MAX_STEPS,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        max_length=None,                          # <-- the key VLM gotcha
        bf16=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        optim="adamw_torch_fused",
        report_to="none",
        push_to_hub=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,              # keep `images` for the collator
        dataset_kwargs={"skip_prepare_dataset": True},  # we already formatted
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        peft_config=lora,
        processing_class=processor,
    )
    print("[train] starting…", flush=True)
    trainer.train()
    print("[train] complete.", flush=True)

    print(f"[push] pushing adapter to {HUB_REPO}…", flush=True)
    trainer.push_to_hub(
        commit_message="VLM SFT + LoRA on ChartQA (10% slice, 150 steps)",
    )
    print("[push] done.", flush=True)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
