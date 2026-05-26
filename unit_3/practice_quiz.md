# Unit 3 — Practice Quiz (Vision Language Models)

Drawn from the four Unit 3 chapters: introduction, using pretrained VLMs,
fine-tuning VLMs, hands-on with SmolVLM2.

> Phrasing on the HF site (`smol-course/unit_4_quiz` per the chapter) will
> differ — explain *why* the right answer is right. ≥ 12/15 = comfortable
> margin.

---

## Quiz territory — VLMs

**1.** A Vision Language Model is best described as:
- a) An LLM that has been fine-tuned to refuse image-related requests.
- b) A model that takes interleaved text and image inputs and produces text outputs — combining a vision encoder with a language decoder.
- c) An image-only classifier.
- d) A speech-to-text model.

**2.** The canonical VLM used throughout Unit 3's hands-on is:
- a) GPT-4V
- b) `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- c) BLIP-2
- d) Stable Diffusion

**3.** The dataset used in the SmolVLM hands-on is:
- a) ImageNet
- b) `HuggingFaceM4/ChartQA` (chart visual question answering)
- c) COCO captions only
- d) A text-only QA dataset

**4.** When fine-tuning VLMs with TRL's `SFTTrainer`, the dataset must include:
- a) Only a `text` column
- b) Both an `images` column **and** a `messages` column whose content lists include both image dicts and text dicts
- c) A `pixels` column with raw float arrays
- d) A `caption` column only

**5.** The critical SFTConfig setting the chapter warns about when training VLMs:
- a) `bf16=False`
- b) **`max_length=None`** — otherwise truncation can drop image tokens
- c) `report_to="trackio"` — required
- d) `gradient_checkpointing=False`

**6.** Loading a VLM in `transformers` typically uses:
- a) `AutoModelForCausalLM`
- b) `AutoModelForImageTextToText` (+ `AutoProcessor` for inputs)
- c) `AutoModelForImageClassification`
- d) `AutoTokenizer` alone

**7.** A typical LoRA `target_modules` choice for VLM fine-tuning is:
- a) Every Linear in the model
- b) Just `q_proj` and `v_proj` (narrower than typical LLM LoRA, lighter)
- c) Only the vision encoder's convolutions
- d) Only `lm_head`

**8.** Quantization is "**especially relevant**" for VLMs because:
- a) Images compress better than text.
- b) Image features inflate memory demands during training and inference, so 8-bit / 4-bit weight quantization buys headroom that matters more than for pure LLMs.
- c) Quantized models are required for ChartQA.
- d) Quantization is required to load a tokenizer.

**9.** The TRL CLI command for VLM SFT is:
- a) `trl vlm-sft …`
- b) `trl sft --model_name_or_path … --dataset_name …` (same `trl sft` command as LLMs)
- c) `trl image …`
- d) `peft sft …`

**10.** When applying the chat template to a VLM input, the user-turn `content` is:
- a) A single string
- b) A list of typed parts, e.g. `[{"type": "image", "image": img}, {"type": "text", "text": "How many bars?"}]`
- c) A binary blob
- d) Three separate columns

**11.** When SFT vs DPO is the right approach for a VLM:
- a) SFT for task-specific specialization (visual QA, captioning); DPO when you have preference-labelled outputs and care about subjective / multi-choice alignment.
- b) Always SFT.
- c) Always DPO.
- d) Neither — VLMs cannot be fine-tuned.

**12.** A pitfall the chapter explicitly calls out for VLM SFT:
- a) Setting `bf16=True` is unsafe for VLMs.
- b) **Overfitting** when the fine-tune set is narrow / small; remedies include using a subset for early validation and stopping early.
- c) VLMs cannot use gradient accumulation.
- d) LoRA is incompatible with VLMs.

**13.** "Effective batch" with `per_device_train_batch_size=4` and `gradient_accumulation_steps=4` is:
- a) 1
- b) 4
- c) **16**
- d) 32

**14.** Two memory-efficiency techniques the chapter recommends **combining** for consumer-GPU VLM training:
- a) LoRA + 4-bit (or 8-bit) quantization + gradient checkpointing
- b) Full fine-tune + fp32 + no accumulation
- c) CPU-only training
- d) `torch.compile()` only — nothing else

**15.** What `processor.apply_chat_template(messages, add_generation_prompt=True)` produces, given a user turn that contains both an image part and a text part:
- a) The image's bytes only.
- b) A single text string with the VLM's role / image / text markers (e.g. `<|im_start|>User:<image>How many bars?<end_of_utterance>\nAssistant:`) ready to pair with the actual image tensor at the processor call.
- c) A NumPy tensor of pixel values.
- d) An error — VLM tokenizers don't apply chat templates.

---

## Answers

1. **b** — VLM = vision encoder + LM decoder, takes interleaved image+text in, produces text out.
2. **b** — `HuggingFaceTB/SmolVLM2-2.2B-Instruct` is the chapter's reference VLM throughout.
3. **b** — ChartQA. The chapter uses a 10 % slice for fast iteration.
4. **b** — `images` column AND a `messages` column with image/text content parts. `SFTTrainer` needs both.
5. **b** — `max_length=None`. Otherwise image tokens can be truncated and training silently degrades.
6. **b** — `AutoModelForImageTextToText` + `AutoProcessor` (the processor wraps tokenizer + image processor).
7. **b** — Just `q_proj` + `v_proj` is the recipe shown; lighter than a typical LLM `q/k/v/o + MLP` set.
8. **b** — Image features add memory pressure; quantization (esp. 4-bit) is the most impactful single knob.
9. **b** — `trl sft …` — the same CLI command handles VLMs when the dataset is in the right shape and the model is loaded with the right Auto class.
10. **b** — Multimodal content is a *list* of typed parts: `{"type": "image", "image": ...}` and `{"type": "text", "text": ...}`.
11. **a** — SFT for task specialization; DPO when preferences matter (subjective / multi-choice / human-aligned).
12. **b** — Narrow / small fine-tune sets cause overfitting. Validate on a subset early and stop early.
13. **c** — `per_device_batch * grad_accum` = `4 * 4 = 16`.
14. **a** — LoRA + 4-bit/8-bit quant + gradient checkpointing — the standard "consumer GPU" trio.
15. **b** — Chat template produces a text string with the VLM's image/role markers; the actual image tensor goes alongside via the same processor.

---

## After the real quiz

The chapter points at `smol-course/unit_4_quiz` for Unit 3. (The course site
re-numbered the units since the toctree — Evaluation is "Unit 2" on the marketing
page but renumbered "Unit 4" in the live `units/en/`, hence the off-by-one in the
quiz URL.) Pass ≥ 70% → check the box and update PLAN.md.
