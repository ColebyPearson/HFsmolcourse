# Unit 1 — Practice Quiz (Instruction Tuning)

Drawn from the 7 chapters: introduction, chat templates, SFT, LoRA/PEFT,
practical exercises, HF Jobs, submit your final project.

> Phrasing on the HF site (`smol-course/unit_1_quiz`) will differ — make sure
> you can *explain* the right answer. ≥ 12/15 = comfortable margin.

---

## Quiz territory — Instruction Tuning

**1.** Which statement best captures what SFT does, per the unit?
- a) Teaches the base model entirely new facts about the world.
- b) Reshapes how the model applies existing pre-trained knowledge — instruction patterns, response styles, formats — without teaching new facts.
- c) Replaces pre-training; SFT alone is enough to train a model from scratch.
- d) Is only useful for code-generation tasks.

**2.** The SmolLM3 instruction-following pipeline goes:
- a) Base → preference alignment → SFT
- b) Base → SFT (on curated instruction data, including SmolTalk2) → preference alignment (e.g. APO)
- c) Base + RAG only, no fine-tuning
- d) SFT only — SmolLM3 has no preference-alignment stage

**3.** The TRL class used to drive SFT in the course is:
- a) `transformers.Trainer`
- b) `trl.SFTTrainer`
- c) `peft.PeftTrainer`
- d) `accelerate.SFT`

**4.** Recommended **starting** learning rate for SFT'ing SmolLM3 in the unit:
- a) `5e-5` (conservative / stable)
- b) `1e-3`
- c) `1e-2`
- d) `0.1`

**5.** Which dataset-size guidance does the unit give?
- a) ≥ 1,000 high-quality examples is the **minimum**; 10,000+ for robust performance
- b) Always at least 1 million examples
- c) Quantity matters more than quality
- d) 100 examples is plenty

**6.** Per the unit, healthy SFT loss curves are characterised by:
- a) Training loss decreasing while validation loss stays constant
- b) Both losses decreasing together with a small gap between them
- c) Training loss spiking repeatedly
- d) Validation loss higher than training loss by orders of magnitude

**7.** Which pattern indicates **overfitting** during SFT?
- a) Training loss flat, validation loss flat
- b) Training loss continues to decrease while validation loss starts to *increase*
- c) Both losses oscillate identically
- d) Loss never moves

**8.** A SmolLM3-style conversational training example uses:
- a) `<|im_start|>role` … `<|im_end|>` blocks (per-message turn markers)
- b) HTML tags around each turn
- c) JSON tool calls only
- d) Raw text with no role boundaries

**9.** PEFT (LoRA) reduces *trainable* parameters by roughly:
- a) ~10% — modest savings
- b) ~50%
- c) ~90% — typically about an order of magnitude less
- d) 99.9999% — virtually no parameters train

**10.** A reasonable LoRA starting point per the unit:
- a) `r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM"`
- b) `r=1, alpha=1`
- c) `r=512, alpha=2048` — always go big
- d) LoRA configuration doesn't matter; defaults are universal

**11.** "Merging the adapter" after a LoRA SFT means:
- a) Bolting on a second adapter
- b) Combining adapter weights with the frozen base into a single set of weights so inference doesn't need to load adapters separately
- c) Concatenating two datasets
- d) Joining `train` and `validation` splits

**12.** A typical dataset format SFTTrainer accepts is:
- a) Only `{"text": ...}`
- b) Only `{"prompt": ..., "completion": ...}`
- c) Both **standard** and **conversational** forms — language-modeling (`text`), prompt-completion, or `messages: [{role, content}, ...]` — and applies chat templates automatically for conversational data
- d) A custom binary format only

**13.** "Gradient accumulation" exists in the SFT recipe because:
- a) It speeds up the optimiser
- b) It lets you simulate a larger effective batch size when per-device memory is tight — e.g. `per_device_train_batch_size=2 × gradient_accumulation_steps=8` ≈ effective batch 16
- c) It's required by Trackio
- d) It disables backprop

**14.** **Hugging Face Jobs** is presented in the unit as:
- a) A way to install local packages
- b) A managed cloud training surface so you don't have to provision GPUs yourself — submit, monitor, retrieve the trained model
- c) A replacement for git
- d) A separate model checkpoint format

**15.** Before starting SFT, the unit's "should I fine-tune?" checklist asks (among other things):
- a) Have you tried prompt-engineering an already-instruction-tuned model first?
- b) Do you have at least 1 GPU-year of compute?
- c) Can you pre-train a base model from scratch?
- d) Have you written your own transformer architecture?

---

## Answers

1. **b** — SFT primarily reshapes *behaviour*; the pre-training step is what installs world knowledge. The unit cites Wei et al. and Ouyang et al. on this point.
2. **b** — Base → SFT on SmolTalk2-style data → preference alignment (APO for SmolLM3). The course's Unit 2 is the preference-alignment stage.
3. **b** — `trl.SFTTrainer` is the course's go-to; it composes Transformers + Datasets + Accelerate + PEFT for you.
4. **a** — `5e-5` is the conservative starting point the unit calls out for SmolLM3. `1e-4` is "balanced"; `2e-4` is "aggressive."
5. **a** — 1,000 minimum; 10,000+ for robust. Quality > quantity is the explicit emphasis.
6. **b** — Both decreasing with a small gap = good generalisation. Big and widening gap = overfitting.
7. **b** — Train loss ↓ while validation loss ↑ is the signature of overfitting. Fix: fewer steps, more/diverse data, early stopping, regularisation.
8. **a** — SmolLM3's chat template uses `<|im_start|>role` … `<|im_end|>` per turn (you can see this in the dataset sample — it's already templated for you in `smoltalk2_everyday_convs_think`).
9. **c** — ~90% reduction is the figure the unit cites; for the GPT-3 175B case the paper showed up to 10,000× fewer trainable params and 3× less GPU memory.
10. **a** — `r=16, alpha=32, dropout=0.05, task_type=CAUSAL_LM` is the example used; the unit also mentions starting small (r=4–8) when iterating.
11. **b** — `merge_and_unload()` combines adapter into the base; resulting weights deploy without PEFT at inference time.
12. **c** — Both standard and conversational dataset formats. SFTTrainer applies the chat template automatically on conversational data.
13. **b** — Gradient accumulation lets a 16 GB GPU train at a larger *effective* batch by accumulating gradients across multiple forward passes before each optimiser step.
14. **b** — Managed cloud training. You submit a job; HF runs it on its GPUs; you get the trained model back. Useful when you don't have local GPU access.
15. **a** — Try prompting an existing instruction model first; SFT is a meaningful engineering investment and only worth it when prompting won't get the behaviour / format / domain you need.

---

## After the real quiz

Pass ≥ 70% on `https://huggingface.co/spaces/smol-course/unit_1_quiz` → you've
satisfied the **Fundamentals certificate** requirement. The remaining cert
("Completion") needs all released units + a final project (Unit 1 chapter 6
introduces what the project submission looks like).
