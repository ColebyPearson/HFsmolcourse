# Unit 2 — Practice Quiz (Preference Alignment / DPO)

Drawn from the four Unit 2 chapters: introduction, Direct Preference
Optimization, Advanced DPO Techniques, DPO Hands-on Implementation.

> Phrasing on the HF site will differ — explain *why* the right answer is
> right. ≥ 12/15 = comfortable margin.

---

## Quiz territory — DPO

**1.** Compared to classic RLHF (reward model + PPO), DPO's main simplification is:
- a) It runs entirely on CPU.
- b) It skips the separate reward-model training and the PPO loop, optimising the policy directly from preference pairs with a binary cross-entropy loss.
- c) It requires more steps than PPO for the same alignment quality.
- d) It eliminates the need for any human preferences.

**2.** DPO requires which prior step before you can run it?
- a) Pre-training from scratch on web data.
- b) Instruction-tune the base model with SFT (so the SFT model becomes the *reference* policy `π_ref` and the starting point for `π_θ`).
- c) Quantize the model to 4-bit first.
- d) Nothing — DPO replaces SFT entirely.

**3.** Each example in a DPO preference dataset contains, at minimum:
- a) `text`, `label`
- b) `prompt`, `chosen`, `rejected`
- c) `question`, `answer`, `score`
- d) `instruction`, `response`

**4.** In TRL the trainer you use is:
- a) `transformers.Trainer`
- b) `trl.SFTTrainer`
- c) `trl.DPOTrainer` (with `trl.DPOConfig`)
- d) `trl.PPOTrainer`

**5.** Recommended `beta` (β) range in DPOConfig:
- a) 0.001–0.01
- b) **0.1–0.5**
- c) 1.0–5.0
- d) 10–100

**6.** What does β control?
- a) The learning rate.
- b) The strength of preference optimisation vs staying close to the reference policy — lower β = more conservative (closer to `π_ref`); higher β = stronger alignment but more risk of overfitting / capability loss.
- c) Token sampling temperature at inference.
- d) Whether to use LoRA.

**7.** DPO learning rate compared to SFT learning rate is typically:
- a) Higher (~1e-3) for faster alignment
- b) **Lower** (~5e-7 to 5e-6) to prevent catastrophic forgetting and instability
- c) Identical
- d) Always 0.1

**8.** In the DPO loss
$L = -\mathbb{E}[\log \sigma(\beta \log \pi_\theta(y_w|x)/\pi_{\text{ref}}(y_w|x) - \beta \log \pi_\theta(y_l|x)/\pi_{\text{ref}}(y_l|x))]$,
what role does `π_ref` play?
- a) It's a reward model trained separately on preferences.
- b) It's the **reference policy** — usually the SFT model — held frozen during DPO so the new policy `π_θ` doesn't drift arbitrarily far from it.
- c) It's the rejected response distribution.
- d) It's irrelevant in modern DPO.

**9.** A common symptom of DPO **over**fitting and the remedy:
- a) Model becomes repetitive or loses general capabilities → lower β, reduce training time, or diversify the dataset.
- b) Model output is identical to the SFT model → raise β to 100.
- c) Loss never decreases → train longer at higher LR.
- d) Model becomes faster at inference → no action needed.

**10.** A common symptom of DPO **under**alignment and the remedy:
- a) Capability loss → lower β.
- b) Little to no improvement vs the SFT reference despite training → increase β, improve dataset quality, or extend training.
- c) Validation loss explodes → reduce dataset size.
- d) The model still uses chat templates → disable templates.

**11.** Recommended dataset size for DPO (per the unit):
- a) ~100 pairs is plenty.
- b) **Minimum ~1,000 high-quality preference pairs** for domain tasks; ~10,000+ for robust alignment.
- c) Always exactly 50,000 pairs.
- d) Quantity matters more than quality.

**12.** DPO training stability tips the unit calls out:
- a) Use gradient clipping; early-stop on plateau/regression; compare outputs vs the reference model regularly.
- b) Disable gradient clipping for speed.
- c) Always run for 100 epochs no matter what.
- d) Skip evaluation between epochs.

**13.** A representative preference dataset the course points at for hands-on DPO:
- a) `imagenet`
- b) `trl-lib/ultrafeedback_binarized` or `Anthropic/hh-rlhf` (preference / harmlessness datasets)
- c) `wikitext-103`
- d) The same SmolTalk2 dataset used for SFT — no special preference data needed

**14.** The `DPOTrainer` accepts a conversational dataset and:
- a) Errors out unless you pre-apply the chat template.
- b) Automatically applies the chat template; supports both **standard** and **conversational** preference formats and both **explicit** and **implicit** prompts (recommended: explicit).
- c) Requires you to disable the tokenizer.
- d) Only accepts plain CSV files.

**15.** The unit's `WARNING` about resources for the DPO hands-on:
- a) "You can run DPO on CPU comfortably."
- b) DPO training is compute-heavy; **HF Jobs requires Pro/Team/Enterprise**; locally you'll need ≥ 16 GB VRAM for SmolLM3-3B.
- c) "You can skip the SFT step."
- d) Inference is free but training requires a subscription to OpenAI.

---

## Answers

1. **b** — Skip the reward model and PPO; binary cross-entropy directly on preference pairs.
2. **b** — SFT first; the SFT model becomes `π_ref` and the warm-start for `π_θ`. The course says DPO "requires SFT to adapt the model to the target domain."
3. **b** — `prompt`, `chosen`, `rejected`. (TRL also supports implicit-prompt forms where prompt is extracted from chosen/rejected; explicit is recommended.)
4. **c** — `trl.DPOTrainer` driven by `trl.DPOConfig`. Same pattern as SFTTrainer/SFTConfig.
5. **b** — 0.1–0.5. Outside that you usually misalign or barely shift the model.
6. **b** — β controls the strength of preference optimisation vs reference adherence. The KL-like term `log π_θ / π_ref` is what β multiplies.
7. **b** — ~5e-7 to 5e-6 — orders of magnitude lower than SFT. Higher LR risks catastrophic forgetting.
8. **b** — `π_ref` is the reference policy (frozen SFT). The loss is a log-odds-ratio between `π_θ` and `π_ref` on chosen vs rejected.
9. **a** — Overfitting → lower β, reduce training, diversify data. The model "becoming repetitive or losing general capabilities" is the canonical symptom.
10. **b** — Underalignment → raise β, improve data, train longer. (Opposite knob from #9.)
11. **b** — 1,000 minimum / 10,000+ recommended; quality > quantity.
12. **a** — Gradient clipping, early stopping, reference-model output comparison. All explicitly called out under "Training Stability."
13. **b** — UltraFeedback / hh-rlhf-style preference datasets are the canonical DPO data. The chapter embeds `trl-lib/ultrafeedback_binarized` as the live example and uses `Anthropic/hh-rlhf` in the hands-on.
14. **b** — Both standard and conversational; both explicit and implicit prompts; chat template applied automatically. Explicit prompts are recommended.
15. **b** — DPO training is compute-heavy. The unit explicitly warns that HF Jobs requires Pro+ and local training needs ≥ 16 GB VRAM for SmolLM3-3B.

---

## After the real quiz

The course's Unit 2 quiz (when published) tests preference alignment
fundamentals. Pass → you're closer to **Completion** (need all released
units + final project).
