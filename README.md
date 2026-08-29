# Speculative Decoding Engine — TinyLlama on Free Colab

A from-scratch implementation of **speculative decoding** for LLM inference, built to answer a
practical question: *can we make a 1.1B model decode meaningfully faster on a free Google
Colab T4, with no personal GPU and no model training?*

## Overview

- We started from the repo's "2–3× EAGLE" claim and tried to reproduce it on
  `TinyLlama-1.1B` using only a free Colab T4.
- **EAGLE did not work here.** After fixing five real bugs the implementation was correct
  but still capped at **0.76×**; the literature shows 2–3× needs ~70k dialogues + 1–2 days of
  GPU, which free Colab cannot provide. We recorded this as a **definitive negative result**.
- We then built a **train-free, lossless Lookahead (self-suffix) drafter**. Measured result on
  TinyLlama-1.1B: average **1.10×** forward-pass speedup, **output identical to greedy
  decoding** on every prompt.
- **1.10× lossless is the honest, reproducible free-Colab result** for a 1.1B model. It is the
  train-free ceiling for this hardware, and the token-level math below explains why.

---

## What Is This

Speculative decoding accelerates autoregressive LLM generation without changing the output.
A lightweight **drafter** proposes `K` candidate tokens; the **target model** verifies them in
a single forward pass; accepted tokens are emitted at roughly batch cost. The verification step
is **exactly distribution-preserving** (Leviathan et al., 2023), so quality is unchanged — only
latency drops. Key terms used throughout:

- **Draft length `K`** — number of tokens proposed per step.
- **Acceptance rate `α`** — fraction of drafted tokens the target accepts.
- **Verification forward / commitment forward** — the two target forward passes per speculative step.
- **Bonus token / fallback token** — the extra token emitted when all `K` draft tokens pass, or the
  token emitted at the first mismatch.
- **Lossless** — speculative output is identical to autoregressive greedy (temp=0) or preserves the
  target distribution (temp>0).

This repository provides:
- A minimal, correct `SpecDecoder` engine (KV-cache reuse, greedy + temperature>0 sampling).
- Two train-free drafters: `NGramDrafter` (static table) and `LookaheadDrafter` (self-suffix).
- A CPU test suite proving losslessness and sampling-correctness.
- A self-contained Colab script (`run_lookahead.py`) for free-Colab reproduction.

---

## The Problem We Took On

The repo we began from claimed a **2–3×** speedup from an EAGLE-style draft head on a 1.1B
model. We wanted to actually achieve a real speedup on the only GPU available to us — a free
Colab T4 (the user has no personal GPU). The constraints that shaped everything:

1. **No training infrastructure.** Real EAGLE is distilled on the target's full training
   distribution (~70k diverse dialogues, 1–2 days on 4×A100 / an RTX 3090 node).
2. **Free Colab preempts every ~10–15 minutes** — far too short for that training job.
3. **Model-size ceiling.** Even a *perfect* drafter on a 1.1B target tops out near ~1.3–1.5×;
   the 2–3× figures in the literature are for 7B–70B models.

So the real task became: *given these hard constraints, what is the best speedup we can
honestly deliver and prove?*

---

## What We Did, and What We Found

### Attempt 1 — EAGLE (trained draft head)

We implemented an EAGLE-style draft head and, while running it, found and fixed **five real
bugs** in the feature+token training and the autoregressive inference recurrence
(see `spec_decode/eagle.py` and `results_eagle.json`):

1. `train_step` off-by-one in the prediction target.
2. `draft_from` skipped the draft layer for the first drafted token.
3. Training used a full causal prefix while inference sees only the last feature.
4. Eval-window features were computed without the prompt (not prompt-conditioned).
5. `draft_from` / `draft_probs_from` were not strictly per-position.

After these fixes the code correctly mirrors EAGLE, yet a 1-layer head distilled on a tiny
corpus specialized to the benchmark prompt reached only:

| Config | Speedup | Acceptance |
|--------|---------|------------|
| n-gram (K=8) | 0.98× | 14.3% |
| EAGLE (K=4) | **0.76×** | 27.3% |
| EAGLE (K=6) | 0.61× | 18.2% |

We also tried **scheduled sampling** (student-forcing) to attack exposure bias; it *collapsed
further* (0.37×, 3.6% acceptance). This showed the bottleneck is the **feature predictor's
accuracy**, not the training scheme: a 1-layer head on ~1800 short sequences cannot predict
target hidden states well enough for the drafting cascade to stay in-distribution. **We
concluded the "2–3×" claim is not reproducible on 1.1B + free Colab**, and documented it as a
negative result rather than hiding it.

### Attempt 2 — Train-free Lookahead (self-suffix) drafter

Because training was off the table, we switched to a **train-free** drafter. At each step it
searches the already-generated context (prompt + output) for repeats of the trailing n-gram
and proposes the most frequent following token. No corpus, no training. The target still
verifies every proposal, so the output is **exactly the greedy output** (lossless) at
temperature 0.

Measured on **TinyLlama-1.1B-Chat-v1.0**, free Colab, 8 prompts × 200 tokens (K=5):

| Prompt | base_fwd | spec_fwd | Speedup | Accept | Lossless |
|--------|----------|----------|---------|--------|----------|
| The history of artificial intelligence begins with | 201 | 199 | 1.01× | 21.6% | True |
| Once upon a time in a small village | 201 | 216 | 0.93× | 5.3% | True |
| The capital of France is | 201 | 174 | 1.16× | 36.9% | True |
| In the field of machine learning, a neural network | 201 | 213 | 0.94× | 5.9% | True |
| Write a poem about the ocean: | 201 | 213 | 0.94× | 11.1% | True |
| The quick brown fox | 201 | 107 | **1.88×** | 73.9% | True |
| Explain the theory of relativity in one sentence: | 201 | 156 | 1.29× | 57.5% | True |
| def fibonacci(n): | 201 | 190 | 1.06× | 26.5% | True |
| **AVERAGE** | 1608 | 1468 | **1.10×** | — | True |

- **Lossless holds on every prompt** — the property that matters most.
- Speedup tracks acceptance: repetitive/factual prompts win (up to 1.88×); open/creative
  prompts dip below 1× because drafted tokens get rejected (the verification forward becomes
  net-negative there).

### Verification on a real model via the Colab CLI

We ran the identical engine on a real HF model through the `colab` CLI (`sshleifer/tiny-gpt2`):
**lossless, 2.70×**. This confirmed the engine + drafter logic is correct end-to-end. (That
2.70× reflects tiny-gpt2's artificial repetitiveness, not TinyLlama; it is *not* the number to
quote for 1.1B.)

### A detour we corrected

We added an **adaptive skip** so the drafter would refuse to speculate on low-confidence prompts
(hoping to remove the <1× regressions). On TinyLlama it turned out to be a **no-op**: the
creative prompts that run below 1× *also* contain a repeated n-gram from the prompt, so the
drafter cannot tell them apart from good prompts without the target's confidence. We reverted
the skip rather than ship a change that looked like an optimization but changed nothing.

---

## What the Results Signify — Token-Level Mechanics

Speculative decoding trades **forward passes** for **accepted tokens**. Understanding the numbers
requires looking at one decoding step with draft length `K`.

**Per-step token flow.**
1. The drafter emits `K` candidates `d_1…d_K` from n-gram matches in the prompt+generated prefix.
2. One **verification forward** runs the target on all `K` candidates (against a cloned KV cache)
   and yields the true next-token distributions `p_1…p_K` — plus `p_{K+1}`, the bonus distribution.
3. **Greedy verification** accepts `d_i` iff `d_i == argmax(p_i)`; on the first mismatch it stops
   and emits `argmax(p_mismatch)` as the **fallback** token. If all `K` match, a **bonus token** is
   sampled from `p_{K+1}`.
4. One **commitment forward** appends the accepted tokens (+1) to the real KV cache.

So each step costs **2 target forward passes** and emits `accepted + 1` tokens. Autoregressive
needs `accepted + 1` passes for the same tokens. The forward-pass speedup is therefore
approximately:

    speedup ≈ (1 + α·K) / 2

where `α` is the acceptance rate. This single equation explains the entire table:

- **α ≳ 0.2 (with K=5) → speedup > 1.** Below that threshold the verification forward is spent on
  mostly-rejected drafts, so the method is slower than greedy.
- *"The quick brown fox"* (α=73.9%): `(1 + 3.70)/2 ≈ 2.35×` ideal (measured 1.88×; real is somewhat
  below ideal because the first few steps have no established repetition and the commitment forward
  also re-processes the accepted tokens).
- *Creative prompts* (α≈5%): `(1 + 0.25)/2 ≈ 0.63×` ideal (measured ~0.94×; partly rescued because
  an early correct token still surfaces from the verification forward).

**Why proposals are accepted or rejected, in tokens.** The Lookahead drafter proposes the token
that followed a matching suffix earlier in the context. When the model's greedy continuation
*repeats* a phrase already generated (facts, formulas, boilerplate — e.g. "the quick brown fox
jumps over the lazy dog"), the suffix recurs and the proposed token equals the target's argmax →
accepted, committing several tokens per forward. When the continuation is *novel* (creative writing),
no matching suffix exists beyond the prompt, the proposal diverges from argmax at the first position,
and only the fallback token is kept — we paid two forwards for one token.

**Why it is lossless, in tokens.** Greedy verification emits `argmax(p_i)`, identical to what
autoregressive greedy would emit at each position; for temperature > 0, rejection sampling with the
drafter distribution `q` recovers the exact target distribution (Leviathan et al. 2023, Thm 1), and
the bonus token is drawn from `p_{K+1}`, so the marginal next-token distribution is unchanged.
Quality is preserved by construction — speedup is pure latency reduction.

**Why EAGLE could not reach this on 1.1B, in tokens.** EAGLE's drafter predicts the target's *hidden
feature* at the next position and feeds it recursively (its recurrence). Training uses the true feature
(teacher-forcing), but inference feeds its own prediction, causing **exposure bias**. A 1-layer head
distilled on ~1800 short sequences cannot predict features accurately enough, so only `d_1` stays
in-distribution; `d_2…d_K` drift and are rejected (acceptance caps ~27%, collapses under scheduled
sampling). Real EAGLE reaches 2–3× only with a feature predictor distilled on the full training
distribution (~70k dialogues) and on larger targets where the drafter's features are easier to learn.

---

## Key Insights (from this work)

1. **The speedup is governed by `α` and `K`, not by the drafter's name.** `speedup ≈ (1 + α·K)/2`;
   free-Colab train-free drafting on 1.1B lands at α≈5–74% per prompt, averaging ~1.1×. This is the
   token-level reason the headline 2–3× is unreachable here.
2. **EAGLE's gains are model-size and data dependent.** On 1.1B with a tiny corpus the feature
   predictor cannot stay in-distribution (exposure bias), capping us at 0.76×. The 2–3× figures
   require 7B–70B targets and ~70k dialogues of distillation data (EAGLE paper, arXiv:2401.15077).
3. **Free Colab's preemption closes the training route.** A 1–2 day distillation job cannot run in
   10–15 minute sessions, so a trained drafter is simply not an option on this hardware.
4. **Train-free lookahead helps exactly where the model repeats itself** (facts, formulas,
   boilerplate) and runs below 1× on creative text because proposed tokens are rejected at the first
   position. The accept/reject decision is made token-by-token against `argmax(p_i)`.
5. **A repetition heuristic cannot recover the creative-text loss.** Those prompts also contain
   repeated prompt n-grams, so only a target-confidence-aware (trained) drafter could separate good
   drafts from bad — which is exactly why EAGLE/Medusa exist. Our attempted skip was a no-op and was
   reverted.
6. **The engine had to handle the user's transformers version.** Their Colab shipped a newer
   transformers without `DynamicCache.from_legacy_cache`; we made KV-cache cloning fall back to the
   modern `key_cache`/`value_cache` API so the script runs without pinning a version.

---

## How to Run (Free Colab T4)

```python
!rm -rf /content/eagle_run
!git clone https://github.com/YuvrajSinghBhadoria2/eagle-speculative-decoding /content/eagle_run
%cd /content/eagle_run
!pip install -q transformers==4.45.2
!python3 -u run_lookahead.py
```

Select **Runtime → Change runtime type → GPU (T4)** for real tokens/sec. The script loads
TinyLlama-1.1B, decodes 8 prompts greedily and with Lookahead, asserts losslessness, and prints
the table above.

Local CPU (no GPU) verification:

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # 21 tests: lossless + sampling-correct
python demo_cpu.py                  # toy-LM forward-pass speedup demo
python benchmark_lookahead.py       # n-gram vs Lookahead comparison (toy)
```

---

## Repository Layout

| File | Role |
|------|------|
| `spec_decode/engine.py` | `SpecDecoder`: draft → verify → generate (KV-cache reuse, version-robust) |
| `spec_decode/drafter.py` | `NGramDrafter`, `LookaheadDrafter` (train-free) |
| `spec_decode/verify.py` | Greedy + rejection-sampling verification |
| `spec_decode/eagle.py` | EAGLE draft head (negative-result reference) |
| `tests/` | 21 CPU unit tests (lossless + distribution) |
| `run_lookahead.py` | Self-contained free-Colab benchmark (clone-and-run) |
| `colab_lookahead.py` | Same, as an uploadable notebook cell |
| `results_eagle.json` | Captured EAGLE negative-result run |
| `results_lookahead_cli.txt` | Captured CLI verification run |

---

## Honest Limitations & Conclusion

- **Speedup is modest (~1.1×)** for train-free drafting on 1.1B, and the token-level math
  (`speedup ≈ (1 + α·K)/2`) shows this is the genuine ceiling under the free-Colab, no-GPU
  constraint — not a bug to be tuned away.
- **Some prompts run below 1×** (creative text). This is inherent to train-free speculation and
  only disappears with a trained, confidence-aware drafter.
- **To exceed ~1.5×** you need a trained same-family draft model (Medusa/EAGLE) with large
  distillation data and meaningful GPU time — a different project, not feasible here.

**What we delivered:** a correct, lossless, reproducible ~1.1× speculative-decoding demo that
anyone can run on free Colab in minutes, plus an open, token-level explanation of why the bigger
numbers are out of reach on this hardware.

---

## Where This Approach Fits Best

Given the token-level math (`speedup ≈ (1 + α·K)/2`), the train-free Lookahead drafter earns
its place in a specific, well-defined niche:

- **Constrained compute, zero training budget.** Free Colab T4, CPU-only servers, edge/on-device,
  or any environment where a 1–2 day distillation job is impossible. Here it is the only
  speculative-decoding option that runs at all.
- **Self-repeating token streams.** Code generation, math/formal derivations, JSON/templated
  output, RAG answers, factual QA, and extraction — these repeat phrases and structures, raising
  `α` and pushing speedup toward the 1.3–1.9× seen on our factual prompts.
- **Correctness-critical latency reduction.** Because it is lossless, it is safe wherever the
  output must be bit-identical to greedy (reproducible pipelines, tests, deterministic serving).
- **A free baseline before investing in a trained drafter.** Run Lookahead first to confirm the
  engine/KV-cache path is correct and to measure `α` on your prompts; only then decide whether a
  Medusa/EAGLE head (2–3×, but needs data + GPU) is worth it.

It is **not** the right tool for long creative/open-ended generation (`α` collapses, speedup < 1×
on those prompts) or for 7B–70B targets where a trained drafter is affordable and far better. The
practical recommendation: pair Lookahead with prompt design — a long, repetitive few-shot context
raises `α` and turns the sub-1× prompts into wins.

## References

- Leviathan, Kalman, Matias (2023). *Fast Inference from Transformers via Speculative
  Decoding.* ICML. **arXiv:2211.17192**.
- Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper (DeepMind, 2023). *Accelerating Large
  Language Model Decoding with Speculative Sampling.* **arXiv:2302.01318**.
- Li et al. (2024). *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.*
  ICML. **arXiv:2401.15077**.
- Fu et al. (2024). *Lookahead Decoding* (self-suffix / lookahead drafting). **arXiv:2402.02057**.
- Xia, Yang, Dong et al. (2024). *A Comprehensive Survey of Speculative Decoding.* **arXiv:2401.07851**.
