# V2c_tail — Next Steps: Ablation First, Then Longer Run

## TL;DR (what to do)

1. **First, run ONE ablation** — a no-anchor version of V2c_tail — to check whether the
   cross-lingual gain comes from the ANCHORS or just from the extra LEARNED LINEAR LAYER.
2. **Only if the anchors prove necessary**, train V2c_tail LONGER (to ~30k, matching the
   existing 30k baseline) to see if the gain keeps growing.

**Do NOT train V2c_tail longer yet.** If a plain linear layer reproduces the gain, a longer run
just confirms the wrong thing. The ablation is cheaper and tells you what you actually have.

---

## Background (why we're doing this)

V2c_tail (top-k anchor concat + tail slot, at the embedding layer) was the ONLY architecture-only
variant that beat the baseline on cross-lingual retrieval:

- Pooled BLI (T6, P@1) on the distant, low-loanword pairs (zh + ar + ru): **baseline 0.222 vs
  V2c_tail 0.271**, delta +0.049, z=4.59, p=4.5e-6. All three pairs improve. This is significant
  and NOT a loanword artifact (different scripts, cognate-filtered).
- No LM-loss cost (3.2484 vs baseline 3.2511).

BUT there is a catch that must be resolved before trusting it:

- V2c_tail's temperature stayed **frozen (~14.7, near-uniform selection)** — the anchors were
  NOT sharply selected; they were near-uniformly blended and then reshaped by the `Linear_out`
  layer (and the tail slot).
- So the gain might come from the **learned linear reshaping**, NOT from anchor routing. If a
  plain linear layer with NO anchors gives the same gain, then "EmbHub" is not the mechanism —
  a learned transform is. That is a very different result.

The ablation below settles this.

---

## STEP 1 — The Ablation (run this FIRST)

**Goal:** does the cross-lingual gain need the anchors, or does a matched learned linear layer
(no anchors) reproduce it?

**What to build — a NO-ANCHOR arm that keeps V2c_tail's TRANSFORM but removes the anchors.**
This is the key to a CLEAN control. V2c_tail is `output = Linear_comb([x ; top-k anchor slots ;
tail slot])` — a SINGLE linear layer (no nonlinearity) applied to x concatenated with the anchor
content. The ablation must remove ONLY the anchors, changing nothing else — so use the SAME
single-linear transform on x alone, with the anchor/tail slots deleted:

```python
# No-anchor ablation block. x = token embedding, d = 1024.
# V2c_tail:  output = Linear_comb([x ; slot_1..slot_k ; tail])   (single linear, no GELU)
# Ablation:  output = Linear_no_anchor(x)                        (same kind of transform, no anchors)
output = Linear_no_anchor(x)     # d -> d single linear on the token embedding
```

- **Do NOT add a GELU / make it a 2-layer MLP.** V2c_tail's combiner is a SINGLE linear with no
  nonlinearity; adding one here would confound "anchors vs no anchors" with "nonlinearity vs
  none". Match V2c_tail's transform TYPE (single linear), just drop the anchor inputs.
- No anchors, no retrieval, no temperature — a learned linear reshaping of the token embedding.
- **Safe init:** initialize `Linear_no_anchor = Identity` (so `output = x` at step 0, then it
  drifts as it trains). Verify: at step 0, assert `output == x` to tolerance; over the first
  ~100 steps, loss tracks the baseline.
- Parameter note: this single d->d linear has FEWER params than V2c_tail's combiner (which takes
  a wider concat input). That is fine and even conservative — if a SMALLER no-anchor transform
  already reproduces the gain, the anchors are definitively not needed. (If you want a tighter
  param match, widen with `output = Linear([x ; x])` — still no anchors, no nonlinearity.)

**Config:** identical to V2c_tail — Qwen3-0.6B from scratch, same data, 8×H200, LR 3e-4 cosine
(min_lr_rate 0.1, warmup 500), weight decay 0.1, **6500 steps**, checkpoints at
1500/3250/5500/6500. (No temperature parameter here, so no 75× LR group needed.)

**Baseline to compare against:** the existing no-hub baseline (already trained).

**The one metric that decides it — T6 (BLI, CSLS P@1) on the pooled distant set (zh+ar+ru):**
- Compute pooled P@1 for the ablation arm at step 6500, with a 95% CI (binomial:
  `1.96*sqrt(p*(1-p)/n)`), and a two-proportion z-test vs baseline.
- Also report V2c_tail's already-known pooled number (0.271) for side-by-side.

**How to read the result:**

| Ablation pooled P@1 vs baseline | Meaning | What to do next |
|---|---|---|
| ~SAME as V2c_tail (≈0.27, beats baseline) | The gain is from the LEARNED LINEAR, not anchors. "EmbHub" is not the mechanism. | Do NOT train V2c_tail longer. Reframe: "a learned transform improves distant-lang BLI", or pivot to finetuning (Section 3 of next-directions doc). |
| ~BASELINE (≈0.22, no gain) | The ANCHORS are doing the work. V2c_tail is a real anchor-based positive. | Proceed to STEP 2 (train V2c_tail longer). |
| In between | Partial — anchors help but linear also contributes. | Judgment call; likely proceed to STEP 2 but note the linear's share. |

---

## STEP 2 — Longer V2c_tail Run (ONLY if Step 1 says anchors matter)

**Goal:** the V2c_tail gain crossed the baseline only between steps 5500-6500, on a still-rising
curve. Does it keep WIDENING against the (plateaued) baseline, or converge back to parity?

**What to run:** extend V2c_tail training to **~30k steps** (matching the existing 30k baseline).
Same config as the original V2c_tail run. Checkpoint regularly (e.g. every ~3-4k steps) so the
trajectory is visible.

**Compare against:** the existing 30k baseline.

**Metrics at each checkpoint:** pooled distant-set BLI (T6, with CI + z-test vs baseline) as the
headline; plus MEXA (T8) and per-language BLI for corroboration; plus training loss (confirm no
LM cost).

**How to read:**
- Gap KEEPS WIDENING against baseline → genuine, growing architecture-only cross-lingual effect.
  Strong result; V2c_tail becomes the thing to develop.
- Gap CONVERGES back to parity → the 6500 crossover was a transient (two curves moving at
  different speeds); architecture-only is a negative after all → pivot to finetuning.

---

## Notes / cautions

- **Ablation before long run, always.** A widening gap in a longer V2c_tail run is meaningless
  if a plain linear layer produces the same gain — you must rule that out first.
- **Frozen temperature is expected here, not a bug.** In V2c_tail the `Linear_out` makes the
  temperature redundant (it handles scaling), so the scale sitting at init is a real property,
  not a wiring failure.
- **Report distant pairs (zh/ar/ru) separately from Latin/loanword-heavy pairs (vi/de).** The
  distant pairs are where a real cross-lingual effect shows (artifacts can't explain them);
  en-vi is n=58 and uninformative — do not read it.
- **This result already cleared the traps that caught earlier false positives:** it's on BLI
  (behavioral, rank-based — not the gameable Probe B), cognate-filtered, significant on the
  pooled distant set, weakly corroborated by MEXA. That's why it's worth confirming rather than
  dismissing.
