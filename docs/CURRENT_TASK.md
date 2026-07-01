# Current Task — last updated 2026-07-01

> The in-progress / foreground work. Read `PROJECT_NOTES.md` first for background, then this for "where exactly are we." When a task finishes, distill its conclusion into the PROJECT_NOTES timeline and clear this file for the next task.

## Goal

Test whether stronger hub coupling (higher training alpha) improves cross-lingual alignment. Also test whether the effect changes with longer training (full 30K steps).

## Status — where we are RIGHT NOW

**Alpha variant experiments completed through full 30K-step training.** S3_a015 (α=0.15) and S3_a02 (α=0.20) trained to 6500 steps, then S3_a02 and baseline extended to full 30K steps. Results: hub consistently ~0.005 below baseline — longer training does not help. Thinking about what to try next.

## What was completed (previous task, 2026-06-20 → 2026-06-26)

The per-step Probe 2 pipeline was rebuilt from scratch and extended significantly. Full results in `docs/EXPERIMENT_RESULTS.md`.

### Completed steps:

1. **Data download + sampling** on 4x A100 machine. Sharded output to avoid OOM (`prepare_data.py` updated with `--flush-every`). ✓
2. **Trained S3** (alpha=0.05) to step 6500 with token-id dump. ✓
3. **Decoded token IDs → text → per-step word counts** (steps 1500/3250/5500/6500). ✓
4. **Probe 2 with MUSE translations** — ran Test A (anchor weight overlap) + added **Test B** (post-hub embedding cosine similarity at alpha=0.0/0.05/0.1/0.2/0.3; fixes the mis-measured Test B noted in PROJECT_NOTES). ✓
5. **Built LLM translation pipeline** (`build_translations_llm.py`) using GPT-4o — 4,804 tuples, much higher quality than MUSE (handles multi-word Vietnamese, no truncation). ✓
6. **Probe 2 with LLM translations** — all-words + single-token-only variants. ✓
7. **Trained baseline** (no EmbHub, identical hyperparameters) to step 6500. ✓
8. **Baseline comparison** — ran Probe 2 Test B on baseline checkpoints. ✓

### New experiments (2026-06-26 → 2026-07-01, on 8x H200):

9. **Data re-downloaded** on new 8x H200 machine. 287GB total, English in 35 shards. ✓
10. **Trained S3_a015 (α=0.15) and S3_a02 (α=0.20)** to step 6500 with token-id dump. ✓
11. **Ran Probe 2 Test B (single-token)** on S3_a015 and S3_a02 at steps 1500/3250/5500/6500. Used committed `resources/frequent_translations_llm.json` (no need to rebuild translations). ✓
12. **Extended baseline and S3_a02 to full 30K steps** with checkpoints every 2500 steps (7500, 10000, ..., 30000). ✓
13. **Ran Probe 2 on full 30K training progression** for both baseline and S3_a02 at all checkpoint steps. ✓

### Code changes made:

- `prepare_data.py`: sharded output (`--flush-every`, default=1) to avoid OOM on limited-RAM machines. No early `os.makedirs` (crash recovery safe). `HF_TOKEN` from env var.
- `train.py` + `smoke_train.py`: shard-aware data loading (detects `shard_*` subdirs, loads each, concatenates). Backward-compatible with old single-dir layout.
- `run_smoke_tests.py`: added S3 alpha variants (`S3_a01` through `S3_a10`). Config updated for current machine. Added 30s sleep between arms, `WANDB_MODE=offline`, removed `train/loss` from summary.
- `anchor_probe2_muse_no_loan_word.py`: added Test B (post-hub embedding cosine at multiple alphas), `--baseline` flag (loads model without EmbHub, Test B at alpha=0.0 only), `--single-token-only` filter.
- `diagnostics/build_translations_llm.py`: NEW — replaces MUSE with LLM-generated translations. Supports Gemini/OpenAI, batched, resumable, deduplicates across multiple freq files.
- `diagnostics/smoke_callback.py`: removed stale `train/loss` from CSV (was from wrong step). Loss is in `trainer_state.json` instead.
- `scripts/setup_env.sh`: NEW — installs miniconda + all conda envs (`embeddings_hub`, `fasttext_env`, `eval`) non-interactively.
- `scripts/train_qwen3_0.6b.sh` + `baseline.sh`: updated for 8x H200 (batch=16, grad_accum=4, 160 workers, nvme paths, `WANDB_MODE=offline`).
- `resources/frequent_translations_llm.json`: NEW — committed LLM translations (4,804 tuples) so probe can run on any machine without the decode/count/translate pipeline.
- `commands.sh`: EC2 remote runner entry point — the new machine pulls the repo and runs this file.
- `docs/EXPERIMENT_RESULTS.md`: NEW — full results with all numbers from MUSE/LLM/single-token/baseline runs (4x A100 experiments).
- `docs/EXPERIMENT_RESULTS_COMPARISON.md`: NEW — detailed comparison of baseline vs S3_a015 vs S3_a02 (8x H200 experiments).

### Key findings:

- **Test B gap grows with training** (step 1500→6500): +0.013 → +0.057 on single-token words at alpha=0.05. Highly significant.
- **BUT baseline shows nearly identical growth**: +0.011 → +0.055. The cross-lingual structure comes from the base model, not the hub.
- **Hub's marginal contribution**: ~0.002 above baseline on single-token, within noise on all-words.
- **Gap declines with alpha > 0.05**: expected since model trained at alpha=0.05 only.
- **Single-token measurement is 3-4x cleaner** than all-words (mean-pooling dilutes signal).
- **LLM translations strictly better than MUSE** for this measurement.

### Alpha variant results (step 6500, single-token, all on 8x H200):

| Arm | Training α | Gap @ α=0.0 (step 6500) |
|-----|-----------|------------------------|
| baseline | N/A | +0.0504 |
| S3_a015 | 0.15 | +0.0457 |
| S3_a02 | 0.20 | +0.0462 |

(S3 α=0.05 was only trained on the 4x A100 machine: +0.0571 vs A100 baseline +0.0550 — marginal +0.002 difference, not reproduced on H200.)

S3_a015 and S3_a02 show LOWER gaps than baseline at step 6500. The hub with higher alpha hurts cross-lingual alignment at the embedding level, not helps.

### Full 30K-step training results (baseline vs S3_a02, single-token, Test B gap @ α=0.0):

NOTE: S3_a02 was re-trained fresh for the 30K run (old 6500-step run moved to S3_a02_old). Same config/seed but training non-determinism means slightly different weights — step 6500 gap is +0.0477 here vs +0.0462 in the first run above.

| Step | Baseline | S3_a02 | Difference |
|------|----------|--------|------------|
| 1500 | +0.0117 | +0.0124 | +0.0007 |
| 3250 | +0.0312 | +0.0293 | -0.0019 |
| 6500 | +0.0504 | +0.0477 | -0.0027 |
| 10000 | +0.0608 | +0.0574 | -0.0034 |
| 15000 | +0.0654 | +0.0607 | -0.0047 |
| 20000 | +0.0661 | +0.0615 | -0.0046 |
| 25000 | +0.0661 | +0.0614 | -0.0047 |
| 30000 | +0.0661 | +0.0614 | -0.0047 |

At S3_a02's trained alpha (0.20), the gap is even slightly worse: +0.0603 at step 30000.

### Key findings from alpha variant + full training experiments:

- **Both models plateau ~step 17500.** Baseline at +0.066, S3_a02 at +0.061. No further improvement with longer training.
- **S3_a02 is consistently ~0.005 below baseline** from step 15000 onward. The gap stabilizes, not converges.
- **Hub contribution at inference (α=0.2) makes it slightly worse** — S3_a02 gap drops from +0.061 (α=0.0) to +0.060 (α=0.2).
- **Higher training alpha does NOT improve cross-lingual alignment.** Both α=0.15 and α=0.20 perform worse than baseline, not better.
- **The "longer training might help" hypothesis is now tested** — it doesn't. The gap is permanent.

### Interpretation:

Higher alpha forces the hub to contribute more to the embedding, but over full training the model adapts and LM loss recovers (unlike the short S9/S10 runs which showed large loss differences at ~1000 steps). Despite this adaptation, cross-lingual alignment is still worse than baseline — the hub's contribution doesn't help translations share more structure.

The from-scratch setting lacks gradient pressure to align translations. Without parallel data or cross-lingual signal, there is no reason for "dog" and "chó" to route to the same anchors — and they don't.

**Open question:** Are there other approaches within the from-scratch setting that could provide the missing cross-lingual gradient signal? Or is the finetuning track (adding EmbHub to a pretrained multilingual model where cross-lingual structure already exists) the right next step?

## Current pipeline — new machine (8x H200, CUDA 13.0)

### Machine setup:
- Envs: `embeddings_hub` (cu130), `fasttext_env`, `eval` (lm-eval-harness) — installed via `scripts/setup_env.sh`
- Data: `/opt/dlami/nvme/embhub_data/Qwen_Qwen3-0.6B/{train,eval}`
- Checkpoints: `/opt/dlami/nvme/smoke_test_outputs/{S3,baseline,S3_a02,...}`

### Step 0 — Data: ✓ DONE
287GB on nvme. English in 35 shards, other languages single shard each.

### Step 1 — Train: ✓ DONE
Trained S3_a015 (α=0.15) and S3_a02 (α=0.20) to step 6500. Then extended baseline and S3_a02 to full 30K steps.

### Step 2 — Probe 2: ✓ DONE
Ran single-token Test B on S3_a015, S3_a02 at steps 1500/3250/5500/6500. Then ran full progression (1500→30000) for baseline and S3_a02.

Used committed `resources/frequent_translations_llm.json` — no decode/count/translate pipeline needed.

### Completed results:

| Arm | Training α | Gap @ α=0.0, step 6500 | Gap @ α=0.0, step 30000 |
|-----|-----------|------------------------|-------------------------|
| baseline | N/A | +0.0550 (A100) / +0.0504 (H200) | +0.0661 (H200) |
| S3 | 0.05 | +0.0571 (A100 only) | — |
| S3_a015 | 0.15 | +0.0457 (H200) | — |
| S3_a02 | 0.20 | +0.0462 (H200, 1st run) / +0.0477 (H200, 30K re-run) | +0.0614 (H200) |

Note: A100 and H200 baseline numbers differ slightly (~0.005) — different machines, different batch configs (2×64×4 vs 16×4×8, same effective batch). S3_a02 was re-trained fresh for the 30K run. The relative comparisons within each machine are what matter.

### Eval results (PPL + benchmarks, step 6500, H200):

Ran `eval_parallel.py` on baseline, S3_a015, S3_a02 at steps 1500/3250/5500/6500.

**PPL at step 6500** — all three models are virtually identical:

| Lang | Baseline | S3_a015 | S3_a02 |
|------|----------|---------|--------|
| en | 24.00 | 24.05 | 24.05 |
| vi | 23.21 | 23.25 | 23.25 |
| zh | 117.94 | 119.20 | 118.12 |
| ru | 21.48 | 21.41 | 21.41 |
| de | 32.03 | 31.98 | 32.07 |
| ar | 27.97 | 27.93 | 27.87 |

**Benchmarks at step 6500** — no meaningful differences. Most tasks near chance (XNLI ~0.33, Belebele ~0.25, HellaSwag ~0.27, PAWS ~0.50). Model is too small (0.6B) and undertrained (6500 steps) for benchmarks to distinguish the three arms. Full results in `eval_ppl.json` and `eval_benchmarks.json` per checkpoint.

### Not yet trained:
- S3_a025 (α=0.25), S3_a03 (α=0.30), S3_a05 (α=0.50), S3_a10 (α=1.0) — configured in `run_smoke_tests.py` but not run yet

## Decisions already made (don't re-litigate)

- **LLM translations over MUSE.** GPT-4o produces strictly better translations. MUSE is single-word only with ~60% untranslated Vietnamese.
- **Single-token-only is the primary metric.** All-words mean-pooling dilutes signal; multi-token effects show after transformer layers, not at embedding level.
- **Decode and count are two scripts in two envs by design** — fasttext multiprocessing + transformers = deadlock risk.
- **Test B is the key measurement.** Test A (anchor weight overlap) is alpha-independent and shows weak signal.

## Dead ends (don't repeat)

- `datasets.map` for the decode step — ~3× slower for this workload.
- fasttext batch-predict — API safety under multiprocessing was uncertain; use per-text predict.
- MUSE dictionaries — truncated Vietnamese, wrong translations, ~60% untranslated. Use LLM translations.
- Alpha=0.05 from-scratch with no baseline comparison — can't distinguish hub effect from base model learning.
- Higher training alpha (0.15, 0.20) from-scratch — LM loss not hurt much over full training (unlike the short S9/S10 runs at ~1000 steps which showed loss ~4.3 vs ~3.9), but Test B gap is consistently lower than baseline. S3_a02 (α=0.20) tested to 30K steps: -0.003 below baseline at step 6500, widening to -0.005 by step 15000, then stable. Training longer does not help — both models plateau by step 17500, gap doesn't close.
