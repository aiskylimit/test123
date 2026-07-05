"""Run probe tests (T2/T4/T5/T6/T7/T8) in parallel across GPUs.

Loads each checkpoint ONCE per GPU, runs all applicable tests, saves
per-model results. Comparison across models is done separately afterward.

Each worker is fully independent — no cross-model dependencies.

Usage:
    python run_probe_tests.py
    python run_probe_tests.py --gpus 0 1 2 3
    python run_probe_tests.py --arms V6f_128 V3_emb --steps 1500 6500
"""

import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Process, Queue


DEFAULT_CKPT_BASE = "/opt/dlami/nvme/smoke_test_outputs_v3"
DEFAULT_BASELINE = "/opt/dlami/nvme/smoke_test_outputs/baseline/checkpoint-6500"
DEFAULT_EVAL_DIR = "/opt/dlami/nvme/embhub_data/Qwen_Qwen3-0.6B/eval"
DEFAULT_ARMS = ["V6f_128", "V5_mid10", "V3_emb", "V2_emb", "V2c_tail_emb"]
DEFAULT_STEPS = [1500, 3250, 5500, 6500]

T5_LAYERS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 27]


def run_tests_for_checkpoint(gpu_id, ckpt_path, arm_name, step,
                              eval_dir, output_dir, is_baseline=False):
    """Run all applicable tests for one checkpoint on one GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from diagnostics.test_utils import (
        get_tokenizer, load_checkpoint, load_translations,
        filter_loanwords_per_pair,
    )

    device = "cuda"
    tokenizer = get_tokenizer(ckpt_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    translations = load_translations(single_token_only=False)
    translations = filter_loanwords_per_pair(translations)

    tag = f"{arm_name}_step{step}" if not is_baseline else "baseline"
    label = f"[GPU {gpu_id}] {tag}"
    print(f"{label}: loading checkpoint...")

    model, hub_info = load_checkpoint(
        ckpt_path, device=device, baseline=is_baseline,
    )
    has_hub = hub_info["has_hub"]

    print(f"{label}: loaded (hub_type={hub_info['hub_type']}, "
          f"placement={hub_info['placement']})")

    # --- Probe A/B: anchor overlap + embedding cosine (all models) ---
    print(f"{label}: Probe A/B...")
    from diagnostics.test_probe_ab import run_probe_ab, format_results as pab_format
    pab_result = run_probe_ab(model, hub_info, tokenizer, translations, device)
    pab_result["checkpoint"] = ckpt_path
    pab_result["arm"] = arm_name
    pab_result["step"] = step
    pab_result["hub_type"] = hub_info["hub_type"]
    out = os.path.join(output_dir, f"probe_ab_{tag}.json")
    with open(out, "w") as f:
        json.dump(pab_result, f, indent=2, default=str)
    print(f"{label}: Probe A/B done -> {out}")
    print(pab_format(pab_result, tag))

    # --- T5: layer sweep (all models) ---
    print(f"{label}: T5 layer sweep...")
    from diagnostics.test_t5_layer_sweep import run_layer_sweep, format_results as t5_format
    layers = [None] + T5_LAYERS
    per_layer = run_layer_sweep(
        model, tokenizer, translations, eval_dir, layers, device,
    )
    out = os.path.join(output_dir, f"t5_{tag}.json")
    with open(out, "w") as f:
        json.dump({"per_layer": per_layer, "checkpoint": ckpt_path,
                    "arm": arm_name, "step": step}, f, indent=2)
    print(f"{label}: T5 done -> {out}")
    print(t5_format(per_layer))

    # --- T2: language decodability (hub only — needs anchor weights) ---
    has_anchors = has_hub and hub_info["hub_type"] != "linear_ablation"
    if has_anchors:
        print(f"{label}: T2 decodability...")
        from diagnostics.test_t2_lang_decodability import (
            run_decodability, format_results as t2_format,
        )
        t2_result = run_decodability(model, hub_info, tokenizer, eval_dir, device)
        t2_result["checkpoint"] = ckpt_path
        t2_result["hub_type"] = hub_info["hub_type"]
        t2_result["arm"] = arm_name
        t2_result["step"] = step
        out = os.path.join(output_dir, f"t2_{tag}.json")
        with open(out, "w") as f:
            json.dump(t2_result, f, indent=2)
        print(f"{label}: T2 done -> {out}")
        print(t2_format(t2_result))

    # --- T4: contribution share (hub only — needs hub internals) ---
    if has_anchors:
        print(f"{label}: T4 contribution share...")
        from diagnostics.test_t4_contribution_share import (
            run_contribution_share, format_results as t4_format,
        )
        t4_result = run_contribution_share(
            model, hub_info, tokenizer, eval_dir, device,
        )
        t4_result["checkpoint"] = ckpt_path
        t4_result["hub_type"] = hub_info["hub_type"]
        t4_result["arm"] = arm_name
        t4_result["step"] = step
        out = os.path.join(output_dir, f"t4_{tag}.json")
        with open(out, "w") as f:
            json.dump(t4_result, f, indent=2)
        print(f"{label}: T4 done -> {out}")
        print(t4_format(t4_result))

    # --- T6: BLI retrieval (all models) ---
    print(f"{label}: T6 BLI retrieval...")
    from diagnostics.test_t6_bli_retrieval import (
        run_bli_all_langs, summarize_results as t6_summarize,
    )
    t6_results = run_bli_all_langs(
        model, hub_info, tokenizer, translations, device,
    )
    t6_out = {
        "checkpoint": ckpt_path,
        "arm": arm_name,
        "step": step,
        "hub_type": hub_info["hub_type"],
        "placement": hub_info["placement"],
        "layer_idx": hub_info["layer_idx"],
        "per_lang": t6_results,
        "summary": t6_summarize(t6_results),
    }
    out = os.path.join(output_dir, f"t6_{tag}.json")
    with open(out, "w") as f:
        json.dump(t6_out, f, indent=2)
    print(f"{label}: T6 done -> {out}")

    # --- T7: generative code-switch (all models, step 6500 only) ---
    if step == 6500 or is_baseline:
        print(f"{label}: T7 code-switch...")
        from diagnostics.test_t7_generative_codeswitch import run_codeswitch
        t7_results = run_codeswitch(
            model, hub_info, tokenizer, translations, device,
        )
        t7_out = {
            "checkpoint": ckpt_path,
            "arm": arm_name,
            "step": step,
            "hub_type": hub_info["hub_type"],
            **t7_results,
        }
        out = os.path.join(output_dir, f"t7_{tag}.json")
        with open(out, "w") as f:
            json.dump(t7_out, f, indent=2)
        print(f"{label}: T7 done -> {out}")

    # --- T8: MEXA (all models, step 6500 only) ---
    if step == 6500 or is_baseline:
        print(f"{label}: T8 MEXA...")
        from diagnostics.test_t8_mexa import (
            load_flores_from_dir, run_mexa, NON_EN_LANGS,
        )
        flores_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resources", "flores200",
        )
        flores_sentences = {}
        for lang in ["en"] + list(NON_EN_LANGS):
            sents = load_flores_from_dir(flores_dir, lang)
            if sents:
                flores_sentences[lang] = sents

        if "en" in flores_sentences:
            mexa_results = run_mexa(
                model, hub_info, tokenizer, flores_sentences, device,
            )
            t8_out = {
                "checkpoint": ckpt_path,
                "arm": arm_name,
                "step": step,
                "hub_type": hub_info["hub_type"],
                **mexa_results,
            }
            out = os.path.join(output_dir, f"t8_{tag}.json")
            with open(out, "w") as f:
                json.dump(t8_out, f, indent=2, default=str)
            print(f"{label}: T8 done -> {out}")
        else:
            print(f"{label}: T8 skipped (no FLORES data)")

    del model
    torch.cuda.empty_cache()
    print(f"{label}: all tests complete")


def worker(gpu_id, job_queue):
    """Worker process: pull jobs from queue, run tests on assigned GPU."""
    while True:
        try:
            job = job_queue.get_nowait()
        except Exception:
            break

        ckpt_path, arm_name, step, eval_dir, output_dir, is_baseline = job
        try:
            run_tests_for_checkpoint(
                gpu_id, ckpt_path, arm_name, step,
                eval_dir, output_dir, is_baseline,
            )
        except Exception as e:
            tag = f"{arm_name}/step-{step}" if not is_baseline else "baseline"
            print(f"[GPU {gpu_id}] ERROR on {tag}: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Run probe tests in parallel across GPUs",
    )
    parser.add_argument("--ckpt-base", default=DEFAULT_CKPT_BASE)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR)
    parser.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    parser.add_argument("--steps", nargs="+", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--gpus", nargs="+", type=int, default=None,
                        help="GPU IDs to use (default: all available)")
    parser.add_argument("--output-dir", default="temp")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Detect GPUs without importing torch (avoids CUDA init in parent)
    if args.gpus is not None:
        gpu_ids = args.gpus
    else:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True, text=True,
            )
            gpu_ids = [int(x.strip()) for x in result.stdout.strip().split("\n")
                       if x.strip()]
        except Exception:
            gpu_ids = [0]
    print(f"Using GPUs: {gpu_ids}")

    # Build job list
    jobs = []

    # Baseline
    if os.path.isdir(args.baseline):
        jobs.append((args.baseline, "baseline", 6500,
                      args.eval_dir, args.output_dir, True))

    # Hub checkpoints
    for arm in args.arms:
        for step in args.steps:
            ckpt_path = os.path.join(args.ckpt_base, arm, f"checkpoint-{step}")
            if os.path.isdir(ckpt_path):
                jobs.append((ckpt_path, arm, step,
                              args.eval_dir, args.output_dir, False))
            else:
                print(f"  Skipping {arm}/checkpoint-{step} (not found)")

    n_hub = sum(1 for j in jobs if not j[5])
    print(f"Total jobs: {len(jobs)} (1 baseline + {n_hub} hub checkpoints)")
    rounds = (len(jobs) + len(gpu_ids) - 1) // len(gpu_ids)
    print(f"Estimated: ~{rounds} rounds on {len(gpu_ids)} GPUs")

    # Pre-download FLORES before spawning workers
    has_6500 = any(s == 6500 for s in args.steps)
    if has_6500 or os.path.isdir(args.baseline):
        flores_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resources", "flores200",
        )
        if not os.path.isfile(os.path.join(flores_dir, "eng_Latn.txt")):
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                print("Pre-downloading FLORES data...")
                from diagnostics.test_t8_mexa import download_flores_hf
                download_flores_hf(flores_dir, hf_token)
            else:
                print("WARNING: FLORES not found and no HF_TOKEN. T8 will skip.")

    # Run
    job_queue = Queue()
    for job in jobs:
        job_queue.put(job)

    start_time = time.time()

    processes = []
    for gpu_id in gpu_ids:
        p = Process(target=worker, args=(gpu_id, job_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    elapsed = time.time() - start_time
    print(f"\nAll done in {elapsed:.0f}s ({elapsed/60:.1f} minutes)")
    print(f"Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
