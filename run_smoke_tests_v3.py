"""Run V3/V5/V2-concat/V2-topk smoke-test arms sequentially.

Each arm runs smoke_train_v3.py via accelerate launch.
Monitors smoke_metrics.csv and terminates at target step.

Usage:
  python run_smoke_tests_v3.py --arms V5_mid10 V3_emb --stop-at-step 6500
  python run_smoke_tests_v3.py --arms V2_emb V2b_emb --stop-at-step 6500
"""

import argparse
import csv
import logging
import os
import signal
import subprocess
import sys
import time

logger = logging.getLogger(__name__)

ARM_CONFIGS = {
    # V5: V3 block at mid-layer (the doc's first-priority run)
    "V5_mid10": {"hub_type": "v3", "placement": "mid", "layer_idx": 10, "num_heads": 1, "num_hub_embeddings": 128},
    "V5_mid14": {"hub_type": "v3", "placement": "mid", "layer_idx": 14, "num_heads": 1, "num_hub_embeddings": 128},
    "V5_mid6":  {"hub_type": "v3", "placement": "mid", "layer_idx": 6,  "num_heads": 1, "num_hub_embeddings": 128},
    # V3: V3 block at embedding (doc's second-priority run)
    "V3_emb":   {"hub_type": "v3", "placement": "embedding", "num_heads": 1, "num_hub_embeddings": 128},
    # V4: multi-head V3
    "V4_emb_h4":   {"hub_type": "v3", "placement": "embedding", "num_heads": 4},
    "V4_mid10_h4": {"hub_type": "v3", "placement": "mid", "layer_idx": 10, "num_heads": 4},
    # V2-concat
    "V2_emb":    {"hub_type": "v2_concat", "placement": "embedding", "use_mlp": False, "num_hub_embeddings": 128},
    "V2b_emb":   {"hub_type": "v2_concat", "placement": "embedding", "use_mlp": True},
    "V2_mid10":  {"hub_type": "v2_concat", "placement": "mid", "layer_idx": 10, "use_mlp": False},
    # V2c-topk
    "V2c_emb":       {"hub_type": "v2_topk", "placement": "embedding", "top_k": 10, "tail_mode": "none"},
    "V2c_tail_emb":  {"hub_type": "v2_topk", "placement": "embedding", "top_k": 10, "tail_mode": "tail"},
    "V2c_tail_mid10": {"hub_type": "v2_topk", "placement": "mid", "layer_idx": 10, "top_k": 10, "tail_mode": "tail"},
    # V6: stochastic replacement (embedding layer only)
    "V6_1000":  {"hub_type": "v6", "placement": "embedding", "num_hub_embeddings": 1000, "top_k": 10, "batch_size": 8, "grad_accum": 8},
    "V6_128":   {"hub_type": "v6", "placement": "embedding", "num_hub_embeddings": 128,  "top_k": 10, "batch_size": 8, "grad_accum": 8},
    # V6f: factorized concept/residual (small codebook)
    "V6f_128":  {"hub_type": "v6f", "placement": "embedding", "num_hub_embeddings": 128, "top_k": 10, "batch_size": 8, "grad_accum": 8},
    "V6f_64":   {"hub_type": "v6f", "placement": "embedding", "num_hub_embeddings": 64,  "top_k": 10, "batch_size": 8, "grad_accum": 8},
    # Linear ablation: no anchors, just a learned d->d linear (control for V2c_tail)
    "linear_ablation": {"hub_type": "linear_ablation", "placement": "embedding", "num_hub_embeddings": 0},
}

ALL_ARMS = list(ARM_CONFIGS.keys())
DEFAULT_OUTPUT_BASE = "/opt/dlami/nvme/smoke_test_outputs_v3"


def build_cmd(arm_name, arm_cfg, output_dir, data_dir, save_token_ids=False):
    cmd = [
        "accelerate", "launch", "smoke_train_v3.py",
        "--config_name", "Qwen/Qwen3-0.6B",
        "--tokenizer_name", "Qwen/Qwen3-0.6B",
        "--data_dir", data_dir,
        "--block_size", "2048",
        "--preprocessing_num_workers", "160",
        "--hub_type", arm_cfg["hub_type"],
        "--num_hub_embeddings", str(arm_cfg.get("num_hub_embeddings", 1000)),
        "--placement", arm_cfg.get("placement", "embedding"),
        "--layer_idx", str(arm_cfg.get("layer_idx", 10)),
        "--num_heads", str(arm_cfg.get("num_heads", 1)),
        "--gate_bias_init", str(arm_cfg.get("gate_bias_init", -5.0)),
        "--scale_init", str(arm_cfg.get("scale_init", 14.0)),
        "--output_dir", output_dir,
        "--seed", "42",
        "--bf16",
        "--ddp_timeout", "21600",
        "--per_device_train_batch_size", str(arm_cfg.get("batch_size", 16)),
        "--gradient_accumulation_steps", str(arm_cfg.get("grad_accum", 4)),
        "--num_train_epochs", "1",
        "--learning_rate", "3e-4",
        "--lr_scheduler_type", "cosine_with_min_lr",
        "--lr_scheduler_kwargs", '{"min_lr_rate": 0.1}',
        "--warmup_steps", "500",
        "--weight_decay", "0.1",
        "--adam_beta1", "0.9",
        "--adam_beta2", "0.95",
        "--max_grad_norm", "1.0",
        "--logging_steps", "10",
        "--save_steps", "250",
        "--dataloader_num_workers", "8",
        "--run_name", f"v3-{arm_name}",
        "--report_to", "wandb",
        "--scale_lr_mult", "75",
        "--scale_no_wd",
    ]

    if arm_cfg.get("use_mlp"):
        cmd.append("--use_mlp")
    if arm_cfg.get("top_k"):
        cmd += ["--top_k", str(arm_cfg["top_k"])]
    if arm_cfg.get("tail_mode", "none") != "none":
        cmd += ["--tail_mode", arm_cfg["tail_mode"]]
    if arm_cfg.get("weighting"):
        cmd += ["--weighting", arm_cfg["weighting"]]
    if arm_cfg.get("r_budget"):
        cmd += ["--r_budget", str(arm_cfg["r_budget"])]
    if arm_cfg.get("p_only"):
        cmd += ["--p_only", str(arm_cfg["p_only"])]
    if arm_cfg.get("p_both"):
        cmd += ["--p_both", str(arm_cfg["p_both"])]
    if arm_cfg.get("anneal_steps"):
        cmd += ["--anneal_steps", str(arm_cfg["anneal_steps"])]

    if save_token_ids:
        cmd += ["--save_token_ids", "--token_ids_output_dir", os.path.join(output_dir, "token_ids")]

    return cmd


def ensure_gpus_free(max_attempts=10, sleep_between=30):
    """Wait until all GPUs have 0 processes. Kill stragglers if needed."""

    for attempt in range(max_attempts):
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.warning(f"    nvidia-smi failed (code {result.returncode}): {result.stderr.strip()}")
            time.sleep(sleep_between)
            continue

        pids = list(dict.fromkeys(p.strip() for p in result.stdout.strip().split("\n") if p.strip()))

        if not pids:
            logger.info(f"    GPUs are free (attempt {attempt + 1})")
            return True

        logger.info(f"    GPUs still in use ({len(pids)} processes: {pids}), attempt {attempt + 1}/{max_attempts}")

        killed_groups = set()
        for pid in pids:
            try:
                pid_int = int(pid)
                pgid = os.getpgid(pid_int)
                if pgid not in killed_groups:
                    os.killpg(pgid, signal.SIGTERM)
                    killed_groups.add(pgid)
                    logger.info(f"    Sent SIGTERM to process group {pgid} (from pid {pid})")
            except (ProcessLookupError, PermissionError, ValueError, OSError):
                pass

        time.sleep(5)

        for pid in pids:
            try:
                pid_int = int(pid)
                pgid = os.getpgid(pid_int)
                os.killpg(pgid, signal.SIGKILL)
                logger.info(f"    Sent SIGKILL to process group {pgid} (from pid {pid})")
            except (ProcessLookupError, PermissionError, ValueError, OSError):
                pass
            try:
                os.kill(int(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, ValueError, OSError):
                pass

        time.sleep(max(0, sleep_between - 5))

    logger.error("    Failed to free GPUs after all attempts!")
    return False


def wait_for_step(proc, smoke_csv_path, target_step, poll_interval=10):
    while proc.poll() is None:
        time.sleep(poll_interval)
        if not os.path.isfile(smoke_csv_path):
            continue
        try:
            with open(smoke_csv_path) as f:
                rows = list(csv.DictReader(f))
            if rows:
                last_step = int(rows[-1].get("step", 0))
                if last_step >= target_step:
                    logger.info(f"    Reached step {last_step} >= {target_step}, waiting 120s...")
                    time.sleep(120)
                    logger.info(f"    Sending SIGTERM...")
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except OSError:
                        pass
                    try:
                        proc.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"    Process did not exit after SIGTERM+60s, sending SIGKILL...")
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except OSError:
                            pass
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            logger.error(f"    Process {proc.pid} stuck after SIGKILL, giving up (ensure_gpus_free will handle)")
                    return last_step
        except (ValueError, OSError):
            continue
    return -1


def main():
    parser = argparse.ArgumentParser(description="Run V3 smoke tests")
    parser.add_argument("--arms", nargs="+", default=["V5_mid10", "V3_emb"], choices=ALL_ARMS)
    parser.add_argument("--output-base", default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--data-dir", default="/opt/dlami/nvme/embhub_data/Qwen_Qwen3-0.6B/train")
    parser.add_argument("--stop-at-step", type=int, default=6500)
    parser.add_argument("--save-token-ids", action="store_true")
    parser.add_argument("--log", default="smoke_tests_v3.log")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.log, mode="w")],
    )

    os.makedirs(args.output_base, exist_ok=True)
    completed = []
    start_time = time.time()

    logger.info(f"Running {len(args.arms)} V3 smoke arms sequentially")
    logger.info(f"Arms: {args.arms}")
    logger.info(f"Stop at step: {args.stop_at_step}")

    for arm_name in args.arms:
        arm_cfg = ARM_CONFIGS[arm_name]
        output_dir = os.path.join(args.output_base, arm_name)
        log_path = os.path.join(args.output_base, f"{arm_name}.log")
        smoke_csv_path = os.path.join(output_dir, "smoke_metrics.csv")

        cmd = build_cmd(arm_name, arm_cfg, output_dir, args.data_dir,
                        save_token_ids=args.save_token_ids)

        logger.info(f"  Ensuring GPUs are free before {arm_name}...")
        if not ensure_gpus_free():
            logger.error(f"  SKIP {arm_name}: could not free GPUs")
            completed.append({"arm": arm_name, "config": arm_cfg, "status": "SKIPPED (GPUs busy)", "elapsed": 0})
            continue

        logger.info(f"  START: {arm_name} ({arm_cfg})")
        arm_start = time.time()

        env = os.environ.copy()
        env["NCCL_NVLS_ENABLE"] = "0"
        env["WANDB_PROJECT"] = "cross_lingual_embedding_hub"
        env["WANDB_MODE"] = "offline"

        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(
                cmd, stdout=log_file, stderr=subprocess.STDOUT,
                preexec_fn=os.setsid, env=env,
            )
            last_step = wait_for_step(proc, smoke_csv_path, args.stop_at_step)

        arm_elapsed = time.time() - arm_start
        if last_step >= args.stop_at_step:
            status = f"STOPPED at step {last_step}"
        elif proc.returncode == 0:
            status = "OK (finished epoch)"
        else:
            status = f"FAILED (code {proc.returncode})"

        logger.info(f"  DONE:  {arm_name} — {status}  [{arm_elapsed:.0f}s]")
        if proc.returncode and proc.returncode not in (0, -signal.SIGTERM):
            logger.info(f"         See log: {log_path}")

        completed.append({"arm": arm_name, "config": arm_cfg, "status": status, "elapsed": arm_elapsed})

        if arm_name != args.arms[-1]:
            logger.info(f"    Waiting 10s for GPU processes to wind down...")
            time.sleep(10)

    total_elapsed = time.time() - start_time
    logger.info(f"\nAll {len(completed)} arms done in {total_elapsed:.0f}s")

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    for job in completed:
        csv_path = os.path.join(args.output_base, job["arm"], "smoke_metrics.csv")
        final_line = ""
        if os.path.isfile(csv_path):
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            if rows:
                last = rows[-1]
                def _safe_float(v, default=0.0):
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        return default
                final_line = (f"step={last.get('step', '?')} "
                              f"scale={_safe_float(last.get('embhub/logit_scale')):.2f} "
                              f"entropy={_safe_float(last.get('embhub/entropy')):.4f} "
                              f"norm_ratio={_safe_float(last.get('embhub/norm_ratio')):.4f}")
        logger.info(f"\n  {job['arm']} ({job['config']['hub_type']}@{job['config'].get('placement', 'emb')}) "
                     f"[{job['elapsed']:.0f}s]:")
        logger.info(f"    {final_line}" if final_line else f"    No metrics")


if __name__ == "__main__":
    main()
