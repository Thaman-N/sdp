"""
"So What" Experiment: FIM-Weighted EWC Regularization
======================================================
Elastic Weight Consolidation (Kirkpatrick et al. 2017) applied to
recursive self-distillation, using your already-computed per-block FIM
values as the importance weights.

MOTIVATION:
  Your paper shows that high-FIM blocks are the geometrically sensitive
  anchor points of the network. EWC's premise is exactly this: penalise
  drift more strongly in high-importance (high-FIM) parameters, allowing
  low-importance parameters to move freely.

  This is the direct algorithmic prescription from the FIM-drift paradox:
    - High-FIM early blocks: strong elastic constraint → drift suppressed
    - Low-FIM late blocks:   weak elastic constraint  → drift allowed
  Unlike freezing (rigid) or ortho (disruptive), EWC is a SOFT constraint
  that operates DURING training, not post-hoc.

LOSS FUNCTION:
  L_total = L_synthetic + λ * Σ_b [ log10(FIM_b) * mean((W_b - W_0_b)²) ]

  Using log10(FIM) prevents the structural anomaly blocks (11, 28, 29)
  from overwhelming the regularisation (raw FIM ratio 1527/27 = 58x,
  log10 ratio = 3.18/1.44 = 2.2x — manageable).

  Using mean squared diff (not sum) normalises across different layer
  sizes so all blocks contribute comparably to the penalty.

FIM SOURCE:
  The script reads your existing Gen0 FIM JSON from results/.
  It falls back to hard-coded Gen0 values if the JSON is not found.

Usage:
  python so_what_ewc.py --generations 5 --lambda_ewc 500
  python so_what_ewc.py --generations 5 --lambda_ewc 100  # weaker
  python so_what_ewc.py --generations 5 --lambda_ewc 1000 # stronger
"""

import os
import gc
import re
import json
import math
import argparse
import subprocess

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_from_disk
from torch.utils.data import DataLoader

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_ID   = "HuggingFaceTB/SmolLM2-135M"
MAX_LENGTH = 256
BATCH_SIZE = 8
LR         = 5e-5

# Gen0 per-block FIM totals (attn + mlp) — fallback if JSON not found.
# These are the verified values from your Gen0 FIM run.
GEN0_FIM_FALLBACK = {
    0:  36.69,  1:  46.34,  2:  85.78,  3:  38.59,  4:  20.31,
    5:  32.96,  6:  17.25,  7:  31.86,  8:  18.69,  9:  17.97,
    10: 19.95,  11: 1527.54, 12: 8.66,  13: 7.51,  14: 8.46,
    15: 11.22,  16: 10.13,  17: 8.12,  18: 7.72,  19: 25.30,
    20: 4.06,   21: 4.26,   22: 3.37,  23: 2.24,  24: 3.14,
    25: 2.09,   26: 2.15,   27: 10.64, 28: 1606.82, 29: 618.43,
}


# ── HELPERS ──────────────────────────────────────────────────────────────────

def get_fim_script():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in [
        os.path.join(current_dir, "perblock_fim.py"),
        "perblock_fim.py",
    ]:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("perblock_fim.py not found")


def load_gen0_fim():
    """
    Load per-block FIM from the Gen0 FIM JSON produced by perblock_fim.py.
    Returns dict {block_int: total_fim_float}.
    Falls back to hard-coded values if JSON not found.
    """
    candidates = [
        "results/smart_ortho_gen_0/perblock_fim.json",
        "results/ortho_drift_gen_0/perblock_fim.json",
        "results/frozen_late_gen_0/perblock_fim.json",
        "results/smollm_treatment_gen_0/perblock_fim.json",
        "results/ewc_gen_0/perblock_fim.json",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                block_fim = {}
                for k, v in data.items():
                    if k.isdigit():
                        b = int(k)
                        # JSON has {"attn_top_eigenvalue": X, "mlp_top_eigenvalue": Y}
                        # or {"top_eigenvalue": X} depending on version
                        if "attn_top_eigenvalue" in v and "mlp_top_eigenvalue" in v:
                            block_fim[b] = v["attn_top_eigenvalue"] + v["mlp_top_eigenvalue"]
                        elif "top_eigenvalue" in v:
                            block_fim[b] = v["top_eigenvalue"]
                if len(block_fim) >= 20:
                    print(f"  Loaded Gen0 FIM from {path} ({len(block_fim)} blocks)")
                    return block_fim
            except Exception as e:
                print(f"  Warning: could not parse {path}: {e}")
    print("  Using hard-coded Gen0 FIM fallback values")
    return GEN0_FIM_FALLBACK


def build_fim_weights(gen0_fim: dict, model: torch.nn.Module) -> dict:
    """
    For each trainable parameter, compute its EWC importance weight:
        weight = log10(FIM_b)
    where b is the block number containing that parameter.
    Returns dict {param_name: scalar_weight}.
    """
    weights = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Extract block number from param name
        m = re.search(r'\.layers\.(\d+)\.', name)
        if m is None:
            # embed_tokens, lm_head — not in any block, skip
            continue
        b = int(m.group(1))
        fim_val = gen0_fim.get(b, 1.0)
        # log10 scaling to prevent anomaly blocks dominating
        weights[name] = math.log10(max(fim_val, 1.0))
    return weights


def compute_ewc_loss(model, base_state, fim_weights):
    """
    EWC penalty: Σ_param [ fim_weight_param * mean((W_current - W_base)²) ]
    Using MEAN (not sum) so all layers contribute comparably regardless of size.
    """
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if name not in fim_weights:
            continue
        if name not in base_state:
            continue
        w0 = base_state[name].to(param.device).to(param.dtype)
        diff_sq_mean = ((param - w0) ** 2).mean()
        penalty = penalty + fim_weights[name] * diff_sq_mean
    return penalty


# ── MAIN ─────────────────────────────────────────────────────────────────────

def run_ewc_treatment(generations: int, lambda_ewc: float):
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fim_script = get_fim_script()
    prefix     = f"ewc_lambda{int(lambda_ewc)}"

    print(f"\nEWC TREATMENT: lambda={lambda_ewc}")
    print(f"  Results prefix: {prefix}")
    print(f"  Device: {device}")

    # ── Load Gen0 FIM ─────────────────────────────────────────────────────
    gen0_fim = load_gen0_fim()
    n_blocks = len(gen0_fim)
    total_fim = sum(gen0_fim.values())
    print(f"  Gen0 FIM loaded: {n_blocks} blocks, total={total_fim:.1f}")

    # ── Gen 0 baseline ────────────────────────────────────────────────────
    gen0_dir = f"results/{prefix}_gen_0"
    os.makedirs(gen0_dir, exist_ok=True)

    reuse_dirs = [
        f"{gen0_dir}/perblock_fim.json",
        "results/smollm_treatment_gen_0/perblock_fim.json",
        "results/ortho_drift_gen_0/perblock_fim.json",
    ]
    if any(os.path.exists(p) for p in reuse_dirs):
        print("✅ Gen0 FIM exists, skipping")
    else:
        print("Running Gen0 FIM...")
        subprocess.run([
            "python", fim_script,
            "--model_path", MODEL_ID,
            "--output_dir", gen0_dir,
            "--disable_flash_attn",
            "--num_batches", "5",
            "--num_eigenvalues", "20",
        ], check=True)

    # ── Load base state (Gen0 weights as EWC anchor) ──────────────────────
    print("\nLoading Gen0 base model weights as EWC anchor...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True
    )
    # Store as float32 CPU tensors — will move to device during penalty compute
    base_state = {k: v.clone().cpu() for k, v in base_model.state_dict().items()}
    del base_model
    gc.collect()
    print(f"  Anchored {len(base_state)} parameter tensors")

    # ── Generations 1–N ───────────────────────────────────────────────────
    for gen in range(1, generations + 1):
        print(f"\n{'='*80}")
        print(f"EWC TREATMENT (λ={lambda_ewc}): Generation {gen}")
        print(f"{'='*80}")

        model_dir  = f"models/{prefix}_gen_{gen}"
        result_dir = f"results/{prefix}_gen_{gen}"
        data_path  = f"data/treatment_synthetic_gen_{gen}"

        if os.path.exists(f"{result_dir}/perblock_fim.json"):
            print(f"✅ Generation {gen} already complete, skipping")
            continue

        if not os.path.exists(data_path):
            print(f"❌ Synthetic data not found: {data_path}")
            print("   Run normal treatment first to generate data.")
            return

        # Source model: base for Gen1, EWC-trained Gen(N-1) for Gen2+
        source = MODEL_ID if gen == 1 else f"models/{prefix}_gen_{gen-1}"

        # ── Phase 1: Train with EWC loss ──────────────────────────────────
        if os.path.exists(model_dir) and os.path.exists(f"{model_dir}/config.json"):
            print(f"[PHASE 1] ✅ Trained model exists at {model_dir}")
        else:
            print(f"[PHASE 1] Training Gen {gen} with EWC (λ={lambda_ewc})...")
            dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            )

            model = AutoModelForCausalLM.from_pretrained(
                source, dtype=dtype,
            ).to(device)

            # Build FIM importance weights for EWC
            fim_weights = build_fim_weights(gen0_fim, model)
            n_weighted  = len(fim_weights)
            print(f"  EWC weights built for {n_weighted} parameter tensors")
            # Show a few examples
            sample_keys = sorted(fim_weights.keys())[:3]
            for k in sample_keys:
                print(f"    {k.split('layers.')[-1]:<35} weight={fim_weights[k]:.3f}")

            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            tokenizer.pad_token    = tokenizer.eos_token
            tokenizer.padding_side = "right"

            dataset = load_from_disk(data_path)
            dataset = dataset.map(
                lambda x: tokenizer(
                    x["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=MAX_LENGTH,
                ),
                batched=True,
                remove_columns=["text"],
            )
            dataset.set_format("torch")
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

            optim = torch.optim.AdamW(model.parameters(), lr=LR)
            sched = get_linear_schedule_with_warmup(
                optim,
                num_warmup_steps=100,
                num_training_steps=len(loader),
            )

            model.train()
            running_synth = 0.0
            running_ewc   = 0.0

            for step, batch in enumerate(tqdm(loader, desc=f"Gen {gen} (EWC λ={lambda_ewc})")):
                batch = {k: v.to(device) for k, v in batch.items()}

                # Synthetic language modelling loss
                loss_synth = model(**batch, labels=batch["input_ids"]).loss

                # EWC regularisation loss
                loss_ewc = compute_ewc_loss(model, base_state, fim_weights)

                loss_total = loss_synth + lambda_ewc * loss_ewc

                loss_total.backward()
                optim.step()
                sched.step()
                optim.zero_grad()

                running_synth += loss_synth.item()
                running_ewc   += loss_ewc.item()

                if (step + 1) % 500 == 0:
                    avg_s = running_synth / 500
                    avg_e = running_ewc   / 500
                    print(f"\n  Step {step+1}: synth={avg_s:.4f} "
                          f"ewc={avg_e:.6f} "
                          f"ewc_contrib={lambda_ewc*avg_e:.4f} "
                          f"total={avg_s + lambda_ewc*avg_e:.4f}")
                    running_synth = 0.0
                    running_ewc   = 0.0

            os.makedirs(model_dir, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            print(f"  Saved to {model_dir}")

            del model, optim, sched
            torch.cuda.empty_cache()
            gc.collect()

        # ── Phase 2: FIM ──────────────────────────────────────────────────
        print(f"[PHASE 2] Running FIM on {model_dir}...")
        subprocess.run([
            "python", fim_script,
            "--model_path", model_dir,
            "--output_dir", result_dir,
            "--disable_flash_attn",
            "--num_batches", "5",
            "--num_eigenvalues", "20",
        ], check=True)

        print(f"✅ Generation {gen} complete!")

    print("\n" + "="*80)
    print("EWC TREATMENT COMPLETE")
    print("Next steps:")
    print(f"  1. Add 'ewc_lambda{int(lambda_ewc)}_gen_': 'SmolLM EWC λ={lambda_ewc}' to eval mapping")
    print(f"  2. Run evaluate_all_metrics_master.py")
    print(f"  3. Run parameter_drift.py on ewc models")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FIM-weighted EWC regularisation for recursive self-distillation"
    )
    parser.add_argument(
        "--generations", type=int, default=5,
        help="Number of recursive generations (default: 5)"
    )
    parser.add_argument(
        "--lambda_ewc", type=float, default=500.0,
        help="EWC regularisation strength. Try 100 (weak), 500 (default), 1000 (strong)"
    )
    args = parser.parse_args()
    run_ewc_treatment(args.generations, args.lambda_ewc)
