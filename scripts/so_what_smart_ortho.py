"""
"So What" Experiment: Smart Freeze + Drift Orthogonalization
=============================================================
Combines the lessons from the two previous "so what" experiments:

WHAT WENT WRONG BEFORE:
  - Frozen Late:  blanket freeze of blocks 20-29 relocated collapse
                  to early blocks instead of preventing it
  - Ortho Drift:  ortho hit blocks 28 and 29 which have PRE-EXISTING
                  high-FIM anomalies in the base model (Block28=1604,
                  Block29=617 at Gen0). This destabilized Block 11
                  (Gen2 spike to 5951) and caused runaway curvature.

THIS EXPERIMENT:
  - FREEZE blocks 11, 28, 29: these are base-model structural anomalies
    with pre-existing high FIM. Never touch their weights, never let
    gradient pile into them.
  - ORTHO blocks 20-27: genuine high-drift late blocks with no
    pre-existing FIM anomalies. Apply drift orthogonalization here.
  - TRAIN blocks 0-19 (minus 11): fully trainable, no intervention.

Expected outcome:
  - Ortho cleans synthetic drift from late blocks WITHOUT hitting
    the structural anomaly blocks
  - Freeze prevents gradient from destabilizing the anomaly blocks
  - Together: collapse slowdown without landscape destabilization

Usage:
  python so_what_smart_ortho.py --generations 5
"""

import os
import gc
import subprocess
import argparse

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_from_disk
from torch.utils.data import DataLoader

# ── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_ID   = "HuggingFaceTB/SmolLM2-135M"
MAX_LENGTH = 256
BATCH_SIZE = 8
LR         = 5e-5

# Blocks with PRE-EXISTING high-FIM in the base model — freeze these,
# never apply ortho here (they are structural, not collapse-induced)
FREEZE_BLOCKS = [11, 28, 29]

# Genuine high-drift late blocks with no pre-existing anomaly — apply
# drift orthogonalization here ONLY (not 28/29)
ORTHO_BLOCKS  = list(range(20, 28))   # blocks 20, 21, 22, 23, 24, 25, 26, 27

N_DIRECTIONS  = 1   # number of top drift directions to remove per matrix


# ── HELPERS ─────────────────────────────────────────────────────────────────

def get_fim_script():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in [
        os.path.join(current_dir, "perblock_fim.py"),
        "perblock_fim.py",
    ]:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("perblock_fim.py not found")


def freeze_anomaly_blocks(model, freeze_blocks):
    """
    Freeze the pre-existing structural anomaly blocks so neither
    gradient nor orthogonalization ever touches them.
    Returns the number of frozen parameters.
    """
    frozen = 0
    for name, param in model.named_parameters():
        if any(f".layers.{b}." in name for b in freeze_blocks):
            param.requires_grad = False
            frozen += param.numel()
    return frozen


def orthogonalize_against_drift(model, base_model_id, ortho_blocks, n_directions=1):
    """
    For each weight matrix W [out_dim, in_dim] in ortho_blocks:
      1. D = W_current - W_base
      2. u = top left singular vector of SVD(D)   shape [out_dim]
      3. W_clean = W - u.unsqueeze(1) @ (u.unsqueeze(0) @ W)

    Operates entirely in float32 on CPU.
    Skips blocks in FREEZE_BLOCKS (structural anomalies).
    """
    print("    Loading base model for drift computation...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    base_state = {k: v.clone() for k, v in base_model.state_dict().items()}
    del base_model
    gc.collect()

    current_state = model.state_dict()
    modified  = 0
    skipped   = 0

    for name, current_W in current_state.items():
        # Only process ortho blocks — explicitly skip freeze blocks
        in_ortho  = any(f".layers.{b}." in name for b in ortho_blocks)
        in_freeze = any(f".layers.{b}." in name for b in FREEZE_BLOCKS)

        if not in_ortho or in_freeze:
            continue
        if current_W.ndim < 2:
            continue
        if name not in base_state:
            continue

        base_W = base_state[name].float()          # CPU float32
        curr_W = current_W.float().cpu()           # CPU float32

        D = curr_W - base_W
        if D.norm() < 1e-8:
            continue

        out_dim = curr_W.shape[0]
        D_2d    = D.reshape(out_dim, -1)
        W_2d    = curr_W.reshape(out_dim, -1)

        try:
            U, _, _ = torch.linalg.svd(D_2d, full_matrices=False)
            W_clean = W_2d.clone()
            for i in range(min(n_directions, U.shape[1])):
                u       = U[:, i]                              # [out_dim]
                proj    = u.unsqueeze(1) @ (u.unsqueeze(0) @ W_clean)
                W_clean = W_clean - proj

            current_state[name] = (
                W_clean.reshape(curr_W.shape).to(current_W.dtype)
            )
            modified += 1

        except Exception:
            skipped += 1
            continue

    model.load_state_dict(current_state)
    print(f"    Orthogonalized {modified} matrices in blocks {ortho_blocks}")
    if skipped:
        print(f"    Skipped {skipped} matrices")
    return model


# ── MAIN ─────────────────────────────────────────────────────────────────────

def run_smart_ortho(generations):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fim_script = get_fim_script()
    prefix    = "smart_ortho"

    # ── Gen 0 baseline ────────────────────────────────────────────────────
    gen0_dir = f"results/{prefix}_gen_0"
    os.makedirs(gen0_dir, exist_ok=True)

    reuse_dirs = [
        f"{gen0_dir}/perblock_fim.json",
        "results/smollm_treatment_gen_0/perblock_fim.json",
        "results/ortho_drift_gen_0/perblock_fim.json",
        "results/frozen_late_gen_0/perblock_fim.json",
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

    # ── Generations 1–N ───────────────────────────────────────────────────
    for gen in range(1, generations + 1):
        print(f"\n{'='*80}")
        print(f"SMART ORTHO TREATMENT: Generation {gen}")
        print(f"  Frozen blocks (structural anomalies): {FREEZE_BLOCKS}")
        print(f"  Ortho blocks  (drift cleaned):        {ORTHO_BLOCKS}")
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

        # Source model: base for Gen1, cleaned Gen(N-1) for Gen2+
        source = MODEL_ID if gen == 1 else f"models/{prefix}_gen_{gen-1}"

        # ── Phase 1: Train ──────────────────────────────────────────────
        if os.path.exists(model_dir) and os.path.exists(f"{model_dir}/config.json"):
            print("[PHASE 1] ✅ Trained model exists, skipping training")
        else:
            print(f"[PHASE 1] Training Gen {gen} (freeze={FREEZE_BLOCKS})...")
            dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            )

            model = AutoModelForCausalLM.from_pretrained(
                source, dtype=dtype,
            ).to(device)

            # Freeze structural anomaly blocks before training
            frozen_params = freeze_anomaly_blocks(model, FREEZE_BLOCKS)
            total_params  = sum(p.numel() for p in model.parameters())
            trainable     = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Total params:     {total_params/1e6:.1f}M")
            print(f"  Frozen params:    {frozen_params/1e6:.1f}M  (blocks {FREEZE_BLOCKS})")
            print(f"  Trainable params: {trainable/1e6:.1f}M")

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

            # Only pass trainable parameters to the optimizer
            optim = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=LR
            )
            sched = get_linear_schedule_with_warmup(
                optim,
                num_warmup_steps=100,
                num_training_steps=len(loader),
            )

            model.train()
            for batch in tqdm(loader, desc=f"Gen {gen} training"):
                batch = {k: v.to(device) for k, v in batch.items()}
                loss  = model(**batch, labels=batch["input_ids"]).loss
                loss.backward()
                optim.step()
                sched.step()
                optim.zero_grad()

            # ── Phase 2: Orthogonalize blocks 20-27 ────────────────────
            print(f"\n[PHASE 2] Applying drift ortho to blocks {ORTHO_BLOCKS}...")
            model = model.float().cpu()      # float32 on CPU for stable SVD
            model = orthogonalize_against_drift(
                model, MODEL_ID, ORTHO_BLOCKS, N_DIRECTIONS
            )
            model = model.to(dtype).to(device)

            os.makedirs(model_dir, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            print(f"  Saved to {model_dir}")

            del model, optim, sched
            torch.cuda.empty_cache()
            gc.collect()

        # ── Phase 3: FIM ────────────────────────────────────────────────
        print(f"[PHASE 3] Running FIM on {model_dir}...")
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
    print("SMART ORTHO TREATMENT COMPLETE")
    print("Next steps:")
    print("  1. Add 'smart_ortho_gen_': 'SmolLM Smart Ortho' to eval script mapping")
    print("  2. Run evaluate_all_metrics_master.py")
    print("  3. Run parameter_drift.py on smart_ortho models")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smart freeze + drift ortho experiment"
    )
    parser.add_argument(
        "--generations", type=int, default=5,
        help="Number of recursive generations (default: 5)"
    )
    args = parser.parse_args()
    run_smart_ortho(args.generations)
