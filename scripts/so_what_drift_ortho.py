"""
"So What" Experiment: Drift Direction Orthogonalization
========================================================
Abliteration-inspired mitigation for recursive collapse.

After each training generation, the late blocks have drifted in a
specific direction from Gen0. By orthogonalizing those weights against
the principal drift direction, we remove the accumulated synthetic
noise before the next generation of training.

Algorithm:
  For each high-drift block b at generation N:
    1. Compute drift matrix: D = W_N - W_0
    2. Get top drift direction: u = top left singular vector of SVD(D)
       u has shape [out_dim] — direction in output weight space
    3. Orthogonalize: W_clean = W_N - outer(u,u) @ W_N
       = W_N - u.unsqueeze(1) @ (u.unsqueeze(0) @ W_N)
    4. Save cleaned model, train Gen N+1 from it

Usage:
  python so_what_drift_ortho.py --generations 5
"""

import os
import torch
import argparse
import subprocess
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_from_disk
from torch.utils.data import DataLoader

# ============================================================
# CONFIG
# ============================================================
MODEL_ID     = "HuggingFaceTB/SmolLM2-135M"
MAX_LENGTH   = 256
BATCH_SIZE   = 8
LR           = 5e-5

# High-drift late blocks in SmolLM sequential architecture
ORTHO_BLOCKS = list(range(20, 30))

# Number of top drift directions to remove per weight matrix
N_DIRECTIONS = 1


def get_script_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in [
        os.path.join(current_dir, "perblock_fim.py"),
        "perblock_fim.py",
        os.path.join(current_dir, "scripts", "perblock_fim.py"),
    ]:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("perblock_fim.py not found")


def orthogonalize_against_drift(model, base_model_id, ortho_blocks, n_directions=1):
    """
    For each weight matrix W [out_dim, in_dim] in the specified blocks:
      1. Compute drift D = W_current - W_base
      2. SVD(D) -> top left singular vector u [out_dim]
      3. W_clean = W - u.unsqueeze(1) @ (u.unsqueeze(0) @ W)
         (removes rank-1 component along principal drift direction)
    """
    print(f"\n  Loading base model for drift computation...")
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
    modified_count = 0
    skipped_count  = 0

    for name, current_W in current_state.items():
        # Check if param belongs to one of the ortho blocks
        in_ortho_block = any(
            f'.layers.{b}.' in name or f'.h.{b}.' in name
            for b in ortho_blocks
        )
        if not in_ortho_block:
            continue
        if current_W.ndim < 2:
            continue
        if name not in base_state:
            continue

        # Float32 on CPU for stable SVD
        base_W = base_state[name].float()
        curr_W = current_W.float().cpu()

        D = curr_W - base_W
        if D.norm() < 1e-8:
            continue

        # Reshape to 2D: [out_dim, in_flat]
        orig_shape = curr_W.shape
        out_dim    = orig_shape[0]
        D_2d       = D.reshape(out_dim, -1)
        W_2d       = curr_W.reshape(out_dim, -1)

        try:
            U, S, Vh = torch.linalg.svd(D_2d, full_matrices=False)
            # U: [out_dim, min(out_dim, in_flat)]

            W_clean = W_2d.clone()
            for i in range(min(n_directions, U.shape[1])):
                u = U[:, i]   # [out_dim] — normalized, top drift direction

                # Remove component of W along u in output space:
                # proj = outer(u, u) @ W = u.unsqueeze(1) @ (u.unsqueeze(0) @ W)
                # shapes: [out_dim,1] @ [1,in_flat] = [out_dim, in_flat]
                proj    = u.unsqueeze(1) @ (u.unsqueeze(0) @ W_clean)
                W_clean = W_clean - proj

            current_state[name] = W_clean.reshape(orig_shape).to(current_W.dtype)
            modified_count += 1

        except Exception:
            skipped_count += 1
            continue

    model.load_state_dict(current_state)
    print(f"  Orthogonalized {modified_count} weight matrices in blocks {ortho_blocks}")
    if skipped_count:
        print(f"  Skipped {skipped_count} matrices (zero drift or SVD issue)")
    return model


def run_ortho_treatment(generations):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fim_script = get_script_path()

    prefix       = "ortho_drift"
    gen0_results = f"results/{prefix}_gen_0"
    normal_gen0  = "results/smollm_treatment_gen_0"
    os.makedirs(gen0_results, exist_ok=True)

    # ===== GEN 0 BASELINE =====
    if (os.path.exists(f"{gen0_results}/perblock_fim.json") or
            os.path.exists(f"{normal_gen0}/perblock_fim.json")):
        print("✅ Gen0 FIM exists, skipping")
    else:
        print("Running Gen0 FIM...")
        subprocess.run([
            "python", fim_script,
            "--model_path", MODEL_ID,
            "--output_dir", gen0_results,
            "--disable_flash_attn",
            "--num_batches", "5",
            "--num_eigenvalues", "20",
        ], check=True)

    for gen in range(1, generations + 1):
        print(f"\n{'='*80}")
        print(f"DRIFT ORTHOGONALIZATION TREATMENT: Generation {gen}")
        print(f"{'='*80}")

        model_dir  = f"models/{prefix}_gen_{gen}"
        result_dir = f"results/{prefix}_gen_{gen}"
        data_path  = f"data/treatment_synthetic_gen_{gen}"

        if os.path.exists(f"{result_dir}/perblock_fim.json"):
            print(f"✅ Generation {gen} complete, skipping...")
            continue

        if not os.path.exists(data_path):
            print(f"Synthetic data not found at {data_path}. Run normal treatment first.")
            return

        # Source: Gen0 base for Gen1, cleaned Gen(N-1) for Gen2+
        source_model = MODEL_ID if gen == 1 else f"models/{prefix}_gen_{gen-1}"

        # ===== TRAIN =====
        if os.path.exists(model_dir) and os.path.exists(f"{model_dir}/config.json"):
            print(f"[PHASE 2] ✅ Ortho model exists at {model_dir}")
        else:
            print(f"[PHASE 2] Training Gen {gen}...")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

            model = AutoModelForCausalLM.from_pretrained(
                source_model, dtype=dtype,
            ).to(device)

            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            tokenizer.pad_token    = tokenizer.eos_token
            tokenizer.padding_side = "right"

            dataset = load_from_disk(data_path)
            dataset = dataset.map(
                lambda x: tokenizer(
                    x["text"], truncation=True,
                    padding="max_length", max_length=MAX_LENGTH,
                ),
                batched=True, remove_columns=["text"],
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
            for batch in tqdm(loader, desc=f"Training Gen {gen}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                loss  = model(**batch, labels=batch["input_ids"]).loss
                loss.backward()
                optim.step()
                sched.step()
                optim.zero_grad()

            # ===== ORTHOGONALIZE =====
            print(f"\n[PHASE 2b] Applying drift orthogonalization...")
            model = model.float().cpu()   # float32 on CPU for stable SVD
            model = orthogonalize_against_drift(
                model, MODEL_ID, ORTHO_BLOCKS, N_DIRECTIONS
            )
            model = model.to(dtype).to(device)

            os.makedirs(model_dir, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)

            del model, optim, sched
            torch.cuda.empty_cache()
            gc.collect()

        # ===== FIM =====
        print(f"[PHASE 3] Running FIM...")
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
    print("ORTHOGONALIZATION TREATMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=5)
    args = parser.parse_args()
    run_ortho_treatment(args.generations)