"""
"So What" Experiment: FIM-Guided Layer Freezing
================================================
Tests the causal claim: are the high-drift late layers the mechanism
driving recursive collapse?

Experiment design:
  - Run SmolLM2-135M recursive treatment for 5 gens as normal (already done)
  - Run SAME experiment but freeze blocks 20-29 (high-drift late layers)
    during each training step
  - Compare perplexity trajectories

If freezing the high-drift layers slows collapse:
  → FIM-drift analysis correctly identifies the CAUSAL mechanism
  → Not just a correlation but a mechanistic finding
  → Directly answers "so what" — you can slow collapse by constraining
    the layers the FIM analysis identifies as collapse vectors

If freezing makes NO difference:
  → The drift is a symptom not a cause
  → Still publishable — negative result with mechanistic explanation

Run:  python so_what_layer_freeze.py --generations 5
"""

import os
import torch
import argparse
import subprocess
import gc
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader

# ============================================================
# CONFIG
# ============================================================
MODEL_ID      = "HuggingFaceTB/SmolLM2-135M"
MAX_LENGTH    = 256
BATCH_SIZE    = 8
GEN_BATCH_SIZE = 32
LR            = 5e-5
SAMPLES       = 50000

# Blocks to FREEZE during training (identified as high-drift in sequential models)
# SmolLM has 30 blocks. From drift analysis, blocks 20-29 are the high-drift
# late layers that absorb gradient pressure. We freeze them to test causality.
FREEZE_BLOCKS = list(range(20, 30))  # blocks 20-29

def get_script_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "perblock_fim.py")
    if not os.path.exists(script_path):
        if os.path.exists("perblock_fim.py"):
            return "perblock_fim.py"
        raise FileNotFoundError(f"Could not find perblock_fim.py")
    return script_path


def freeze_late_blocks(model, freeze_blocks):
    """Freeze specified transformer blocks — they still forward but get no grad."""
    frozen_params = 0
    for name, param in model.named_parameters():
        # Check if param belongs to one of the freeze blocks
        for b in freeze_blocks:
            if f'.layers.{b}.' in name or f'.h.{b}.' in name:
                param.requires_grad = False
                frozen_params += param.numel()
                break
    return frozen_params


def run_frozen_treatment(generations):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fim_script = get_script_path()

    prefix = "frozen_late"
    base_results_dir = f"results/{prefix}_gen_0"
    os.makedirs(base_results_dir, exist_ok=True)

    # ===== GEN 0 BASELINE (shared with normal treatment) =====
    normal_gen0 = "results/smollm_treatment_gen_0"
    if os.path.exists(f"{normal_gen0}/perblock_fim.json"):
        print(f"✅ Reusing existing Gen 0 FIM from normal treatment")
    elif os.path.exists(f"{base_results_dir}/perblock_fim.json"):
        print(f"✅ Gen 0 FIM exists")
    else:
        print(f"Running Gen 0 FIM...")
        subprocess.run([
            "python", fim_script,
            "--model_path", MODEL_ID,
            "--output_dir", base_results_dir,
            "--disable_flash_attn",
            "--num_batches", "5",
            "--num_eigenvalues", "20",
        ], check=True)

    # ===== REUSE SYNTHETIC DATA FROM NORMAL TREATMENT =====
    # Generation is stochastic and uses the same source model for each gen,
    # so we can reuse the same synthetic datasets for fair comparison.

    for gen in range(1, generations + 1):
        print(f"\n{'='*80}")
        print(f"FROZEN LATE LAYERS TREATMENT: Generation {gen}")
        print(f"Frozen blocks: {FREEZE_BLOCKS}")
        print(f"{'='*80}")

        model_dir  = f"models/{prefix}_gen_{gen}"
        result_dir = f"results/{prefix}_gen_{gen}"
        # Reuse synthetic data from normal treatment run
        data_path  = f"data/treatment_synthetic_gen_{gen}"

        if os.path.exists(f"{result_dir}/perblock_fim.json"):
            print(f"✅ Generation {gen} complete, skipping...")
            continue

        # ----- PHASE 1: Data (reuse from normal treatment) -----
        if os.path.exists(data_path):
            print(f"[PHASE 1] ✅ Reusing synthetic data from normal treatment")
        else:
            print(f"[PHASE 1] Normal treatment data not found at {data_path}")
            print(f"          Run normal treatment first to generate data.")
            return

        # ----- PHASE 2: Train with late layers FROZEN -----
        if os.path.exists(model_dir) and os.path.exists(f"{model_dir}/config.json"):
            print(f"[PHASE 2] ✅ Frozen model exists at {model_dir}")
        else:
            print(f"[PHASE 2] Training Gen {gen} with late blocks frozen...")

            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            ).to(device)

            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            tokenizer.pad_token    = tokenizer.eos_token
            tokenizer.padding_side = "right"

            # FREEZE LATE BLOCKS
            frozen = freeze_late_blocks(model, FREEZE_BLOCKS)
            total  = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Total params:    {total/1e6:.1f}M")
            print(f"  Frozen params:   {frozen/1e6:.1f}M ({100*frozen/total:.1f}%)")
            print(f"  Trainable params:{trainable/1e6:.1f}M ({100*trainable/total:.1f}%)")

            dataset = load_from_disk(data_path)
            dataset = dataset.map(
                lambda x: tokenizer(x["text"], truncation=True,
                                    padding="max_length", max_length=MAX_LENGTH),
                batched=True, remove_columns=["text"]
            )
            dataset.set_format("torch")
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

            # Only optimize trainable params
            optim = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=LR
            )
            sched = get_linear_schedule_with_warmup(
                optim, num_warmup_steps=100, num_training_steps=len(loader)
            )

            model.train()
            for batch in tqdm(loader, desc=f"Training Gen {gen} (frozen)"):
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch, labels=batch["input_ids"]).loss
                loss.backward()
                optim.step()
                sched.step()
                optim.zero_grad()

            os.makedirs(model_dir, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)

            del model, optim, sched
            torch.cuda.empty_cache()
            gc.collect()

        # ----- PHASE 3: FIM Analysis -----
        print(f"[PHASE 3] Running FIM on frozen model...")
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
    print("FROZEN TREATMENT COMPLETE")
    print("Compare results/ to see perplexity and FIM-drift differences")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=5)
    args = parser.parse_args()
    run_frozen_treatment(args.generations)
