"""
TREATMENT: Recursive Training on microsoft/phi-1_5 with FIM Analysis
Fixes:
  - bfloat16 for generation model (float16 unstable on Blackwell/RTX 5090)
  - repetition_penalty + top_p to prevent degenerate distributions
  - explicit pad_token_id and left-padding for generation
  - trust_remote_code=True throughout
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

# --- CONFIG ---
MODEL_ID = "tiiuae/falcon-7b"
MAX_LENGTH = 256
BATCH_SIZE = 8
GEN_BATCH_SIZE = 32
LR = 5e-5
SAMPLES = 50000


def get_script_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "perblock_fim.py")
    if not os.path.exists(script_path):
        if os.path.exists("perblock_fim.py"):
            return "perblock_fim.py"
        else:
            raise FileNotFoundError(f"Could not find perblock_fim.py at {script_path}.")
    return script_path


def run_recursive_treatment(generations):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fim_script = get_script_path()

    base_results_dir = "results/falcon-7b_treatment_gen_0"
    os.makedirs(base_results_dir, exist_ok=True)

    # ===================================================================
    # GENERATION 0: BASELINE ANALYSIS
    # ===================================================================
    if os.path.exists(f"{base_results_dir}/perblock_fim.json"):
        print(f"✅ Gen 0 FIM analysis already exists, skipping...")
    else:
        print(f"Running FIM analysis on base model: {MODEL_ID}...")
        try:
            subprocess.run([
                "python", fim_script,
                "--model_path", MODEL_ID,
                "--output_dir", base_results_dir,
                "--disable_flash_attn",
                "--num_batches", "5",
                "--num_eigenvalues", "20",
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Gen 0 Analysis Failed: {e}")
            return

    # ===================================================================
    # RECURSIVE LOOP (Gen 1 to N)
    # ===================================================================
    for gen in range(1, generations + 1):
        print(f"\n{'='*80}\nTREATMENT: Generation {gen}\n{'='*80}")

        model_dir = f"models/falcon-7b_treatment_gen_{gen}"
        result_dir = f"results/falcon-7b_treatment_gen_{gen}"
        data_path  = f"data/falcon-7b_treatment_synthetic_gen_{gen}"

        if os.path.exists(f"{result_dir}/perblock_fim.json"):
            print(f"✅ Generation {gen} complete, skipping...")
            continue

        # ---------------------------------------------------------------
        # PHASE 1: GENERATE SYNTHETIC DATA
        # ---------------------------------------------------------------
        source_model_path = MODEL_ID if gen == 1 else f"models/falcon-7b_treatment_gen_{gen-1}"

        if os.path.exists(data_path):
            print(f"[PHASE 1] ✅ Synthetic data exists at {data_path}")
        else:
            print(f"[PHASE 1] Generating data from: {source_model_path}")

            # KEY FIX: bfloat16 instead of float16 — float16 produces
            # NaN/inf logits on Blackwell (RTX 5090) with Phi-1.5's
            # parallel attention architecture, collapsing the probability
            # distribution and triggering the multinomial CUDA assert.
            gen_model = AutoModelForCausalLM.from_pretrained(
                source_model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                trust_remote_code=True,
            ).to(device)

            gen_tok = AutoTokenizer.from_pretrained(
                source_model_path, trust_remote_code=True
            )
            # Phi-1.5 has no native pad token — set to eos
            gen_tok.pad_token    = gen_tok.eos_token
            gen_tok.pad_token_id = gen_tok.eos_token_id
            gen_tok.padding_side = "left"  # required for batch generation

            prompts = [
                "Once upon a time",
                "The scientist discovered",
                "The system detected",
                "Explain the concept of",
                "In a world where",
                "The young child asked",
            ]

            stories = []
            gen_model.eval()

            with torch.no_grad():
                pbar = tqdm(total=SAMPLES, desc="Generating")
                while len(stories) < SAMPLES:
                    batch_prompts = [
                        random.choice(prompts) for _ in range(GEN_BATCH_SIZE)
                    ]
                    inputs = gen_tok(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                    ).to(device)

                    outputs = gen_model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.95,
                        repetition_penalty=1.1,
                        pad_token_id=gen_tok.eos_token_id,
                        eos_token_id=gen_tok.eos_token_id,
                    )

                    decoded = gen_tok.batch_decode(
                        outputs, skip_special_tokens=True
                    )
                    stories.extend(decoded)
                    pbar.update(len(decoded))
                pbar.close()

            os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
            Dataset.from_dict({"text": stories[:SAMPLES]}).save_to_disk(data_path)
            print(f"✅ Saved {SAMPLES} samples to {data_path}")

            del gen_model
            torch.cuda.empty_cache()
            gc.collect()

        # ---------------------------------------------------------------
        # PHASE 2: TRAIN ON SYNTHETIC DATA
        # ---------------------------------------------------------------
        if os.path.exists(model_dir) and os.path.exists(f"{model_dir}/config.json"):
            print(f"[PHASE 2] ✅ Model exists at {model_dir}")
        else:
            print(f"[PHASE 2] Training Generation {gen}...")

            # Training uses bfloat16 too for consistency and stability
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                attn_implementation="eager",
                trust_remote_code=True,
            ).to(device)

            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID, trust_remote_code=True
            )
            tokenizer.pad_token    = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.padding_side = "right"  # right-padding preferred for training

            if tokenizer.pad_token is None:
                model.config.pad_token_id = tokenizer.eos_token_id

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
            progress_bar = tqdm(loader, desc=f"Training Gen {gen}")
            for batch in progress_bar:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch, labels=batch["input_ids"]).loss
                loss.backward()
                optim.step()
                sched.step()
                optim.zero_grad()
                progress_bar.set_postfix({"loss": f"{loss.item():.3f}"})

            os.makedirs(model_dir, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            print(f"✅ Saved model to {model_dir}")

            del model, optim, sched
            torch.cuda.empty_cache()
            gc.collect()

        # ---------------------------------------------------------------
        # PHASE 3: PER-BLOCK FIM ANALYSIS
        # ---------------------------------------------------------------
        if os.path.exists(f"{result_dir}/perblock_fim.json"):
            print(f"[PHASE 3] ✅ FIM results exist, skipping.")
        else:
            print(f"[PHASE 3] Running per-block FIM analysis...")
            torch.cuda.empty_cache()
            gc.collect()
            try:
                subprocess.run([
                    "python", fim_script,
                    "--model_path", model_dir,
                    "--output_dir", result_dir,
                    "--disable_flash_attn",
                    "--num_batches", "5",
                    "--num_eigenvalues", "20",
                ], check=True)
                print(f"✅ Generation {gen} complete!")
            except subprocess.CalledProcessError as e:
                print(f"❌ FIM analysis failed for Gen {gen}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=5)
    args = parser.parse_args()

    print("=" * 80)
    print(f"TREATMENT: RECURSIVE TRAINING ON {MODEL_ID} (FIM ANALYSIS)")
    print("=" * 80)
    run_recursive_treatment(args.generations)
    print("\n" + "=" * 80)
    print("TREATMENT COMPLETE!")
    print("=" * 80)