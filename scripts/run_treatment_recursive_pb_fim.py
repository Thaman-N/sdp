"""
TREATMENT: Recursive Training on google/gemma-3-1b-it with FIM Analysis
Includes explicit Generation 0 analysis.
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
MODEL_ID = "Qwen/Qwen3.5-0.8B" 
MAX_LENGTH = 256
BATCH_SIZE = 8
GEN_BATCH_SIZE = 32
LR = 5e-5
SAMPLES = 50000 

def get_script_path():
    """Dynamically find perblock_fim.py relative to this script"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "perblock_fim.py")
    if not os.path.exists(script_path):
        # Fallback: check current working directory if not found in scripts dir
        if os.path.exists("perblock_fim.py"):
            return "perblock_fim.py"
        else:
            raise FileNotFoundError(f"Could not find perblock_fim.py at {script_path} or in current directory.")
    return script_path

def run_recursive_treatment(generations):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fim_script = get_script_path()
    print(f"Using analysis script: {fim_script}")
    
    # Define base paths
    base_results_dir = "results/Qwen3.5-0.8B_treatment_gen_0"
    os.makedirs(base_results_dir, exist_ok=True)

    # ========================================================================
    # GENERATION 0: BASELINE ANALYSIS
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"BASELINE: Generation 0 (Base Model Analysis)")
    print(f"{'='*80}")
    
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
            print("✅ Gen 0 Analysis Complete")
        except subprocess.CalledProcessError as e:
            print(f"❌ Gen 0 Analysis Failed: {e}")
            # We exit here because if Gen 0 fails, we probably have a configuration issue
            return 

    # ========================================================================
    # RECURSIVE LOOP (Gen 1 to N)
    # ========================================================================
    for gen in range(1, generations + 1):
        print(f"\n{'='*80}")
        print(f"TREATMENT: Generation {gen}")
        print(f"{'='*80}")
        
        # Define paths for this generation
        model_dir = f"models/Qwen3.5-0.8B_treatment_gen_{gen}"
        result_dir = f"results/Qwen3.5-0.8B_treatment_gen_{gen}"
        data_path = f"data/Qwen3.5-0.8B_treatment_synthetic_gen_{gen}"
        
        # Check completion
        if os.path.exists(f"{result_dir}/perblock_fim.json"):
            print(f"✅ Generation {gen} already complete (found FIM results), skipping...")
            continue
        
        # --- PHASE 1: GENERATE SYNTHETIC DATA ---
        source_model_path = MODEL_ID if gen == 1 else f"models/Qwen3.5-0.8B_treatment_gen_{gen-1}"
        
        if os.path.exists(data_path):
            print(f"[PHASE 1] ✅ Synthetic data already exists at {data_path}, skipping generation...")
        else:
            print(f"\n[PHASE 1] Generating data from: {source_model_path}")
            
            # Load generation model
            gen_model = AutoModelForCausalLM.from_pretrained(
                source_model_path,
                torch_dtype=torch.float16,
                attn_implementation="eager" 
            ).to(device)
            
            gen_tok = AutoTokenizer.from_pretrained(source_model_path)
            gen_tok.padding_side = "left"
            if gen_tok.pad_token is None:
                gen_tok.pad_token = gen_tok.eos_token
            
            prompts = [
                "Once upon a time", "The scientist discovered", "In the future", 
                "The system detected", "Explain the concept of", "Write a story about",
                "The cat sat on", "Why is the sky", "To be or not to be",
                "The algorithm optimized", "Deep learning is", "The president announced"
            ]
            
            stories = []
            gen_model.eval()
            
            print(f"Generating {SAMPLES} synthetic samples...")
            with torch.no_grad():
                pbar = tqdm(total=SAMPLES, desc="Generating")
                while len(stories) < SAMPLES:
                    batch_prompts = [random.choice(prompts) for _ in range(GEN_BATCH_SIZE)]
                    inputs = gen_tok(batch_prompts, return_tensors="pt", padding=True).to(device)
                    
                    outputs = gen_model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=0.8,
                        top_k=50,
                        pad_token_id=gen_tok.pad_token_id,
                        eos_token_id=gen_tok.eos_token_id
                    )
                    
                    decoded = gen_tok.batch_decode(outputs, skip_special_tokens=True)
                    stories.extend(decoded)
                    pbar.update(len(decoded))
                pbar.close()
            
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            Dataset.from_dict({"text": stories[:SAMPLES]}).save_to_disk(data_path)
            print(f"✅ Saved {SAMPLES} stories to {data_path}")
            
            del gen_model
            torch.cuda.empty_cache()
            gc.collect()
        
        # --- PHASE 2: TRAIN ON SYNTHETIC DATA ---
        if os.path.exists(model_dir) and os.path.exists(f"{model_dir}/config.json"):
            print(f"\n[PHASE 2] ✅ Model already exists at {model_dir}, skipping training...")
        else:
            print(f"\n[PHASE 2] Training Generation {gen}...")
            
            # Reset to base model for training
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, 
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                attn_implementation="eager"
            ).to(device)
            
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            tokenizer.padding_side = "right"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = tokenizer.eos_token_id
            
            dataset = load_from_disk(data_path)
            
            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=MAX_LENGTH,
                    return_tensors="pt"
                )
            
            dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
            dataset.set_format("torch")
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            
            optim = torch.optim.AdamW(model.parameters(), lr=LR)
            num_training_steps = len(loader)
            sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=100, num_training_steps=num_training_steps)
            
            model.train()
            progress_bar = tqdm(loader, desc=f"Training Gen {gen}")
            
            for batch in progress_bar:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch, labels=batch["input_ids"])
                
                loss = outputs.loss
                loss.backward()
                
                optim.step()
                sched.step()
                optim.zero_grad()
                
                progress_bar.set_postfix({"loss": loss.item()})
            
            print(f"\n[PHASE 3] Saving model to {model_dir}...")
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            
            del model
            del optim
            del sched
            torch.cuda.empty_cache()
            gc.collect()
        
        # --- PHASE 3: PER-BLOCK FIM ANALYSIS ---
        if os.path.exists(f"{result_dir}/perblock_fim.json"):
            print(f"\n[PHASE 4] ✅ Per-block FIM results already exist, skipping...")
        else:
            print(f"\n[PHASE 4] Running per-block FIM analysis...")
            print("🧹 Clearing VRAM before FIM analysis...")
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
            except subprocess.CalledProcessError as e:
                print(f"❌ Analysis failed for Gen {gen}: {e}")
        
        print(f"\n✅ Generation {gen} complete!")

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