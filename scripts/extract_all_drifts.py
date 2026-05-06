import os
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

# ==========================================
# CHOOSE YOUR MODEL (Uncomment ONE block)
# ==========================================

# --- OPTION A: LLAMA 3.2 (1B) ---
# HF_MODEL_ID = "meta-llama/Llama-3.2-1B" 
# MODEL_FOLDER_PREFIX = "llama_treatment_gen_"
# NUM_BLOCKS = 16 
# LAYER_PREFIX = "model.layers" 

# --- OPTION B: GPT-2 (124M) ---
# HF_MODEL_ID = "gpt2"
# MODEL_FOLDER_PREFIX = "gpt2_treatment_gen_"
# NUM_BLOCKS = 12
# LAYER_PREFIX = "transformer.h"

HF_MODEL_ID = "tiiuae/falcon-7b"
MODEL_FOLDER_PREFIX = "falcon_treatment_gen_"
NUM_BLOCKS = 32
LAYER_PREFIX = "transformer.h"

# ==========================================

BASE_DIR = r"/workspace/hessian-spectral-analysis/models"

print(f"Loading Gen 0 directly from HuggingFace ({HF_MODEL_ID})...")
base_model = AutoModelForCausalLM.from_pretrained(HF_MODEL_ID, device_map="cpu")
sd_0 = base_model.state_dict()

all_drifts = {}

# Loop through Generations 1 to 5
for gen in range(1, 6):
    folder_name = f"{MODEL_FOLDER_PREFIX}{gen}"
    gen_path = os.path.join(BASE_DIR, folder_name, "model.safetensors")
    
    print(f"\n--- Processing {folder_name} ---")
    if not os.path.exists(gen_path):
        print(f"ERROR: Could not find {gen_path}. Skipping.")
        continue
        
    sd_n = load_file(gen_path)
    drift_array = []

    for i in range(NUM_BLOCKS):
        block_keys = [k for k in sd_0.keys() if f"{LAYER_PREFIX}.{i}." in k]
        
        sum_sq_baseline = 0.0
        sum_sq_diff = 0.0
        
        for k in block_keys:
            k_local = k
            if k not in sd_n:
                # Handle safetensors prefix dropping
                k_local = k.replace("model.", "") if k.startswith("model.") else f"model.{k}"
                if k_local not in sd_n:
                    continue

            w0 = sd_0[k].float()
            wn = sd_n[k_local].float()
            
            sum_sq_baseline += torch.sum(w0 ** 2).item()
            sum_sq_diff += torch.sum((wn - w0) ** 2).item()
            
        if sum_sq_baseline > 0:
            block_drift = (sum_sq_diff ** 0.5) / (sum_sq_baseline ** 0.5)
        else:
            block_drift = 0.0
            
        drift_array.append(round(block_drift, 5))
        
    all_drifts[f"Gen {gen}"] = drift_array
    print(f"Done calculating Gen {gen}.")

print("\n" + "="*50)
print(f"SUCCESS! HERE ARE YOUR {HF_MODEL_ID} DRIFT ARRAYS:")
print("="*50)
for gen_name, array in all_drifts.items():
    print(f"{gen_name} = [{', '.join(map(str, array))}]")