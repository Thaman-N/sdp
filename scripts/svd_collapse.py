import torch
from transformers import AutoModelForCausalLM
import argparse
import json
import re
import os

def get_layer_info(name):
    match = re.search(r'\.(layers|h|blocks)\.(\d+)\.', name)
    if not match:
        return None, None
    block_idx = int(match.group(2))
    name_lower = name.lower()
    
    if any(x in name_lower for x in ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'c_attn', 'c_proj']):
        component = 'attn'
    elif any(x in name_lower for x in ['mlp', 'fc', 'up_proj', 'down_proj', 'gate_proj', 'c_fc']):
        component = 'mlp'
    else:
        component = 'other'
    return block_idx, component

def compute_svd(model_path, output_path):
    print(f"Loading Model for SVD: {model_path}")
    
    # We load in float32 for stable SVD math, using CPU to avoid OOM
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map="cpu")
    
    results = {"blocks": {}}
    
    print("Computing Singular Value Decompositions (This takes a minute)...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if len(param.shape) < 2:
                continue # Skip 1D vectors (biases, layernorms)
                
            block_idx, component = get_layer_info(name)
            if block_idx is None or component == 'other':
                continue
                
            W = param.float()
            
            # Compute Singular Values (S)
            S = torch.linalg.svdvals(W)
            
            # Effective Rank Formula: Shannon Entropy of the Singular Values
            # This is the gold standard for measuring true dimensionality of a neural network layer
            S_norm = S / S.sum()
            S_norm = S_norm[S_norm > 0] # Avoid log(0)
            entropy_rank = torch.exp(-torch.sum(S_norm * torch.log(S_norm))).item()
            
            if str(block_idx) not in results["blocks"]:
                results["blocks"][str(block_idx)] = {"attn": [], "mlp": []}
                
            results["blocks"][str(block_idx)][component].append({
                "layer_name": name.split('.')[-1],
                "effective_rank": entropy_rank
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved SVD results to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the model (HF ID or local folder)")
    parser.add_argument("--out", type=str, required=True, help="Path to save JSON output")
    args = parser.parse_args()
    compute_svd(args.model, args.out)