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

def compute_advanced_svd(model_path, output_path):
    print(f"Loading Model for Advanced SVD: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map="cpu")
    
    results = {"blocks": {}}
    
    print("Computing Advanced Singular Value Decompositions...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if len(param.shape) < 2:
                continue 
                
            block_idx, component = get_layer_info(name)
            if block_idx is None or component == 'other':
                continue
                
            W = param.float()
            
            # ---> THE FIX: Flatten any 3D+ tensors to strict 2D matrices <---
            if W.ndim > 2:
                W = W.reshape(W.shape[0], -1)
                
            S = torch.linalg.svdvals(W)
            
            # --- METRIC 1: Roy-Vetterli Effective Rank (Entropy) ---
            S_norm = S / S.sum()
            S_norm_safe = S_norm[S_norm > 0] 
            entropy_rank = torch.exp(-torch.sum(S_norm_safe * torch.log(S_norm_safe))).item()
            
            # --- METRIC 2: Threshold Rank (0.01 * Max Singular Value) ---
            threshold = 0.01 * S[0]
            threshold_rank = torch.sum(S > threshold).item()
            
            # --- METRIC 3: 99% Explained Variance Rank ---
            cumulative_variance = torch.cumsum(S**2, dim=0) / torch.sum(S**2)
            var_99_rank = torch.searchsorted(cumulative_variance, 0.99).item() + 1
            
            # --- METRIC 4: Condition Number (Max / Min) ---
            s_min = S[-1] if S[-1] > 1e-7 else torch.tensor(1e-7)
            condition_number = (S[0] / s_min).item()
            
            if str(block_idx) not in results["blocks"]:
                results["blocks"][str(block_idx)] = {"attn": [], "mlp": []}
                
            results["blocks"][str(block_idx)][component].append({
                "layer_name": name.split('.')[-1],
                "effective_rank": entropy_rank,
                "threshold_rank": threshold_rank,
                "variance_99_rank": var_99_rank,
                "condition_number": condition_number
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved Advanced SVD results to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    compute_advanced_svd(args.model, args.out)