import torch
from transformers import AutoModelForCausalLM
import argparse
import json
import re
import os
from collections import defaultdict

def get_layer_info(name):
    """
    Parses the parameter name to find its block index and whether it belongs 
    to Attention or MLP. Works across Llama, Gemma, Qwen, GPT-2, etc.
    """
    # Look for patterns like .layers.5. or .h.5.
    match = re.search(r'\.(layers|h|blocks)\.(\d+)\.', name)
    if not match:
        return None, None
    
    block_idx = int(match.group(2))
    name_lower = name.lower()
    
    # Classify as Attention or MLP
    if any(x in name_lower for x in ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'c_attn', 'c_proj']):
        component = 'attn'
    elif any(x in name_lower for x in ['mlp', 'fc', 'up_proj', 'down_proj', 'gate_proj', 'c_fc']):
        component = 'mlp'
    else:
        component = 'other'
        
    return block_idx, component

def compute_drift(base_model_path, target_model_path, output_path):
    print(f"Loading Base Model (Gen 0): {base_model_path}")
    # Load in fp16 to save system RAM. We compute norms in fp32 later.
    model_0 = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map="cpu")
    
    print(f"Loading Target Model (Gen X): {target_model_path}")
    model_x = AutoModelForCausalLM.from_pretrained(target_model_path, torch_dtype=torch.float16, device_map="cpu")

    dict_0 = dict(model_0.named_parameters())
    dict_x = dict(model_x.named_parameters())

    # We will accumulate the sum of squared differences and sum of squared base weights
    # Formula: Relative Frobenius Norm = sqrt( sum((W_x - W_0)^2) ) / sqrt( sum(W_0^2) )
    stats = defaultdict(lambda: {'attn_diff_sq': 0.0, 'attn_base_sq': 0.0, 
                                 'mlp_diff_sq': 0.0, 'mlp_base_sq': 0.0})

    print("Computing block-wise parameter drift...")
    with torch.no_grad():
        for name, p_0 in dict_0.items():
            if name not in dict_x:
                continue
            
            p_x = dict_x[name]
            block_idx, component = get_layer_info(name)
            
            if block_idx is None or component == 'other':
                continue # Skip embeddings, layernorms, and final lm_head
            
            # Convert to fp32 for stable mathematical norm calculation
            p_0_f32 = p_0.float()
            p_x_f32 = p_x.float()
            
            diff_sq = torch.sum((p_x_f32 - p_0_f32) ** 2).item()
            base_sq = torch.sum(p_0_f32 ** 2).item()
            
            if component == 'attn':
                stats[block_idx]['attn_diff_sq'] += diff_sq
                stats[block_idx]['attn_base_sq'] += base_sq
            elif component == 'mlp':
                stats[block_idx]['mlp_diff_sq'] += diff_sq
                stats[block_idx]['mlp_base_sq'] += base_sq

    # Finalize the math (take the square roots)
    results = {"blocks": {}}
    for block_idx in sorted(stats.keys()):
        s = stats[block_idx]
        
        attn_drift = (s['attn_diff_sq'] ** 0.5) / (s['attn_base_sq'] ** 0.5) if s['attn_base_sq'] > 0 else 0
        mlp_drift = (s['mlp_diff_sq'] ** 0.5) / (s['mlp_base_sq'] ** 0.5) if s['mlp_base_sq'] > 0 else 0
        
        results["blocks"][block_idx] = {
            "attn_relative_drift": float(attn_drift),
            "mlp_relative_drift": float(mlp_drift)
        }
        print(f"Block {block_idx:2d} | Attn Drift: {attn_drift:.4%} | MLP Drift: {mlp_drift:.4%}")

    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved drift results to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True, help="Path to Gen 0 model")
    parser.add_argument("--target", type=str, required=True, help="Path to Gen X model")
    parser.add_argument("--out", type=str, required=True, help="Path to save JSON output")
    args = parser.parse_args()
    
    compute_drift(args.base, args.target, args.out)