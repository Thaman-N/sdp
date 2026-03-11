import json
import os
import numpy as np

def get_averages(filepath):
    if not os.path.exists(filepath): return None
    with open(filepath, 'r') as f: data = json.load(f)
        
    metrics = {"effective_rank": [], "threshold_rank": [], "variance_99_rank": [], "condition_number": []}
    
    for block_idx, components in data.get("blocks", {}).items():
        for comp_type in ["attn", "mlp"]:
            for layer in components.get(comp_type, []):
                for k in metrics.keys():
                    if k in layer:
                        metrics[k].append(layer[k])
                        
    if not metrics["effective_rank"]: return None
    return {k: np.mean(v) for k, v in metrics.items()}

def main():
    models = {
        "SmolLM Treatment": ("smollm_gen_0.json", "smollm_treatment_gen_5.json"),
        "GPT-2 Treatment": ("gpt2_gen_0.json", "gpt2_treatment_gen_5.json"),
        "Qwen 2.5 0.5B": ("qwen2_5_gen_0.json", "qwen2_5_treatment_gen_5.json"),
        "Qwen 3.5 0.8B": ("qwen3_5_gen_0.json", "qwen3_5_treatment_gen_5.json"),
        "Llama 1B": ("llama_gen_0.json", "llama_treatment_gen_5.json"),
        "Gemma 3 1B": ("gemma_gen_0.json", "gemma_treatment_gen_5.json")
    }
    
    print("\n" + "="*110)
    print(f"{'Model Architecture':<20} | {'Eff. Rank (Δ%)':<20} | {'Thresh. Rank (Δ%)':<20} | {'99% Var Rank (Δ%)':<20} | {'Condition Num (Δ%)':<20}")
    print("-" * 110)
    
    for name, (base_file, target_file) in models.items():
        base_path = os.path.join("results/svd", base_file)
        target_path = os.path.join("results/svd", target_file)
        
        base_metrics = get_averages(base_path)
        target_metrics = get_averages(target_path)
        
        if base_metrics and target_metrics:
            # Effective Rank
            eff_pct = ((base_metrics['effective_rank'] - target_metrics['effective_rank']) / base_metrics['effective_rank']) * 100
            eff_str = f"{-eff_pct:+.2f}%"
            
            # Threshold Rank
            thresh_pct = ((base_metrics['threshold_rank'] - target_metrics['threshold_rank']) / base_metrics['threshold_rank']) * 100
            thresh_str = f"{-thresh_pct:+.2f}%"
            
            # 99% Variance Rank
            var_pct = ((base_metrics['variance_99_rank'] - target_metrics['variance_99_rank']) / base_metrics['variance_99_rank']) * 100
            var_str = f"{-var_pct:+.2f}%"
            
            # Condition Number
            cond_pct = ((target_metrics['condition_number'] - base_metrics['condition_number']) / base_metrics['condition_number']) * 100
            cond_str = f"+{cond_pct:.1f}%"
            
            print(f"{name:<20} | {eff_str:<20} | {thresh_str:<20} | {var_str:<20} | {cond_str:<20}")
        else:
            print(f"{name:<20} | {'N/A':<20} | {'N/A':<20} | {'N/A':<20} | {'N/A':<20}")
            
    print("="*110 + "\n")

if __name__ == "__main__":
    main()