import json
import os
import numpy as np

def calculate_average_rank(filepath):
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    ranks = []
    for block_idx, components in data.get("blocks", {}).items():
        for comp_type in ["attn", "mlp"]:
            for layer in components.get(comp_type, []):
                ranks.append(layer["effective_rank"])
                
    if not ranks:
        return None
    return np.mean(ranks)

def main():
    models = {
        "SmolLM Treatment": ("smollm_gen_0.json", "smollm_treatment_gen_5.json"),
        "SmolLM Control A": ("smollm_gen_0.json", "smollm_control_a_gen_5.json"),
        "SmolLM Control B": ("smollm_gen_0.json", "smollm_control_b_gen_5.json"),
        "GPT-2": ("gpt2_gen_0.json", "gpt2_treatment_gen_5.json"),
        "Qwen 2.5 0.5B": ("qwen2_5_gen_0.json", "qwen2_5_treatment_gen_5.json"),
        "Qwen 3.5 0.8B": ("qwen3_5_gen_0.json", "qwen3_5_treatment_gen_5.json"),
        "Llama 1B": ("llama_gen_0.json", "llama_treatment_gen_5.json"),
        "Gemma 1B": ("gemma_gen_0.json", "gemma_treatment_gen_5.json")
    }
    
    print("\n" + "="*60)
    print(f"{'Model Architecture':<20} | {'Gen 0 Rank':<10} | {'Gen 5 Rank':<10} | {'Collapse %':<10}")
    print("-" * 60)
    
    for name, (base_file, target_file) in models.items():
        base_path = os.path.join("results/svd", base_file)
        target_path = os.path.join("results/svd", target_file)
        
        base_rank = calculate_average_rank(base_path)
        target_rank = calculate_average_rank(target_path)
        
        if base_rank and target_rank:
            collapse_pct = ((base_rank - target_rank) / base_rank) * 100
            print(f"{name:<20} | {base_rank:<10.2f} | {target_rank:<10.2f} | -{collapse_pct:.2f}%")
        else:
            print(f"{name:<20} | {'Missing Data':<10} | {'Missing Data':<10} | N/A")
            
    print("="*60 + "\n")

if __name__ == "__main__":
    main()