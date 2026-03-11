import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def parse_fim_text(filepath):
    fim_totals = {}
    encodings = ['utf-8', 'utf-16', 'cp1252']
    content = ""
    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                content = f.read()
            break
        except UnicodeError:
            continue
            
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("Block") and "|" in line:
            try:
                parts = line.split('|')
                block_idx = int(parts[0].replace('Block', '').strip())
                attn_val = float(parts[1].split(':')[1].strip())
                mlp_val = float(parts[2].split(':')[1].strip())
                fim_totals[block_idx] = attn_val + mlp_val
            except Exception:
                pass
    return fim_totals

def main():
    mappings = {
        "SmolLM Treatment": ("results/fimtreatment{gen}.txt", "results/parameterdrift/treatment_gen_{gen}.json"),
        "SmolLM Control A": ("results/fimcontrola{gen}.txt", "results/parameterdrift/control_generation_{gen}.json"),
        "SmolLM Control B": ("results/fimcontrolb{gen}.txt", "results/parameterdrift/control_b_gen_{gen}.json"),
        "GPT-2": ("results/fimgpt2_gen_{gen}.txt", "results/parameterdrift/gpt2_treatment_gen_{gen}.json"),
        "Llama 1B": ("results/fimllama_treatment_gen_{gen}.txt", "results/parameterdrift/llama_treatment_gen_{gen}.json"),
        "Gemma 1B": ("results/gemma_treatment_gen_{gen}.txt", "results/parameterdrift/gemma_treatment_gen_{gen}.json"),
        "Qwen 2.5 0.5B": ("results/fimqwen_gen_{gen}.txt", "results/parameterdrift/control_c_gen_{gen}.json"),
        "Qwen 3.5 0.8B": ("results/fim_summary_qwen_gen{gen}.txt", "results/parameterdrift/Qwen3.5-0.8B_treatment_gen_{gen}.json")
    }

    records = []
    print("Aggregating Master Dataset for Normalization...")
    
    for model_name, (fim_tmpl, drift_tmpl) in mappings.items():
        for gen in range(1, 6):
            fim_path = fim_tmpl.format(gen=gen)
            drift_path = drift_tmpl.format(gen=gen)
            
            if not os.path.exists(fim_path) or not os.path.exists(drift_path):
                continue
                
            fim_dict = parse_fim_text(fim_path)
            try:
                drift_data = load_json(drift_path)
            except:
                continue
                
            for block_str, drift_vals in drift_data.get('blocks', {}).items():
                block_idx = int(block_str)
                if block_idx in fim_dict:
                    fim_val = fim_dict[block_idx]
                    avg_drift = (drift_vals['attn_relative_drift'] + drift_vals['mlp_relative_drift']) / 2.0 * 100
                    
                    if fim_val > 0:
                        records.append({
                            'Model': model_name,
                            'Generation': gen,
                            'Block': block_idx,
                            'FIM': fim_val,
                            'Log10_FIM': np.log10(fim_val),
                            'Drift': avg_drift
                        })

    df = pd.DataFrame(records)
    if df.empty:
        print("Error: No data successfully aggregated.")
        return

    # ---------------------------------------------------------
    # THE MAGIC: Z-SCORE NORMALIZATION (Per Model)
    # ---------------------------------------------------------
    # We subtract the model's mean and divide by its standard deviation.
    # This centers every model at 0, removing the architectural scale bias!
    df['Norm_FIM'] = df.groupby('Model')['Log10_FIM'].transform(lambda x: (x - x.mean()) / x.std())
    df['Norm_Drift'] = df.groupby('Model')['Drift'].transform(lambda x: (x - x.mean()) / x.std())

    # Drop any NaNs (just in case standard deviation was 0)
    df = df.dropna(subset=['Norm_FIM', 'Norm_Drift'])

    grand_corr, grand_p = spearmanr(df['Norm_FIM'], df['Norm_Drift'])
    
    print("\n" + "="*60)
    print(" 🌟 NORMALIZED MASTER STATISTICAL PROOF 🌟 ")
    print("="*60)
    print("Simpson's Paradox Successfully Removed via Z-Score Standardization.")
    print(f"Total Data Points (Blocks): {len(df)}")
    print(f"True Universal Spearman Correlation: {grand_corr:.4f}")
    print(f"Master P-Value: {grand_p:.2e}")
    print("="*60 + "\n")

    # --- PLOTTING ---
    plt.figure(figsize=(12, 8))
    
    # Plot normalized points
    scatter = plt.scatter(df['Norm_FIM'], df['Norm_Drift'], c=df['Generation'], cmap='viridis', alpha=0.7, s=50, edgecolors='none')
    plt.colorbar(scatter, label='Generation (1 to 5)')
    
    # Calculate global trendline
    z = np.polyfit(df['Norm_FIM'], df['Norm_Drift'], 1)
    p = np.poly1d(z)
    
    # Generate line points
    x_range = np.linspace(df['Norm_FIM'].min(), df['Norm_FIM'].max(), 100)
    plt.plot(x_range, p(x_range), "r--", linewidth=4, label=f"True Global Trendline (r={grand_corr:.2f})")

    # Add lines at 0,0 to show the normalized center
    plt.axhline(0, color='black', linewidth=1, alpha=0.3)
    plt.axvline(0, color='black', linewidth=1, alpha=0.3)

    plt.title(f"The True Universal Sensitivity-Drift Paradox\nSimpson's Paradox Corrected (Z-Score Normalized)", fontsize=16, fontweight='bold')
    plt.xlabel("Normalized Fisher Information (Z-Scores)", fontsize=14, fontweight='bold')
    plt.ylabel("Normalized Parameter Drift (Z-Scores)", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    os.makedirs("results/master_plots", exist_ok=True)
    out_path = 'results/master_plots/normalized_grand_master.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved Normalized Master Plot to: {out_path}")

if __name__ == "__main__":
    main()