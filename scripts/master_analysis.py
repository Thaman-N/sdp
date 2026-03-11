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
    print("Aggregating Master Dataset...")
    
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

    os.makedirs("results/master_plots", exist_ok=True)

    # ---------------------------------------------------------
    # 1. THE CORRELATION MATRIX (Model vs Generation)
    # ---------------------------------------------------------
    print("\n--- Generating Correlation Matrix ---")
    matrix_data = []
    for model in df['Model'].unique():
        row = {'Model': model}
        for gen in range(1, 6):
            subset = df[(df['Model'] == model) & (df['Generation'] == gen)]
            if len(subset) > 2:
                corr, _ = spearmanr(subset['Log10_FIM'], subset['Drift'])
                row[f'Gen {gen}'] = corr
            else:
                row[f'Gen {gen}'] = np.nan
        matrix_data.append(row)
    
    corr_df = pd.DataFrame(matrix_data).set_index('Model')
    print(corr_df.round(4).to_string())

    plt.figure(figsize=(10, 6))
    cax = plt.matshow(corr_df, cmap='coolwarm', vmin=-1, vmax=1, fignum=1)
    plt.colorbar(cax, label='Spearman Correlation')
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, fontsize=10)
    plt.yticks(range(len(corr_df.index)), corr_df.index, fontsize=10)
    
    # Add text annotations to the heatmap
    for i in range(len(corr_df.index)):
        for j in range(len(corr_df.columns)):
            val = corr_df.iloc[i, j]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.2f}", ha='center', va='center', 
                         color='white' if abs(val) > 0.5 else 'black', fontweight='bold')

    plt.title('Sensitivity-Drift Paradox Correlation Matrix', pad=20, fontsize=14, fontweight='bold')
    plt.savefig('results/master_plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # 2. MASTER PER-GENERATION PLOTS
    # ---------------------------------------------------------
    print("\n--- Generating Per-Generation Plots ---")
    colors = plt.cm.tab10(np.linspace(0, 1, len(df['Model'].unique())))
    model_colors = dict(zip(df['Model'].unique(), colors))

    for gen in range(1, 6):
        subset_gen = df[df['Generation'] == gen]
        if subset_gen.empty: continue
        
        overall_corr, p_val = spearmanr(subset_gen['Log10_FIM'], subset_gen['Drift'])
        print(f"Gen {gen} Master Correlation (All Models): {overall_corr:.4f} (p={p_val:.2e})")

        plt.figure(figsize=(10, 6))
        for model in subset_gen['Model'].unique():
            subset_model = subset_gen[subset_gen['Model'] == model]
            plt.scatter(subset_model['Log10_FIM'], subset_model['Drift'], 
                        label=model, color=model_colors[model], alpha=0.7, s=50, edgecolors='k')

        z = np.polyfit(subset_gen['Log10_FIM'], subset_gen['Drift'], 1)
        p = np.poly1d(z)
        plt.plot(subset_gen['Log10_FIM'], p(subset_gen['Log10_FIM']), "k--", linewidth=2, label=f"Trendline (r={overall_corr:.2f})")

        plt.title(f"Generation {gen}: Sensitivity vs. Drift (All Models)\nMaster Spearman: {overall_corr:.2f}", fontsize=14, fontweight='bold')
        plt.xlabel("Log10 Fisher Information (Sensitivity)", fontsize=12, fontweight='bold')
        plt.ylabel("Parameter Drift (%)", fontsize=12, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'results/master_plots/master_gen_{gen}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # ---------------------------------------------------------
    # 3. THE GRAND MASTER PLOT (All Models, All Gens)
    # ---------------------------------------------------------
    grand_corr, grand_p = spearmanr(df['Log10_FIM'], df['Drift'])
    print(f"\nGRAND MASTER CORRELATION (All Data Points): {grand_corr:.4f} (p={grand_p:.2e})")

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['Log10_FIM'], df['Drift'], c=df['Generation'], cmap='viridis', alpha=0.6, s=40, edgecolors='none')
    plt.colorbar(scatter, label='Generation (1 to 5)')
    
    z = np.polyfit(df['Log10_FIM'], df['Drift'], 1)
    p = np.poly1d(z)
    plt.plot(df['Log10_FIM'], p(df['Log10_FIM']), "r--", linewidth=3, label=f"Global Trendline (r={grand_corr:.2f})")

    plt.title(f"The Universal Sensitivity-Drift Paradox\n{len(df)} Blocks Across All Architectures & Generations", fontsize=16, fontweight='bold')
    plt.xlabel("Log10 Fisher Information Magnitude (Sensitivity)", fontsize=14, fontweight='bold')
    plt.ylabel("Parameter Drift (%)", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results/master_plots/grand_master_all.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nAll master plots saved to 'results/master_plots/' directory!")

if __name__ == "__main__":
    main()