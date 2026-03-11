import json
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def parse_fim_text(filepath):
    fim_totals = {}
    # Windows PowerShell uses UTF-16 for the > operator. We will try both encodings to be safe.
    encodings = ['utf-8', 'utf-16', 'cp1252']
    content = ""
    
    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                content = f.read()
            break # Stop if successful
        except UnicodeError:
            continue
            
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("Block") and "|" in line:
            parts = line.split('|')
            block_idx = int(parts[0].replace('Block', '').strip())
            attn_val = float(parts[1].split(':')[1].strip())
            mlp_val = float(parts[2].split(':')[1].strip())
            fim_totals[block_idx] = attn_val + mlp_val
            
    return fim_totals

def plot_paradox(fim_text_path, drift_path, output_path, model_name):
    print(f"Loading FIM Summary from: {fim_text_path}")
    fim_dict = parse_fim_text(fim_text_path)
    
    print(f"Loading Drift data from: {drift_path}")
    drift_data = load_json(drift_path)
    
    blocks = []
    fim_totals = []
    drift_totals = []
    
    for block_str, drift_vals in drift_data.get('blocks', {}).items():
        block_idx = int(block_str)
        if block_idx in fim_dict:
            blocks.append(block_idx)
            fim_totals.append(fim_dict[block_idx])
            # Combine Attn and MLP drift (average)
            avg_drift = (drift_vals['attn_relative_drift'] + drift_vals['mlp_relative_drift']) / 2.0
            drift_totals.append(avg_drift * 100) # Convert to Percentage

    if not blocks:
        print("Error: Could not align blocks. Check your FIM text file format.")
        return

    # Sort by block index
    blocks, fim_totals, drift_totals = zip(*sorted(zip(blocks, fim_totals, drift_totals)))
    
    # Calculate Correlation
    corr, p_value = spearmanr(fim_totals, drift_totals)
    print(f"\n--- Statistical Proof ---")
    print(f"Spearman Correlation: {corr:.4f} (p-value: {p_value:.4e})")
    
    # --- PLOTTING ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_fim = 'tab:red'
    ax1.set_xlabel('Transformer Block Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fisher Information Magnitude (Sensitivity)', color=color_fim, fontsize=12, fontweight='bold')
    ax1.bar(blocks, fim_totals, color=color_fim, alpha=0.6, label='FIM (Curvature)')
    ax1.tick_params(axis='y', labelcolor=color_fim)
    
    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()  
    color_drift = 'tab:blue'
    ax2.set_ylabel('Parameter Drift (%)', color=color_drift, fontsize=12, fontweight='bold')  
    ax2.plot(blocks, drift_totals, color=color_drift, marker='o', linewidth=3, markersize=8, label='Physical Drift')
    ax2.tick_params(axis='y', labelcolor=color_drift)

    plt.title(f"{model_name} (Gen 5) - The Sensitivity-Drift Paradox\nSpearman Correlation: {corr:.2f}", fontsize=14, fontweight='bold')
    
    # Add grid and layout
    ax1.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()  

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Saved Paradox Plot to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fim", type=str, required=True, help="Path to FIM text summary file")
    parser.add_argument("--drift", type=str, required=True, help="Path to parameter drift json")
    parser.add_argument("--out", type=str, required=True, help="Path to save PNG plot")
    parser.add_argument("--name", type=str, default="Model", help="Name for plot title")
    args = parser.parse_args()
    
    plot_paradox(args.fim, args.drift, args.out, args.name)