#!/usr/bin/env python3
"""
Compare baseline metrics with FIM analysis across generations
Visualizes patterns and correlations between traditional metrics and FIM eigenvalues
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import argparse

def load_baseline_metrics(baseline_path):
    """Load baseline metrics from JSON file"""
    with open(baseline_path, 'r') as f:
        return json.load(f)

def load_fim_metrics(fim_path):
    """Load FIM metrics from JSON file"""
    with open(fim_path, 'r') as f:
        return json.load(f)

def extract_block_metrics(baseline_data, block_idx, component='attention'):
    """Extract specific metrics for a block component"""
    if block_idx >= len(baseline_data['blocks']):
        return {}
    
    block_data = baseline_data['blocks'][block_idx]
    return block_data.get(component, {})

def extract_fim_eigenvalues(fim_data, block_idx, component='attention'):
    """Extract FIM eigenvalues for comparison"""
    if 'blocks' not in fim_data or block_idx >= len(fim_data['blocks']):
        return []
    
    block_data = fim_data['blocks'][block_idx]
    if component in block_data and 'eigenvalues' in block_data[component]:
        return block_data[component]['eigenvalues']
    return []

def compare_generations(baseline_files, fim_files, output_dir='comparison_plots'):
    """Compare metrics across generations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load all data
    baseline_data = []
    fim_data = []
    
    for baseline_file, fim_file in zip(baseline_files, fim_files):
        baseline_data.append(load_baseline_metrics(baseline_file))
        fim_data.append(load_fim_metrics(fim_file))
    
    num_generations = len(baseline_data)
    num_blocks = baseline_data[0]['num_blocks']
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Baseline Metrics vs FIM Eigenvalues Across Generations', fontsize=16)
    
    # Collect data for plotting
    generations = list(range(num_generations))
    
    # Plot 1: Frobenius norm vs FIM top eigenvalue (Block 0 Attention)
    baseline_fro = []
    fim_top = []
    
    for gen_idx in range(num_generations):
        block_0_attn = extract_block_metrics(baseline_data[gen_idx], 0, 'attention')
        baseline_fro.append(block_0_attn.get('frobenius_norm', 0))
        
        fim_eigenvals = extract_fim_eigenvalues(fim_data[gen_idx], 0, 'attention')
        fim_top.append(max(fim_eigenvals) if fim_eigenvals else 0)
    
    ax = axes[0, 0]
    ax.plot(generations, baseline_fro, 'b-o', label='Frobenius Norm', linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(generations, fim_top, 'r-s', label='FIM Top Eigenvalue', linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Frobenius Norm', color='b')
    ax2.set_ylabel('FIM Top Eigenvalue', color='r')
    ax.set_title('Block 0 Attention:\nFrobenius vs FIM')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Plot 2: Effective Rank vs FIM Eigenvalue Count
    baseline_rank = []
    fim_nonzero_count = []
    
    for gen_idx in range(num_generations):
        block_0_attn = extract_block_metrics(baseline_data[gen_idx], 0, 'attention')
        baseline_rank.append(block_0_attn.get('effective_rank', 0))
        
        fim_eigenvals = extract_fim_eigenvalues(fim_data[gen_idx], 0, 'attention')
        fim_nonzero_count.append(sum(1 for x in fim_eigenvals if x > 1e-3))
    
    ax = axes[0, 1]
    ax.plot(generations, baseline_rank, 'g-o', label='Effective Rank', linewidth=2)
    ax.plot(generations, fim_nonzero_count, 'm-s', label='FIM Nonzero Count', linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Rank/Count')
    ax.set_title('Block 0 Attention:\nRank Comparison')
    ax.legend()
    
    # Plot 3: Condition Number vs FIM Ratio
    baseline_cond = []
    fim_ratio = []
    
    for gen_idx in range(num_generations):
        block_0_attn = extract_block_metrics(baseline_data[gen_idx], 0, 'attention')
        cond_num = block_0_attn.get('condition_number', float('inf'))
        baseline_cond.append(min(cond_num, 1000))  # Cap for visualization
        
        fim_eigenvals = extract_fim_eigenvalues(fim_data[gen_idx], 0, 'attention')
        if len(fim_eigenvals) > 1 and fim_eigenvals[-1] > 1e-8:
            ratio = fim_eigenvals[0] / fim_eigenvals[-1]
        else:
            ratio = 1000
        fim_ratio.append(min(ratio, 1000))  # Cap for visualization
    
    ax = axes[0, 2]
    ax.plot(generations, baseline_cond, 'c-o', label='Condition Number', linewidth=2)
    ax.plot(generations, fim_ratio, 'y-s', label='FIM Eigenvalue Ratio', linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Ratio (capped at 1000)')
    ax.set_title('Block 0 Attention:\nCondition Number vs FIM Ratio')
    ax.legend()
    ax.set_yscale('log')
    
    # Plot 4-6: Same metrics for different blocks or MLP
    # Block 11 MLP (middle layer)
    baseline_fro_mlp = []
    fim_top_mlp = []
    
    target_block = min(11, num_blocks - 1)  # Use block 11 or last block if fewer
    
    for gen_idx in range(num_generations):
        block_mlp = extract_block_metrics(baseline_data[gen_idx], target_block, 'mlp')
        baseline_fro_mlp.append(block_mlp.get('frobenius_norm', 0))
        
        fim_eigenvals = extract_fim_eigenvalues(fim_data[gen_idx], target_block, 'mlp')
        fim_top_mlp.append(max(fim_eigenvals) if fim_eigenvals else 0)
    
    ax = axes[1, 0]
    ax.plot(generations, baseline_fro_mlp, 'b-o', label='Frobenius Norm', linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(generations, fim_top_mlp, 'r-s', label='FIM Top Eigenvalue', linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Frobenius Norm', color='b')
    ax2.set_ylabel('FIM Top Eigenvalue', color='r')
    ax.set_title(f'Block {target_block} MLP:\nFrobenius vs FIM')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Plot 5: Correlation heatmap
    correlation_data = []
    
    for gen_idx in range(num_generations):
        gen_metrics = []
        
        # Collect baseline metrics for block 0
        block_0_attn = extract_block_metrics(baseline_data[gen_idx], 0, 'attention')
        gen_metrics.extend([
            block_0_attn.get('frobenius_norm', 0),
            block_0_attn.get('effective_rank', 0),
            min(block_0_attn.get('condition_number', 1000), 1000)
        ])
        
        # Collect FIM metrics for block 0
        fim_eigenvals = extract_fim_eigenvalues(fim_data[gen_idx], 0, 'attention')
        gen_metrics.extend([
            max(fim_eigenvals) if fim_eigenvals else 0,
            sum(1 for x in fim_eigenvals if x > 1e-3),
            fim_eigenvals[0] / fim_eigenvals[-1] if len(fim_eigenvals) > 1 and fim_eigenvals[-1] > 1e-8 else 1
        ])
        
        correlation_data.append(gen_metrics)
    
    correlation_df = pd.DataFrame(correlation_data, columns=[
        'Fro_Norm', 'Eff_Rank', 'Cond_Num', 
        'FIM_Top', 'FIM_Count', 'FIM_Ratio'
    ])
    
    ax = axes[1, 1]
    sns.heatmap(correlation_df.corr(), annot=True, cmap='RdBu_r', center=0, ax=ax)
    ax.set_title('Correlation Matrix:\nBaseline vs FIM Metrics')
    
    # Plot 6: Layer-wise comparison (final generation)
    final_gen = num_generations - 1
    block_indices = list(range(min(10, num_blocks)))  # Show first 10 blocks
    
    baseline_fro_layers = []
    fim_top_layers = []
    
    for block_idx in block_indices:
        block_attn = extract_block_metrics(baseline_data[final_gen], block_idx, 'attention')
        baseline_fro_layers.append(block_attn.get('frobenius_norm', 0))
        
        fim_eigenvals = extract_fim_eigenvalues(fim_data[final_gen], block_idx, 'attention')
        fim_top_layers.append(max(fim_eigenvals) if fim_eigenvals else 0)
    
    ax = axes[1, 2]
    x_pos = np.arange(len(block_indices))
    width = 0.35
    
    ax.bar(x_pos - width/2, baseline_fro_layers, width, label='Frobenius Norm', alpha=0.7)
    ax2 = ax.twinx()
    ax2.bar(x_pos + width/2, fim_top_layers, width, label='FIM Top Eigenvalue', alpha=0.7, color='red')
    
    ax.set_xlabel('Block Index')
    ax.set_ylabel('Frobenius Norm', color='blue')
    ax2.set_ylabel('FIM Top Eigenvalue', color='red')
    ax.set_title(f'Layer-wise Comparison\n(Generation {final_gen})')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'B{i}' for i in block_indices])
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_fim_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlation_df

def main():
    parser = argparse.ArgumentParser(description='Compare baseline metrics with FIM analysis')
    parser.add_argument('--baseline_files', nargs='+', required=True, 
                      help='Baseline metrics JSON files (in generation order)')
    parser.add_argument('--fim_files', nargs='+', required=True,
                      help='FIM metrics JSON files (in generation order)')
    parser.add_argument('--output_dir', default='comparison_plots',
                      help='Output directory for plots')
    
    args = parser.parse_args()
    
    if len(args.baseline_files) != len(args.fim_files):
        print("Error: Number of baseline files must match number of FIM files")
        return
    
    correlation_df = compare_generations(args.baseline_files, args.fim_files, args.output_dir)
    
    print("\nCorrelation Matrix:")
    print(correlation_df.corr())
    
    print(f"\nPlots saved to {args.output_dir}/")

if __name__ == '__main__':
    main()