"""
L2 Norm Trajectory Analysis
Computes per-block Frobenius norm for Attn and MLP weight matrices
across all generations for all models.

Tells us: are weights rotating (stable norm), expanding, or compressing
during recursive training?

Output: norm_trajectories.json + norm_trajectories.csv
"""

import os
import re
import json
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from transformers import AutoModelForCausalLM

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = r"D:\Thaman\Work\hessian-spectral-analysis\models"

MODEL_CONFIGS = {
    "SmolLM_Trt": {
        "hf_id":    "HuggingFaceTB/SmolLM2-135M",
        "folders":  {i: f"treatment_gen_{i}" for i in range(1, 6)},
    },
    "SmolLM_CtrlA": {
        "hf_id":    "HuggingFaceTB/SmolLM2-135M",
        "folders":  {i: f"control_generation_{i}" for i in range(1, 6)},
    },
    "SmolLM_CtrlB": {
        "hf_id":    "HuggingFaceTB/SmolLM2-135M",
        "folders":  {i: f"control_b_gen_{i}" for i in range(1, 6)},
    },
    "GPT2": {
        "hf_id":    "gpt2",
        "folders":  {i: f"gpt2_treatment_gen_{i}" for i in range(1, 6)},
    },
    "Llama": {
        "hf_id":    "meta-llama/Llama-3.2-1B",
        "folders":  {i: f"llama_treatment_gen_{i}" for i in range(1, 6)},
    },
    "Qwen2.5": {
        "hf_id":    "Qwen/Qwen2.5-0.5B",
        "folders":  {i: f"control_c_gen_{i}" for i in range(1, 6)},
    },
    "Qwen3.5": {
        "hf_id":    "Qwen/Qwen3.5-0.8B",
        "folders":  {i: f"Qwen3.5-0.8B_treatment_gen_{i}" for i in range(1, 6)},
    },
    "Gemma": {
        "hf_id":    "google/gemma-3-1b-it",
        "folders":  {i: f"gemma_treatment_gen_{i}" for i in range(1, 6)},
    },
}

OUTPUT_DIR = "results/norm_trajectories"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LAYER PARSING — same logic as parameter_drift.py
# ============================================================
def get_layer_info(name):
    match = re.search(r'\.(layers|h|blocks)\.(\d+)\.', name)
    if not match:
        return None, None
    block_idx = int(match.group(2))
    name_lower = name.lower()
    if any(x in name_lower for x in
           ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj',
            'o_proj', 'c_attn', 'c_proj']):
        component = 'attn'
    elif any(x in name_lower for x in
             ['mlp', 'fc', 'up_proj', 'down_proj', 'gate_proj', 'c_fc']):
        component = 'mlp'
    else:
        component = 'other'
    return block_idx, component


# ============================================================
# NORM COMPUTATION
# ============================================================
def compute_norms(model_path):
    """
    Returns dict: {block_idx: {attn_norm, mlp_norm}}
    Norm = Frobenius norm of each weight matrix, summed per block per component.
    Computed in float32 for numerical stability.
    """
    print(f"    Loading: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )

    # accumulate sum of squared norms per block/component
    stats = defaultdict(lambda: {
        'attn_norm_sq': 0.0,
        'mlp_norm_sq':  0.0,
    })

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim < 2:
                continue
            block_idx, component = get_layer_info(name)
            if block_idx is None or component == 'other':
                continue

            W = param.float()
            if W.ndim > 2:
                W = W.reshape(W.shape[0], -1)

            norm_sq = torch.sum(W ** 2).item()

            if component == 'attn':
                stats[block_idx]['attn_norm_sq'] += norm_sq
            elif component == 'mlp':
                stats[block_idx]['mlp_norm_sq'] += norm_sq

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    results = {}
    for block_idx in sorted(stats.keys()):
        s = stats[block_idx]
        results[block_idx] = {
            'attn_norm': float(s['attn_norm_sq'] ** 0.5),
            'mlp_norm':  float(s['mlp_norm_sq']  ** 0.5),
        }
    return results


# ============================================================
# MAIN LOOP
# ============================================================
all_results = {}
rows = []

for model_name, cfg in MODEL_CONFIGS.items():
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")

    model_results = {}

    # --- Gen 0: load from HuggingFace ---
    print(f"  Gen 0 (base): {cfg['hf_id']}")
    try:
        gen0_norms = compute_norms(cfg['hf_id'])
        model_results[0] = gen0_norms
        for block, v in gen0_norms.items():
            rows.append({
                'Model': model_name, 'Gen': 0, 'Block': block,
                'Attn_Norm': v['attn_norm'], 'MLP_Norm': v['mlp_norm'],
                'Total_Norm': v['attn_norm'] + v['mlp_norm'],
            })
    except Exception as e:
        print(f"    ERROR Gen 0: {e}")
        gen0_norms = None

    # --- Gen 1-5: load from local folders ---
    for gen, folder in cfg['folders'].items():
        path = os.path.join(BASE_DIR, folder)
        if not os.path.exists(path):
            print(f"  Gen {gen}: MISSING ({path})")
            continue
        if not os.path.exists(os.path.join(path, 'config.json')):
            print(f"  Gen {gen}: SKIPPING — no config.json in {path}")
            continue

        print(f"  Gen {gen}: {folder}")
        try:
            norms = compute_norms(path)
            model_results[gen] = norms
            for block, v in norms.items():
                rows.append({
                    'Model': model_name, 'Gen': gen, 'Block': block,
                    'Attn_Norm': v['attn_norm'], 'MLP_Norm': v['mlp_norm'],
                    'Total_Norm': v['attn_norm'] + v['mlp_norm'],
                })
        except Exception as e:
            print(f"    ERROR Gen {gen}: {e}")

    all_results[model_name] = model_results

    # --- Print summary for this model ---
    if gen0_norms and model_results:
        print(f"\n  NORM TRAJECTORY SUMMARY (mean total norm across blocks):")
        print(f"  {'Gen':<6} {'Mean_Attn_Norm':>16} {'Mean_MLP_Norm':>14} "
              f"{'vs Gen0 (%)':>12}")
        g0_total = np.mean([v['attn_norm'] + v['mlp_norm']
                            for v in gen0_norms.values()])
        for gen in sorted(model_results.keys()):
            norms = model_results[gen]
            mean_attn = np.mean([v['attn_norm'] for v in norms.values()])
            mean_mlp  = np.mean([v['mlp_norm']  for v in norms.values()])
            mean_total = mean_attn + mean_mlp
            pct_change = (mean_total - g0_total) / g0_total * 100
            print(f"  {gen:<6} {mean_attn:>16.2f} {mean_mlp:>14.2f} "
                  f"{pct_change:>+11.2f}%")

# ============================================================
# SAVE OUTPUTS
# ============================================================
# JSON: full per-block detail
json_path = os.path.join(OUTPUT_DIR, "norm_trajectories.json")
# convert int keys to str for JSON
json_safe = {
    model: {
        str(gen): {str(b): v for b, v in blocks.items()}
        for gen, blocks in gens.items()
    }
    for model, gens in all_results.items()
}
with open(json_path, 'w') as f:
    json.dump(json_safe, f, indent=2)
print(f"\nSaved full JSON: {json_path}")

# CSV: flat table for easy plotting
csv_path = os.path.join(OUTPUT_DIR, "norm_trajectories.csv")
df = pd.DataFrame(rows)
df.to_csv(csv_path, index=False)
print(f"Saved CSV:       {csv_path}")

# ============================================================
# CROSS-MODEL SUMMARY TABLE
# ============================================================
print("\n\n" + "="*70)
print("CROSS-MODEL SUMMARY: Mean total norm change Gen0 -> Gen5")
print("="*70)
print(f"{'Model':<18} {'Gen0':>10} {'Gen1':>10} {'Gen3':>10} "
      f"{'Gen5':>10} {'G0->G5 %':>10}")
for model_name, model_results in all_results.items():
    if 0 not in model_results:
        continue
    def mean_total(gen):
        if gen not in model_results:
            return float('nan')
        return np.mean([v['attn_norm'] + v['mlp_norm']
                        for v in model_results[gen].values()])
    g0 = mean_total(0)
    g1 = mean_total(1)
    g3 = mean_total(3)
    g5 = mean_total(5)
    pct = (g5 - g0) / g0 * 100 if not np.isnan(g5) else float('nan')
    print(f"{model_name:<18} {g0:>10.2f} {g1:>10.2f} {g3:>10.2f} "
          f"{g5:>10.2f} {pct:>+9.2f}%")
