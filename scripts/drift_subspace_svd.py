"""
Drift Subspace Dimensionality Analysis
=======================================
Computes the effective rank of the weight drift (ΔW = W_gen5 - W_gen0)
per block per model, then compares sequential vs parallel architectures.

Hypothesis:
  Sequential collapse = low-rank drift (AdamW v_t funnels gradient into
  a small number of principal directions in late low-FIM blocks)
  Parallel collapse   = higher-rank drift (positive FIM-drift correlation
  means drift spreads more uniformly across all singular directions)

No new training runs required — uses Gen0 (HuggingFace) and Gen5 (saved).

Output:
  results/drift_subspace/svd_results.json
  results/drift_subspace/svd_summary.csv
  Console summary table
"""

import os, re, json, gc
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────
BASE_DIR    = r"D:\Thaman\Work\hessian-spectral-analysis\models"
OUTPUT_DIR  = "results/drift_subspace"
VARIANCE_THRESHOLD = 0.95   # effective rank = min dims to explain 95% variance
TOP_K_REPORT = [1, 3, 5, 10, 20]  # report cumulative variance at these ranks

MODELS = {
    "SmolLM_Trt":  {
        "base":  "HuggingFaceTB/SmolLM2-135M",
        "gen5":  "treatment_gen_5",
        "arch":  "sequential",
        "n_blocks": 30,
    },
    "SmolLM_CtrlA": {
        "base":  "HuggingFaceTB/SmolLM2-135M",
        "gen5":  "control_generation_5",
        "arch":  "sequential",
        "n_blocks": 30,
    },
    "SmolLM_CtrlB": {
        "base":  "HuggingFaceTB/SmolLM2-135M",
        "gen5":  "control_b_gen_5",
        "arch":  "sequential",
        "n_blocks": 30,
    },
    "GPT2": {
        "base":  "gpt2",
        "gen5":  "gpt2_treatment_gen_5",
        "arch":  "sequential",
        "n_blocks": 12,
    },
    "Llama": {
        "base":  "meta-llama/Llama-3.2-1B",
        "gen5":  "llama_treatment_gen_5",
        "arch":  "sequential",
        "n_blocks": 16,
    },
    "Phi15": {
        "base":  "microsoft/phi-1_5",
        "gen5":  "phi-1_5_treatment_gen_5",
        "arch":  "parallel",
        "n_blocks": 24,
    },
    "Pythia": {
        "base":  "EleutherAI/pythia-1.4b",
        "gen5":  "pythia-1.4b_treatment_gen_5",
        "arch":  "parallel",
        "n_blocks": 24,
    },
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── HELPERS ───────────────────────────────────────────────────────────────

def get_block_index(name):
    """Extract block index from parameter name. Returns None if not a block param."""
    for pattern in [r'\.layers\.(\d+)\.', r'\.h\.(\d+)\.', r'\.blocks\.(\d+)\.']:
        m = re.search(pattern, name)
        if m:
            return int(m.group(1))
    return None


def is_weight_matrix(name, tensor):
    """Only process 2D weight matrices (not biases, norms, embeddings)."""
    return (tensor.ndim == 2
            and 'embed' not in name.lower()
            and 'lm_head' not in name.lower()
            and 'norm' not in name.lower()
            and 'ln_' not in name.lower())


def effective_rank(singular_values, threshold=0.95):
    """
    Minimum k such that top-k singular values explain >= threshold
    of total Frobenius norm squared (= sum of squared singular values).
    """
    sv_sq = singular_values ** 2
    total = sv_sq.sum()
    if total == 0:
        return 0
    cumvar = np.cumsum(sv_sq) / total
    k = int(np.searchsorted(cumvar, threshold)) + 1
    return min(k, len(singular_values))


def cumvar_at_k(singular_values, ks):
    """Fraction of variance explained by top-k singular values."""
    sv_sq = singular_values ** 2
    total = sv_sq.sum()
    if total == 0:
        return {k: 0.0 for k in ks}
    result = {}
    for k in ks:
        result[k] = float(np.sum(sv_sq[:k]) / total)
    return result


def analyse_model(model_name, cfg):
    gen5_path = os.path.join(BASE_DIR, cfg["gen5"])
    print(f"\n{'='*65}")
    print(f"Model: {model_name}  ({cfg['arch'].upper()})")
    print(f"  Base: {cfg['base']}")
    print(f"  Gen5: {gen5_path}")

    if not os.path.exists(gen5_path):
        print(f"  ✗ Gen5 path not found, skipping.")
        return None

    # Load both models on CPU to avoid VRAM limits
    print("  Loading Gen0 (base)...")
    m0 = AutoModelForCausalLM.from_pretrained(
        cfg["base"], torch_dtype=torch.float32,
        device_map="cpu", low_cpu_mem_usage=True
    )
    print("  Loading Gen5...")
    m5 = AutoModelForCausalLM.from_pretrained(
        gen5_path, torch_dtype=torch.float32,
        device_map="cpu", low_cpu_mem_usage=True
    )

    d0 = dict(m0.named_parameters())
    d5 = dict(m5.named_parameters())

    # Accumulate per-block SVD results
    # block_data[b] = list of (matrix_name, singular_values_array)
    block_data = defaultdict(list)

    with torch.no_grad():
        for name, p0 in d0.items():
            if name not in d5:
                continue
            b = get_block_index(name)
            if b is None:
                continue
            if not is_weight_matrix(name, p0):
                continue

            delta = (d5[name].float() - p0.float()).numpy()

            # SVD — use full_matrices=False for efficiency
            # For large matrices cap at rank 50 with randomised SVD
            rows, cols = delta.shape
            k = min(rows, cols, 50)
            try:
                if min(rows, cols) <= 512:
                    # Full SVD for small matrices
                    _, s, _ = np.linalg.svd(delta, full_matrices=False)
                else:
                    # Randomised SVD for large matrices (Phi/Pythia MLPs)
                    from sklearn.utils.extmath import randomized_svd
                    _, s, _ = randomized_svd(delta, n_components=k,
                                             random_state=42)
            except Exception as e:
                print(f"    SVD failed for {name}: {e}")
                continue

            mat_label = name.split(f".{b}.")[-1] if f".{b}." in name else name
            block_data[b].append((mat_label, s))

    del m0, m5
    gc.collect()

    # Aggregate per block
    block_results = {}
    for b in sorted(block_data.keys()):
        mats = block_data[b]
        if not mats:
            continue

        # Per-matrix effective rank, then average
        eff_ranks   = []
        cumvars_agg = {k: [] for k in TOP_K_REPORT}

        for mat_label, s in mats:
            er = effective_rank(s, VARIANCE_THRESHOLD)
            eff_ranks.append(er)
            cv = cumvar_at_k(s, TOP_K_REPORT)
            for k in TOP_K_REPORT:
                cumvars_agg[k].append(cv[k])

        block_results[b] = {
            "mean_effective_rank": float(np.mean(eff_ranks)),
            "max_effective_rank":  float(np.max(eff_ranks)),
            "min_effective_rank":  float(np.min(eff_ranks)),
            "n_matrices":          len(eff_ranks),
            "cumvar": {
                str(k): float(np.mean(cumvars_agg[k]))
                for k in TOP_K_REPORT
            }
        }

    # Summary stats
    all_er = [v["mean_effective_rank"] for v in block_results.values()]
    n_blocks_actual = len(all_er)
    early_er = [block_results[b]["mean_effective_rank"]
                for b in sorted(block_results.keys())
                if b < cfg["n_blocks"]//2]
    late_er  = [block_results[b]["mean_effective_rank"]
                for b in sorted(block_results.keys())
                if b >= cfg["n_blocks"]//2]

    summary = {
        "model":          model_name,
        "arch":           cfg["arch"],
        "n_blocks":       n_blocks_actual,
        "mean_eff_rank":  float(np.mean(all_er)),
        "early_mean_er":  float(np.mean(early_er)) if early_er else 0,
        "late_mean_er":   float(np.mean(late_er))  if late_er  else 0,
        "mean_cumvar_1":  float(np.mean([v["cumvar"]["1"]  for v in block_results.values()])),
        "mean_cumvar_5":  float(np.mean([v["cumvar"]["5"]  for v in block_results.values()])),
        "mean_cumvar_10": float(np.mean([v["cumvar"]["10"] for v in block_results.values()])),
        "blocks":         block_results,
    }

    # Print per-block table
    print(f"\n  {'Block':<7} {'EffRank':>8} {'Top1%':>8} {'Top5%':>8} {'Top10%':>8}")
    print(f"  {'-'*45}")
    for b in sorted(block_results.keys()):
        r = block_results[b]
        print(f"  {b:<7} {r['mean_effective_rank']:>8.1f} "
              f"{r['cumvar']['1']*100:>7.1f}% "
              f"{r['cumvar']['5']*100:>7.1f}% "
              f"{r['cumvar']['10']*100:>7.1f}%")

    print(f"\n  SUMMARY: mean_eff_rank={summary['mean_eff_rank']:.1f} | "
          f"early={summary['early_mean_er']:.1f} late={summary['late_mean_er']:.1f} | "
          f"top1={summary['mean_cumvar_1']*100:.1f}% "
          f"top5={summary['mean_cumvar_5']*100:.1f}% "
          f"top10={summary['mean_cumvar_10']*100:.1f}%")

    return summary


# ── MAIN ──────────────────────────────────────────────────────────────────

all_results = {}
for model_name, cfg in MODELS.items():
    result = analyse_model(model_name, cfg)
    if result:
        all_results[model_name] = result

# Save JSON
out_json = os.path.join(OUTPUT_DIR, "svd_results.json")
with open(out_json, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved full results to {out_json}")

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────
print("\n" + "="*75)
print("DRIFT SUBSPACE DIMENSIONALITY — CROSS-MODEL SUMMARY")
print(f"(Effective rank = min dims to explain {VARIANCE_THRESHOLD*100:.0f}% of Frobenius drift)")
print("="*75)
print(f"\n{'Model':<16} {'Arch':<11} {'MeanER':>8} {'EarlyER':>9} {'LateER':>8} "
      f"{'Top1%':>7} {'Top5%':>7} {'Top10%':>8}")
print("-"*75)

seq_means, par_means = [], []
rows = []
for name, r in all_results.items():
    arch = r["arch"]
    print(f"  {name:<14} {arch:<11} {r['mean_eff_rank']:>8.1f} "
          f"{r['early_mean_er']:>9.1f} {r['late_mean_er']:>8.1f} "
          f"{r['mean_cumvar_1']*100:>6.1f}% "
          f"{r['mean_cumvar_5']*100:>6.1f}% "
          f"{r['mean_cumvar_10']*100:>7.1f}%")
    rows.append({
        "Model": name, "Architecture": arch,
        "Mean_EffRank": round(r["mean_eff_rank"], 2),
        "Early_EffRank": round(r["early_mean_er"], 2),
        "Late_EffRank": round(r["late_mean_er"], 2),
        "Top1_VarPct": round(r["mean_cumvar_1"]*100, 1),
        "Top5_VarPct": round(r["mean_cumvar_5"]*100, 1),
        "Top10_VarPct": round(r["mean_cumvar_10"]*100, 1),
    })
    if arch == "sequential":
        seq_means.append(r["mean_eff_rank"])
    else:
        par_means.append(r["mean_eff_rank"])

# Save CSV
df = pd.DataFrame(rows)
out_csv = os.path.join(OUTPUT_DIR, "svd_summary.csv")
df.to_csv(out_csv, index=False)
print(f"\nSaved summary CSV to {out_csv}")

# Architecture comparison
if seq_means and par_means:
    print(f"\n{'─'*75}")
    print(f"  Sequential mean effective rank: {np.mean(seq_means):.2f}  (n={len(seq_means)})")
    print(f"  Parallel   mean effective rank: {np.mean(par_means):.2f}  (n={len(par_means)})")
    from scipy import stats as sp_stats
    if len(seq_means) >= 2 and len(par_means) >= 2:
        t, p = sp_stats.ttest_ind(par_means, seq_means, equal_var=False)
        print(f"  Welch t = {t:.3f}, p = {p:.4f}")
    else:
        diff = np.mean(par_means) - np.mean(seq_means)
        print(f"  Difference: {diff:+.2f} (need more models for significance test)")

print(f"""
INTERPRETATION GUIDE:
  Low effective rank  (e.g. 1-3):  Collapse is highly structured — drift
    concentrates in a few principal directions. Geometry is "quasi-1D rotation."
  
  High effective rank (e.g. 10+):  Collapse is diffuse — drift spreads across
    many independent directions. Less structured, less predictable.

  Expected finding:
    Sequential: low effective rank in late blocks (low-FIM drift, v_t
    suppresses high-FIM early blocks, leaving a structured residual)
    Parallel: higher effective rank (drift unconstrained, spreads widely)
  
  Early vs Late block split:
    Sequential: late blocks should have LOWER rank (they drift most via
    Frobenius norm but in fewer directions — concentrated runaway)
    Parallel: more uniform effective rank across early/late
""")