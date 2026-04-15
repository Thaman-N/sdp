"""
Drift Direction Cosine Similarity Across Generations
=====================================================
For each block in each model, computes the cosine similarity between
consecutive-generation incremental drift vectors:

  δ_n = W_gen_n - W_gen_{n-1}      (incremental drift, NOT cumulative)
  cos(n→n+1) = dot(δ_n, δ_{n+1}) / (||δ_n|| ||δ_{n+1}||)

If sequential models show cos ≈ 0 (random direction each generation)
and parallel models show cos > 0 (drift reinforces same direction),
this is a new geometric result that:
  1. Explains WHY ortho interventions failed
  2. Explains WHY parallel collapse compounds faster
  3. Provides a direction-based account of the FIM-drift sign difference

Memory efficient: loads only 2 models at a time, computes δ on-the-fly,
discards each generation after use. Uses fp16 to halve RAM.

Output:
  results/drift_cosine/cosine_results.json
  results/drift_cosine/cosine_summary.csv
"""

import os, re, json, gc
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────
BASE_DIR   = r"D:\Thaman\Work\hessian-spectral-analysis\models"
OUTPUT_DIR = "results/drift_cosine"

# Which generation pairs to compute cosine similarity for
# cos(1→2) means: similarity between δ₁=(Gen1-Gen0) and δ₂=(Gen2-Gen1)
GEN_PAIRS  = [(1,2), (2,3), (3,4), (4,5)]

MODELS = {
    "SmolLM_Trt": {
        "base":   "HuggingFaceTB/SmolLM2-135M",
        "prefix": "treatment_gen_",
        "arch":   "sequential",
        "n_blocks": 30,
    },
    "SmolLM_CtrlA": {
        "base":   "HuggingFaceTB/SmolLM2-135M",
        "prefix": "control_generation_",
        "arch":   "sequential",
        "n_blocks": 30,
    },
    "GPT2": {
        "base":   "gpt2",
        "prefix": "gpt2_treatment_gen_",
        "arch":   "sequential",
        "n_blocks": 12,
    },
    "Llama": {
        "base":   "meta-llama/Llama-3.2-1B",
        "prefix": "llama_treatment_gen_",
        "arch":   "sequential",
        "n_blocks": 16,
    },
    "Phi15": {
        "base":   "microsoft/phi-1_5",
        "prefix": "phi-1_5_treatment_gen_",
        "arch":   "parallel",
        "n_blocks": 24,
    },
    "Pythia": {
        "base":   "EleutherAI/pythia-1.4b",
        "prefix": "pythia-1.4b_treatment_gen_",
        "arch":   "parallel",
        "n_blocks": 24,
    },
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── HELPERS ───────────────────────────────────────────────────────────────

def get_block_index(name):
    for pattern in [r'\.layers\.(\d+)\.', r'\.h\.(\d+)\.', r'\.blocks\.(\d+)\.']:
        m = re.search(pattern, name)
        if m:
            return int(m.group(1))
    return None


def is_weight_matrix(name, tensor):
    """Only process 2D weight matrices."""
    return (tensor.ndim == 2
            and 'embed' not in name.lower()
            and 'lm_head' not in name.lower()
            and 'norm' not in name.lower()
            and 'ln_' not in name.lower())


def load_state(path_or_id):
    """Load model state dict as float32 on CPU."""
    m = AutoModelForCausalLM.from_pretrained(
        path_or_id,
        torch_dtype=torch.float16,   # load fp16 to save RAM
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    # Upcast to float32 for accurate cosine computation
    sd = {k: v.float() for k, v in m.state_dict().items()}
    del m
    gc.collect()
    return sd


def compute_delta(sd_a, sd_b):
    """
    Compute per-block incremental drift vectors.
    Returns dict: {block_idx: {mat_name: flat_delta_np_array}}
    """
    deltas = defaultdict(dict)
    for name, pa in sd_a.items():
        if name not in sd_b:
            continue
        b = get_block_index(name)
        if b is None:
            continue
        if not is_weight_matrix(name, pa):
            continue
        pb = sd_b[name]
        d = (pb - pa).numpy().flatten().astype(np.float32)
        mat_label = name.split(f".{b}.")[-1] if f".{b}." in name else name
        deltas[b][mat_label] = d
    return deltas


def cosine_sim(v1, v2):
    """Cosine similarity between two flat vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return float('nan')
    return float(np.dot(v1, v2) / (n1 * n2))


# ── PER-MODEL ANALYSIS ────────────────────────────────────────────────────

def analyse_model(model_name, cfg):
    print(f"\n{'='*65}")
    print(f"Model: {model_name}  ({cfg['arch'].upper()})")

    gen_paths = {}
    gen_paths[0] = cfg["base"]  # HuggingFace ID or local
    for g in range(1, 6):
        p = os.path.join(BASE_DIR, f"{cfg['prefix']}{g}")
        if not os.path.exists(p):
            print(f"  ✗ Missing: {p}")
            return None
        gen_paths[g] = p

    print(f"  All 6 generation paths found ✓")

    # We need generations 0-5 to compute 4 consecutive cosine similarities.
    # Load sequentially: keep current delta, load next, compute next delta,
    # compute cosine, discard previous delta.

    # Step 1: load Gen0 and Gen1, compute δ₁
    print(f"  Loading Gen0...")
    sd_prev = load_state(gen_paths[0])
    print(f"  Loading Gen1...")
    sd_curr = load_state(gen_paths[1])
    delta_prev = compute_delta(sd_prev, sd_curr)  # δ₁ = Gen1 - Gen0
    del sd_prev
    gc.collect()

    # block_cosines[b][gen_pair_label] = cosine_value
    block_cosines = defaultdict(dict)

    for g in range(2, 6):
        print(f"  Loading Gen{g}...")
        sd_next = load_state(gen_paths[g])
        delta_curr = compute_delta(sd_curr, sd_next)  # δ_g = Gen_g - Gen_{g-1}

        # Compute cosine similarity per block, per matrix
        pair_label = f"cos_{g-1}_{g}"
        common_blocks = set(delta_prev.keys()) & set(delta_curr.keys())

        for b in sorted(common_blocks):
            cos_vals = []
            common_mats = set(delta_prev[b].keys()) & set(delta_curr[b].keys())
            for mat in common_mats:
                c = cosine_sim(delta_prev[b][mat], delta_curr[b][mat])
                if not np.isnan(c):
                    cos_vals.append(c)
            if cos_vals:
                block_cosines[b][pair_label] = float(np.mean(cos_vals))

        # Advance: curr becomes prev
        del sd_curr
        gc.collect()
        sd_curr  = sd_next
        delta_prev = delta_curr
        del sd_next
        gc.collect()

    del sd_curr
    gc.collect()

    # Aggregate results per block
    block_results = {}
    for b in sorted(block_cosines.keys()):
        pair_vals = list(block_cosines[b].values())
        if pair_vals:
            block_results[b] = {
                "mean_cosine":   float(np.mean(pair_vals)),
                "std_cosine":    float(np.std(pair_vals)),
                "pair_cosines":  block_cosines[b],
            }

    # Overall summary
    all_cos = [v["mean_cosine"] for v in block_results.values()]
    n_blocks = cfg["n_blocks"]
    early = [block_results[b]["mean_cosine"]
             for b in sorted(block_results.keys()) if b < n_blocks//2]
    late  = [block_results[b]["mean_cosine"]
             for b in sorted(block_results.keys()) if b >= n_blocks//2]

    summary = {
        "model":           model_name,
        "arch":            cfg["arch"],
        "mean_cosine":     float(np.mean(all_cos)),
        "std_cosine":      float(np.std(all_cos)),
        "early_mean_cos":  float(np.mean(early)) if early else 0.0,
        "late_mean_cos":   float(np.mean(late))  if late  else 0.0,
        "pct_positive":    float(np.mean([c > 0 for c in all_cos]) * 100),
        "blocks":          block_results,
    }

    # Print block table
    print(f"\n  {'Block':<7} {'MeanCos':>10} {'Std':>8}  {'Sign':>6}")
    print(f"  {'-'*38}")
    for b in sorted(block_results.keys()):
        r = block_results[b]
        sign = "+" if r["mean_cosine"] > 0.01 else ("-" if r["mean_cosine"] < -0.01 else "~0")
        print(f"  {b:<7} {r['mean_cosine']:>10.5f} {r['std_cosine']:>8.5f}  {sign:>6}")

    print(f"\n  SUMMARY: mean_cos={summary['mean_cosine']:.5f} | "
          f"early={summary['early_mean_cos']:.5f} late={summary['late_mean_cos']:.5f} | "
          f"{summary['pct_positive']:.0f}% positive")

    return summary


# ── MAIN ──────────────────────────────────────────────────────────────────

all_results = {}
for model_name, cfg in MODELS.items():
    result = analyse_model(model_name, cfg)
    if result:
        all_results[model_name] = result

# Save JSON
out_json = os.path.join(OUTPUT_DIR, "cosine_results.json")
with open(out_json, "w") as f:
    json.dump(all_results, f, indent=2)

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────
print("\n" + "="*72)
print("DRIFT DIRECTION COSINE SIMILARITY — CROSS-MODEL SUMMARY")
print("cos(δ_n, δ_{n+1}) averaged over blocks and generation pairs 1→2, 2→3, 3→4, 4→5")
print("="*72)
print(f"\n{'Model':<16} {'Arch':<11} {'MeanCos':>10} {'EarlyCos':>10} "
      f"{'LateCos':>9} {'%Pos':>7}")
print("-"*65)

seq_cos, par_cos = [], []
rows = []
for name, r in all_results.items():
    arch = r["arch"]
    print(f"  {name:<14} {arch:<11} {r['mean_cosine']:>10.5f} "
          f"{r['early_mean_cos']:>10.5f} {r['late_mean_cos']:>9.5f} "
          f"{r['pct_positive']:>6.0f}%")
    rows.append({
        "Model":         name,
        "Architecture":  arch,
        "Mean_Cosine":   round(r["mean_cosine"], 6),
        "Early_Cosine":  round(r["early_mean_cos"], 6),
        "Late_Cosine":   round(r["late_mean_cos"], 6),
        "Pct_Positive":  round(r["pct_positive"], 1),
    })
    if arch == "sequential":
        seq_cos.append(r["mean_cosine"])
    else:
        par_cos.append(r["mean_cosine"])

# Save CSV
df = pd.DataFrame(rows)
out_csv = os.path.join(OUTPUT_DIR, "cosine_summary.csv")
df.to_csv(out_csv, index=False)
print(f"\nSaved to {out_csv} and {out_json}")

# Architecture comparison
from scipy import stats as sp
if seq_cos and par_cos:
    print(f"\n{'─'*72}")
    print(f"  Sequential mean cosine: {np.mean(seq_cos):.5f} ± {np.std(seq_cos):.5f}")
    print(f"  Parallel   mean cosine: {np.mean(par_cos):.5f} ± {np.std(par_cos):.5f}")
    if len(seq_cos) >= 2 and len(par_cos) >= 2:
        t, p = sp.ttest_ind(par_cos, seq_cos, equal_var=False)
        print(f"  Welch t = {t:.3f}, p = {p:.4f}")
    else:
        print(f"  Difference: {np.mean(par_cos)-np.mean(seq_cos):+.5f}")

print("""
INTERPRETATION:
  If mean_cosine ≈ 0:  Drift direction is random each generation.
                       Consecutive δ's are nearly orthogonal.
                       → Ortho CANNOT work (nothing to project out)

  If mean_cosine > 0:  Drift reinforces a consistent direction.
                       Each generation nudges weights in the same way.
                       → Compounding collapse (parallel archetype)

  If mean_cosine < 0:  Drift partially reverses each generation.
                       Oscillatory behaviour — unlikely but possible
                       in heavy EWC-like scenarios.

  Key split to look for:
    Sequential ~0 vs Parallel >0  →  NEW FINDING, supports main claim
    Both ~0                        →  Null (ortho fails for all)
    Both >0                        →  Null (no architecture difference)
""")