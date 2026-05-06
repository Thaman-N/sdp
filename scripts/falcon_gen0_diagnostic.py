"""
Falcon-7B Gen0 FIM Profile Diagnostic
======================================
Computes per-block FIM for Falcon-7B base model (no training) and
compares the profile against Phi-1.5 and Pythia-1.4B Gen0 FIM.

If Falcon's FIM profile (early blocks high, late blocks low) resembles
Phi-1.5 and Pythia, the positive FIM-drift correlation is very likely
to replicate. If the profile is structurally different, MQA or scale
may be a confound worth investigating before committing A100 time.

This requires only a forward pass through Falcon-7B — no training.
Estimated time: 20-30 minutes on RTX 5090.

Output:
  results/falcon_gen0_fim/perblock_fim.json
  Console comparison table vs Phi-1.5 and Pythia Gen0
"""

import os, gc, json, re, subprocess
import torch
import numpy as np

BASE_DIR   = r"D:\Thaman\Work\hessian-spectral-analysis"
FALCON_ID  = "tiiuae/falcon-7b"
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "falcon_gen0_fim")
FIM_SCRIPT = os.path.join(BASE_DIR, "scripts", "perblock_fim.py")

# Gen0 FIM files for comparison models
PHI_FIM_PATH    = os.path.join(BASE_DIR, "results", "phi-1_5_treatment_gen_0", "perblock_fim.json")
PYTHIA_FIM_PATH = os.path.join(BASE_DIR, "results", "pythia-1.4b_treatment_gen_0", "perblock_fim.json")
SMOLLM_FIM_PATH = os.path.join(BASE_DIR, "results", "treatment_gen_0", "perblock_fim.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_fim(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    out = {}
    for i, bd in enumerate(data.get("blocks", [])):
        if not isinstance(bd, dict):
            continue
        b    = bd.get("block_idx", i)
        attn = bd.get("attention", {})
        mlp  = bd.get("mlp", {})
        at   = float(attn.get("top", 0)) if isinstance(attn, dict) and "error" not in attn else 0.0
        mt   = float(mlp.get("top",  0)) if isinstance(mlp,  dict) and "error" not in mlp  else 0.0
        if at > 0 or mt > 0:
            out[b] = at + mt
    return out


def profile_stats(fim_dict):
    """Compute profile statistics for a FIM dict."""
    if not fim_dict:
        return {}
    blocks  = sorted(fim_dict.keys())
    values  = [fim_dict[b] for b in blocks]
    n       = len(blocks)
    early   = values[:n//2]
    late    = values[n//2:]
    return {
        "n_blocks":       n,
        "min":            min(values),
        "max":            max(values),
        "range_ratio":    max(values) / (min(values) + 1e-12),
        "early_mean":     np.mean(early),
        "late_mean":      np.mean(late),
        "early_late_ratio": np.mean(early) / (np.mean(late) + 1e-12),
        "log10_range":    np.log10(max(values) + 1e-12) - np.log10(min(values) + 1e-12),
    }


# ── STEP 1: Run FIM on Falcon Gen0 ────────────────────────────────────────────
falcon_fim_path = os.path.join(OUTPUT_DIR, "perblock_fim.json")

if os.path.exists(falcon_fim_path):
    print("Falcon Gen0 FIM already computed, loading...")
else:
    print("="*60)
    print("Computing Falcon-7B Gen0 FIM (no training required)")
    print("="*60)
    torch.cuda.empty_cache()
    gc.collect()
    subprocess.run([
        "python", FIM_SCRIPT,
        "--model_path", FALCON_ID,
        "--output_dir", OUTPUT_DIR,
        "--disable_flash_attn",
        "--num_batches", "5",
        "--num_eigenvalues", "20",
    ], check=True)
    print("Falcon FIM complete.")

# ── STEP 2: Load all FIM profiles ─────────────────────────────────────────────
fim_falcon  = load_fim(falcon_fim_path)
fim_phi     = load_fim(PHI_FIM_PATH)
fim_pythia  = load_fim(PYTHIA_FIM_PATH)
fim_smollm  = load_fim(SMOLLM_FIM_PATH)

models = {
    "Falcon-7B  (Par)": fim_falcon,
    "Phi-1.5    (Par)": fim_phi,
    "Pythia-1.4B(Par)": fim_pythia,
    "SmolLM-135M(Seq)": fim_smollm,
}

# ── STEP 3: Profile comparison ────────────────────────────────────────────────
print("\n" + "="*70)
print("FIM PROFILE COMPARISON — GEN0")
print("="*70)

stats_all = {}
for name, fim in models.items():
    if fim is None:
        print(f"  {name}: FIM file not found — skipping")
        continue
    s = profile_stats(fim)
    stats_all[name] = s
    print(f"\n  {name}")
    print(f"    Blocks:          {s['n_blocks']}")
    print(f"    FIM range:       {s['min']:.1f} to {s['max']:.1f}  ({s['range_ratio']:.0f}x)")
    print(f"    log10 range:     {s['log10_range']:.2f} decades")
    print(f"    Early mean FIM:  {s['early_mean']:.1f}")
    print(f"    Late mean FIM:   {s['late_mean']:.1f}")
    print(f"    Early/late ratio:{s['early_late_ratio']:.2f}x")

# ── STEP 4: Normalised block profile ─────────────────────────────────────────
print("\n" + "="*70)
print("NORMALISED FIM PROFILE (% of max, every 4th block)")
print("High early + low late = same pattern as Phi/Pythia = positive correlation likely")
print("="*70)

print(f"\n{'Block':<8}", end="")
for name in models:
    if stats_all.get(name):
        print(f"  {name[:12]:>12}", end="")
print()
print("-"*60)

# Find max block across all models for alignment
max_blocks = max(len(f) for f in models.values() if f is not None)
for b in range(0, max_blocks, 4):
    print(f"  {b:<6}", end="")
    for name, fim in models.items():
        if fim is None or name not in stats_all:
            print(f"  {'—':>12}", end="")
            continue
        if b in fim:
            pct = fim[b] / stats_all[name]['max'] * 100
            print(f"  {pct:>11.1f}%", end="")
        else:
            print(f"  {'—':>12}", end="")
    print()

# ── STEP 5: Diagnosis ─────────────────────────────────────────────────────────
print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

if fim_falcon and fim_phi and fim_pythia:
    f_stats  = stats_all.get("Falcon-7B  (Par)", {})
    ph_stats = stats_all.get("Phi-1.5    (Par)", {})
    py_stats = stats_all.get("Pythia-1.4B(Par)", {})

    # Key metric: does early FIM dominate late FIM (like Phi/Pythia)?
    f_el  = f_stats.get('early_late_ratio', 0)
    ph_el = ph_stats.get('early_late_ratio', 0)
    py_el = py_stats.get('early_late_ratio', 0)

    print(f"\n  Early/late FIM ratio:")
    print(f"    Falcon:  {f_el:.2f}x")
    print(f"    Phi-1.5: {ph_el:.2f}x")
    print(f"    Pythia:  {py_el:.2f}x")

    if f_el > 1.5:
        print(f"""
  VERDICT: Falcon's FIM profile shows early-block dominance similar to
  Phi-1.5 and Pythia (early/late ratio {f_el:.1f}x). Under parallel
  residual streams, high-FIM early blocks will accumulate large v_t
  but cannot redirect gradient (simultaneous attention+MLP on shared
  residual). Positive FIM-drift correlation is LIKELY.
  
  Proceed with Falcon-7B training on A100.
""")
    elif f_el > 1.0:
        print(f"""
  VERDICT: Falcon shows mild early-block FIM dominance ({f_el:.1f}x),
  weaker than Phi-1.5 ({ph_el:.1f}x) and Pythia ({py_el:.1f}x).
  MQA may be dampening the FIM contrast. Positive correlation is
  PLAUSIBLE but may be weaker in magnitude. Proceed with caution
  and expect smaller rho values than Pythia.
""")
    else:
        print(f"""
  VERDICT: Falcon's FIM profile does NOT show early-block dominance
  (early/late ratio {f_el:.1f}x). This is structurally different from
  Phi-1.5 ({ph_el:.1f}x) and Pythia ({py_el:.1f}x). MQA or scale
  may be significantly confounding the FIM structure.
  
  Consider investigating the FIM structure further before committing
  A100 time. The positive correlation may still appear but the
  mechanism may operate differently.
""")