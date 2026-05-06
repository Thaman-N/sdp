"""
Pythia vs Phi-1.5 FIM Profile Comparison
==========================================
Checks whether the per-block FIM structure of Pythia-1.4B (which we
have 5 generations of data for) resembles Phi-1.5 Gen0 FIM.

If their FIM profiles are structurally similar (both show early-block
dominance, similar log10 FIM range), this confirms that the two parallel
families are operating under the same structural conditions, and that
the positive correlation result is robust across parallel architectures
with different training recipes.

Also computes: Spearman correlation between Pythia Gen0 FIM ranks and
Phi-1.5 Gen0 FIM ranks (normalised by block position) to see if the
two models have similar sensitivity hierarchies.

Uses existing Gen0 FIM files — no new computation needed.

Output: Console report only (no new files needed)
"""

import os, json
import numpy as np
from scipy import stats

BASE_DIR = r"D:\Thaman\Work\hessian-spectral-analysis"

# Gen0 FIM paths for all models
FIM_PATHS = {
    "SmolLM_Trt  (Seq, 30b)": os.path.join(BASE_DIR, "results", "treatment_gen_0",               "perblock_fim.json"),
    "GPT2        (Seq, 12b)": os.path.join(BASE_DIR, "results", "gpt2_treatment_gen_0",            "perblock_fim.json"),
    "Llama       (Seq, 16b)": os.path.join(BASE_DIR, "results", "llama_treatment_gen_0",           "perblock_fim.json"),
    "Phi-1.5     (Par, 24b)": os.path.join(BASE_DIR, "results", "phi-1_5_treatment_gen_0",         "perblock_fim.json"),
    "Pythia-1.4B (Par, 24b)": os.path.join(BASE_DIR, "results", "pythia-1.4b_treatment_gen_0",     "perblock_fim.json"),
}


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


print("="*70)
print("PER-BLOCK FIM PROFILE COMPARISON ACROSS ALL MODELS")
print("="*70)

all_fim = {}
for name, path in FIM_PATHS.items():
    fim = load_fim(path)
    if fim is None:
        print(f"  {name}: file not found at {path}")
        continue
    all_fim[name] = fim
    blocks  = sorted(fim.keys())
    values  = [fim[b] for b in blocks]
    n       = len(blocks)
    early   = np.mean(values[:n//2])
    late    = np.mean(values[n//2:])
    log10v  = [np.log10(v + 1e-12) for v in values]
    log10rng = max(log10v) - min(log10v)
    el_ratio = early / (late + 1e-12)

    print(f"\n  {name}")
    print(f"    FIM range: {min(values):.1f} to {max(values):.1f}  "
          f"({max(values)/(min(values)+1e-12):.0f}x linear, {log10rng:.1f} log10 decades)")
    print(f"    Early mean: {early:.1f}  Late mean: {late:.1f}  "
          f"Early/late: {el_ratio:.2f}x")

# ── Block-position normalised FIM rank correlation ───────────────────────────
print("\n" + "="*70)
print("NORMALISED FIM RANK CORRELATION BETWEEN MODELS")
print("Block position normalised to [0,1] for cross-model comparison")
print("="*70)

def normalised_profile(fim_dict):
    """Return FIM values at normalised block positions [0,1]."""
    blocks = sorted(fim_dict.keys())
    n = len(blocks)
    positions = [b / (n - 1) for b in range(n)]
    values    = [fim_dict[b] for b in blocks]
    return np.array(positions), np.array(values)

# Compare Phi vs Pythia profile similarity
if "Phi-1.5     (Par, 24b)" in all_fim and "Pythia-1.4B (Par, 24b)" in all_fim:
    _, phi_vals    = normalised_profile(all_fim["Phi-1.5     (Par, 24b)"])
    _, pythia_vals = normalised_profile(all_fim["Pythia-1.4B (Par, 24b)"])

    # Both have 24 blocks so direct comparison
    rho_phi_pythia, p = stats.spearmanr(
        np.log10(phi_vals + 1e-12),
        np.log10(pythia_vals + 1e-12)
    )
    print(f"\n  Phi-1.5 vs Pythia-1.4B FIM rank correlation:")
    print(f"    Spearman rho = {rho_phi_pythia:+.4f}  p = {p:.4f}")
    if rho_phi_pythia > 0.7:
        print(f"    SIMILAR profiles: both models have the same sensitivity")
        print(f"    hierarchy. Falcon is likely to follow the same pattern.")
    elif rho_phi_pythia > 0.4:
        print(f"    MODERATELY similar: same general trend but differences")
        print(f"    in specific block ranks. Positive correlation still likely.")
    else:
        print(f"    DISSIMILAR: Phi and Pythia have different sensitivity")
        print(f"    hierarchies despite both being parallel. Model family")
        print(f"    differences matter more than just parallel classification.")

# Compare sequential vs parallel profile shape
print(f"\n  Sequential vs Parallel early/late ratio comparison:")
for name, fim in all_fim.items():
    blocks = sorted(fim.keys())
    n = len(blocks)
    values = [fim[b] for b in blocks]
    early = np.mean(values[:n//2])
    late  = np.mean(values[n//2:])
    arch  = "Par" if "Par" in name else "Seq"
    ratio = early / (late + 1e-12)
    bar   = "█" * int(ratio * 3)
    print(f"    {name[:20]:<20} {arch}  early/late = {ratio:.2f}x  {bar}")

# ── FIM log10 range comparison ────────────────────────────────────────────────
print(f"\n" + "="*70)
print("LOG10 FIM RANGE — KEY METRIC FOR EWC AND CORRELATION STRENGTH")
print("="*70)
print(f"Larger range = stronger Spearman correlation possible")
print(f"Also determines EWC effectiveness (need >1 decade for selectivity)\n")

for name, fim in all_fim.items():
    values  = list(fim.values())
    log10v  = [np.log10(v + 1e-12) for v in values]
    log10rng = max(log10v) - min(log10v)
    bar = "█" * int(log10rng * 4)
    print(f"  {name[:22]:<22}  {log10rng:.2f} log10 decades  {bar}")

print(f"""
INTERPRETATION:
  If both parallel models (Phi, Pythia) show similar log10 FIM range
  and early/late dominance, and Falcon's Gen0 diagnostic shows the same,
  then Falcon's positive correlation is structural and predictable.

  The EWC failure threshold is approximately 1.4-3.2x across blocks
  (measured in your experiments). Any model with log10 range < 0.5
  decades (3x) would likely show the same EWC failure pattern.
""")