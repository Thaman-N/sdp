"""
SGD Ablation: Recursive Self-Distillation without AdamW
========================================================
Trains SmolLM2-135M for 5 generations on the same synthetic data as the
treatment condition, but using pure SGD instead of AdamW.

Hypothesis (from v_t mechanism):
  Under pure SGD (no second moment accumulation), the FIM-drift correlation
  should be near zero or weakly positive: high-FIM blocks receive large
  gradients and drift proportionally, since there is no v_t to suppress them.
  This completes the optimizer-axis characterisation:

    beta2=0.9    -> +0.46* (fast decay, weak suppression)
    beta2=0.999  -> -0.51** (standard AdamW)
    beta2=0.9999 -> -0.06  (over-saturation, no differential)
    SGD          -> ~0 or weakly positive (no v_t at all)

Key design choices:
  - Reuses existing treatment synthetic data (data/treatment_synthetic_gen_N)
    so no new data generation needed. Same data = cleanest comparison to
    treatment condition.
  - SGD with momentum=0 (pure gradient descent, no momentum term).
  - Learning rate matched to AdamW lr=5e-5. Note: SGD and AdamW use
    fundamentally different effective step sizes so this is not a perfect
    match, but the comparison is about the presence/absence of v_t, not
    step size calibration.
  - Per-block FIM computed identically to treatment condition.

Output:
  models/sgd_ablation_gen_1-5/
  results/sgd_ablation_gen_1-5/perblock_fim.json
  results/sgd_ablation_summary.json   (Spearman rho per generation)
"""

import os
import gc
import json
import subprocess
import torch
import numpy as np
from tqdm import tqdm
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_from_disk
from torch.utils.data import DataLoader

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_ID    = "HuggingFaceTB/SmolLM2-135M"
BASE_DIR    = r"D:\Thaman\Work\hessian-spectral-analysis"
DATA_PREFIX = os.path.join(BASE_DIR, "data", "treatment_synthetic_gen_")
MODEL_OUT   = os.path.join(BASE_DIR, "models", "sgd_ablation_gen_")
RESULT_OUT  = os.path.join(BASE_DIR, "results", "sgd_ablation_gen_")
SUMMARY_OUT = os.path.join(BASE_DIR, "results", "sgd_ablation_summary.json")
FIM_SCRIPT  = os.path.join(BASE_DIR, "scripts", "perblock_fim.py")

MAX_LENGTH  = 256
BATCH_SIZE  = 8
LR          = 5e-5       # Same as treatment AdamW lr
MOMENTUM    = 0.0        # Pure SGD, no momentum
GENERATIONS = 5


def train_one_generation(gen, source_model_path, data_path, model_out_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n  [TRAIN] Gen {gen} | source: {source_model_path}")
    print(f"  [TRAIN] data: {data_path}")
    print(f"  [TRAIN] output: {model_out_path}")

    model = AutoModelForCausalLM.from_pretrained(
        source_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        attn_implementation="eager"
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(source_model_path)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    dataset = load_from_disk(data_path)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    dataset.set_format("torch")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ── Pure SGD, no momentum, no weight decay ────────────────────────────────
    # This is the cleanest test of the v_t hypothesis: AdamW v_t is entirely
    # absent. weight_decay=0 ensures no AdamW-style decoupled decay either.
    optim = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=0.0
    )

    # Warmup scheduler (same as treatment for fair comparison)
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=100,
        num_training_steps=len(loader)
    )

    model.train()
    pbar = tqdm(loader, desc=f"  SGD Gen {gen}")
    total_loss = 0.0

    for step, batch in enumerate(pbar):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        loss.backward()
        optim.step()
        sched.step()
        optim.zero_grad()
        total_loss += loss.item()

        if step % 100 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    print(f"  [TRAIN] Gen {gen} avg loss: {avg_loss:.4f}")

    os.makedirs(model_out_path, exist_ok=True)
    model.save_pretrained(model_out_path)
    tokenizer.save_pretrained(model_out_path)
    print(f"  [TRAIN] Saved to {model_out_path}")

    del model, optim, sched
    torch.cuda.empty_cache()
    gc.collect()


def run_fim(model_path, result_dir):
    """Run per-block FIM analysis using existing perblock_fim.py script."""
    os.makedirs(result_dir, exist_ok=True)
    print(f"\n  [FIM] Running on {model_path}")
    torch.cuda.empty_cache()
    gc.collect()

    subprocess.run([
        "python", FIM_SCRIPT,
        "--model_path", model_path,
        "--output_dir", result_dir,
        "--disable_flash_attn",
        "--num_batches", "5",
        "--num_eigenvalues", "20",
    ], check=True)
    print(f"  [FIM] Done → {result_dir}")


def load_fim_results(result_dir):
    """
    Load per-block FIM values from perblock_fim.json.
    Structure: {"architecture": ..., "blocks": [{"block_idx": 0,
                "attention": {"top": X, ...}, "mlp": {"top": Y, ...}}, ...]}
    Returns: {block_idx: attn_top + mlp_top}
    """
    path = os.path.join(result_dir, "perblock_fim.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)

    fim_per_block = {}
    blocks_list = data.get("blocks", [])
    for i, block_data in enumerate(blocks_list):
        if not isinstance(block_data, dict):
            continue
        b = block_data.get("block_idx", i)
        attn_top = 0.0
        mlp_top  = 0.0
        attn = block_data.get("attention", {})
        mlp  = block_data.get("mlp", {})
        if isinstance(attn, dict) and "error" not in attn:
            attn_top = float(attn.get("top", 0))
        if isinstance(mlp, dict) and "error" not in mlp:
            mlp_top = float(mlp.get("top", 0))
        if attn_top > 0 or mlp_top > 0:
            fim_per_block[b] = attn_top + mlp_top
    return fim_per_block


def compute_drift(model_path_0, model_path_n):
    """Compute relative Frobenius drift per block vs Gen0."""
    import re
    from collections import defaultdict

    def load_sd(path):
        m = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float32,
            device_map="cpu", low_cpu_mem_usage=True
        )
        sd = {k: v.clone() for k, v in m.state_dict().items()}
        del m
        gc.collect()
        return sd

    print(f"  [DRIFT] Loading Gen0 weights...")
    sd0 = load_sd(model_path_0)
    print(f"  [DRIFT] Loading Gen{model_path_n} weights...")
    sdn = load_sd(model_path_n)

    block_norms  = defaultdict(float)
    block_drifts = defaultdict(float)

    for name, p0 in sd0.items():
        if name not in sdn:
            continue
        # Extract block index
        m = None
        for pat in [r'\.layers\.(\d+)\.', r'\.h\.(\d+)\.', r'\.blocks\.(\d+)\.']:
            m = re.search(pat, name)
            if m:
                break
        if m is None:
            continue
        b = int(m.group(1))
        # Only weight matrices
        if p0.ndim < 2:
            continue
        if any(x in name.lower() for x in ['embed', 'lm_head', 'norm', 'ln_']):
            continue

        pn = sdn[name]
        drift = (pn.float() - p0.float()).norm(p='fro').item()
        base  = p0.float().norm(p='fro').item()
        block_drifts[b] += drift ** 2
        block_norms[b]  += base  ** 2

    relative_drift = {}
    for b in block_drifts:
        if block_norms[b] > 0:
            relative_drift[b] = (block_drifts[b] ** 0.5) / (block_norms[b] ** 0.5)

    del sd0, sdn
    gc.collect()
    return relative_drift


def compute_spearman(fim_dict, drift_dict):
    """Compute Spearman rho between log10(FIM_b) and drift_b."""
    common = sorted(set(fim_dict.keys()) & set(drift_dict.keys()))
    if len(common) < 5:
        return float('nan'), float('nan'), len(common)

    log_fim = np.array([np.log10(fim_dict[b] + 1e-12) for b in common])
    drift   = np.array([drift_dict[b] for b in common])
    rho, p  = stats.spearmanr(log_fim, drift)
    return float(rho), float(p), len(common)


# ── MAIN ──────────────────────────────────────────────────────────────────────

print("=" * 70)
print("SGD ABLATION: RECURSIVE SELF-DISTILLATION WITHOUT AdamW v_t")
print(f"Model: {MODEL_ID}")
print(f"Optimizer: SGD (momentum={MOMENTUM}, lr={LR}, weight_decay=0)")
print(f"Reusing treatment synthetic data from: {DATA_PREFIX}N")
print("=" * 70)

# Gen0 FIM (reuse from treatment results if available)
gen0_fim_dir = os.path.join(BASE_DIR, "results", "treatment_gen_0")
if not os.path.exists(os.path.join(gen0_fim_dir, "perblock_fim.json")):
    print("\nGen0 FIM not found in treatment results, running fresh...")
    run_fim(MODEL_ID, gen0_fim_dir)

fim_gen0 = load_fim_results(gen0_fim_dir)
print(f"\nGen0 FIM loaded: {len(fim_gen0)} blocks")

summary = {}

for gen in range(1, GENERATIONS + 1):
    print(f"\n{'='*70}")
    print(f"GENERATION {gen}")
    print(f"{'='*70}")

    data_path    = f"{DATA_PREFIX}{gen}"
    model_out    = f"{MODEL_OUT}{gen}"
    result_dir   = f"{RESULT_OUT}{gen}"
    fim_json     = os.path.join(result_dir, "perblock_fim.json")

    # Check data exists
    if not os.path.exists(data_path):
        print(f"  ✗ Treatment data not found at {data_path}")
        print(f"    Run the treatment experiment first.")
        break

    # Source model: Gen0 base for Gen1, previous SGD model for Gen2+
    source_model = MODEL_ID if gen == 1 else f"{MODEL_OUT}{gen-1}"

    # Train if needed
    if os.path.exists(model_out) and os.path.exists(os.path.join(model_out, "config.json")):
        print(f"  ✓ Model already exists at {model_out}, skipping training")
    else:
        train_one_generation(gen, source_model, data_path, model_out)

    # FIM if needed
    if os.path.exists(fim_json):
        print(f"  ✓ FIM already exists at {fim_json}, skipping")
    else:
        run_fim(model_out, result_dir)

    # Load FIM
    fim_n = load_fim_results(result_dir)
    if fim_n is None:
        print(f"  ✗ FIM results not found after analysis, skipping Gen {gen}")
        continue

    # Compute drift vs Gen0
    print(f"\n  [DRIFT] Computing relative Frobenius drift vs Gen0...")
    drift_n = compute_drift(MODEL_ID, model_out)

    # Spearman correlation
    rho, p, n_blocks = compute_spearman(fim_n, drift_n)
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
    print(f"\n  Gen {gen}: rho={rho:+.4f}{sig}  p={p:.4f}  n={n_blocks} blocks")

    summary[gen] = {
        "spearman_rho": rho,
        "p_value": p,
        "n_blocks": n_blocks,
        "significant": p < 0.05
    }

# ── SAVE AND PRINT SUMMARY ────────────────────────────────────────────────────
with open(SUMMARY_OUT, "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*70)
print("SGD ABLATION — SPEARMAN FIM-DRIFT CORRELATION SUMMARY")
print("="*70)
print(f"\n{'Gen':<6} {'Rho':>10} {'p-value':>12} {'Sig':>5} {'n_blocks':>10}")
print("-"*45)
for gen, r in summary.items():
    sig = "**" if r['p_value'] < 0.01 else ("*" if r['p_value'] < 0.05 else "")
    print(f"  {gen:<4} {r['spearman_rho']:>+10.4f} {r['p_value']:>12.4f} "
          f"{sig:>5} {r['n_blocks']:>10}")

print(f"""
INTERPRETATION GUIDE:
  Near zero or weakly positive rho: v_t hypothesis confirmed.
    High-FIM blocks drift proportionally to gradient magnitude
    when the adaptive suppression mechanism is absent.

  Still negative: unexpected. Would suggest the negative correlation
    has a component independent of v_t (e.g. gradient direction effects).
    Would require revising the mechanistic account.

  Strongly positive: also interesting — SGD without weight decay might
    amplify high-FIM block drift more than standard AdamW parallel models,
    since there is no preconditioner at all.

Compare against treatment condition (AdamW beta2=0.999):
  SmolLM Trt G1-G5: -0.40*, -0.43*, -0.44*, -0.52**, -0.49**

Results saved to: {SUMMARY_OUT}
""")