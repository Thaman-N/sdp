"""
AdamW Beta2 Ablation Study (Optimized)
Tests whether the FIM-Drift anticorrelation is specific to AdamW's
second moment accumulation by varying beta2.

Optimizations vs original:
  - Gen1 synthetic data is generated ONCE and shared across all beta2 conditions
    (valid because all three use the same base model as generator)
  - Samples reduced from 50k to 20k (sufficient for correlation signal)
  - Total estimated runtime: ~10-11 hours sequential

beta2 controls how aggressively the second moment (v_t) is accumulated.
  - Low beta2 (0.90):   fast v_t accumulation  -> stronger/faster bottleneck
  - Default  (0.999):   standard behavior
  - High beta2 (0.9999): slow v_t              -> weaker bottleneck

Requires: perblock_fim.py in same directory.
Output:   results/beta2_ablation/
"""

import os
import gc
import json
import re
import subprocess
import random
import torch
import numpy as np
from scipy import stats
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================
MODEL_ID    = "microsoft/phi-1_5"
MAX_LENGTH  = 256
BATCH_SIZE  = 8
GEN_BATCH   = 32
SAMPLES     = 50000          # reduced from 50k — sufficient for signal
GENERATIONS = 5

BETA2_VALUES = [0.90, 0.999, 0.9999]   # drop 0.9999, add 1.00

BASE_RESULTS = "results/beta2_ablation"
BASE_MODELS  = "models/beta2_ablation"
BASE_DATA    = "data/beta2_ablation"

# Shared Gen1 data path — generated once, reused by all beta2 conditions
# SHARED_GEN1_DATA = os.path.join(BASE_DATA, "shared_gen1_data")
SHARED_GEN1_DATA = r"D:\Thaman\Work\hessian-spectral-analysis\data\phi-1_5_treatment_synthetic_gen_1"

os.makedirs(BASE_RESULTS, exist_ok=True)
os.makedirs(BASE_MODELS,  exist_ok=True)
os.makedirs(BASE_DATA,    exist_ok=True)

PROMPTS = [
    "Once upon a time", "The scientist discovered", "In the future",
    "The algorithm optimized", "Deep learning is", "The cat sat on",
]

# ============================================================
# HELPERS
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
        return block_idx, 'attn'
    elif any(x in name_lower for x in
             ['mlp', 'fc', 'up_proj', 'down_proj', 'gate_proj', 'c_fc']):
        return block_idx, 'mlp'
    return block_idx, 'other'


def compute_drift(base_path, target_path):
    """Relative Frobenius norm drift per block, always vs base MODEL_ID."""
    print(f"    Computing drift: {base_path} -> {target_path}")
    m0 = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.float16, device_map="cpu")
    mx = AutoModelForCausalLM.from_pretrained(
        target_path, torch_dtype=torch.float16, device_map="cpu")
    d0 = dict(m0.named_parameters())
    dx = dict(mx.named_parameters())

    accum = defaultdict(lambda: {
        'attn_diff_sq': 0., 'attn_base_sq': 0.,
        'mlp_diff_sq':  0., 'mlp_base_sq':  0.,
    })
    with torch.no_grad():
        for name, p0 in d0.items():
            if name not in dx:
                continue
            block_idx, component = get_layer_info(name)
            if block_idx is None or component == 'other':
                continue
            p0f = p0.float()
            pxf = dx[name].float()
            diff_sq = torch.sum((pxf - p0f) ** 2).item()
            base_sq = torch.sum(p0f ** 2).item()
            if component == 'attn':
                accum[block_idx]['attn_diff_sq'] += diff_sq
                accum[block_idx]['attn_base_sq'] += base_sq
            elif component == 'mlp':
                accum[block_idx]['mlp_diff_sq'] += diff_sq
                accum[block_idx]['mlp_base_sq'] += base_sq
    del m0, mx
    gc.collect()

    results = {}
    for b in sorted(accum.keys()):
        s = accum[b]
        attn = (s['attn_diff_sq']**0.5 / s['attn_base_sq']**0.5
                if s['attn_base_sq'] > 0 else 0.)
        mlp  = (s['mlp_diff_sq'] **0.5 / s['mlp_base_sq'] **0.5
                if s['mlp_base_sq']  > 0 else 0.)
        results[b] = {'attn_relative_drift': attn, 'mlp_relative_drift': mlp}
    return results


def parse_fim_json(path):
    """
    Parse perblock_fim.json from perblock_fim.py.
    Output format: {"blocks": [{"block_idx": 0, "attention": {"top": X, ...},
                                                "mlp":       {"top": Y, ...}}, ...]}
    FIM value per block = attention['top'] + mlp['top'] (top eigenvalue each).
    """
    with open(path) as f:
        data = json.load(f)
    blocks = {}
    if 'blocks' not in data:
        return blocks
    for entry in data['blocks']:
        b = int(entry['block_idx'])
        attn_info = entry.get('attention') or {}
        mlp_info  = entry.get('mlp')       or {}
        # skip blocks where FIM computation errored
        if 'error' in attn_info or 'error' in mlp_info:
            continue
        attn_top = attn_info.get('top', 0.0)
        mlp_top  = mlp_info.get('top',  0.0)
        blocks[b] = attn_top + mlp_top
    return blocks


def spearman_corr(fim_d, drift_d):
    common = sorted(set(fim_d.keys()) & set(drift_d.keys()))
    if len(common) < 4:
        return None, None
    x = [fim_d[b] for b in common]
    y = [(drift_d[b]['attn_relative_drift'] +
          drift_d[b]['mlp_relative_drift']) / 2
         for b in common]
    r, p = stats.spearmanr(x, y)
    return round(r, 4), round(p, 4)


def generate_data(source_path, data_path, n_samples):
    """Generate synthetic data from source_path, save to data_path."""
    if os.path.exists(data_path):
        print(f"    Data exists at {data_path}, skipping.")
        return
    print(f"    Generating {n_samples} samples from {source_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen_model = AutoModelForCausalLM.from_pretrained(
        source_path, torch_dtype=torch.float16,
        attn_implementation="eager").to(device)
    gen_tok = AutoTokenizer.from_pretrained(source_path)
    gen_tok.padding_side = "left"
    if gen_tok.pad_token is None:
        gen_tok.pad_token = gen_tok.eos_token

    stories = []
    gen_model.eval()
    with torch.no_grad():
        pbar = tqdm(total=n_samples, desc="Generating")
        while len(stories) < n_samples:
            batch_prompts = [random.choice(PROMPTS) for _ in range(GEN_BATCH)]
            inputs = gen_tok(batch_prompts, return_tensors="pt",
                             padding=True).to(device)
            try:
                outputs = gen_model.generate(
                    **inputs, max_new_tokens=200, do_sample=True,
                    temperature=0.8, top_k=50,
                    pad_token_id=gen_tok.pad_token_id,
                    eos_token_id=gen_tok.eos_token_id,
                )
                decoded = gen_tok.batch_decode(outputs, skip_special_tokens=True)
                stories.extend(decoded)
                pbar.update(len(decoded))
            except Exception as e:
                print(f"Skipping batch due to error: {e}")
                torch.cuda.empty_cache()
                continue
            decoded = gen_tok.batch_decode(outputs, skip_special_tokens=True)
            stories.extend(decoded)
            pbar.update(len(decoded))
        pbar.close()

    os.makedirs(data_path, exist_ok=True)
    Dataset.from_dict({"text": stories[:n_samples]}).save_to_disk(data_path)
    print(f"    Saved {n_samples} samples to {data_path}")
    del gen_model
    gc.collect()
    torch.cuda.empty_cache()


def train_one_generation(data_path, save_path, beta2, source_model_path=None):
    if source_model_path is None:
        source_model_path = MODEL_ID

    if os.path.exists(save_path) and os.path.exists(
            os.path.join(save_path, 'config.json')):
        print(f"    Model exists at {save_path}, skipping training.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Training beta2={beta2} from {source_model_path} -> {save_path}")

    model = AutoModelForCausalLM.from_pretrained(
        source_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported()
                    else torch.float16,
        attn_implementation="eager",
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    dataset = load_from_disk(data_path)

    def tokenize(examples):
        return tokenizer(
            examples["text"], truncation=True,
            padding="max_length", max_length=MAX_LENGTH,
            return_tensors="pt",
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format("torch")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # KEY: only beta2 varies, all other hyperparameters identical
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        betas=(0.9, beta2),
        weight_decay=0.01,
        eps=1e-8,
    )
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=100,
        num_training_steps=len(loader),
    )

    model.train()
    pbar = tqdm(loader, desc=f"Training beta2={beta2}")
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch, labels=batch["input_ids"]).loss
        loss.backward()
        optim.step()
        sched.step()
        optim.zero_grad()
        pbar.set_postfix({"loss": f"{loss.item():.3f}"})

    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    del model, optim, sched
    gc.collect()
    torch.cuda.empty_cache()


def run_fim(model_path, result_dir, fim_script):
    fim_out = os.path.join(result_dir, "perblock_fim.json")
    if os.path.exists(fim_out):
        print(f"    FIM exists, skipping.")
        return fim_out
    print(f"    Running FIM analysis on {model_path}...")
    try:
        subprocess.run([
            "python", fim_script,
            "--model_path", model_path,
            "--output_dir", result_dir,
            "--disable_flash_attn",
            "--num_batches", "5",
            "--num_eigenvalues", "20",
        ], check=True)
        return fim_out
    except subprocess.CalledProcessError as e:
        print(f"    FIM FAILED: {e}")
        return None


# ============================================================
# MAIN
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Samples per generation: {SAMPLES}")
print(f"Beta2 values: {BETA2_VALUES}")
print(f"Generations: {GENERATIONS}")

fim_script = "scripts/perblock_fim.py"
if not os.path.exists(fim_script):
    raise FileNotFoundError(
        "perblock_fim.py not found in current directory."
    )

summary = {}  # beta2 -> gen -> {rho, p}

# ============================================================
# STEP 0: Generate Gen1 data ONCE from base model (shared)
# ============================================================
print(f"\n{'#'*65}")
print(f"STEP 0: Generating shared Gen1 data from base model")
print(f"This is done once and reused by all beta2 conditions.")
print(f"{'#'*65}")
generate_data(MODEL_ID, SHARED_GEN1_DATA, SAMPLES)

# ============================================================
# MAIN LOOP
# ============================================================
for beta2 in BETA2_VALUES:
    b2_tag = str(beta2).replace('.', '_')
    print(f"\n{'#'*65}")
    print(f"BETA2 = {beta2}")
    print(f"{'#'*65}")
    summary[beta2] = {}

    prev_model_path = MODEL_ID  # always drift vs base

    for gen in range(1, GENERATIONS + 1):
        print(f"\n--- Generation {gen} ---")

        model_save = os.path.join(
            BASE_MODELS,  f"pythia_beta2_{b2_tag}_gen{gen}")
        result_dir = os.path.join(
            BASE_RESULTS, f"pythia_beta2_{b2_tag}_gen{gen}")
        os.makedirs(result_dir, exist_ok=True)

        # Data: Gen1 is shared; Gen2 is generated from the Gen1 model
        # of THIS beta2 condition (different models -> different data)
        if gen == 1:
            data_path = os.path.join(
                r"D:\Thaman\Work\hessian-spectral-analysis\data",
                "pythia-1.4b_treatment_synthetic_gen_1"
            )
            print(f"    Using shared Gen1 treatment data: {data_path}")
        else:
            data_path = os.path.join(
                BASE_DATA, f"pythia_beta2_{b2_tag}_gen{gen}"
            )
            gen_source = os.path.join(
                BASE_MODELS, f"pythia_beta2_{b2_tag}_gen{gen-1}"
            )
            generate_data(gen_source, data_path, SAMPLES)

        # Train
        source = MODEL_ID if gen == 1 else os.path.join(
            BASE_MODELS, f"pythia_beta2_{b2_tag}_gen{gen-1}")
        train_one_generation(data_path, model_save, beta2, source_model_path=source)

        # FIM
        fim_out = run_fim(model_save, result_dir, fim_script)

        # Drift — always vs base MODEL_ID so comparisons are fair
        drift_out = os.path.join(result_dir, "drift.json")
        if not os.path.exists(drift_out):
            drift = compute_drift(MODEL_ID, model_save)
            with open(drift_out, 'w') as f:
                json.dump(
                    {"blocks": {str(k): v for k, v in drift.items()}},
                    f, indent=2,
                )
        else:
            print(f"    Drift exists, loading.")
            with open(drift_out) as f:
                drift = {int(k): v
                         for k, v in json.load(f)['blocks'].items()}

        # Correlation
        if fim_out and os.path.exists(fim_out):
            fim_d = parse_fim_json(fim_out)
            r, p = spearman_corr(fim_d, drift)
            summary[beta2][gen] = {'rho': r, 'p': p}
            sig = "* SIGNIFICANT" if p and p < 0.05 else "  not significant"
            print(f"    Spearman rho={r}, p={p}  {sig}")
        else:
            summary[beta2][gen] = {'rho': None, 'p': None}

# ============================================================
# RESULTS TABLE
# ============================================================
print("\n\n" + "="*65)
print("BETA2 ABLATION RESULTS")
print("Drift always measured vs base model (Gen0)")
print("="*65)
print(f"\n{'beta2':<10} {'Gen1_rho':>10} {'Gen1_p':>10} "
      f"{'Gen2_rho':>10} {'Gen2_p':>10}")
print("-"*55)
for beta2 in BETA2_VALUES:
    g1 = summary.get(beta2, {}).get(1, {})
    g2 = summary.get(beta2, {}).get(2, {})
    r1 = g1.get('rho', 'N/A')
    p1 = g1.get('p',   'N/A')
    r2 = g2.get('rho', 'N/A')
    p2 = g2.get('p',   'N/A')
    print(f"{beta2:<10} {str(r1):>10} {str(p1):>10} "
          f"{str(r2):>10} {str(p2):>10}")

# Save
out_path = os.path.join(BASE_RESULTS, "beta2_summary.json")
with open(out_path, 'w') as f:
    json.dump({str(k): v for k, v in summary.items()}, f, indent=2)
print(f"\nSaved summary: {out_path}")

# ============================================================
# INTERPRETATION GUIDE
# ============================================================
print("""
INTERPRETATION GUIDE:
--------------------------------------------------------------
  If rho stays ~-0.5 across all beta2 values:
    -> Correlation is NOT driven by v_t accumulation rate.
       The mechanism is elsewhere (e.g. gradient geometry,
       data distribution shift). Weaker mechanistic claim.

  If rho weakens (toward 0) as beta2 increases:
    -> Slower v_t accumulation reduces the bottleneck.
       Supports the AdamW second-moment claim. Moderate evidence.

  If rho strengthens (more negative) with low beta2:
    -> Faster v_t accumulation tightens the bottleneck faster.
       STRONGEST evidence for the paper's core mechanistic claim.

  If rho flips positive with high beta2:
    -> Extreme case: disabling effective v_t reverses the paradox.
       This would be a very strong mechanistic result.
--------------------------------------------------------------
""")