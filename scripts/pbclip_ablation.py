"""
Per-Block FIM-Inverse Gradient Clipping
========================================
Tests whether gradient history accumulation (v_t) specifically is necessary
for the Sensitivity-Drift Paradox, vs current-step magnitude scaling being
sufficient.

Standard global gradient clipping: clips all parameters uniformly.
This condition: clip norm threshold for each block is inversely proportional
to its Gen0 FIM value. High-FIM blocks get lower clip threshold (their
large gradients are clipped more aggressively), low-FIM blocks get higher
threshold (their small gradients are less constrained).

This mimics what v_t does (suppressing high-FIM block updates) but WITHOUT
accumulating gradient history. It applies the same constraint at every step
regardless of past gradients.

Theoretical distinction:
  If this produces same correlation weakening as v_t reset:
    -> Mechanism is about current-step magnitude scaling, not history
  If correlation is still strongly negative (similar to treatment):
    -> Accumulated history (v_t) specifically is what drives the mechanism
    -> v_t's memory effect is irreplaceable — this is a strong mechanistic claim

Clip norm for block b: clip_norm_b = base_clip * (mean_FIM / FIM_b)
So high-FIM blocks are clipped more, low-FIM blocks less.

Output:
  models/pbclip_gen_1-5/
  results/pbclip_gen_1-5/
  results/pbclip_summary.json
"""

import os, gc, json, re, subprocess
import torch
import numpy as np
from tqdm import tqdm
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_from_disk, load_dataset
from torch.utils.data import DataLoader
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_ID    = "HuggingFaceTB/SmolLM2-135M"
BASE_DIR    = r"D:\Thaman\Work\hessian-spectral-analysis"
DATA_PREFIX = os.path.join(BASE_DIR, "data", "treatment_synthetic_gen_")
MODEL_OUT   = os.path.join(BASE_DIR, "models", "pbclip_gen_")
RESULT_OUT  = os.path.join(BASE_DIR, "results", "pbclip_gen_")
SUMMARY_OUT = os.path.join(BASE_DIR, "results", "pbclip_summary.json")
FIM_SCRIPT  = os.path.join(BASE_DIR, "scripts", "perblock_fim.py")

MAX_LENGTH  = 256
BATCH_SIZE  = 8
LR          = 5e-5
BASE_CLIP   = 1.0    # Base gradient clip norm
GENERATIONS = 5

TREATMENT_RHO = {1: -0.40, 2: -0.43, 3: -0.44, 4: -0.52, 5: -0.49}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_fim_for_clipping(result_dir):
    """Load FIM values for computing per-block clip norms."""
    path = os.path.join(result_dir, "perblock_fim.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    out = {}
    for i, bd in enumerate(data.get("blocks", [])):
        if not isinstance(bd, dict): continue
        b    = bd.get("block_idx", i)
        attn = bd.get("attention", {})
        mlp  = bd.get("mlp", {})
        at   = float(attn.get("top", 0)) if isinstance(attn, dict) and "error" not in attn else 0.0
        mt   = float(mlp.get("top",  0)) if isinstance(mlp,  dict) and "error" not in mlp  else 0.0
        if at > 0 or mt > 0:
            out[b] = at + mt
    return out


def get_block_param_groups(model, fim_dict, base_clip=1.0):
    """
    Build per-block clip norms: clip_b = base_clip * mean_FIM / FIM_b
    High FIM -> lower clip norm (more aggressive clipping)
    Low FIM  -> higher clip norm (less constrained)
    """
    fim_values = np.array(list(fim_dict.values()))
    mean_fim = np.mean(fim_values)

    block_clip_norms = {}
    for b, fim_b in fim_dict.items():
        if fim_b > 0:
            block_clip_norms[b] = base_clip * mean_fim / fim_b
        else:
            block_clip_norms[b] = base_clip

    # Map parameter names to block indices
    param_to_block = {}
    for name, _ in model.named_parameters():
        for pat in [r'\.layers\.(\d+)\.', r'\.h\.(\d+)\.', r'\.blocks\.(\d+)\.']:
            m = re.search(pat, name)
            if m:
                param_to_block[name] = int(m.group(1))
                break

    return param_to_block, block_clip_norms


def clip_gradients_per_block(model, param_to_block, block_clip_norms, base_clip):
    """Apply per-block gradient clipping."""
    # Group parameters by block
    block_params = defaultdict(list)
    other_params = []
    for name, param in model.named_parameters():
        if param.grad is None: continue
        if name in param_to_block:
            block_params[param_to_block[name]].append(param)
        else:
            other_params.append(param)

    total_norm = 0.0
    # Clip each block's parameters
    for b, params in block_params.items():
        clip_norm = block_clip_norms.get(b, base_clip)
        n = torch.nn.utils.clip_grad_norm_(params, max_norm=clip_norm)
        total_norm += n.item() ** 2

    # Clip non-block parameters with base clip
    if other_params:
        torch.nn.utils.clip_grad_norm_(other_params, max_norm=base_clip)

    return total_norm ** 0.5


def train_one_generation(gen, source_model_path, data_path, out_path,
                         fim_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  [TRAIN] Gen {gen} | Per-block FIM-inverse gradient clipping")

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

    # Build per-block clip norms from Gen0 FIM
    param_to_block, block_clip_norms = get_block_param_groups(
        model, fim_dict, BASE_CLIP
    )

    # Log clip norms for first 5 and last 5 blocks
    sorted_blocks = sorted(block_clip_norms.keys())
    print(f"  [CLIP] Sample clip norms (high FIM = low clip):")
    for b in sorted_blocks[:3] + sorted_blocks[-3:]:
        print(f"    Block {b:2d}: FIM={fim_dict.get(b,0):.1f}  clip_norm={block_clip_norms[b]:.4f}")

    dataset = load_from_disk(data_path)
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True,
                         padding="max_length", max_length=MAX_LENGTH)
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    dataset.set_format("torch")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Standard AdamW - optimizer state carries over (same as treatment)
    # The ONLY difference is gradient clipping, not v_t accumulation
    optim = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.01
    )
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=100, num_training_steps=len(loader)
    )

    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"  PBClip Gen {gen}")
    for step, batch in enumerate(pbar):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        loss.backward()

        # Per-block FIM-inverse gradient clipping (instead of global clip)
        clip_gradients_per_block(model, param_to_block, block_clip_norms, BASE_CLIP)

        optim.step()
        sched.step()
        optim.zero_grad()
        total_loss += loss.item()
        if step % 100 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    print(f"  [TRAIN] avg loss: {total_loss/len(loader):.4f}")
    os.makedirs(out_path, exist_ok=True)
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    del model, optim, sched
    torch.cuda.empty_cache()
    gc.collect()


def run_fim(model_path, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    torch.cuda.empty_cache(); gc.collect()
    subprocess.run([
        "python", FIM_SCRIPT,
        "--model_path", model_path,
        "--output_dir", result_dir,
        "--disable_flash_attn",
        "--num_batches", "5",
        "--num_eigenvalues", "20",
    ], check=True)


def load_fim(result_dir):
    path = os.path.join(result_dir, "perblock_fim.json")
    if not os.path.exists(path): return None
    with open(path) as f: data = json.load(f)
    out = {}
    for i, bd in enumerate(data.get("blocks", [])):
        if not isinstance(bd, dict): continue
        b = bd.get("block_idx", i)
        attn = bd.get("attention", {}); mlp = bd.get("mlp", {})
        at = float(attn.get("top", 0)) if isinstance(attn, dict) and "error" not in attn else 0.0
        mt = float(mlp.get("top",  0)) if isinstance(mlp,  dict) and "error" not in mlp  else 0.0
        if at > 0 or mt > 0: out[b] = at + mt
    return out


@torch.no_grad()
def evaluate_ppl(model_path, n_batches=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16,
        device_map=device, low_cpu_mem_usage=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("roneneldan/TinyStories",
                            split=f"validation[:{n_batches*BATCH_SIZE}]")
    def tok(ex):
        return tokenizer(ex["text"], truncation=True,
                         padding="max_length", max_length=MAX_LENGTH)
    dataset = dataset.map(tok, batched=True, remove_columns=["text"])
    dataset.set_format("torch")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= n_batches: break
        ids = batch["input_ids"].to(device)
        labels = ids.clone(); labels[labels == tokenizer.pad_token_id] = -100
        total += model(ids, labels=labels).loss.item(); count += 1
    ppl = float(torch.exp(torch.tensor(total/count)).item())
    del model; torch.cuda.empty_cache(); gc.collect()
    return ppl


def compute_drift(model_path_0, model_path_n):
    def load_sd(p):
        m = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float32,
                device_map="cpu", low_cpu_mem_usage=True)
        sd = {k: v.clone() for k, v in m.state_dict().items()}
        del m; gc.collect(); return sd
    sd0 = load_sd(model_path_0); sdn = load_sd(model_path_n)
    sq_d, sq_b = defaultdict(float), defaultdict(float)
    for name, p0 in sd0.items():
        if name not in sdn: continue
        for pat in [r'\.layers\.(\d+)\.', r'\.h\.(\d+)\.', r'\.blocks\.(\d+)\.']:
            m = re.search(pat, name)
            if m: break
        else: continue
        b = int(m.group(1))
        if p0.ndim < 2: continue
        if any(x in name.lower() for x in ['embed','lm_head','norm','ln_']): continue
        pn = sdn[name]
        sq_d[b] += (pn.float()-p0.float()).norm('fro').item()**2
        sq_b[b] += p0.float().norm('fro').item()**2
    del sd0, sdn; gc.collect()
    return {b: (sq_d[b]**0.5)/(sq_b[b]**0.5) for b in sq_d if sq_b[b]>0}


def spearman(fim, drift):
    common = sorted(set(fim)&set(drift))
    if len(common) < 5: return float('nan'), float('nan')
    r, p = stats.spearmanr(
        [np.log10(fim[b]+1e-12) for b in common],
        [drift[b] for b in common]
    )
    return float(r), float(p)


# ── MAIN ──────────────────────────────────────────────────────────────────────

print("="*70)
print("PER-BLOCK FIM-INVERSE GRADIENT CLIPPING")
print(f"Base clip norm: {BASE_CLIP}")
print("High-FIM blocks clipped more aggressively (inversely proportional)")
print("="*70)

# Load Gen0 FIM for computing clip norms
gen0_fim_dir = os.path.join(BASE_DIR, "results", "treatment_gen_0")
if not os.path.exists(os.path.join(gen0_fim_dir, "perblock_fim.json")):
    run_fim(MODEL_ID, gen0_fim_dir)
fim_gen0 = load_fim_for_clipping(gen0_fim_dir)
print(f"Gen0 FIM loaded: {len(fim_gen0)} blocks")
fim_values = list(fim_gen0.values())
print(f"FIM range: {min(fim_values):.1f} to {max(fim_values):.1f} "
      f"(ratio {max(fim_values)/min(fim_values):.1f}x)")

ppl_gen0 = evaluate_ppl(MODEL_ID)
print(f"Gen0 PPL: {ppl_gen0:.4f}")

if os.path.exists(SUMMARY_OUT):
    with open(SUMMARY_OUT) as f:
        summary = json.load(f)
else:
    summary = {}

for gen in range(1, GENERATIONS + 1):
    if str(gen) in summary:
        print(f"\n  Gen {gen}: already done, skipping")
        continue

    data_path  = f"{DATA_PREFIX}{gen}"
    model_out  = f"{MODEL_OUT}{gen}"
    result_dir = f"{RESULT_OUT}{gen}"

    if not os.path.exists(data_path):
        print(f"  ✗ Data missing: {data_path}")
        break

    src = MODEL_ID if gen == 1 else f"{MODEL_OUT}{gen-1}"

    if os.path.exists(os.path.join(model_out, "config.json")):
        print(f"\n  Gen {gen}: model exists, skipping training")
    else:
        train_one_generation(gen, src, data_path, model_out, fim_gen0)

    if not os.path.exists(os.path.join(result_dir, "perblock_fim.json")):
        run_fim(model_out, result_dir)

    fim_n   = load_fim(result_dir)
    drift_n = compute_drift(MODEL_ID, model_out)
    ppl_n   = evaluate_ppl(model_out)
    rho, p  = spearman(fim_n, drift_n)
    sig     = "**" if p < 0.01 else ("*" if p < 0.05 else "")
    pct     = (ppl_n - ppl_gen0) / ppl_gen0 * 100

    print(f"\n  Gen {gen}: rho={rho:+.4f}{sig}  PPL={ppl_n:.2f} ({pct:+.1f}%)")
    summary[str(gen)] = {
        "spearman_rho": rho, "p_value": p,
        "significant": p < 0.05,
        "perplexity": ppl_n, "ppl_pct_change": pct
    }
    with open(SUMMARY_OUT, "w") as f:
        json.dump(summary, f, indent=2)

print("\n" + "="*70)
print("PER-BLOCK GRADIENT CLIPPING — SUMMARY")
print("="*70)
print(f"\n{'Gen':<6} {'Rho':>10} {'Sig':>5} {'PPL':>10} {'PPL%':>8}  Treatment")
print("-"*55)
for gen in range(1, 6):
    r = summary.get(str(gen), {})
    if not r: continue
    rho = r['spearman_rho']; p = r['p_value']
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
    t_rho = TREATMENT_RHO.get(gen, float('nan'))
    print(f"  {gen:<4} {rho:>+10.4f}{sig:>3}  {r['perplexity']:>10.2f}  "
          f"{r['ppl_pct_change']:>+7.1f}%  ({t_rho:+.2f})")

print(f"\nGen0 PPL: {ppl_gen0:.4f}")
print("""
Interpretation:
  rho similar to treatment (-0.40 to -0.54):
    -> Current-step clipping does NOT replicate v_t effect
    -> Gradient history accumulation is specifically what drives the mechanism
    
  rho weaker/near-zero:
    -> Current-step magnitude scaling is sufficient
    -> v_t's memory effect is not uniquely necessary
    
  rho positive (like SGD):
    -> Clipping high-FIM blocks too aggressively removes the natural
       tendency for those blocks to stabilise via v_t, causing runaway
""")