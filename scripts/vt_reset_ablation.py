"""
v_t Reset Between Generations Ablation
=======================================
Tests whether cross-generation v_t accumulation is necessary for the
Sensitivity-Drift Paradox, or whether within-generation accumulation alone
is sufficient.

Current treatment: optimizer state (v_t, m_t) carries over between generations.
This condition: fresh AdamW initialised at the start of each generation.

Theoretical prediction:
  If cross-generation v_t is necessary:
    - Correlation weakens significantly (closer to zero or positive at Gen1)
    - Within each generation the correlation builds, then resets
    - Overall correlation across 5 generations much weaker than treatment
    
  If within-generation v_t is sufficient:
    - Correlation similar to treatment (50k samples is enough to build the
      differential within one generation)
    - Collapse rate similar to treatment

This is the most direct mechanistic test available: single line change,
zero compute overhead, directly probes the cross-generation component.

Output:
  models/vt_reset_gen_1-5/
  results/vt_reset_gen_1-5/
  results/vt_reset_summary.json
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
MODEL_ID    = "microsoft/phi-1_5"
BASE_DIR    = r"D:\Thaman\Work\hessian-spectral-analysis"
DATA_PREFIX = os.path.join(BASE_DIR, "data", "phi-1_5_treatment_synthetic_gen_")
MODEL_OUT   = os.path.join(BASE_DIR, "models", "phi_vt_reset_gen_")
RESULT_OUT  = os.path.join(BASE_DIR, "results", "phi_vt_reset_gen_")
SUMMARY_OUT = os.path.join(BASE_DIR, "results", "phi_vt_reset_summary.json")
FIM_SCRIPT  = os.path.join(BASE_DIR, "scripts", "perblock_fim.py")

MAX_LENGTH  = 256
BATCH_SIZE  = 8
LR          = 5e-5
GENERATIONS = 5

# Treatment baseline for comparison
TREATMENT_RHO = {1: 0.19, 2: 0.39, 3: 0.47, 4: 0.65, 5: 0.46}
TREATMENT_PPL = {0: 6.05, 5: 7.84}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def train_one_generation(gen, source_model_path, data_path, out_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  [TRAIN] Gen {gen} | source: {source_model_path}")
    print(f"  [KEY] Fresh AdamW initialised — no v_t carryover from previous generation")

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
        return tokenizer(examples["text"], truncation=True,
                         padding="max_length", max_length=MAX_LENGTH)
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    dataset.set_format("torch")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # KEY DIFFERENCE: Fresh optimizer every generation — no v_t carryover
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=LR, betas=(0.9, 0.999), weight_decay=0.01
    )
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=100, num_training_steps=len(loader)
    )

    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"  v_t-reset Gen {gen}")
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

    print(f"  [TRAIN] avg loss: {total_loss/len(loader):.4f}")
    os.makedirs(out_path, exist_ok=True)
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    del model, optim, sched
    torch.cuda.empty_cache()
    gc.collect()


def run_fim(model_path, result_dir):
    os.makedirs(result_dir, exist_ok=True)
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


def load_fim(result_dir):
    path = os.path.join(result_dir, "perblock_fim.json")
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
                            split=f"validation[:{n_batches * BATCH_SIZE}]")
    def tok(ex):
        return tokenizer(ex["text"], truncation=True,
                         padding="max_length", max_length=MAX_LENGTH)
    dataset = dataset.map(tok, batched=True, remove_columns=["text"])
    dataset.set_format("torch")
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= n_batches: break
        ids    = batch["input_ids"].to(device)
        labels = ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        out    = model(ids, labels=labels)
        total += out.loss.item()
        count += 1

    ppl = float(torch.exp(torch.tensor(total / count)).item())
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return ppl


def compute_drift(model_path_0, model_path_n):
    def load_sd(p):
        m = AutoModelForCausalLM.from_pretrained(
            p, torch_dtype=torch.float32,
            device_map="cpu", low_cpu_mem_usage=True
        )
        sd = {k: v.clone() for k, v in m.state_dict().items()}
        del m; gc.collect()
        return sd

    sd0 = load_sd(model_path_0)
    sdn = load_sd(model_path_n)
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
        sq_d[b] += (pn.float() - p0.float()).norm('fro').item() ** 2
        sq_b[b] += p0.float().norm('fro').item() ** 2
    del sd0, sdn; gc.collect()
    return {b: (sq_d[b]**0.5)/(sq_b[b]**0.5) for b in sq_d if sq_b[b] > 0}


def spearman(fim, drift):
    common = sorted(set(fim) & set(drift))
    if len(common) < 5: return float('nan'), float('nan')
    lf = np.array([np.log10(fim[b]+1e-12) for b in common])
    d  = np.array([drift[b] for b in common])
    r, p = stats.spearmanr(lf, d)
    return float(r), float(p)


# ── MAIN ──────────────────────────────────────────────────────────────────────

print("="*70)
print("v_t RESET BETWEEN GENERATIONS")
print("AdamW reinitialised fresh at start of each generation (no state carryover)")
print("="*70)

ppl_gen0 = evaluate_ppl(MODEL_ID)
print(f"Gen0 PPL: {ppl_gen0:.4f}")

gen0_fim_dir = os.path.join(BASE_DIR, "results", "treatment_gen_0")
if not os.path.exists(os.path.join(gen0_fim_dir, "perblock_fim.json")):
    run_fim(MODEL_ID, gen0_fim_dir)
fim0 = load_fim(gen0_fim_dir)

if os.path.exists(SUMMARY_OUT):
    with open(SUMMARY_OUT) as f:
        summary = json.load(f)
    print(f"Loaded existing results for gens: {list(summary.keys())}")
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
        train_one_generation(gen, src, data_path, model_out)

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
print("v_t RESET — SUMMARY")
print("="*70)
print(f"\n{'Gen':<6} {'Rho':>10} {'Sig':>5} {'PPL':>10} {'PPL%':>8}  Treatment rho")
print("-"*55)
for gen in range(1, 6):
    r = summary.get(str(gen), {})
    if not r: continue
    rho = r['spearman_rho']; p = r['p_value']
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
    t_rho = TREATMENT_RHO.get(gen, float('nan'))
    print(f"  {gen:<4} {rho:>+10.4f}{sig:>3}  {r['perplexity']:>10.2f}  "
          f"{r['ppl_pct_change']:>+7.1f}%  ({t_rho:+.2f})")

print(f"\nGen0 PPL: {ppl_gen0:.4f}  |  Treatment Gen5 PPL: 7.84 (+29.5%)")
print(f"\nInterpretation:")
print(f"  If rho similar to treatment: within-generation v_t is sufficient")
print(f"  If rho weaker/positive: cross-generation carryover is necessary")