"""
TCE Gamma Sweep
===============
Sweeps gamma in {0.95, 0.99, 0.999} to find the tradeoff point between
correlation disruption and training quality degradation.

gamma=0.9  (already done) -> catastrophic PPL (120→62), weakened correlation
gamma=0.95 -> moderate masking
gamma=0.99 -> light masking
gamma=0.999 -> minimal masking, near-standard CE

Resume-safe: skips training/FIM if already done for a given gamma.
Each gamma stores models and results in separate directories.

Output:
  models/tce_gamma_{g}_gen_1-5/
  results/tce_gamma_{g}_gen_1-5/perblock_fim.json
  results/tce_sweep_summary.json
"""

import os, gc, json, re, subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
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
FIM_SCRIPT  = os.path.join(BASE_DIR, "scripts", "perblock_fim.py")
SWEEP_OUT   = os.path.join(BASE_DIR, "results", "tce_sweep_summary.json")

MAX_LENGTH  = 256
BATCH_SIZE  = 8
LR          = 5e-5
GENERATIONS = 5

# Gamma values to sweep (0.9 already done, included here for completeness
# but will be skipped via resume logic if already present)
GAMMAS = [0.9, 0.95, 0.99, 0.999]

# Treatment baseline for comparison
TREATMENT_PPL    = {0: 6.05, 1: 6.21, 2: 6.48, 3: 6.79, 4: 7.19, 5: 7.84}
TREATMENT_RHO    = {1: -0.40, 2: -0.43, 3: -0.44, 4: -0.52, 5: -0.49}


# ── TCE LOSS ──────────────────────────────────────────────────────────────────

class TruncatedCrossEntropyLoss(nn.Module):
    def __init__(self, gamma=0.9):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, labels):
        B, T, V = logits.shape
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        flat_logits  = shift_logits.view(-1, V)
        flat_labels  = shift_labels.view(-1)

        per_token_loss = F.cross_entropy(
            flat_logits, flat_labels,
            ignore_index=-100, reduction='none'
        )
        with torch.no_grad():
            probs = torch.exp(-per_token_loss)

        valid_mask      = (flat_labels != -100)
        confidence_mask = (probs < self.gamma)
        final_mask      = valid_mask & confidence_mask

        if final_mask.sum() == 0:
            return F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)

        return per_token_loss[final_mask].mean()


# ── HELPERS ───────────────────────────────────────────────────────────────────

def gamma_tag(gamma):
    """Filesystem-safe gamma tag: 0.999 -> '0_999'"""
    return str(gamma).replace('.', '_')


def model_dir(gamma, gen):
    return os.path.join(BASE_DIR, "models",
                        f"tce_gamma_{gamma_tag(gamma)}_gen_{gen}")


def result_dir(gamma, gen):
    return os.path.join(BASE_DIR, "results",
                        f"tce_gamma_{gamma_tag(gamma)}_gen_{gen}")


def train_one_generation(gen, source_model_path, data_path, out_path, gamma):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  [TRAIN] Gen {gen} gamma={gamma} | source: {source_model_path}")

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

    optim = torch.optim.AdamW(model.parameters(), lr=LR,
                               betas=(0.9, 0.999), weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=100, num_training_steps=len(loader)
    )
    tce = TruncatedCrossEntropyLoss(gamma=gamma)

    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"  TCE gamma={gamma} Gen {gen}")
    for step, batch in enumerate(pbar):
        batch = {k: v.to(device) for k, v in batch.items()}
        ids   = batch['input_ids']
        outputs = model(input_ids=ids)
        labels  = ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        loss = tce(outputs.logits.float(), labels)
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


def run_fim(model_path, res_dir):
    os.makedirs(res_dir, exist_ok=True)
    print(f"  [FIM] {model_path}")
    torch.cuda.empty_cache()
    gc.collect()
    subprocess.run([
        "python", FIM_SCRIPT,
        "--model_path", model_path,
        "--output_dir", res_dir,
        "--disable_flash_attn",
        "--num_batches", "5",
        "--num_eigenvalues", "20",
    ], check=True)


def load_fim(res_dir):
    path = os.path.join(res_dir, "perblock_fim.json")
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
    print(f"  [PPL] {model_path}")
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

    def tok(examples):
        return tokenizer(examples["text"], truncation=True,
                         padding="max_length", max_length=MAX_LENGTH)

    dataset = dataset.map(tok, batched=True, remove_columns=["text"])
    dataset.set_format("torch")
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        ids    = batch["input_ids"].to(device)
        labels = ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        out    = model(ids, labels=labels)
        total += out.loss.item()
        count += 1

    ppl = float(torch.exp(torch.tensor(total / count)).item())
    print(f"  [PPL] {ppl:.4f}")
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
        del m
        gc.collect()
        return sd

    sd0 = load_sd(model_path_0)
    sdn = load_sd(model_path_n)
    sq_drift, sq_base = defaultdict(float), defaultdict(float)

    for name, p0 in sd0.items():
        if name not in sdn:
            continue
        for pat in [r'\.layers\.(\d+)\.', r'\.h\.(\d+)\.', r'\.blocks\.(\d+)\.']:
            m = re.search(pat, name)
            if m:
                break
        else:
            continue
        b = int(m.group(1))
        if p0.ndim < 2:
            continue
        if any(x in name.lower() for x in ['embed', 'lm_head', 'norm', 'ln_']):
            continue
        pn = sdn[name]
        sq_drift[b] += (pn.float() - p0.float()).norm('fro').item() ** 2
        sq_base[b]  += p0.float().norm('fro').item() ** 2

    del sd0, sdn
    gc.collect()
    return {b: (sq_drift[b]**0.5) / (sq_base[b]**0.5)
            for b in sq_drift if sq_base[b] > 0}


def spearman(fim, drift):
    common = sorted(set(fim) & set(drift))
    if len(common) < 5:
        return float('nan'), float('nan')
    lf = np.array([np.log10(fim[b] + 1e-12) for b in common])
    d  = np.array([drift[b] for b in common])
    r, p = stats.spearmanr(lf, d)
    return float(r), float(p)


# ── MAIN ──────────────────────────────────────────────────────────────────────

print("=" * 70)
print("TCE GAMMA SWEEP")
print(f"Gammas: {GAMMAS}")
print("=" * 70)

# Gen0 baseline
ppl_gen0 = evaluate_ppl(MODEL_ID)
print(f"\nGen0 baseline PPL: {ppl_gen0:.4f}")

# Load existing sweep results if any
if os.path.exists(SWEEP_OUT):
    with open(SWEEP_OUT) as f:
        all_results = json.load(f)
    print(f"Loaded existing sweep results for gammas: {list(all_results.keys())}")
else:
    all_results = {}

# Gen0 FIM
gen0_fim_dir = os.path.join(BASE_DIR, "results", "treatment_gen_0")
if not os.path.exists(os.path.join(gen0_fim_dir, "perblock_fim.json")):
    run_fim(MODEL_ID, gen0_fim_dir)
fim0 = load_fim(gen0_fim_dir)
print(f"Gen0 FIM loaded: {len(fim0)} blocks")

for gamma in GAMMAS:
    gtag = gamma_tag(gamma)
    print(f"\n{'='*70}")
    print(f"GAMMA = {gamma}")
    print(f"{'='*70}")

    if gtag not in all_results:
        all_results[gtag] = {}

    # Map gamma=0.9 to existing tce_ablation_gen_N directories
    def _model_dir(gamma, gen):
        if gamma == 0.9:
            return os.path.join(BASE_DIR, "models", f"tce_ablation_gen_{gen}")
        return model_dir(gamma, gen)

    def _result_dir(gamma, gen):
        if gamma == 0.9:
            return os.path.join(BASE_DIR, "results", f"tce_ablation_gen_{gen}")
        return result_dir(gamma, gen)

    for gen in range(1, GENERATIONS + 1):
        if str(gen) in all_results[gtag]:
            print(f"  Gen {gen}: already done, skipping")
            continue

        data_path = f"{DATA_PREFIX}{gen}"
        mdir      = _model_dir(gamma, gen)
        rdir      = _result_dir(gamma, gen)
        fim_json  = os.path.join(rdir, "perblock_fim.json")

        if not os.path.exists(data_path):
            print(f"  ✗ Data missing: {data_path}")
            break

        src = MODEL_ID if gen == 1 else _model_dir(gamma, gen - 1)

        # Train
        if os.path.exists(os.path.join(mdir, "config.json")):
            print(f"  Gen {gen}: model exists, skipping training")
        else:
            train_one_generation(gen, src, data_path, mdir, gamma)

        # FIM
        if os.path.exists(fim_json):
            print(f"  Gen {gen}: FIM exists, skipping")
        else:
            run_fim(mdir, rdir)

        fim_n  = load_fim(rdir)
        drift_n = compute_drift(MODEL_ID, mdir)
        ppl_n   = evaluate_ppl(mdir)
        rho, p  = spearman(fim_n, drift_n)
        sig     = "**" if p < 0.01 else ("*" if p < 0.05 else "")

        pct_ppl = (ppl_n - ppl_gen0) / ppl_gen0 * 100
        print(f"  Gen {gen}: rho={rho:+.4f}{sig}  PPL={ppl_n:.2f} ({pct_ppl:+.1f}%)")

        all_results[gtag][str(gen)] = {
            "spearman_rho": rho,
            "p_value": p,
            "significant": p < 0.05,
            "perplexity": ppl_n,
            "ppl_pct_change": pct_ppl,
        }

        # Save after every generation
        with open(SWEEP_OUT, "w") as f:
            json.dump(all_results, f, indent=2)

# ── FINAL SUMMARY TABLE ───────────────────────────────────────────────────────
print("\n" + "="*70)
print("TCE GAMMA SWEEP — SUMMARY (Gen5)")
print(f"Gen0 baseline PPL: {ppl_gen0:.4f}")
print(f"Treatment Gen5:    PPL=7.84 (+29.5%)  rho=-0.49**")
print("="*70)
print(f"\n{'Gamma':<8} {'G5 rho':>10} {'Sig':>5} {'G5 PPL':>10} {'G5 PPL%':>9}")
print("-"*48)
for gamma in GAMMAS:
    gtag = gamma_tag(gamma)
    r = all_results.get(gtag, {}).get("5", {})
    if not r:
        print(f"  {gamma:<6}  {'—':>10}")
        continue
    rho = r.get("spearman_rho", float('nan'))
    ppl = r.get("perplexity", float('nan'))
    pct = r.get("ppl_pct_change", float('nan'))
    sig = "**" if r.get("p_value", 1) < 0.01 else ("*" if r.get("p_value", 1) < 0.05 else "")
    print(f"  {gamma:<6}  {rho:>+10.4f}{sig:>3}  {ppl:>10.2f}  {pct:>+8.1f}%")

print(f"\n  CE (treatment): rho=-0.49**  PPL=7.84  (+29.5%)")
print(f"\nFull results saved to: {SWEEP_OUT}")