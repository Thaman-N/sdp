"""
KL Distillation from Gen0 (Output Distribution Anchoring)
===========================================================
Tests whether anchoring to Gen0's output distribution prevents collapse,
avoiding EWC's structural failure (flat FIM contrast across blocks).

Loss: L_total = L_CE_synthetic + lambda * KL(p_Gen0 || p_current)

Key differences from EWC:
  1. Operates on OUTPUT DISTRIBUTION not weight parameters
  2. No FIM-weighted penalty — uniform across vocabulary
  3. KL gradient is LARGEST where output distributions diverge most
     (which is where collapse is actively happening)
  4. Avoids EWC's amplification of v_t for high-FIM blocks

The KL term flows through ALL blocks (it's a function of final logits),
but its gradient magnitude is determined by how much the outputs have
drifted from Gen0, not by FIM structure. This is a fundamentally different
regularisation pathway from EWC.

Prediction:
  If output distribution anchoring slows collapse:
    -> KL gradient effectively competes with synthetic CE gradient
    -> High lambda should show less perplexity degradation than EWC
    -> FIM-drift correlation may weaken (KL gradient disrupts v_t hierarchy)
  
  If it fails like EWC:
    -> The KL gradient also feeds v_t (it still generates gradients)
    -> Loop is unbreakable by any gradient-based intervention
    -> This would be a strong theoretical conclusion

Lambda values: 0.1, 0.5, 1.0 (lighter than EWC because KL is already
on the scale of the CE loss, unlike EWC's parameter-space penalty)

Output:
  models/kl_distill_lambda{l}_gen_1-5/
  results/kl_distill_lambda{l}_gen_1-5/
  results/kl_distill_summary.json
"""

import os, gc, json, re, subprocess
import torch
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
SUMMARY_OUT = os.path.join(BASE_DIR, "results", "kl_distill_summary.json")
FIM_SCRIPT  = os.path.join(BASE_DIR, "scripts", "perblock_fim.py")

MAX_LENGTH  = 256
BATCH_SIZE  = 8
LR          = 5e-5
GENERATIONS = 5

# Lambda values to test — lighter than EWC since KL is already normalised
KL_LAMBDAS  = [0.1, 0.5, 1.0]

TREATMENT_RHO = {1: -0.40, 2: -0.43, 3: -0.44, 4: -0.52, 5: -0.49}
TREATMENT_PPL_G5 = 7.84

# ── HELPERS ───────────────────────────────────────────────────────────────────

def model_dir(lam, gen):
    tag = str(lam).replace('.', '_')
    return os.path.join(BASE_DIR, "models", f"kl_distill_lambda{tag}_gen_{gen}")

def result_dir(lam, gen):
    tag = str(lam).replace('.', '_')
    return os.path.join(BASE_DIR, "results", f"kl_distill_lambda{tag}_gen_{gen}")

def lam_tag(lam):
    return str(lam).replace('.', '_')


def train_one_generation(gen, source_model_path, data_path, out_path,
                         gen0_model, kl_lambda, device):
    print(f"\n  [TRAIN] Gen {gen} | KL lambda={kl_lambda}")

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

    optim = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.01
    )
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=100, num_training_steps=len(loader)
    )

    model.train()
    gen0_model.eval()
    total_ce, total_kl, total_loss_val = 0.0, 0.0, 0.0

    pbar = tqdm(loader, desc=f"  KL-distill lambda={kl_lambda} Gen {gen}")
    for step, batch in enumerate(pbar):
        batch = {k: v.to(device) for k, v in batch.items()}
        ids = batch["input_ids"]

        # Current model forward
        outputs = model(input_ids=ids)
        logits_curr = outputs.logits  # (B, T, V)

        # Standard CE loss on synthetic data
        labels = ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        ce_loss = F.cross_entropy(
            logits_curr[:, :-1].reshape(-1, logits_curr.size(-1)),
            ids[:, 1:].reshape(-1),
            ignore_index=-100
        )

        # KL divergence from Gen0 output distribution
        # KL(p_gen0 || p_current) = sum(p_gen0 * log(p_gen0/p_current))
        with torch.no_grad():
            logits_gen0 = gen0_model(input_ids=ids).logits  # (B, T, V)
            log_p_gen0 = F.log_softmax(logits_gen0[:, :-1].float(), dim=-1)

        log_p_curr = F.log_softmax(logits_curr[:, :-1].float(), dim=-1)

        # Only compute KL on non-padding positions
        valid_mask = (ids[:, 1:] != tokenizer.pad_token_id).float()  # (B, T-1)
        p_gen0 = log_p_gen0.exp()
        kl_per_token = (p_gen0 * (log_p_gen0 - log_p_curr)).sum(dim=-1)  # (B, T-1)
        kl_loss = (kl_per_token * valid_mask).sum() / valid_mask.sum().clamp(min=1)

        # Combined loss
        loss = ce_loss + kl_lambda * kl_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        sched.step()
        optim.zero_grad()

        total_ce += ce_loss.item()
        total_kl += kl_loss.item()
        total_loss_val += loss.item()

        if step % 100 == 0:
            pbar.set_postfix({
                "ce": f"{ce_loss.item():.3f}",
                "kl": f"{kl_loss.item():.3f}",
                "total": f"{loss.item():.3f}"
            })

    n = len(loader)
    print(f"  [TRAIN] avg CE={total_ce/n:.4f}  KL={total_kl/n:.4f}  "
          f"total={total_loss_val/n:.4f}")
    os.makedirs(out_path, exist_ok=True)
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    del model, optim, sched
    torch.cuda.empty_cache()
    gc.collect()


def run_fim(model_path, res_dir):
    os.makedirs(res_dir, exist_ok=True)
    torch.cuda.empty_cache(); gc.collect()
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
def evaluate_ppl(model_path, tokenizer_path, n_batches=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16,
        device_map=device, low_cpu_mem_usage=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
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
print("KL DISTILLATION FROM GEN0")
print(f"L_total = L_CE_synthetic + lambda * KL(p_Gen0 || p_current)")
print(f"Lambda values: {KL_LAMBDAS}")
print("="*70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ppl_gen0 = evaluate_ppl(MODEL_ID, MODEL_ID)
print(f"Gen0 PPL: {ppl_gen0:.4f}")

# Load Gen0 FIM
gen0_fim_dir = os.path.join(BASE_DIR, "results", "treatment_gen_0")
if not os.path.exists(os.path.join(gen0_fim_dir, "perblock_fim.json")):
    run_fim(MODEL_ID, gen0_fim_dir)

# Load Gen0 model as frozen teacher (kept in memory throughout)
print("\nLoading Gen0 model as frozen teacher...")
gen0_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map=device,
    low_cpu_mem_usage=True
)
gen0_model.eval()
for p in gen0_model.parameters():
    p.requires_grad = False
print("Gen0 teacher loaded and frozen.")

if os.path.exists(SUMMARY_OUT):
    with open(SUMMARY_OUT) as f:
        all_results = json.load(f)
else:
    all_results = {}

try:
  for kl_lambda in KL_LAMBDAS:
    ltag = lam_tag(kl_lambda)
    print(f"\n{'='*70}")
    print(f"KL LAMBDA = {kl_lambda}")
    print(f"{'='*70}")

    if ltag not in all_results:
        all_results[ltag] = {}

    for gen in range(1, GENERATIONS + 1):
        if str(gen) in all_results[ltag]:
            print(f"  Gen {gen}: already done, skipping")
            continue

        data_path = f"{DATA_PREFIX}{gen}"
        mdir      = model_dir(kl_lambda, gen)
        rdir      = result_dir(kl_lambda, gen)

        if not os.path.exists(data_path):
            print(f"  ✗ Data missing: {data_path}"); break

        src = MODEL_ID if gen == 1 else model_dir(kl_lambda, gen-1)

        if os.path.exists(os.path.join(mdir, "config.json")):
            print(f"  Gen {gen}: model exists, skipping training")
        else:
            train_one_generation(gen, src, data_path, mdir,
                                 gen0_model, kl_lambda, device)

        if not os.path.exists(os.path.join(rdir, "perblock_fim.json")):
            run_fim(mdir, rdir)

        fim_n   = load_fim(rdir)
        drift_n = compute_drift(MODEL_ID, mdir)
        ppl_n   = evaluate_ppl(mdir, MODEL_ID)
        rho, p  = spearman(fim_n, drift_n)
        sig     = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        pct     = (ppl_n - ppl_gen0) / ppl_gen0 * 100

        print(f"  Gen {gen}: rho={rho:+.4f}{sig}  PPL={ppl_n:.2f} ({pct:+.1f}%)")
        all_results[ltag][str(gen)] = {
            "spearman_rho": rho, "p_value": p,
            "significant": p < 0.05,
            "perplexity": ppl_n, "ppl_pct_change": pct,
            "kl_lambda": kl_lambda
        }
        with open(SUMMARY_OUT, "w") as f:
            json.dump(all_results, f, indent=2)

finally:
    # Cleanup Gen0 teacher regardless of errors
    del gen0_model; torch.cuda.empty_cache(); gc.collect()

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("KL DISTILLATION — GEN5 SUMMARY")
print("="*70)
print(f"\n{'Lambda':<10} {'G5 rho':>10} {'Sig':>5} {'G5 PPL':>10} {'G5 PPL%':>9}")
print("-"*48)
for kl_lambda in KL_LAMBDAS:
    ltag = lam_tag(kl_lambda)
    r = all_results.get(ltag, {}).get("5", {})
    if not r: print(f"  {kl_lambda:<8}  {'—':>10}"); continue
    rho = r['spearman_rho']; p = r['p_value']
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
    print(f"  {kl_lambda:<8}  {rho:>+10.4f}{sig:>3}  "
          f"{r['perplexity']:>10.2f}  {r['ppl_pct_change']:>+8.1f}%")

print(f"\n  Treatment: rho=-0.49**  PPL=7.84 (+29.5%)")
print(f"  EWC best:  rho~-0.49    PPL=7.85 (+29.3%)")
print(f"\nKey question: Does KL distillation break the mechanism where EWC failed?")
print(f"  EWC failed because: flat FIM contrast + penalty feeds v_t")
print(f"  KL avoids both: operates on outputs, not weights")