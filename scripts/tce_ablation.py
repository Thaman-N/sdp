"""
Truncated Cross-Entropy (TCE) Intervention
===========================================
Tests whether a confidence-aware loss function disrupts the v_t feedback
loop identified in the Sensitivity-Drift Paradox.

Hypothesis:
  High-confidence tokens generate large, consistent gradient magnitudes
  across training steps — these are exactly the tokens that most aggressively
  build v_t in high-FIM blocks. TCE selectively removes them from the loss,
  potentially reducing differential v_t accumulation between blocks and
  weakening the negative FIM-drift correlation.

  Three possible outcomes:
  1. TCE disrupts the negative FIM-drift correlation AND reduces perplexity
     degradation → mechanistic confirmation: high-confidence gradients drive
     the v_t feedback loop.
  2. TCE preserves the correlation but slows perplexity degradation → TCE
     delays collapse through a different mechanism (diversity preservation)
     without breaking the v_t dynamic.
  3. TCE makes things worse → overconfident tokens are actually informative;
     removing them destabilises training.

TCE formulation (ForTIFAI, Shabgahi et al. 2025, arXiv:2509.08972):
  Standard CE: L = -sum(y * log(p))
  TCE:         L = -sum(y * log(p) * [p < gamma])
  where gamma in (0,1) is a confidence threshold. Tokens where the model's
  predicted probability exceeds gamma are masked from the loss.
  gamma=1.0 recovers standard CE. gamma=0.9 is a reasonable starting point.

Reuses existing treatment synthetic data. No new data generation.

Output:
  models/tce_ablation_gen_1-5/
  results/tce_ablation_gen_1-5/perblock_fim.json
  results/tce_ablation_summary.json
"""

import os
import gc
import json
import re
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_from_disk
from torch.utils.data import DataLoader
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_ID    = "HuggingFaceTB/SmolLM2-135M"
BASE_DIR    = r"D:\Thaman\Work\hessian-spectral-analysis"
DATA_PREFIX = os.path.join(BASE_DIR, "data", "treatment_synthetic_gen_")
MODEL_OUT   = os.path.join(BASE_DIR, "models", "tce_ablation_gen_")
RESULT_OUT  = os.path.join(BASE_DIR, "results", "tce_ablation_gen_")
SUMMARY_OUT = os.path.join(BASE_DIR, "results", "tce_ablation_summary.json")
FIM_SCRIPT  = os.path.join(BASE_DIR, "scripts", "perblock_fim.py")

MAX_LENGTH  = 256
BATCH_SIZE  = 8
LR          = 5e-5          # Same as treatment
GAMMA       = 0.9           # TCE confidence threshold (tokens with p > GAMMA masked)
GENERATIONS = 5


# ── TRUNCATED CROSS ENTROPY ───────────────────────────────────────────────────

class TruncatedCrossEntropyLoss(nn.Module):
    """
    TCE from ForTIFAI (Shabgahi et al. 2025).
    Masks tokens where the model's predicted probability exceeds gamma,
    filtering out high-confidence (likely machine-generated) predictions.

    gamma=1.0 -> standard CE (no masking)
    gamma=0.9 -> mask tokens where model assigns >90% confidence
    gamma=0.7 -> more aggressive masking

    The connection to v_t: high-confidence tokens generate large, consistent
    gradients across batches. These are the primary contributors to v_t
    accumulation in high-FIM blocks. Masking them should reduce the gradient
    magnitude differential between high-FIM and low-FIM blocks.
    """
    def __init__(self, gamma=0.9):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, labels):
        """
        logits: (B, T, V)
        labels: (B, T)  with -100 for positions to ignore
        """
        B, T, V = logits.shape

        # Compute probabilities for the actual next token at each position
        # logits[:, :-1] predicts labels[:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, V)
        shift_labels = labels[:, 1:].contiguous()        # (B, T-1)

        # Flatten
        flat_logits = shift_logits.view(-1, V)           # (B*(T-1), V)
        flat_labels = shift_labels.view(-1)              # (B*(T-1),)

        # Standard per-token CE loss (unreduced)
        per_token_loss = F.cross_entropy(
            flat_logits, flat_labels,
            ignore_index=-100,
            reduction='none'
        )                                                 # (B*(T-1),)

        # Compute predicted probability for the true next token
        # This is exp(-per_token_loss) where loss is not -100-masked
        with torch.no_grad():
            probs = torch.exp(-per_token_loss)           # (B*(T-1),)

        # Mask: keep tokens where predicted probability < gamma
        # (i.e., model is uncertain enough that the token is worth learning from)
        valid_mask = (flat_labels != -100)               # not padding
        confidence_mask = (probs < self.gamma)           # below confidence threshold
        final_mask = valid_mask & confidence_mask

        if final_mask.sum() == 0:
            # Fallback to standard CE if all tokens masked (shouldn't happen)
            return F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)

        masked_loss = per_token_loss[final_mask]
        return masked_loss.mean()


# ── HELPERS ───────────────────────────────────────────────────────────────────

def train_one_generation(gen, source_model_path, data_path, model_out_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  [TRAIN] Gen {gen} | source: {source_model_path}")
    print(f"  [TRAIN] gamma={GAMMA} | data: {data_path}")

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

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=100,
        num_training_steps=len(loader)
    )

    # TCE loss
    tce = TruncatedCrossEntropyLoss(gamma=GAMMA)

    model.train()
    pbar = tqdm(loader, desc=f"  TCE Gen {gen} (gamma={GAMMA})")
    total_loss = 0.0
    total_tokens_masked = 0
    total_tokens_seen = 0

    for step, batch in enumerate(pbar):
        batch = {k: v.to(device) for k, v in batch.items()}
        ids = batch['input_ids']

        # Forward pass — get logits without standard loss
        with torch.cuda.amp.autocast(dtype=torch.bfloat16
                                     if torch.cuda.is_bf16_supported()
                                     else torch.float16):
            outputs = model(input_ids=ids)
            logits = outputs.logits

        # TCE loss with labels = input_ids (causal LM)
        labels = ids.clone()
        # Mask padding tokens from loss
        labels[labels == tokenizer.pad_token_id] = -100

        loss = tce(logits.float(), labels)
        loss.backward()
        optim.step()
        sched.step()
        optim.zero_grad()

        total_loss += loss.item()
        if step % 100 == 0:
            pbar.set_postfix({"tce_loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    print(f"  [TRAIN] Gen {gen} avg TCE loss: {avg_loss:.4f}")

    os.makedirs(model_out_path, exist_ok=True)
    model.save_pretrained(model_out_path)
    tokenizer.save_pretrained(model_out_path)

    del model, optim, sched
    torch.cuda.empty_cache()
    gc.collect()


def run_fim(model_path, result_dir):
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


def load_fim_results(result_dir):
    path = os.path.join(result_dir, "perblock_fim.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    fim_per_block = {}
    for i, block_data in enumerate(data.get("blocks", [])):
        if not isinstance(block_data, dict):
            continue
        b = block_data.get("block_idx", i)
        attn = block_data.get("attention", {})
        mlp  = block_data.get("mlp", {})
        attn_top = float(attn.get("top", 0)) if isinstance(attn, dict) and "error" not in attn else 0.0
        mlp_top  = float(mlp.get("top",  0)) if isinstance(mlp,  dict) and "error" not in mlp  else 0.0
        if attn_top > 0 or mlp_top > 0:
            fim_per_block[b] = attn_top + mlp_top
    return fim_per_block


def compute_drift(model_path_0, model_path_n):
    def load_sd(path):
        m = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float32,
            device_map="cpu", low_cpu_mem_usage=True
        )
        sd = {k: v.clone() for k, v in m.state_dict().items()}
        del m
        gc.collect()
        return sd

    print(f"  [DRIFT] Loading Gen0 and target model...")
    sd0 = load_sd(model_path_0)
    sdn = load_sd(model_path_n)

    block_sq_drift = defaultdict(float)
    block_sq_base  = defaultdict(float)

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
        block_sq_drift[b] += (pn.float() - p0.float()).norm(p='fro').item() ** 2
        block_sq_base[b]  += p0.float().norm(p='fro').item() ** 2

    del sd0, sdn
    gc.collect()
    return {b: (block_sq_drift[b]**0.5) / (block_sq_base[b]**0.5)
            for b in block_sq_drift if block_sq_base[b] > 0}



@torch.no_grad()
def evaluate_perplexity(model_path, n_batches=50):
    """Evaluate perplexity on TinyStories validation set."""
    from datasets import load_dataset as _load_dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [PPL] Evaluating {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16,
        device_map=device, low_cpu_mem_usage=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = _load_dataset("roneneldan/TinyStories",
                             split=f"validation[:{n_batches * BATCH_SIZE}]")
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True,
                         padding="max_length", max_length=MAX_LENGTH)
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    dataset.set_format("torch")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    total_loss, count = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        ids    = batch["input_ids"].to(device)
        labels = ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        out    = model(ids, labels=labels)
        total_loss += out.loss.item()
        count += 1
    ppl = float(torch.exp(torch.tensor(total_loss / count)).item())
    print(f"  [PPL] Perplexity: {ppl:.4f}")
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return ppl


def compute_spearman(fim_dict, drift_dict):
    common = sorted(set(fim_dict.keys()) & set(drift_dict.keys()))
    if len(common) < 5:
        return float('nan'), float('nan'), len(common)
    log_fim = np.array([np.log10(fim_dict[b] + 1e-12) for b in common])
    drift   = np.array([drift_dict[b] for b in common])
    rho, p  = stats.spearmanr(log_fim, drift)
    return float(rho), float(p), len(common)


# ── MAIN ──────────────────────────────────────────────────────────────────────

print("=" * 70)
print("TCE INTERVENTION: TRUNCATED CROSS-ENTROPY LOSS")
print(f"Model: {MODEL_ID}")
print(f"Optimizer: AdamW (standard, beta2=0.999)")
print(f"Loss: TCE with gamma={GAMMA}")
print(f"Hypothesis: masking high-confidence tokens disrupts v_t feedback loop")
print("=" * 70)

# Gen0 FIM — reuse treatment
gen0_fim_dir = os.path.join(BASE_DIR, "results", "treatment_gen_0")
if not os.path.exists(os.path.join(gen0_fim_dir, "perblock_fim.json")):
    print("\nRunning Gen0 FIM...")
    run_fim(MODEL_ID, gen0_fim_dir)

fim_gen0 = load_fim_results(gen0_fim_dir)
print(f"\nGen0 FIM loaded: {len(fim_gen0)} blocks")

summary = {}

# Gen0 baseline perplexity
print("\n[PPL] Computing Gen0 baseline perplexity...")
ppl_gen0 = evaluate_perplexity(MODEL_ID)
print(f"  Gen0 perplexity: {ppl_gen0:.4f}")

for gen in range(1, GENERATIONS + 1):
    print(f"\n{'='*70}")
    print(f"GENERATION {gen}")
    print(f"{'='*70}")

    data_path  = f"{DATA_PREFIX}{gen}"
    model_out  = f"{MODEL_OUT}{gen}"
    result_dir = f"{RESULT_OUT}{gen}"
    fim_json   = os.path.join(result_dir, "perblock_fim.json")

    if not os.path.exists(data_path):
        print(f"  ✗ Treatment data not found at {data_path}. Run treatment first.")
        break

    source_model = MODEL_ID if gen == 1 else f"{MODEL_OUT}{gen-1}"

    if os.path.exists(model_out) and os.path.exists(os.path.join(model_out, "config.json")):
        print(f"  ✓ Model exists at {model_out}, skipping training")
    else:
        train_one_generation(gen, source_model, data_path, model_out)

    if os.path.exists(fim_json):
        print(f"  ✓ FIM exists, skipping")
    else:
        run_fim(model_out, result_dir)

    fim_n = load_fim_results(result_dir)
    if fim_n is None:
        print(f"  ✗ FIM results missing for Gen {gen}")
        continue

    print(f"\n  [DRIFT] Computing relative Frobenius drift vs Gen0...")
    drift_n = compute_drift(MODEL_ID, model_out)

    rho, p, n_blocks = compute_spearman(fim_n, drift_n)
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
    print(f"\n  Gen {gen}: rho={rho:+.4f}{sig}  p={p:.4f}  n={n_blocks} blocks")

    # Perplexity evaluation
    ppl = evaluate_perplexity(model_out)

    summary[gen] = {
        "spearman_rho": rho,
        "p_value": p,
        "n_blocks": n_blocks,
        "significant": p < 0.05,
        "gamma": GAMMA,
        "perplexity": ppl
    }

with open(SUMMARY_OUT, "w") as f:
    json.dump(summary, f, indent=2)

# Print summary
print("\n" + "="*70)
print(f"TCE ABLATION (gamma={GAMMA}) — FIM-DRIFT CORRELATION SUMMARY")
print("="*70)
print(f"\n{'Gen':<6} {'Rho':>10} {'p-value':>12} {'Sig':>5} {'PPL':>10} {'PPL%':>8}")
print("-"*55)
for gen, r in summary.items():
    sig = "**" if r['p_value'] < 0.01 else ("*" if r['p_value'] < 0.05 else "")
    ppl = r.get('perplexity', float('nan'))
    pct = (ppl - ppl_gen0) / ppl_gen0 * 100 if not (ppl != ppl) else float('nan')
    print(f"  {gen:<4} {r['spearman_rho']:>+10.4f} {r['p_value']:>12.4f} {sig:>5} {ppl:>10.2f} {pct:>+7.1f}%")
print(f"\n  Gen0 baseline PPL: {ppl_gen0:.4f}")
print(f"  Treatment Gen5 PPL: 7.84 (+29.5%)")

print(f"""
INTERPRETATION:
  Compare against treatment (AdamW, standard CE):
    G1-G5: -0.40*, -0.43*, -0.44*, -0.52**, -0.49**

  If TCE correlation is WEAKER (closer to zero or positive):
    -> Masking high-confidence tokens disrupts the v_t feedback loop.
       High-confidence gradients are the primary driver of differential
       v_t accumulation across blocks. This is the strongest possible
       confirmation of the mechanism.

  If TCE correlation is SIMILAR (still strongly negative):
    -> The v_t mechanism is robust to confidence-based loss filtering.
       TCE may still delay perplexity collapse through diversity
       preservation, but via a different pathway than v_t suppression.

  If TCE correlation is STRONGER (more negative):
    -> Uncertain tokens actually contribute more to differential v_t.
       Removing confident tokens makes the remaining gradient signal
       more concentrated in high-sensitivity blocks.

  Also note: check perplexity of TCE models vs treatment models to
  assess whether the intervention delays collapse regardless of
  correlation changes.

Results saved to: {SUMMARY_OUT}
""")