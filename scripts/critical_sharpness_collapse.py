"""
Critical Sharpness Analysis for Recursive Self-Distillation
============================================================
Computes two measures per model per generation:

1. RELATIVE CRITICAL SHARPNESS (λc^{Gen0→Genn}):
   How far can you step in Gen_n's update direction before Gen0's loss increases?
   δ_n = W_gen_n - W_gen_{n-1}  (incremental drift, normalised to unit norm)
   ηc = smallest η > 0 such that L_gen0(θ_gen0 - η * δ_n_hat) > L_gen0(θ_gen0)
   λc = 2 / ηc

2. PER-BLOCK RELATIVE CRITICAL SHARPNESS:
   Same but update direction masked to each block's parameters only.

Resume-safe: saves a checkpoint JSON after each generation. If killed and
restarted, skips already-completed (model, generation) pairs.

Output:
  results/critical_sharpness/cs_results.json          (full results)
  results/critical_sharpness/cs_checkpoint.json        (resume checkpoint)
  results/critical_sharpness/cs_summary.csv
  results/critical_sharpness/cs_perblock_summary.csv
"""

import os, re, gc, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR    = r"D:\Thaman\Work\hessian-spectral-analysis\models"
OUTPUT_DIR  = "results/critical_sharpness"

EPS         = 1/16
ETA_INIT    = 1e-3
MAX_EXP     = 40
N_VAL       = 50
BATCH_SIZE  = 4
MAX_LEN     = 256

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
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "cs_checkpoint.json")
RESULTS_PATH    = os.path.join(OUTPUT_DIR, "cs_results.json")

# ── CHECKPOINT HELPERS ────────────────────────────────────────────────────────

def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f, indent=2)


def is_done(checkpoint, model_name, gen):
    return checkpoint.get(model_name, {}).get(str(gen), False)


def mark_done(checkpoint, model_name, gen):
    if model_name not in checkpoint:
        checkpoint[model_name] = {}
    checkpoint[model_name][str(gen)] = True
    save_checkpoint(checkpoint)

# ── ARCHITECTURE HELPERS ──────────────────────────────────────────────────────

def get_model_blocks(model):
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return list(model.transformer.h), 'gpt2'
    elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return list(model.gpt_neox.layers), 'gpt_neox'
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return list(model.model.layers), model.config.model_type.lower()
    else:
        raise ValueError(f"Unknown architecture: {model.config.model_type}")


def get_block_param_names(model, block_idx):
    patterns = [
        rf'\.layers\.{block_idx}\.',
        rf'\.h\.{block_idx}\.',
        rf'\.blocks\.{block_idx}\.',
    ]
    names = []
    for name, _ in model.named_parameters():
        for pat in patterns:
            if re.search(pat, name):
                names.append(name)
                break
    return names

# ── DATA LOADING ──────────────────────────────────────────────────────────────

def make_dataloader(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("roneneldan/TinyStories",
                           split=f"validation[:{N_VAL * BATCH_SIZE}]")
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True,
                         padding="max_length", max_length=MAX_LEN,
                         return_tensors=None)
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format("torch")
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ── LOSS EVALUATION ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_loss(model, dataloader, device, label=""):
    """Evaluate average cross-entropy loss. Prints a dot every 10 batches."""
    loss_fn = nn.CrossEntropyLoss()
    total, count = 0.0, 0
    if label:
        print(f"      [{label}] evaluating loss: ", end="", flush=True)
    for i, batch in enumerate(dataloader):
        if i >= N_VAL:
            break
        ids = batch['input_ids'].to(device)
        out = model(ids)
        logits = out.logits if hasattr(out, 'logits') else out
        loss = loss_fn(logits[:, :-1].reshape(-1, logits.size(-1)),
                       ids[:, 1:].reshape(-1))
        total += loss.item()
        count += 1
        if label and i % 10 == 9:
            print(".", end="", flush=True)
    if label:
        print(f" done ({count} batches, loss={total/count:.4f})")
    return total / count if count > 0 else float('nan')

# ── MODEL PERTURBATION ────────────────────────────────────────────────────────

def perturb_model(model, delta_dict, eta, param_names=None):
    originals = {}
    for name, param in model.named_parameters():
        if param_names is not None and name not in param_names:
            continue
        if name not in delta_dict:
            continue
        originals[name] = param.data.clone()
        param.data.add_(-eta * delta_dict[name].to(param.device))
    return originals


def restore_model(model, originals):
    for name, param in model.named_parameters():
        if name in originals:
            param.data.copy_(originals[name])

# ── LINE SEARCH ───────────────────────────────────────────────────────────────

def line_search_critical_lr(model, dataloader, device, delta_dict,
                             base_loss, param_names=None,
                             eta0=ETA_INIT, label=""):
    """
    Find ηc = smallest η > 0 such that L(θ - η*δ̂) > L(θ).
    Two-phase: exponential bracketing + binary refinement.
    All forward passes, no gradients.
    """
    print(f"      [{label}] line search — base_loss={base_loss:.5f}  η0={eta0:.2e}")

    # ── Phase 1: Exponential search ──────────────────────────────────────────
    eta = eta0
    originals = perturb_model(model, delta_dict, eta, param_names)
    loss_at_eta = evaluate_loss(model, dataloader, device,
                                label=f"{label} exp η={eta:.2e}")
    restore_model(model, originals)

    direction = -1 if loss_at_eta > base_loss else +1
    print(f"      [{label}] initial direction: {'decrease η' if direction==-1 else 'increase η'}")

    eta_lower, eta_upper = None, None
    for step in range(MAX_EXP):
        eta = eta * (2.0 ** direction)
        originals = perturb_model(model, delta_dict, eta, param_names)
        loss_at_eta = evaluate_loss(model, dataloader, device,
                                    label=f"{label} exp step {step+1} η={eta:.2e}")
        restore_model(model, originals)

        if direction == +1 and loss_at_eta > base_loss:
            eta_upper = eta
            eta_lower = eta / 2.0
            print(f"      [{label}] bracketed: [{eta_lower:.2e}, {eta_upper:.2e}]")
            break
        elif direction == -1 and loss_at_eta <= base_loss:
            eta_lower = eta
            eta_upper = eta * 2.0
            print(f"      [{label}] bracketed: [{eta_lower:.2e}, {eta_upper:.2e}]")
            break

    if eta_lower is None or eta_upper is None:
        print(f"      [{label}] WARNING: bracketing failed, returning nan")
        return float('nan')

    # ── Phase 2: Binary search ────────────────────────────────────────────────
    for step in range(20):
        if abs(1 - eta_lower / eta_upper) < EPS:
            break
        eta_mid = 0.5 * (eta_lower + eta_upper)
        originals = perturb_model(model, delta_dict, eta_mid, param_names)
        loss_mid = evaluate_loss(model, dataloader, device,
                                 label=f"{label} bin step {step+1} η={eta_mid:.2e}")
        restore_model(model, originals)
        if loss_mid > base_loss:
            eta_upper = eta_mid
        else:
            eta_lower = eta_mid

    eta_c = 0.5 * (eta_lower + eta_upper)
    lambda_c = 2.0 / eta_c
    print(f"      [{label}] RESULT: ηc={eta_c:.6f}  λc={lambda_c:.3f}")
    return eta_c

# ── MAIN ANALYSIS ─────────────────────────────────────────────────────────────

def analyse_model(model_name, cfg, checkpoint, all_results):
    print(f"\n{'='*70}")
    print(f"Model: {model_name}  ({cfg['arch'].upper()})")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    gen_paths = {0: cfg["base"]}
    for g in range(1, 6):
        p = os.path.join(BASE_DIR, f"{cfg['prefix']}{g}")
        if not os.path.exists(p):
            print(f"  ✗ Missing: {p}")
            return
        gen_paths[g] = p

    # Check if all 5 generations already done
    if all(is_done(checkpoint, model_name, g) for g in range(1, 6)):
        print(f"  ✓ All generations complete, skipping.")
        return

    print("  Loading dataloader...")
    dataloader = make_dataloader(cfg["base"])

    print("  Loading Gen0...")
    model_gen0 = AutoModelForCausalLM.from_pretrained(
        cfg["base"], torch_dtype=torch.float16,
        device_map=device, low_cpu_mem_usage=True
    )
    model_gen0.eval()

    base_loss = evaluate_loss(model_gen0, dataloader, device, label="Gen0 baseline")
    print(f"  Gen0 base loss: {base_loss:.5f}")

    blocks, arch = get_model_blocks(model_gen0)
    n_blocks = len(blocks)
    block_param_names = {b: get_block_param_names(model_gen0, b)
                         for b in range(n_blocks)}

    if model_name not in all_results:
        all_results[model_name] = {
            "model": model_name, "arch": cfg["arch"],
            "n_blocks": n_blocks, "base_loss": base_loss,
            "global": {}, "perblock": {}
        }

    print(f"  Loading Gen0 state dict for delta computation...")
    sd_prev = {k: v.cpu().float().clone()
               for k, v in model_gen0.state_dict().items()}

    # Need to fast-forward sd_prev to the last completed generation
    last_done = 0
    for g in range(1, 6):
        if is_done(checkpoint, model_name, g):
            last_done = g

    if last_done > 0:
        print(f"  Fast-forwarding state dict to Gen{last_done}...")
        for g in range(1, last_done + 1):
            m_tmp = AutoModelForCausalLM.from_pretrained(
                gen_paths[g], torch_dtype=torch.float16,
                device_map='cpu', low_cpu_mem_usage=True
            )
            sd_prev = {k: v.cpu().float().clone()
                       for k, v in m_tmp.state_dict().items()}
            del m_tmp
            gc.collect()
        print(f"  Resuming from Gen{last_done + 1}")

    for g in range(1, 6):
        if is_done(checkpoint, model_name, g):
            print(f"\n  ── Generation {g}: already complete, skipping ──")
            continue

        print(f"\n  ── Generation {g} ──")
        print(f"    Loading Gen{g} state dict...")
        m_gn = AutoModelForCausalLM.from_pretrained(
            gen_paths[g], torch_dtype=torch.float16,
            device_map='cpu', low_cpu_mem_usage=True
        )
        sd_curr = {k: v.cpu().float().clone()
                   for k, v in m_gn.state_dict().items()}
        del m_gn
        gc.collect()

        # Compute delta and global norm
        delta_raw = {}
        delta_flat_all = []
        for name in sd_prev:
            if name in sd_curr:
                d = sd_curr[name] - sd_prev[name]
                delta_raw[name] = d
                if (d.ndim >= 2
                        and 'embed' not in name.lower()
                        and 'norm' not in name.lower()
                        and 'ln_' not in name.lower()):
                    delta_flat_all.append(d.flatten())

        global_norm = torch.cat(delta_flat_all).norm().item()
        print(f"    Global drift norm: {global_norm:.6f}")

        if global_norm < 1e-12:
            print(f"    WARNING: near-zero drift, skipping Gen{g}")
            sd_prev = sd_curr
            mark_done(checkpoint, model_name, g)
            continue

        delta_hat = {k: v / global_norm for k, v in delta_raw.items()}

        # ── Global line search ────────────────────────────────────────────────
        print(f"    [GLOBAL] Starting line search for Gen{g}...")
        eta_c = line_search_critical_lr(
            model_gen0, dataloader, device, delta_hat,
            base_loss=base_loss, param_names=None,
            eta0=ETA_INIT, label=f"{model_name} G{g} global"
        )
        lambda_c = 2.0 / eta_c if not np.isnan(eta_c) else float('nan')
        all_results[model_name]["global"][g] = {
            "eta_c": eta_c, "lambda_c": lambda_c, "drift_norm": global_norm
        }
        print(f"    [GLOBAL] Gen{g}: ηc={eta_c:.6f}  λc={lambda_c:.3f}")

        # ── Per-block line search ─────────────────────────────────────────────
        all_results[model_name]["perblock"][g] = {}
        for b in range(n_blocks):
            print(f"\n    [BLOCK {b:2d}/{n_blocks-1}] Starting line search...")
            pnames = block_param_names[b]
            if not pnames:
                print(f"    [BLOCK {b:2d}] No params found, skipping")
                continue

            block_flat = [delta_raw[pn].flatten()
                          for pn in pnames
                          if pn in delta_raw and delta_raw[pn].ndim >= 2]
            if not block_flat:
                continue

            block_norm = torch.cat(block_flat).norm().item()
            if block_norm < 1e-12:
                all_results[model_name]["perblock"][g][b] = {
                    "eta_c": float('nan'), "lambda_c": float('nan'),
                    "block_drift_norm": 0.0
                }
                continue

            block_delta_hat = {pn: delta_raw[pn] / block_norm
                               for pn in pnames if pn in delta_raw}

            eta_c_b = line_search_critical_lr(
                model_gen0, dataloader, device, block_delta_hat,
                base_loss=base_loss, param_names=set(pnames),
                eta0=ETA_INIT,
                label=f"{model_name} G{g} B{b}"
            )
            lambda_c_b = (2.0 / eta_c_b
                          if not np.isnan(eta_c_b) else float('nan'))

            all_results[model_name]["perblock"][g][b] = {
                "eta_c": eta_c_b,
                "lambda_c": lambda_c_b,
                "block_drift_norm": block_norm
            }
            print(f"    [BLOCK {b:2d}] ηc={eta_c_b:.6f}  "
                  f"λc={lambda_c_b:.3f}  drift={block_norm:.6f}")

        # Save checkpoint and full results after each generation
        sd_prev = sd_curr
        mark_done(checkpoint, model_name, g)

        # Save running results
        serialisable = {}
        for mn, mr in all_results.items():
            serialisable[mn] = {
                k: ({str(gk): gv for gk, gv in v.items()}
                    if isinstance(v, dict) else v)
                for k, v in mr.items()
            }
        with open(RESULTS_PATH, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"\n  ✓ Gen{g} complete. Checkpoint saved.")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model_gen0
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── ENTRY POINT ───────────────────────────────────────────────────────────────

checkpoint = load_checkpoint()
all_results = {}

# Load any existing results
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH, "r") as f:
        all_results = json.load(f)
    print(f"Loaded existing results for: {list(all_results.keys())}")

for model_name, cfg in MODELS.items():
    analyse_model(model_name, cfg, checkpoint, all_results)

# ── FINAL CSVS ────────────────────────────────────────────────────────────────
global_rows, pb_rows = [], []
for model_name, r in all_results.items():
    arch = r.get("arch", "unknown")
    for g, gdata in r.get("global", {}).items():
        global_rows.append({
            "Model": model_name, "Architecture": arch,
            "Generation": int(g),
            "EtaC": gdata.get("eta_c", float('nan')),
            "LambdaC": gdata.get("lambda_c", float('nan')),
            "DriftNorm": gdata.get("drift_norm", float('nan')),
        })
    for g, block_data in r.get("perblock", {}).items():
        for b, bdata in block_data.items():
            pb_rows.append({
                "Model": model_name, "Architecture": arch,
                "Generation": int(g), "Block": int(b),
                "EtaC": bdata.get("eta_c", float('nan')),
                "LambdaC": bdata.get("lambda_c", float('nan')),
                "BlockDriftNorm": bdata.get("block_drift_norm", float('nan')),
            })

pd.DataFrame(global_rows).to_csv(
    os.path.join(OUTPUT_DIR, "cs_summary.csv"), index=False)
pd.DataFrame(pb_rows).to_csv(
    os.path.join(OUTPUT_DIR, "cs_perblock_summary.csv"), index=False)

# ── PRINT GLOBAL SUMMARY ──────────────────────────────────────────────────────
print("\n" + "="*70)
print("GLOBAL RELATIVE CRITICAL SHARPNESS — SUMMARY")
print("="*70)
print(f"\n{'Model':<16} {'Arch':<11} {'G1':>10} {'G2':>10} {'G3':>10} "
      f"{'G4':>10} {'G5':>10}")
print("-"*68)
for model_name, r in all_results.items():
    vals = []
    for g in range(1, 6):
        lc = r.get("global", {}).get(str(g), r.get("global", {}).get(g, {}))
        lc_val = lc.get("lambda_c", float('nan')) if isinstance(lc, dict) else float('nan')
        vals.append(f"{lc_val:>10.2f}" if not np.isnan(lc_val) else f"{'nan':>10}")
    arch = r.get("arch", "?")
    print(f"  {model_name:<14} {arch:<11} {''.join(vals)}")

print(f"""
INTERPRETATION:
  HIGH λc: Gen_n drift direction pushes hard against Gen0 loss landscape.
           Basin wall is close — model is leaving pretraining geometry fast.

  LOW λc:  Gen_n drift direction is compatible with Gen0 loss landscape.
           Basin wall is far — synthetic training is not strongly opposing
           the original geometry.

  Prediction:
    Parallel models: λc INCREASES across generations (compounding pressure)
    Sequential models: λc FLAT or DECREASING
""")