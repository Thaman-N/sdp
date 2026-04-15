import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import glob
import json
import numpy as np
from tqdm import tqdm
import gc

# ============================================================================
# CONFIGURATION
# ============================================================================
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = r"D:\Thaman\Work\hessian-spectral-analysis\models"
CSV_PATH = "results/summary/comprehensive_metrics.csv"
JSON_PATH = "results/summary/comprehensive_metrics.json"

MAX_LENGTH   = 512
EVAL_SAMPLES = 100

BASE_MODELS = {
    "SmolLM Control A":   "HuggingFaceTB/SmolLM2-135M",
    "SmolLM Control B":   "HuggingFaceTB/SmolLM2-135M",
    "SmolLM Treatment":   "HuggingFaceTB/SmolLM2-135M",
    "Qwen 2.5 Control C": "Qwen/Qwen2.5-0.5B",
    "GPT2 Treatment":     "gpt2",
    "Llama Treatment":    "meta-llama/Llama-3.2-1B",
    "Gemma Treatment":    "google/gemma-3-1b-it",
    "Qwen 3.5 Treatment": "Qwen/Qwen3.5-0.8B",
    "Phi-1.5 Treatment":  "microsoft/phi-1_5",
    "Pythia Treatment":   "EleutherAI/pythia-1.4b",
}

MAPPING = {
    "control_generation_":            "SmolLM Control A",
    "control_b_gen_":                 "SmolLM Control B",
    "treatment_gen_":                 "SmolLM Treatment",
    "control_c_gen_":                 "Qwen 2.5 Control C",
    "llama_treatment_gen_":           "Llama Treatment",
    "gemma_treatment_gen_":           "Gemma Treatment",
    "gpt2_treatment_gen_":            "GPT2 Treatment",
    "Qwen3.5-0.8B_treatment_gen_":    "Qwen 3.5 Treatment",
    "phi-1_5_treatment_gen_":         "Phi-1.5 Treatment",
    "pythia-1.4b_treatment_gen_":     "Pythia Treatment",
    "frozen_late_gen_":               "SmolLM Frozen Late",
    "ortho_drift_gen_":               "SmolLM Ortho Drift",
    "smart_ortho_gen_":               "SmolLM Smart Ortho",
    "ewc_lambda100_gen_":             "SmolLM EWC lambda=100",
    "ewc_lambda500_gen_":             "SmolLM EWC lambda=500",
    "ewc_lambda50_gen_":              "SmolLM EWC lambda=50",
}

TEST_PROMPTS = [
    "Once upon a time, there was a little dog.",
    "The sun was shining bright in the sky.",
    "Lily went to the park with her mom.",
    "Once upon a time",
    "In the dark forest",
]

# ============================================================================
# METRICS
# ============================================================================

def calculate_perplexity(model, tokenizer, dataset):
    model.eval()
    nlls = []
    texts = dataset.select(range(min(EVAL_SAMPLES, len(dataset))))["text"]
    with torch.no_grad():
        for text in tqdm(texts[:30], desc="      PPL", leave=False):
            enc = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=MAX_LENGTH)
            ids = enc.input_ids.to(DEVICE)
            if ids.size(1) < 10:
                continue
            nlls.append(model(ids, labels=ids).loss)
    return torch.exp(torch.stack(nlls).mean()).item() if nlls else float("inf")


def calculate_diversity(model, tokenizer):
    model.eval()
    generated = ""
    for p in TEST_PROMPTS[:3]:
        inputs = tokenizer(p, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=50, do_sample=True,
                temperature=0.7, pad_token_id=tokenizer.eos_token_id
            )
        generated += tokenizer.decode(out[0], skip_special_tokens=True) + " "
    tokens = generated.split()
    if len(tokens) < 2:
        return 0.0
    bigrams = list(zip(tokens, tokens[1:]))
    return len(set(bigrams)) / len(bigrams)


def calculate_coherence(model, tokenizer, embed_model):
    model.eval()
    sims = []
    for prompt in TEST_PROMPTS[:3]:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=50, do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        gen = tokenizer.decode(out[0], skip_special_tokens=True).replace(prompt, "").strip()
        if not gen:
            gen = "empty"
        emb = embed_model.encode([prompt, gen], convert_to_tensor=True)
        sims.append(util.pytorch_cos_sim(emb[0], emb[1]).item())
    return float(np.mean(sims))


def evaluate_checkpoint(path, exp_name, gen, val_ds, embed_model):
    print(f"--> {exp_name} | Gen {gen}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        res = {
            "Experiment": exp_name,
            "Generation": gen,
            "Model_Path": path,
            "Perplexity": round(calculate_perplexity(model, tokenizer, val_ds), 4),
            "Diversity":  round(calculate_diversity(model, tokenizer), 4),
            "Coherence":  round(calculate_coherence(model, tokenizer, embed_model), 4),
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return res
    except Exception as e:
        print(f"    Error: {e}")
        return None


# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    # Load existing results — build skip set from (Experiment, Generation) pairs
    if os.path.exists(CSV_PATH):
        existing_df  = pd.read_csv(CSV_PATH)
        already_done = set(zip(existing_df["Experiment"], existing_df["Generation"]))
        all_results  = existing_df.to_dict("records")
        print(f"Loaded {len(existing_df)} existing rows — skipping these.")
    else:
        already_done = set()
        all_results  = []
        print("No existing CSV — evaluating everything from scratch.")

    val_dataset = load_dataset("roneneldan/TinyStories", split="validation")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2").to("cpu")
    new_count   = 0

    # Gen 0 base models
    print("\n" + "="*50 + "\nGEN 0 BASE MODELS\n" + "="*50)
    for exp_name, hf_id in BASE_MODELS.items():
        if (exp_name, 0) in already_done:
            print(f"    SKIP {exp_name} | Gen 0")
            continue
        res = evaluate_checkpoint(hf_id, exp_name, 0, val_dataset, embed_model)
        if res:
            all_results.append(res)
            new_count += 1

    # Local checkpoint folders
    print("\n" + "="*50 + "\nLOCAL GENERATIONS\n" + "="*50)
    for folder in sorted(glob.glob(os.path.join(BASE_DIR, "*"))):
        name     = os.path.basename(folder)
        exp_name = None

        for prefix, label in MAPPING.items():
            if name.startswith(prefix):
                exp_name = label
                break

        if not exp_name:
            continue
        if not os.path.exists(os.path.join(folder, "config.json")):
            continue

        try:
            gen = int(name.split("_")[-1])
        except ValueError:
            continue

        if (exp_name, gen) in already_done:
            print(f"    SKIP {exp_name} | Gen {gen}")
            continue

        res = evaluate_checkpoint(folder, exp_name, gen, val_dataset, embed_model)
        if res:
            all_results.append(res)
            new_count += 1

    # Save
    if new_count == 0:
        print("\nNothing new — CSV is already up to date.")
        return

    df = (
        pd.DataFrame(all_results)
        .sort_values(by=["Experiment", "Generation"])
        .reset_index(drop=True)
    )
    df.to_csv(CSV_PATH, index=False)
    with open(JSON_PATH, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nDone — added {new_count} new rows.")
    print(f"Saved to:\n  {CSV_PATH}\n  {JSON_PATH}")

    # Show only the new rows
    new_keys = set(zip(df["Experiment"], df["Generation"])) - already_done
    new_rows = df[df.apply(
        lambda r: (r["Experiment"], r["Generation"]) in new_keys, axis=1
    )]
    print("\nNew rows:")
    print(new_rows[["Experiment", "Generation", "Perplexity"]].to_string(index=False))


if __name__ == "__main__":
    main()