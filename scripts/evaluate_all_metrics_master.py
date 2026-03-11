"""
MASTER EVALUATION SCRIPT
Calculates ALL metrics for ALL experiments in one comprehensive run.
Handles Gen 0 (base models), Treatment, Control A, Control B, and Control C.

Metrics Calculated:
1. Perplexity (on human validation data)
2. Repetition Ratio (4-gram uniqueness)
3. Uniqueness Score (bigram diversity)
4. Shannon Entropy (vocabulary health)
5. Semantic Coherence (prompt-output similarity)
6. MAUVE Score (distribution quality)
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import glob
import os
import json
import numpy as np
from tqdm import tqdm

# Try to import MAUVE (optional - skip if not installed)
try:
    import mauve
    MAUVE_AVAILABLE = True
except ImportError:
    MAUVE_AVAILABLE = False
    print("⚠️  MAUVE not installed. Install with: pip install mauve-text")
    print("   MAUVE scores will be skipped.\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512
STRIDE = 512
EVAL_SAMPLES = 200  # Number of samples for perplexity
GEN_SAMPLES = 20    # Number of generations for MAUVE/coherence
GEN_LENGTH = 128    # Length of generated text

# Base models for Gen 0
BASE_MODELS = {
    # "Control A (Fresh)": "HuggingFaceTB/SmolLM2-135M",
    # "Control B (Static)": "HuggingFaceTB/SmolLM2-135M", 
    # "Treatment (Recursive)": "HuggingFaceTB/SmolLM2-135M",
    # "Control C (Qwen)": "Qwen/Qwen2.5-0.5B"  # Optional
    "Gemma Treatment": "google/gemma-3-1b-it"
}

# Test prompts for generation tasks
TEST_PROMPTS = [
    "Once upon a time, there was a little dog.",
    "Timmy loved to play with his red ball.",
    "The sun was shining bright in the sky.",
    "Lily went to the park with her mom.",
    "One day, a big bear came to the house.",
    "Once upon a time",
    "The little girl",
    "In the dark forest",
    "The sun was"
]

# ============================================================================
# METRIC FUNCTIONS
# ============================================================================

def calculate_perplexity(model, tokenizer, dataset):
    """
    Calculates Perplexity on human validation data.
    Processes texts individually to avoid context window issues.
    """
    model.eval()
    nlls = []
    
    # Process texts individually instead of concatenating
    texts = dataset.select(range(min(EVAL_SAMPLES, len(dataset))))["text"]
    
    with torch.no_grad():
        for text in tqdm(texts[:50], desc="      Computing PPL", leave=False):  # Limit to 50 for speed
            # Tokenize with truncation
            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH
            )
            
            input_ids = encodings.input_ids.to(DEVICE)
            target_ids = input_ids.clone()
            
            # Skip very short sequences
            if input_ids.size(1) < 10:
                continue
            
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss)
    
    if len(nlls) == 0:
        return float('inf')
    
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


def calculate_repetition_ratio(model, tokenizer):
    """
    Measures repetitive loops using 4-gram analysis.
    Lower ratio = more repetitive (bad).
    """
    model.eval()
    prompt = "Once upon a time,"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=100, 
            do_sample=False,  # Greedy reveals loops
            pad_token_id=tokenizer.eos_token_id
        )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 4-gram uniqueness check
    words = text.split()
    if len(words) < 4:
        return 1.0, text
    
    four_grams = [tuple(words[i:i+4]) for i in range(len(words)-3)]
    unique_4grams = len(set(four_grams))
    total_4grams = len(four_grams)
    
    ratio = unique_4grams / total_4grams if total_4grams > 0 else 1.0
    return ratio, text


def calculate_uniqueness_score(model, tokenizer):
    """
    Generates text and checks bigram diversity (mode collapse detection).
    Higher score = more diverse (good).
    """
    model.eval()
    prompts = ["Once upon a time", "The little girl", "In the dark forest", "The sun was"]
    generated_text = ""
    
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100, 
                do_sample=True, 
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_text += tokenizer.decode(outputs[0], skip_special_tokens=True) + " "
    
    # Calculate unique bigrams / total bigrams
    tokens = generated_text.split()
    if len(tokens) < 2:
        return 0.0
    
    bigrams = list(zip(tokens, tokens[1:]))
    unique_bigrams = len(set(bigrams))
    
    return unique_bigrams / len(bigrams) if len(bigrams) > 0 else 0.0


def calculate_shannon_entropy(model, tokenizer, dataset):
    """
    Calculates Shannon Entropy of model's predictive distribution.
    Measures vocabulary health and confidence spread.
    """
    model.eval()
    text = "\n\n".join(dataset["text"][:5])
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Entropy = -sum(p * log(p))
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    
    return entropy.item()


def calculate_semantic_coherence(model, tokenizer, embed_model, prompts):
    """
    Generates stories and measures semantic similarity to prompts.
    Uses sentence transformers to check if output relates to input.
    """
    model.eval()
    similarities = []
    
    for prompt in prompts:
        # Generate continuation
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=64,
                do_sample=True, 
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the NEW text (remove prompt)
        new_text = generated_text.replace(prompt, "").strip()
        if not new_text:
            new_text = "empty"
        
        # Embed and compare
        embeddings = embed_model.encode([prompt, new_text], convert_to_tensor=True)
        sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        similarities.append(sim.item())
    
    return np.mean(similarities) if similarities else 0.0


def calculate_mauve_score(model, tokenizer, human_texts):
    """
    Calculates MAUVE score comparing generated vs human text distributions.
    Requires mauve-text package.
    """
    if not MAUVE_AVAILABLE:
        return None
    
    model.eval()
    gen_texts = []
    
    # Generate texts using human texts as prompts
    for human_text in human_texts[:GEN_SAMPLES]:
        prompt = human_text[:50]  # First 50 chars as prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=GEN_LENGTH,
                do_sample=True, 
                temperature=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        gen_texts.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    # Calculate MAUVE
    try:
        out_mauve = mauve.compute_mauve(
            p_text=human_texts[:len(gen_texts)],
            q_text=gen_texts,
            device_id=0 if torch.cuda.is_available() else -1,
            max_text_length=256,
            verbose=False
        )
        return out_mauve.mauve
    except Exception as e:
        print(f"      ⚠️  MAUVE calculation failed: {e}")
        return None


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(model_path, exp_name, gen, val_dataset, embed_model, human_texts):
    """
    Runs all metrics on a single model checkpoint.
    """
    print(f"\n{'='*80}")
    print(f"📊 Evaluating: {exp_name} - Generation {gen}")
    print(f"   Model: {model_path}")
    print(f"{'='*80}")
    
    results = {
        "Experiment": exp_name,
        "Generation": gen,
        "Model_Path": model_path
    }
    
    try:
        # Load model
        print("   Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16
        ).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 1. Perplexity
        print("   [1/6] Calculating Perplexity...")
        ppl = calculate_perplexity(model, tokenizer, val_dataset)
        results["Perplexity"] = round(ppl, 4)
        print(f"      ✓ Perplexity: {ppl:.4f}")
        
        # 2. Repetition Ratio (4-grams)
        print("   [2/6] Calculating Repetition Ratio...")
        rep_ratio, sample_text = calculate_repetition_ratio(model, tokenizer)
        results["Repetition_Ratio"] = round(rep_ratio, 4)
        results["Sample_Text"] = sample_text[:100]
        print(f"      ✓ Repetition Ratio: {rep_ratio:.4f}")
        
        # 3. Uniqueness Score (bigrams)
        print("   [3/6] Calculating Uniqueness Score...")
        uniq_score = calculate_uniqueness_score(model, tokenizer)
        results["Uniqueness_Score"] = round(uniq_score, 4)
        print(f"      ✓ Uniqueness Score: {uniq_score:.4f}")
        
        # 4. Shannon Entropy
        print("   [4/6] Calculating Shannon Entropy...")
        entropy = calculate_shannon_entropy(model, tokenizer, val_dataset)
        results["Shannon_Entropy"] = round(entropy, 4)
        print(f"      ✓ Shannon Entropy: {entropy:.4f}")
        
        # 5. Semantic Coherence
        print("   [5/6] Calculating Semantic Coherence...")
        coherence = calculate_semantic_coherence(model, tokenizer, embed_model, TEST_PROMPTS[:5])
        results["Semantic_Coherence"] = round(coherence, 4)
        print(f"      ✓ Semantic Coherence: {coherence:.4f}")
        
        # 6. MAUVE Score (optional)
        if MAUVE_AVAILABLE:
            print("   [6/6] Calculating MAUVE Score...")
            mauve_score = calculate_mauve_score(model, tokenizer, human_texts)
            results["MAUVE_Score"] = round(mauve_score, 4) if mauve_score else None
            print(f"      ✓ MAUVE Score: {mauve_score:.4f}" if mauve_score else "      ⚠️  MAUVE: Skipped")
        else:
            results["MAUVE_Score"] = None
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        
        print(f"   ✅ Completed: {exp_name} Gen {gen}")
        return results
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        results["Error"] = str(e)
        return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("🚀 MASTER EVALUATION SCRIPT - ALL METRICS")
    print("="*80)
    
    # Load resources
    print("\n📚 Loading validation dataset...")
    val_dataset = load_dataset("roneneldan/TinyStories", split="validation")
    human_texts = val_dataset.select(range(GEN_SAMPLES))["text"]
    
    print("🧠 Loading sentence transformer for coherence...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2").to("cpu")
    
    # Load existing results if available
    all_results = []
    existing_results = {}
    csv_path = "results/summary/comprehensive_metrics_fimgemma.csv"
    json_path = "results/summary/comprehensive_metrics_fimgemma.json"
    if os.path.exists(csv_path):
        try:
            df_existing = pd.read_csv(csv_path)
            for _, row in df_existing.iterrows():
                key = (str(row["Experiment"]), int(row["Generation"]), str(row["Model_Path"]))
                existing_results[key] = True
        except Exception as e:
            print(f"⚠️  Failed to load existing CSV: {e}")
    elif os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                json_existing = json.load(f)
            for item in json_existing:
                key = (str(item["Experiment"]), int(item["Generation"]), str(item["Model_Path"]))
                existing_results[key] = True
        except Exception as e:
            print(f"⚠️  Failed to load existing JSON: {e}")

    
    # ========================================================================
    # PART 1: Evaluate Gen 0 (meta-llama/Llama-3.2-1B only)
    # ========================================================================
    print("\n" + "="*80)
    print("PART 1: Llama Gen 0")
    print("="*80)

    for exp_name, model_id in BASE_MODELS.items():
        key = (exp_name, 0, model_id)
        if key in existing_results:
            print(f"⚠️  Skipping base model {exp_name} - already evaluated")
            continue
        results = evaluate_model(
            model_path=model_id,
            exp_name=exp_name,
            gen=0,
            val_dataset=val_dataset,
            embed_model=embed_model,
            human_texts=human_texts
        )
        all_results.append(results)
    
    # ========================================================================
    # PART 2: Evaluate Llama Treatment Models (Gen 1-5)
    # ========================================================================
    print("\n" + "="*80)
    print("PART 2: Llama Treatment Models (Generations 1-5)")
    print("="*80)

    for gen in range(1, 6):
        folder = f"models/gemma_treatment_gen_{gen}"
        if not os.path.exists(os.path.join(folder, "config.json")):
            print(f"⚠️  Skipping {folder} - config.json not found")
            continue
        exp_name = "Gemma Treatment"
        key = (exp_name, gen, folder)
        if key in existing_results:
            print(f"⚠️  Skipping {folder} - already evaluated")
            continue
        results = evaluate_model(
            model_path=folder,
            exp_name=exp_name,
            gen=gen,
            val_dataset=val_dataset,
            embed_model=embed_model,
            human_texts=human_texts
        )
        all_results.append(results)
    
    # ========================================================================
    # SAVE RESULTS (append new results to existing files, avoid duplicates)
    # ========================================================================
    print("\n" + "="*80)
    print("💾 SAVING RESULTS")
    print("="*80)

    os.makedirs("results/summary", exist_ok=True)
    csv_path = "results/summary/comprehensive_metrics_fimgemma.csv"
    json_path = "results/summary/comprehensive_metrics_fimgemma.json"

    # Load existing results again to ensure up-to-date
    existing_rows = []
    if os.path.exists(csv_path):
        try:
            df_existing = pd.read_csv(csv_path)
            existing_rows = df_existing.to_dict(orient="records")
        except Exception as e:
            print(f"⚠️  Failed to reload existing CSV: {e}")
    elif os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                existing_rows = json.load(f)
        except Exception as e:
            print(f"⚠️  Failed to reload existing JSON: {e}")

    # Combine and deduplicate
    combined_results = existing_rows + all_results
    # Deduplicate by (Experiment, Generation, Model_Path)
    seen = set()
    deduped_results = []
    for row in combined_results:
        key = (str(row.get("Experiment")), int(row.get("Generation", 0)), str(row.get("Model_Path")))
        if key not in seen:
            deduped_results.append(row)
            seen.add(key)

    # Sort for consistency
    df = pd.DataFrame(deduped_results)
    df = df.sort_values(by=["Experiment", "Generation"])

    # Save to CSV (overwrite, but with all previous + new results)
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV: {csv_path}")

    # Save to JSON (overwrite, but with all previous + new results)
    with open(json_path, "w") as f:
        json.dump(deduped_results, f, indent=4)
    print(f"✅ Saved JSON: {json_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    print(f"Total models evaluated: {len(all_results)}")
    print(f"Experiments: {df['Experiment'].nunique()}")
    print(f"Generations per experiment: {df.groupby('Experiment')['Generation'].count().to_dict()}")
    
    # Show key metrics table
    print("\n📈 Key Metrics Preview:")
    summary_cols = ["Experiment", "Generation", "Perplexity", "Shannon_Entropy", "Semantic_Coherence"]
    if "MAUVE_Score" in df.columns:
        summary_cols.append("MAUVE_Score")
    print(df[summary_cols].to_string(index=False))
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")


if __name__ == "__main__":
    main()