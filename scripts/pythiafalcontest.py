"""
Quick generation quality check for Pythia-1.4B
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

MODEL_ID = "tiiuae/falcon-7b"

PROMPTS = [
    "Once upon a time",
    "The scientist discovered",
    "In a world where",
    "The young child asked",
    "Explain the concept of",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Loading {MODEL_ID}...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
).to(device)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.pad_token    = tok.eos_token
tok.pad_token_id = tok.eos_token_id
tok.padding_side = "left"

model.eval()
all_text = ""

print("\n" + "="*70)
print("GENERATED SAMPLES (bfloat16, temp=0.7, top_p=0.95, rep_pen=1.1)")
print("="*70)

with torch.no_grad():
    for i, prompt in enumerate(PROMPTS):
        inputs = tok(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
        text = tok.decode(outputs[0], skip_special_tokens=True)
        all_text += text + " "
        print(f"\n[Sample {i+1}] Prompt: '{prompt}'")
        print("-" * 50)
        print(text)

print("\n" + "="*70)
print("DIVERSITY METRICS")
print("="*70)

tokens = all_text.split()
total  = len(tokens)

if total > 1:
    unique_ratio        = len(set(tokens)) / total
    bigrams             = list(zip(tokens, tokens[1:]))
    unique_bigram_ratio = len(set(bigrams)) / len(bigrams)
    top_tokens          = Counter(tokens).most_common(5)

    print(f"Total tokens generated:    {total}")
    print(f"Unique token ratio:        {unique_ratio:.3f}  (>0.4 healthy, <0.2 degenerate)")
    print(f"Unique bigram ratio:       {unique_bigram_ratio:.3f}  (>0.6 healthy, <0.3 degenerate)")
    print(f"Top 5 repeated tokens:     {top_tokens}")
    print()

    if unique_ratio > 0.35 and unique_bigram_ratio > 0.5:
        print("VERDICT: ✅ HEALTHY — suitable for synthetic training data.")
    elif unique_ratio > 0.2:
        print("VERDICT: ⚠️  BORDERLINE — may produce noisy training signal.")
    else:
        print("VERDICT: ❌ DEGENERATE — high repetition, poor training signal.")