"""
Architecture Diagnostic Script
Inspects actual parameter names in Qwen3.5 and Gemma 3 to determine:
  1. What the FIM script actually measured for each block
  2. Whether Gemma's local vs global attention layers are distinguishable
  3. Whether Qwen3.5 DeltaNet blocks have capturable parameters

Run this BEFORE deciding whether to rerun FIM.
Runs on CPU, no training, takes ~5-10 minutes total.
"""

import torch
from transformers import AutoModelForCausalLM, AutoConfig
from collections import defaultdict
import re

# ============================================================
# CONFIG — adjust paths if needed
# ============================================================
MODELS = {
    "Falcon-7B": "tiiuae/falcon-7b",
}

# Keywords from perblock_fim.py — exact copy
ATTN_KEYWORDS = ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj',
                 'o_proj', 'c_attn', 'c_proj']
MLP_KEYWORDS  = ['mlp', 'fc', 'up_proj', 'down_proj', 'gate_proj', 'c_fc']


def get_block_idx(name):
    match = re.search(r'\.(layers|h|blocks)\.(\d+)\.', name)
    if match:
        return int(match.group(2))
    return None


def classify_param(name):
    name_lower = name.lower()
    if any(x in name_lower for x in
           ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
            'self_attn.o_proj', 'attn.c_attn', 'attn.c_proj']):
        return 'attn_exact'
    elif any(x in name_lower for x in
             ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj',
              'mlp.c_fc', 'mlp.c_proj']):
        return 'mlp_exact'
    elif any(x in name_lower for x in ATTN_KEYWORDS):
        return 'attn_fallback'
    elif any(x in name_lower for x in MLP_KEYWORDS):
        return 'mlp_fallback'
    else:
        return 'other'


def diagnose_model(model_name, model_id):
    print(f"\n{'='*70}")
    print(f"DIAGNOSING: {model_name}  ({model_id})")
    print(f"{'='*70}")

    # Load config first to check for special attributes
    config = AutoConfig.from_pretrained(model_id)
    print(f"\nConfig model_type: {config.model_type}")

    # Check for layer_types (Qwen3.5 specific)
    if hasattr(config, 'layer_types'):
        print(f"layer_types: {config.layer_types}")
    if hasattr(config, 'sliding_window'):
        print(f"sliding_window: {config.sliding_window}")
    if hasattr(config, 'attn_implementation'):
        print(f"attn_implementation: {config.attn_implementation}")

    # Load model weights (CPU, no need for GPU)
    print(f"\nLoading model weights (CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    # Collect all parameter names per block
    block_params = defaultdict(list)
    unblocked_params = []

    for name, param in model.named_parameters():
        if param.ndim < 2:
            continue
        block_idx = get_block_idx(name)
        classification = classify_param(name)
        if block_idx is not None:
            block_params[block_idx].append((name, classification, param.numel()))
        else:
            unblocked_params.append((name, classification))

    num_blocks = max(block_params.keys()) + 1 if block_params else 0
    print(f"Total blocks detected: {num_blocks}")

    # Per-block summary
    print(f"\n{'Block':<7} {'Attn_Exact':>12} {'Attn_Fall':>10} "
          f"{'MLP_Exact':>10} {'MLP_Fall':>10} {'Other':>8} {'FIM_would_measure'}")
    print("-" * 75)

    problem_blocks = []
    empty_blocks = []

    for b in range(num_blocks):
        params = block_params.get(b, [])
        counts = defaultdict(int)
        param_counts = defaultdict(int)  # total params per category

        for name, cls, numel in params:
            counts[cls] += 1
            param_counts[cls] += numel

        attn_e = counts['attn_exact']
        attn_f = counts['attn_fallback']
        mlp_e  = counts['mlp_exact']
        mlp_f  = counts['mlp_fallback']
        other  = counts['other']

        # What would FIM actually measure?
        if attn_e > 0 and mlp_e > 0:
            fim_status = "✅ correct (exact match)"
        elif attn_e == 0 and attn_f == 0 and mlp_e > 0:
            fim_status = "⚠️  MLP only (no attn found)"
            problem_blocks.append(b)
        elif attn_e == 0 and attn_f > 0:
            fim_status = "⚠️  fallback attn (may be wrong)"
            problem_blocks.append(b)
        elif attn_e == 0 and attn_f == 0 and mlp_e == 0:
            fim_status = "❌ EMPTY — script would crash/skip"
            empty_blocks.append(b)
        else:
            fim_status = f"? mixed ({attn_e}ae,{attn_f}af,{mlp_e}me)"

        print(f"  {b:<5} {attn_e:>12} {attn_f:>10} {mlp_e:>10} "
              f"{mlp_f:>10} {other:>8}   {fim_status}")

    # Show unique parameter name patterns for problem blocks
    if problem_blocks or empty_blocks:
        print(f"\n⚠️  PROBLEM BLOCKS: {sorted(set(problem_blocks + empty_blocks))}")
        print("\nSample parameter names from first problem block:")
        first_prob = sorted(set(problem_blocks + empty_blocks))[0]
        for name, cls, numel in block_params.get(first_prob, [])[:15]:
            print(f"  [{cls:>14}] {name}  ({numel:,} params)")

    # Also show a clean block for comparison
    clean_blocks = [b for b in range(num_blocks)
                    if b not in problem_blocks and b not in empty_blocks]
    if clean_blocks:
        print(f"\nSample parameter names from clean block {clean_blocks[0]}:")
        for name, cls, numel in block_params.get(clean_blocks[0], [])[:10]:
            print(f"  [{cls:>14}] {name}  ({numel:,} params)")

    # Gemma: check if local vs global is distinguishable
    if model_name == "Gemma":
        print(f"\nGEMMA SLIDING WINDOW CHECK:")
        if hasattr(config, 'sliding_window_pattern'):
            print(f"  sliding_window_pattern: {config.sliding_window_pattern}")
        # Check if config has layer-level attention type info
        for attr in ['layer_types', 'attn_types', 'attention_types',
                     'sliding_window_pattern', 'attn_logit_softcapping',
                     'query_pre_attn_scalar']:
            if hasattr(config, attr):
                print(f"  {attr}: {getattr(config, attr)}")

        # Try to find if any parameter distinguishes local from global
        print("\n  Checking if local/global distinction exists in param names...")
        all_attn_names = []
        for b in range(num_blocks):
            for name, cls, _ in block_params.get(b, []):
                if 'attn' in name.lower() or 'attention' in name.lower():
                    short = name.split('.')[-2] + '.' + name.split('.')[-1]
                    if short not in all_attn_names:
                        all_attn_names.append(short)
        print(f"  Unique attn param suffixes: {all_attn_names}")
        print("\n  NOTE: If all blocks share identical param names,")
        print("  local vs global is ONLY distinguishable via config,")
        print("  not from parameter structure alone.")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"\n{'='*70}")
    print(f"DONE: {model_name}")
    print(f"{'='*70}")


# ============================================================
# RUN
# ============================================================
for model_name, model_id in MODELS.items():
    try:
        diagnose_model(model_name, model_id)
    except Exception as e:
        print(f"\nERROR diagnosing {model_name}: {e}")
        import traceback
        traceback.print_exc()

print("\n\nFINAL SUMMARY:")
print("  Run complete. Check each model's ✅/⚠️/❌ status above.")
print("  ✅ = FIM script measured correct params")
print("  ⚠️ = FIM script ran but may have measured wrong/mixed params")
print("  ❌ = FIM script found nothing — block was skipped or crashed")
