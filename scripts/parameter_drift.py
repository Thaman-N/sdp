"""
Parameter Drift Analysis (Block-wise Relative Frobenius Norm)
=============================================================
Default: combined attn + MLP drift per block.
--split-mqa: additionally splits attention into Q-only vs KV-only
             for diagnosing MQA models like Falcon-7B.

Usage:
  # Default:
  python parameter_drift.py \
      --base "tiiuae/falcon-7b" \
      --target path/to/falcon-7b_treatment_gen_1 \
      --out results/drift_gen_1.json

  # With MQA split:
  python parameter_drift.py \
      --base "tiiuae/falcon-7b" \
      --target path/to/falcon-7b_treatment_gen_1 \
      --out results/drift_gen_1_split.json \
      --split-mqa
"""

import os, re, json, argparse
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM


def patch_falcon_config(config):
    """
    Patch missing attributes that newer transformers versions expect
    but old Falcon checkpoints don't have.
    """
    bridges = {
        "max_position_embeddings":  ("n_positions",            2048),
        "n_positions":              ("max_position_embeddings", 2048),
        "num_attention_heads":      ("n_head",                  71),
        "n_head":                   ("num_attention_heads",     71),
        "num_kv_heads":             ("n_head_kv",               1),
        "n_head_kv":                ("num_kv_heads",            1),
        "ffn_hidden_size":          (None,                      18176),
        "hidden_act":               ("activation",             "gelu"),
        "activation":               ("hidden_act",             "gelu"),
        "parallel_attn":            (None,                      True),
        "num_ln_in_parallel_attn":  (None,                      1),
        "new_decoder_architecture": (None,                      False),
    }
    for attr, (alt, default) in bridges.items():
        if not hasattr(config, attr):
            val = getattr(config, alt, default) if alt else default
            setattr(config, attr, val)
    return config


def load_model(path, is_local=False):
    """
    Load model robustly for Falcon.
    
    For the BASE model (HuggingFace ID): use trust_remote_code=True
    which loads the legacy Falcon code bundled with the checkpoint.
    This is self-consistent and avoids the native FalconConfig mismatch.
    
    For the TARGET model (local saved checkpoint): we must NOT pass any
    external config — doing so causes transformers to reinitialise weights
    randomly rather than loading them from the checkpoint. Instead we:
    1. Load the config.json from the local directory
    2. Patch any missing attributes in-place
    3. Pass the patched local config to from_pretrained
    This preserves the checkpoint's weight loading while satisfying
    the newer transformers attribute requirements.
    """
    if not is_local:
        # Base model from HuggingFace — use legacy remote code path
        return AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    else:
        # Local saved checkpoint — patch its own config, don't override it
        from transformers import AutoConfig
        local_config = AutoConfig.from_pretrained(
            os.path.abspath(path),
            trust_remote_code=False,  # needed to read Falcon config class
        )
        local_config = patch_falcon_config(local_config)

        return AutoModelForCausalLM.from_pretrained(
            os.path.abspath(path),
            config=local_config,          # patched local config
            torch_dtype=torch.float16,
            device_map="cpu",
            local_files_only=True,
            low_cpu_mem_usage=True,
            # no trust_remote_code: use native transformers Falcon
            # which matches what the patched config now expects
        )


def get_layer_info(name):
    match = re.search(r'\.(layers|h|blocks)\.(\d+)\.', name)
    if not match:
        return None, None

    block_idx = int(match.group(2))
    n = name.lower()

    if any(x in n for x in ['embed', 'lm_head', 'norm', 'ln_',
                              'layernorm', 'word_embed']):
        return None, None

    if any(x in n for x in ['mlp', 'fc', 'up_proj', 'down_proj',
                              'gate_proj', 'c_fc', 'ffn', 'feed_forward']):
        return block_idx, 'mlp'

    if 'query_key_value' in n or ('c_attn' in n and 'c_proj' not in n):
        return block_idx, 'attn_fused'

    if any(x in n for x in ['q_proj', 'q_attn']):
        return block_idx, 'attn_q'
    if any(x in n for x in ['k_proj', 'v_proj']):
        return block_idx, 'attn_kv'
    if any(x in n for x in ['o_proj', 'out_proj', 'c_proj', 'dense']):
        return block_idx, 'attn_out'
    if any(x in n for x in ['attn', 'attention']):
        return block_idx, 'attn'

    return block_idx, 'other'


def compute_drift(base_path, target_path, output_path, split_mqa=False):
    # Falcon-7B MQA dimensions
    N_HEADS    = 71
    N_KV_HEADS = 1
    HEAD_DIM   = 64   # 4544 / 71 = 64
    Q_SZ       = N_HEADS    * HEAD_DIM   # 4544
    KV_SZ      = N_KV_HEADS * HEAD_DIM  # 64
    TOTAL      = Q_SZ + 2 * KV_SZ       # 4672

    print(f"Loading base model:   {base_path}")
    model_0 = load_model(base_path, is_local=False)
    print(f"Base model loaded. Verifying weights are not random...")
    # Sanity check: first param should not be near-zero mean for a real model
    first_p = next(model_0.parameters())
    print(f"  First param mean abs: {first_p.float().abs().mean().item():.4f} "
          f"(should be ~0.01-0.05 for real weights, ~0.08 for random fp16)")

    print(f"\nLoading target model: {target_path}")
    model_x = load_model(target_path, is_local=True)
    print(f"Target model loaded. Verifying weights differ from base...")
    dict_0 = dict(model_0.named_parameters())
    dict_x = dict(model_x.named_parameters())
    # Quick check: first shared param should differ
    first_name = next(iter(dict_0))
    if first_name in dict_x:
        diff = (dict_x[first_name].float() - dict_0[first_name].float()).abs().mean().item()
        base_scale = dict_0[first_name].float().abs().mean().item()
        print(f"  First param mean abs diff: {diff:.6f} vs base scale {base_scale:.4f}")
        if diff < 1e-6:
            print("  WARNING: Models appear identical — target may not have loaded correctly!")
        elif diff > base_scale * 0.5:
            print("  WARNING: Diff very large relative to base — possible random init!")
        else:
            print("  OK: Models differ by expected small amount.")

    stats = defaultdict(lambda: {k: 0.0 for k in [
        'attn_diff_sq', 'attn_base_sq',
        'mlp_diff_sq',  'mlp_base_sq',
        'q_diff_sq',    'q_base_sq',
        'kv_diff_sq',   'kv_base_sq',
        'out_diff_sq',  'out_base_sq',
    ]})

    print("\nComputing drift...")
    with torch.no_grad():
        for name, p0 in dict_0.items():
            if name not in dict_x:
                continue
            px = dict_x[name]
            block_idx, comp = get_layer_info(name)
            if block_idx is None or comp in ('other', None):
                continue

            p0f = p0.float()
            pxf = px.float()
            diff_sq = torch.sum((pxf - p0f) ** 2).item()
            base_sq = torch.sum(p0f ** 2).item()
            s = stats[block_idx]

            if comp == 'mlp':
                s['mlp_diff_sq'] += diff_sq
                s['mlp_base_sq'] += base_sq

            elif comp == 'attn_fused' and split_mqa:
                # Unpack fused QKV for Falcon-7B
                # Shape: [Q_SZ + KV_SZ + KV_SZ, hidden_dim] = [4672, 4544]
                if pxf.shape[0] >= TOTAL:
                    p0_q  = p0f[:Q_SZ];              px_q  = pxf[:Q_SZ]
                    p0_k  = p0f[Q_SZ:Q_SZ+KV_SZ];   px_k  = pxf[Q_SZ:Q_SZ+KV_SZ]
                    p0_v  = p0f[Q_SZ+KV_SZ:TOTAL];  px_v  = pxf[Q_SZ+KV_SZ:TOTAL]

                    s['q_diff_sq']  += torch.sum((px_q - p0_q)**2).item()
                    s['q_base_sq']  += torch.sum(p0_q**2).item()
                    s['kv_diff_sq'] += torch.sum((px_k - p0_k)**2).item()
                    s['kv_diff_sq'] += torch.sum((px_v - p0_v)**2).item()
                    s['kv_base_sq'] += torch.sum(p0_k**2).item()
                    s['kv_base_sq'] += torch.sum(p0_v**2).item()
                s['attn_diff_sq'] += diff_sq
                s['attn_base_sq'] += base_sq

            elif comp == 'attn_fused':
                # Non-split: treat fused QKV as combined attn
                s['attn_diff_sq'] += diff_sq
                s['attn_base_sq'] += base_sq

            elif comp == 'attn_q':
                s['q_diff_sq']    += diff_sq
                s['q_base_sq']    += base_sq
                s['attn_diff_sq'] += diff_sq
                s['attn_base_sq'] += base_sq

            elif comp == 'attn_kv':
                s['kv_diff_sq']   += diff_sq
                s['kv_base_sq']   += base_sq
                s['attn_diff_sq'] += diff_sq
                s['attn_base_sq'] += base_sq

            elif comp == 'attn_out':
                s['out_diff_sq']  += diff_sq
                s['out_base_sq']  += base_sq
                s['attn_diff_sq'] += diff_sq
                s['attn_base_sq'] += base_sq

            else:
                s['attn_diff_sq'] += diff_sq
                s['attn_base_sq'] += base_sq

    def safe(d, b):
        return float((d**0.5) / (b**0.5)) if b > 0 else 0.0

    results = {"blocks": {}}
    for b in sorted(stats):
        s = stats[b]
        attn = safe(s['attn_diff_sq'], s['attn_base_sq'])
        mlp  = safe(s['mlp_diff_sq'],  s['mlp_base_sq'])
        entry = {"attn_relative_drift": attn, "mlp_relative_drift": mlp}

        if split_mqa:
            q   = safe(s['q_diff_sq'],   s['q_base_sq'])
            kv  = safe(s['kv_diff_sq'],  s['kv_base_sq'])
            out = safe(s['out_diff_sq'], s['out_base_sq'])
            entry.update({
                "attn_q_drift":   q,
                "attn_kv_drift":  kv,
                "attn_out_drift": out,
            })
            print(f"Block {b:2d} | Attn {attn:.4%} "
                  f"(Q {q:.4%}  KV {kv:.4%}  Out {out:.4%}) | MLP {mlp:.4%}")
        else:
            print(f"Block {b:2d} | Attn Drift: {attn:.4%} | MLP Drift: {mlp:.4%}")

        results["blocks"][str(b)] = entry

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base",      required=True,
                   help="Base model path or HuggingFace ID (Gen0)")
    p.add_argument("--target",    required=True,
                   help="Target model path (Gen N)")
    p.add_argument("--out",       required=True,
                   help="Output JSON path")
    p.add_argument("--split-mqa", action="store_true", default=False,
                   help="Split attn drift into Q vs KV (for Falcon MQA diagnosis)")
    args = p.parse_args()
    compute_drift(args.base, args.target, args.out, args.split_mqa)