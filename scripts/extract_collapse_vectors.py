import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import gc

# Strict GPU check
assert torch.cuda.is_available(), "CUDA is not available."
DEVICE = "cuda"
print(f"Using: {torch.cuda.get_device_name(0)}")

BASE_DIR = r"D:\Thaman\Work\hessian-spectral-analysis\models"

# Master configuration with your exact IDs and folder mappings
MODEL_CONFIGS = {
    "llama": {
        "hf_id": "meta-llama/Llama-3.2-1B",
        "folder_prefix": "llama_treatment_gen_",
        "layer_prefix": "model.layers"
    },
    "gpt2": {
        "hf_id": "gpt2",
        "folder_prefix": "gpt2_treatment_gen_",
        "layer_prefix": "transformer.h"
    },
    "smollm": {
        "hf_id": "HuggingFaceTB/SmolLM2-135M",
        "folder_prefix": "treatment_gen_",
        "layer_prefix": "model.layers"
    },
    "qwen2_5": {
        "hf_id": "Qwen/Qwen2.5-0.5B",
        "folder_prefix": "control_c_gen_", 
        "layer_prefix": "model.layers"
    },
    "qwen3_5": {
        "hf_id": "Qwen/Qwen3.5-0.8B",
        "folder_prefix": "Qwen3.5-0.8B_treatment_gen_",
        "layer_prefix": "model.layers"
    },
    "gemma": {
        "hf_id": "google/gemma-3-1b-it",
        "folder_prefix": "gemma_treatment_gen_",
        "layer_prefix": "model.layers"
    }
}

PROMPTS = [
    "The most important scientific discovery of the century is",
    "In a shocking turn of events, the government decided to",
    "To build a fully functional web application, you need to",
    "The history of the Roman Empire teaches us that",
    "Once upon a time in a distant galaxy, there was a"
]

def get_span_target_layers(hf_id, folder_prefix, layer_prefix, start_pct=0.50, end_pct=0.80):
    """Calculates a block of layers with a manual override for new models."""
    # MANUAL OVERRIDE for Qwen 3.5 or other unrecognized IDs
    if "qwen3_5" in hf_id.lower() or "0.8b" in hf_id.lower():
        total_layers = 24
    else:
        try:
            config = AutoConfig.from_pretrained(hf_id)
            if hasattr(config, "num_hidden_layers"):
                total_layers = config.num_hidden_layers
            elif hasattr(config, "n_layer"):
                total_layers = config.n_layer
            else:
                total_layers = 24 # Final fallback
        except:
            total_layers = 24

    start_idx = int(total_layers * start_pct)
    end_idx = int(total_layers * end_pct)
    layer_paths = [f"{layer_prefix}.{i}" for i in range(start_idx, end_idx + 1)]
    return layer_paths, total_layers, start_idx, end_idx

def get_span_activations(model, tokenizer, prompts, layer_paths):
    """Hooks multiple layers and flattens result into a 1D vector."""
    activations = []
    hooks = []
    
    def get_hook_fn():
        def hook_fn(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            # [1, seq_len, hidden_dim] -> [hidden_dim]
            activations.append(hidden_states[0, -1, :].detach())
        return hook_fn

    for l_path in layer_paths:
        layer = dict(model.named_modules())[l_path]
        hooks.append(layer.register_forward_hook(get_hook_fn()))

    model.eval()
    prompt_vectors = []
    
    with torch.no_grad():
        for text in prompts:
            activations.clear()
            inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
            model(**inputs)
            # Average across the layers for this prompt
            prompt_vectors.append(torch.stack(activations).mean(dim=0))
            
    for h in hooks: h.remove()
    # Average across all prompts and ensure it is 1D
    return torch.stack(prompt_vectors).mean(dim=0).squeeze()

# ==========================================
# EXECUTION LOOP
# ==========================================
summary_results = {}

for model_name, config in MODEL_CONFIGS.items():
    print(f"\n{'='*50}\nANALYZING SPAN: {model_name.upper()}\n{'='*50}")
    try:
        l_paths, tot, start, end = get_span_target_layers(config['hf_id'], config['folder_prefix'], config['layer_prefix'])
        print(f"  -> {tot} layers found. Using span {start}-{end}.")

        tokenizer = AutoTokenizer.from_pretrained(config['hf_id'])
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

        print(f"Loading Gen 0...")
        base_model = AutoModelForCausalLM.from_pretrained(config['hf_id'], torch_dtype=torch.float16).to(DEVICE)
        base_vec = get_span_activations(base_model, tokenizer, PROMPTS, l_paths)
        del base_model
        gc.collect(); torch.cuda.empty_cache()

        vectors = {}
        for gen in range(1, 6):
            f_name = f"{config['folder_prefix']}{gen}"
            p = os.path.join(BASE_DIR, f_name)
            if not os.path.exists(p): continue
                
            print(f"Loading Gen {gen}...")
            m = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float16, local_files_only=True).to(DEVICE)
            vectors[f"Gen_{gen}"] = get_span_activations(m, tokenizer, PROMPTS, l_paths) - base_vec
            del m
            gc.collect(); torch.cuda.empty_cache()

        if "Gen_1" in vectors and "Gen_5" in vectors:
            v1, v5 = vectors["Gen_1"], vectors["Gen_5"]
            summary_results[model_name] = {
                "Mag1": torch.norm(v1).item(),
                "Mag5": torch.norm(v5).item(),
                "Sim": F.cosine_similarity(v1.unsqueeze(0), v5.unsqueeze(0)).item()
            }
    except Exception as e:
        print(f"Error: {e}")
        summary_results[model_name] = {"Error": str(e)}

# ==========================================
# SUMMARY
# ==========================================
print("\n" + "="*65)
print(f"{'MODEL':<12} | {'GEN 1 MAG':<10} | {'GEN 5 MAG':<10} | {'SPAN SIM'}")
print("="*65)
for m, s in summary_results.items():
    if "Error" in s: print(f"{m.upper():<12} | {s['Error']}")
    else: print(f"{m.upper():<12} | {s['Mag1']:<10.4f} | {s['Mag5']:<10.4f} | {s['Sim']:.4f}")