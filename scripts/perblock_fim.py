"""
Per-Block Fisher Information Matrix (FIM) Analysis
FIM = E[∇log p · ∇log p^T] - measures parameter sensitivity to data
"""

import torch
import json
import argparse
from pathlib import Path
from scipy.stats import gaussian_kde
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# ============================================================================
# Core FIM Computation
# ============================================================================

def get_fim_eigenvalues(model, loss_fn, dataloader, num_batches, device, 
                        n_eigenvalues, param_list, compute_density=False):
    """
    Compute top FIM eigenvalues for a specific parameter list.
    
    FIM = E[∇log p · (∇log p)^T] where gradients are wrt log-likelihood.
    Uses Lanczos to find top eigenvalues without forming the full matrix.
    """
    num_params = sum(p.numel() for p in param_list)
    
    # --- Configuration ---
    if compute_density:
        target_k = 100
        lanczos_m = 120 
    else:
        target_k = n_eigenvalues
        lanczos_m = max(20, 2 * n_eigenvalues)

    # Safety for massive layers
    if num_params > 100_000_000:
        lanczos_m = min(lanczos_m, 20)
        target_k = min(target_k, lanczos_m - 2)
        print(f"    -> Large layer detected. Capping Lanczos steps to {lanczos_m}.")

    def fim_vector_product(vector):
        """
        Compute F*v where F is the Fisher Information Matrix.
        
        F*v = E[(g^T v) * g] where g = ∇log p(y|x, θ)
        
        This is MUCH cheaper than Hessian - only first-order gradients!
        """
        fvp_sum = None
        count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            model.zero_grad()
            inputs = batch['input_ids'].to(device)
            labels = batch['input_ids'].to(device)
            
            output = model(inputs)
            logits = output.logits if hasattr(output, 'logits') else output
            
            # Cross-entropy loss (negative log-likelihood)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # First derivative: g = ∇log p
            grads = torch.autograd.grad(loss, param_list, create_graph=False)
            flat_grad = torch.cat([g.contiguous().view(-1) for g in grads])
            
            # Compute g^T v (scalar)
            gvp = torch.sum(flat_grad * vector)
            
            # FIM-vector product: (g^T v) * g
            fvp = gvp * flat_grad
            
            if fvp_sum is None:
                fvp_sum = fvp
            else:
                fvp_sum += fvp
            count += 1
            
            # Cleanup
            del grads, flat_grad, gvp, fvp, loss, logits, output
            
        return (fvp_sum / count) if count > 0 else fvp_sum

    # --- PyTorch Native Lanczos Iteration ---
    print(f"    - Running Lanczos on FIM (m={lanczos_m})...")
    
    T = torch.zeros(lanczos_m, lanczos_m)
    V = torch.zeros(lanczos_m, num_params, device=device)
    
    # Initial random vector
    v_start = torch.randn(num_params, device=device)
    v_start = v_start / torch.norm(v_start)
    V[0] = v_start
    
    beta = 0
    
    for j in range(lanczos_m - 1):
        # w = F * v_j
        w = fim_vector_product(V[j])
        
        # alpha = w . v_j
        alpha = torch.dot(w, V[j])
        
        # Orthogonalize
        w = w - alpha * V[j]
        if j > 0:
            w = w - beta * V[j-1]
        
        # Re-orthogonalization
        for i in range(j + 1):
            proj = torch.dot(w, V[i])
            w = w - proj * V[i]
            
        beta = torch.norm(w)
        
        # Update tridiagonal matrix
        T[j, j] = alpha.cpu()
        T[j, j+1] = beta.cpu()
        T[j+1, j] = beta.cpu()
        
        if beta < 1e-6:
            print("    -> Lanczos converged early.")
            break
            
        V[j+1] = w / beta
        del w
        
        if j % 5 == 0:
            torch.cuda.empty_cache()

    # Solve tridiagonal system
    T_final = T[:lanczos_m-1, :lanczos_m-1].numpy()
    eigvals, _ = np.linalg.eigh(T_final)
    
    # Sort descending (FIM is PSD, all eigenvalues should be >= 0)
    eigvals = np.sort(eigvals)[::-1]
    top_k_eigs = eigvals[:target_k]
    
    # Density estimation
    density_x, density_y = None, None
    if compute_density and len(eigvals) >= 10:
        try:
            kde = gaussian_kde(eigvals)
            x_min, x_max = eigvals.min(), eigvals.max()
            x_range = x_max - x_min
            if x_range < 1e-6: x_range = 1.0
                
            density_x = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 20)
            density_y = kde(density_x)
        except Exception as e:
            print(f"    Warning: Density calculation failed: {e}")

    del V
    torch.cuda.empty_cache()

    return top_k_eigs, density_x, density_y


# ============================================================================
# Architecture Detection (SAME AS HESSIAN VERSION)
# ============================================================================

def get_model_blocks(model):
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h, 'gpt2'
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        model_type = model.config.model_type.lower()
        return model.model.layers, model_type
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        return model.model.decoder.layers, 'opt'
    else:
        raise ValueError("Unknown architecture")

def get_block_params(block, arch, param_type):
    params = []
    
    if param_type == 'attention':
        if arch == 'gpt2':
            keywords = ['attn.c_attn', 'attn.c_proj']
        elif arch in ['llama', 'mistral', 'smollm', 'smollm2', 'qwen2']:
            keywords = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']
        elif arch == 'opt':
            keywords = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.out_proj']
        else:
            keywords = ['attn', 'attention', 'self_attn']
    
    elif param_type == 'mlp':
        if arch == 'gpt2':
            keywords = ['mlp.c_fc', 'mlp.c_proj']
        elif arch in ['llama', 'mistral', 'smollm', 'smollm2', 'qwen2']:
            keywords = ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
        elif arch == 'opt':
            keywords = ['fc1', 'fc2']
        else:
            keywords = ['mlp', 'ffn']
    
    for name, param in block.named_parameters():
        if any(kw in name for kw in keywords):
            params.append(param)
    
    return params

def get_output_params(model):
    """Get output layer parameters, handling different architectures and weight tying."""
    params = []
    
    # Try different possible output parameter names
    output_keywords = [
        'lm_head.weight',           # Most modern models
        'embed_tokens.weight',      # Some models with weight tying  
        'transformer.wte.weight',   # GPT-2 style
        'wte.weight',              # Alternative GPT-2
        'embed_out.weight',        # Some other variants
    ]
    
    # Collect all parameter names for debugging
    all_param_names = [name for name, _ in model.named_parameters()]
    
    # Try to find output parameters
    for keyword in output_keywords:
        for name, param in model.named_parameters():
            if keyword in name:
                params.append(param)
                print(f"    Found output param: {name} ({param.shape})")
                break
        if params:  # Stop after finding the first match
            break
    
    # If no output parameters found, print debug info
    if not params:
        print("    Debug: No output parameters found")
        print("    Available parameter names containing 'embed', 'lm_head', or 'wte':")
        relevant_names = [name for name in all_param_names 
                         if any(kw in name.lower() for kw in ['embed', 'lm_head', 'wte', 'head'])]
        for name in relevant_names[:10]:  # Show first 10
            print(f"      - {name}")
        if len(relevant_names) > 10:
            print(f"      ... and {len(relevant_names) - 10} more")
    
    return params


# ============================================================================
# Main Analysis (ADAPTED FOR FIM)
# ============================================================================

def analyze_perblock_fim(model_path, output_dir, num_batches=5, num_eigenvalues=20, 
                         device='auto', disable_flash_attn=False, compute_density=False,
                         force=False, skip_output=False):
    """Run per-block FIM analysis."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "perblock_fim.json"  # Changed filename
    
    # Load existing results
    existing_results = None
    if output_file.exists() and not force:
        print(f"📂 Found existing FIM results: {output_file}")
        try:
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
            print(f"   Will resume from existing results")
        except Exception as e:
            print(f"   Warning: Could not load: {e}")
    
    # Device setup
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    
    # Load model
    print(f"📦 Loading: {model_path}")
    model_kwargs = {'torch_dtype': torch.float32}
    if disable_flash_attn:
        model_kwargs['attn_implementation'] = 'eager'
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model = model.to(device)
    model.eval()
    
    blocks, arch = get_model_blocks(model)
    num_blocks = len(blocks)
    print(f"🏗️  Architecture: {arch} | {num_blocks} blocks")
    
    # Prepare data
    print("📚 Loading dataset...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset("roneneldan/TinyStories", split="validation[:200]")
    
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", 
                        max_length=256, return_tensors=None)
    
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format("torch")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Initialize results
    if existing_results:
        results = existing_results
        if 'blocks' not in results:
            results['blocks'] = []
    else:
        results = {
            'architecture': arch,
            'num_blocks': num_blocks,
            'blocks': [],
            'output': None
        }
    
    def is_complete(block_data, layer_type):
        if not block_data or layer_type not in block_data:
            return False
        layer_data = block_data[layer_type]
        if not layer_data or 'error' in layer_data:
            return False
        return 'eigenvalues' in layer_data and layer_data['eigenvalues']
    
    # Analyze blocks
    total_analyses = num_blocks * 2 + (0 if skip_output else 1)
    current = 0
    
    for block_idx in range(num_blocks):
        block = blocks[block_idx]
        
        block_results = None
        for br in results['blocks']:
            if br.get('block_idx') == block_idx:
                block_results = br
                break
        
        if block_results is None:
            block_results = {
                'block_idx': block_idx,
                'attention': None,
                'mlp': None
            }
            results['blocks'].append(block_results)
        
        # Attention
        if not is_complete(block_results, 'attention'):
            print(f"\n{'='*60}")
            print(f"[{current+1}/{total_analyses}] Block {block_idx} - Attention (FIM)")
            print(f"{'='*60}")
            current += 1
            
            try:
                attn_params = get_block_params(block, arch, 'attention')
                if attn_params:
                    num_params = sum(p.numel() for p in attn_params)
                    print(f"  Parameters: {num_params:,}")
                    
                    eigenvalues, density_x, density_y = get_fim_eigenvalues(
                        model, loss_fn, dataloader, num_batches, device,
                        num_eigenvalues, attn_params, compute_density
                    )
                    
                    eigs_sorted = np.sort(eigenvalues)[::-1]
                    
                    block_results['attention'] = {
                        'num_params': num_params,
                        'eigenvalues': eigs_sorted.tolist(),
                        'top': float(eigs_sorted[0]),
                        'bottom': float(eigs_sorted[-1]),
                        'median': float(np.median(eigs_sorted)),
                        'trace': float(eigs_sorted.sum())
                    }
                    
                    if density_x is not None:
                        block_results['attention']['plot_x'] = density_x.tolist()
                        block_results['attention']['plot_y'] = density_y.tolist()
                    
                    print(f"  ✅ λ_max={eigs_sorted[0]:.2f}, λ_med={np.median(eigs_sorted):.2f}")
                else:
                    block_results['attention'] = {'error': 'No parameters'}
            except Exception as e:
                print(f"  ❌ {e}")
                import traceback
                traceback.print_exc()
                block_results['attention'] = {'error': str(e)}
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            print(f"⏭️  Block {block_idx} attention FIM complete")
            current += 1
        
        # MLP
        if not is_complete(block_results, 'mlp'):
            print(f"\n{'='*60}")
            print(f"[{current+1}/{total_analyses}] Block {block_idx} - MLP (FIM)")
            print(f"{'='*60}")
            current += 1
            
            try:
                mlp_params = get_block_params(block, arch, 'mlp')
                if mlp_params:
                    num_params = sum(p.numel() for p in mlp_params)
                    print(f"  Parameters: {num_params:,}")
                    
                    eigenvalues, density_x, density_y = get_fim_eigenvalues(
                        model, loss_fn, dataloader, num_batches, device,
                        num_eigenvalues, mlp_params, compute_density
                    )
                    
                    eigs_sorted = np.sort(eigenvalues)[::-1]
                    
                    block_results['mlp'] = {
                        'num_params': num_params,
                        'eigenvalues': eigs_sorted.tolist(),
                        'top': float(eigs_sorted[0]),
                        'bottom': float(eigs_sorted[-1]),
                        'median': float(np.median(eigs_sorted)),
                        'trace': float(eigs_sorted.sum())
                    }
                    
                    if density_x is not None:
                        block_results['mlp']['plot_x'] = density_x.tolist()
                        block_results['mlp']['plot_y'] = density_y.tolist()
                    
                    print(f"  ✅ λ_max={eigs_sorted[0]:.2f}, λ_med={np.median(eigs_sorted):.2f}")
                else:
                    block_results['mlp'] = {'error': 'No parameters'}
            except Exception as e:
                print(f"  ❌ {e}")
                import traceback
                traceback.print_exc()
                block_results['mlp'] = {'error': str(e)}
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            print(f"⏭️  Block {block_idx} MLP FIM complete")
            current += 1
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Output layer
    if not skip_output:
        output_complete = (results.get('output') and 
                          'eigenvalues' in results.get('output', {}) and
                          'error' not in results.get('output', {}))
        
        if not output_complete:
            print(f"\n{'='*60}")
            print(f"[{total_analyses}/{total_analyses}] Output Layer (FIM)")
            print(f"{'='*60}")
            
            try:
                output_params = get_output_params(model)
                if output_params:
                    num_params = sum(p.numel() for p in output_params)
                    print(f"  Parameters: {num_params:,}")
                    
                    eigenvalues, density_x, density_y = get_fim_eigenvalues(
                        model, loss_fn, dataloader, num_batches, device,
                        num_eigenvalues, output_params, compute_density
                    )
                    
                    eigs_sorted = np.sort(eigenvalues)[::-1]
                    
                    results['output'] = {
                        'num_params': num_params,
                        'eigenvalues': eigs_sorted.tolist(),
                        'top': float(eigs_sorted[0]),
                        'bottom': float(eigs_sorted[-1]),
                        'median': float(np.median(eigs_sorted)),
                        'trace': float(eigs_sorted.sum())
                    }
                    
                    if density_x is not None:
                        results['output']['plot_x'] = density_x.tolist()
                        results['output']['plot_y'] = density_y.tolist()
                    
                    print(f"  ✅ λ_max={eigs_sorted[0]:.2f}")
                else:
                    results['output'] = {'error': 'No parameters'}
            except Exception as e:
                print(f"  ❌ {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                results['output'] = {'error': str(e)}
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            print(f"⏭️  Output FIM complete")
    else:
        print(f"\n⏭️  Skipping output layer")
        if 'output' not in results:
            results['output'] = {'skipped': True}
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Complete! Saved to: {output_file}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-Block FIM Analysis")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_batches', type=int, default=5)
    parser.add_argument('--num_eigenvalues', type=int, default=20)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--disable_flash_attn', action='store_true')
    parser.add_argument('--compute_density', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--skip_output', action='store_true')
    
    args = parser.parse_args()
    
    analyze_perblock_fim(
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_batches=args.num_batches,
        num_eigenvalues=args.num_eigenvalues,
        device=args.device,
        disable_flash_attn=args.disable_flash_attn,
        compute_density=args.compute_density,
        force=args.force,
        skip_output=args.skip_output
    )