"""
Per-Block Layer-wise Hessian Analysis
Analyzes each transformer block's attention and MLP separately.

This gives spatial resolution of where collapse happens in the network.
"""

import torch
import json
import argparse
from pathlib import Path
from torch.autograd import grad
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.stats import gaussian_kde
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# ============================================================================
# Core Hessian Computation (Simple, No Bugs)
# ============================================================================

def get_hessian_eigenvalues(model, loss_fn, dataloader, num_batches, device, 
                            n_eigenvalues, param_list, compute_density=False):
    """
    Compute top Hessian eigenvalues for a specific parameter list.
    
    Uses a PyTorch-native Lanczos implementation to avoid SciPy/ARPACK 
    integer overflow issues on large layers.
    """
    from scipy.stats import gaussian_kde
    import numpy as np
    import torch
    
    num_params = sum(p.numel() for p in param_list)
    
    # --- Configuration ---
    # We need 'm' Lanczos vectors to find 'k' eigenvalues. 
    # Rule of thumb: m >= 2*k.
    
    if compute_density:
        target_k = 100
        lanczos_m = 120 
    else:
        target_k = n_eigenvalues
        lanczos_m = max(20, 2 * n_eigenvalues)

    # Safety for massive layers (Output layer ~136M params)
    if num_params > 100_000_000:
        lanczos_m = min(lanczos_m, 20) # Hard cap for safety on 24GB cards
        target_k = min(target_k, lanczos_m - 2)
        print(f"    -> Large layer detected. Capping Lanczos steps to {lanczos_m} to save VRAM.")

    def hessian_vector_product(vector):
        """Compute H*v by iterating through batches"""
        hvp_sum = None
        count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            model.zero_grad()
            inputs = batch['input_ids'].to(device)
            labels = batch['input_ids'].to(device)
            
            output = model(inputs)
            logits = output.logits if hasattr(output, 'logits') else output
            
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # First derivative
            grads = torch.autograd.grad(loss, param_list, create_graph=True, retain_graph=True)
            flat_grad = torch.cat([g.contiguous().view(-1) for g in grads])
            
            # Dot product
            gvp = torch.sum(flat_grad * vector)
            
            # Second derivative (HVP)
            hvp = torch.autograd.grad(gvp, param_list, retain_graph=False)
            hvp_flat = torch.cat([h.contiguous().view(-1) for h in hvp])
            
            if hvp_sum is None:
                hvp_sum = hvp_flat
            else:
                hvp_sum += hvp_flat
            count += 1
            
            # Aggressive cleanup
            del grads, flat_grad, gvp, hvp, hvp_flat, loss, logits, output
            
        return (hvp_sum / count) if count > 0 else hvp_sum

    # --- PyTorch Native Lanczos Iteration ---
    
    print(f"    - Running PyTorch Native Lanczos (m={lanczos_m})...")
    
    # 1. Initialize storage
    # T is the tridiagonal matrix (small, on CPU)
    T = torch.zeros(lanczos_m, lanczos_m)
    # V is the basis vectors (large, on GPU)
    V = torch.zeros(lanczos_m, num_params, device=device)
    
    # 2. Initial random vector
    v_start = torch.randn(num_params, device=device)
    v_start = v_start / torch.norm(v_start)
    V[0] = v_start
    
    # 3. Iteration
    beta = 0
    
    for j in range(lanczos_m - 1):
        # w = H * v_j
        w = hessian_vector_product(V[j])
        
        # alpha = w . v_j
        alpha = torch.dot(w, V[j])
        
        # Orthogonalize: w = w - alpha * V[j]
        w = w - alpha * V[j]
        if j > 0:
            w = w - beta * V[j-1]
        
        # Re-orthogonalization (Gram-Schmidt)
        for i in range(j + 1):
            proj = torch.dot(w, V[i])
            w = w - proj * V[i]
            
        beta = torch.norm(w)
        
        # Update Tridiagonal Matrix T (CPU side is fine)
        T[j, j] = alpha.cpu()
        T[j, j+1] = beta.cpu()
        T[j+1, j] = beta.cpu()
        
        if beta < 1e-6:
            print("    -> Lanczos converged early.")
            break
            
        V[j+1] = w / beta
        
        # Clean up w
        del w
        
        # FIXED: Use 'j' instead of 'batch_idx'
        if j % 5 == 0:
            torch.cuda.empty_cache()

    # 4. Solve the small tridiagonal system
    T_final = T[:lanczos_m-1, :lanczos_m-1].numpy()
    eigvals, _ = np.linalg.eigh(T_final)
    
    # Sort descending
    eigvals = np.sort(eigvals)[::-1]
    
    # Select top k
    top_k_eigs = eigvals[:target_k]
    
    # --- Density Estimation ---
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

    # Clean up massive V tensor immediately
    del V
    torch.cuda.empty_cache()

    return top_k_eigs, density_x, density_y

# ============================================================================
# Architecture Detection
# ============================================================================

def get_model_blocks(model):
    """
    Get transformer blocks for different architectures.
    
    Returns:
        (blocks, architecture_name)
    """
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT2, GPT-Neo
        return model.transformer.h, 'gpt2'
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Llama, Mistral, Qwen2, SmolLM
        model_type = model.config.model_type.lower()
        return model.model.layers, model_type
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        # OPT
        return model.model.decoder.layers, 'opt'
    else:
        raise ValueError("Unknown architecture - cannot find transformer blocks")


def get_block_params(block, arch, param_type):
    """
    Get parameters for a specific block and type.
    
    Args:
        block: Single transformer block
        arch: Architecture name
        param_type: 'attention' or 'mlp'
    
    Returns:
        List of parameters
    """
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
    
    else:
        raise ValueError(f"Unknown param_type: {param_type}")
    
    # Collect parameters matching keywords
    for name, param in block.named_parameters():
        if any(kw in name for kw in keywords):
            params.append(param)
    
    return params


def get_output_params(model):
    """Get output layer parameters (handles tied embeddings)"""
    tied = getattr(model.config, 'tie_word_embeddings', False)
    params = []
    
    if tied:
        # Look for embed_tokens
        keyword = 'embed_tokens'
    else:
        # Look for lm_head
        keyword = 'lm_head'
    
    for name, param in model.named_parameters():
        if keyword in name:
            params.append(param)
    
    return params


# ============================================================================
# Main Analysis
# ============================================================================

def analyze_perblock(model_path, output_dir, num_batches=5, num_eigenvalues=20, 
                     device='auto', disable_flash_attn=False, compute_density=False,
                     force=False, skip_output=False):
    """
    Run per-block Hessian analysis.
    
    Args:
        force: If True, ignore existing results and recompute everything
        skip_output: If True, skip the output layer (useful for large models)
        compute_density: If True, compute eigenvalue density (slower)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "perblock_hessian.json"
    
    # Load existing results if present and not forcing
    existing_results = None
    if output_file.exists() and not force:
        print(f"📂 Found existing results: {output_file}")
        try:
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
            print(f"   Loaded {len(existing_results.get('blocks', []))} completed blocks")
            print(f"   Will skip completed blocks and resume from failures")
        except Exception as e:
            print(f"   Warning: Could not load existing results: {e}")
            existing_results = None
    elif force:
        print(f"🔄 Force mode: Ignoring existing results")
    
    # Device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    if device == 'cpu':
        print("⚠️  CPU mode - will be slow")
        disable_flash_attn = True
    
    # Load model
    print(f"📦 Loading: {model_path}")
    model_kwargs = {'torch_dtype': torch.float32}
    if disable_flash_attn:
        model_kwargs['attn_implementation'] = 'eager'
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model = model.to(device)
    model.eval()
    
    # Get architecture info
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
    
    # Initialize results structure
    if existing_results:
        results = existing_results
        # Ensure structure exists
        if 'blocks' not in results:
            results['blocks'] = []
        if 'architecture' not in results:
            results['architecture'] = arch
        if 'num_blocks' not in results:
            results['num_blocks'] = num_blocks
    else:
        results = {
            'architecture': arch,
            'num_blocks': num_blocks,
            'blocks': [],
            'output': None
        }
    
    # Helper to check if block analysis is complete
    def is_complete(block_data, layer_type):
        if not block_data or layer_type not in block_data:
            return False
        layer_data = block_data[layer_type]
        if not layer_data or 'error' in layer_data:
            return False
        return 'eigenvalues' in layer_data and layer_data['eigenvalues']
    
    # Analyze each block
    total_analyses = num_blocks * 2 + (0 if skip_output else 1)
    current = 0
    
    for block_idx in range(num_blocks):
        block = blocks[block_idx]
        
        # Find or create block results
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
            print(f"[{current+1}/{total_analyses}] Block {block_idx} - Attention")
            print(f"{'='*60}")
            current += 1
            
            try:
                attn_params = get_block_params(block, arch, 'attention')
                if attn_params:
                    num_params = sum(p.numel() for p in attn_params)
                    print(f"  Parameters: {num_params:,}")
                    
                    eigenvalues, density_x, density_y = get_hessian_eigenvalues(
                        model, loss_fn, dataloader, num_batches, device,
                        num_eigenvalues, attn_params, compute_density
                    )
                    
                    # Sort descending
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
                    print("  ⚠️  No attention parameters found")
                    block_results['attention'] = {'error': 'No parameters found'}
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                block_results['attention'] = {'error': str(e)}
            
            # Save after attention
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            print(f"⏭️  Block {block_idx} attention already complete, skipping")
            current += 1
        
        # MLP
        if not is_complete(block_results, 'mlp'):
            print(f"\n{'='*60}")
            print(f"[{current+1}/{total_analyses}] Block {block_idx} - MLP")
            print(f"{'='*60}")
            current += 1
            
            try:
                mlp_params = get_block_params(block, arch, 'mlp')
                if mlp_params:
                    num_params = sum(p.numel() for p in mlp_params)
                    print(f"  Parameters: {num_params:,}")
                    
                    eigenvalues, density_x, density_y = get_hessian_eigenvalues(
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
                    print("  ⚠️  No MLP parameters found")
                    block_results['mlp'] = {'error': 'No parameters found'}
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                block_results['mlp'] = {'error': str(e)}
            
            # Save after MLP
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            print(f"⏭️  Block {block_idx} MLP already complete, skipping")
            current += 1
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Output layer
    if not skip_output:
        # Check if output already complete
        output_complete = (results.get('output') and 
                          'eigenvalues' in results.get('output', {}) and
                          results['output']['eigenvalues'] and
                          'error' not in results.get('output', {}))
        
        if not output_complete:
            print(f"\n{'='*60}")
            print(f"[{total_analyses}/{total_analyses}] Output Layer")
            print(f"{'='*60}")
            
            try:
                output_params = get_output_params(model)
                if output_params:
                    num_params = sum(p.numel() for p in output_params)
                    print(f"  Parameters: {num_params:,}")
                    
                    print(f"  Computing eigenvalues...")
                    eigenvalues, density_x, density_y = get_hessian_eigenvalues(
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
                    
                    print(f"  ✅ λ_max={eigs_sorted[0]:.2f}, λ_med={np.median(eigs_sorted):.2f}")
                else:
                    print("  ⚠️  No output parameters found")
                    results['output'] = {'error': 'No parameters found'}
            except RuntimeError as e:
                error_msg = str(e)
                print(f"  ❌ RuntimeError: {error_msg}")
                if "out of memory" in error_msg.lower():
                    print(f"  💡 Try: --skip-output or reduce --num_batches")
                import traceback
                traceback.print_exc()
                results['output'] = {'error': error_msg, 'error_type': 'RuntimeError'}
            except Exception as e:
                print(f"  ❌ Error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                results['output'] = {'error': str(e), 'error_type': type(e).__name__}
            
            # Save after output
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            print(f"⏭️  Output layer already complete, skipping")
    else:
        print(f"\n⏭️  Skipping output layer (--skip-output flag)")
        if 'output' not in results or results['output'] is None:
            results['output'] = {'skipped': True}
    
    # Final save
    output_file = output_dir / "perblock_hessian.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Complete! Saved to: {output_file}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for block_data in results['blocks']:
        idx = block_data['block_idx']
        attn = block_data['attention']
        mlp = block_data['mlp']
        
        attn_str = f"λ={attn['top']:7.1f}" if attn and 'top' in attn else "Error/None"
        mlp_str = f"λ={mlp['top']:7.1f}" if mlp and 'top' in mlp else "Error/None"
        
        print(f"Block {idx:2d}  Attn: {attn_str:15s}  MLP: {mlp_str:15s}")
    
    if results['output'] and 'top' in results['output']:
        print(f"Output     λ={results['output']['top']:7.1f}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-Block Hessian Analysis")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_batches', type=int, default=5)
    parser.add_argument('--num_eigenvalues', type=int, default=20)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--disable_flash_attn', action='store_true',
                       help='Disable Flash Attention (use standard attention)')
    parser.add_argument('--compute_density', action='store_true',
                       help='Compute eigenvalue density (computes 100 eigenvalues, slower)')
    parser.add_argument('--force', action='store_true',
                       help='Force recompute all blocks, ignore existing results')
    parser.add_argument('--skip_output', action='store_true',
                       help='Skip output layer (useful for large models where output layer OOMs)')
    
    args = parser.parse_args()
    
    analyze_perblock(
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