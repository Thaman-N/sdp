#!/usr/bin/env python3
"""
Layer-wise baseline metrics for comparison with FIM analysis
Computes traditional collapse indicators: weight norms, gradient norms, 
effective rank, condition numbers for transformer blocks
"""

import torch
import torch.nn as nn
import numpy as np
import json
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_effective_rank(matrix, threshold=1e-3):
    """Compute effective rank based on singular value threshold"""
    if matrix.numel() == 0:
        return 0
    
    # Handle different tensor shapes
    if len(matrix.shape) > 2:
        matrix = matrix.view(-1, matrix.shape[-1])
    
    try:
        U, s, V = torch.svd(matrix.float())
        # Normalize by largest singular value
        s_normalized = s / (s[0] + 1e-8)
        effective_rank = (s_normalized > threshold).sum().item()
        return effective_rank
    except:
        return 0

def compute_condition_number(matrix):
    """Compute condition number (σ_1/σ_r)"""
    if matrix.numel() == 0:
        return float('inf')
    
    if len(matrix.shape) > 2:
        matrix = matrix.view(-1, matrix.shape[-1])
    
    try:
        s = torch.svd(matrix.float())[1]
        if len(s) == 0 or s[-1] < 1e-8:
            return float('inf')
        return (s[0] / s[-1]).item()
    except:
        return float('inf')

def compute_stable_rank(matrix):
    """Compute stable rank (||A||_F^2 / ||A||_2^2)"""
    if matrix.numel() == 0:
        return 0
    
    frobenius_norm = torch.norm(matrix, 'fro')
    spectral_norm = torch.norm(matrix, 2)
    
    if spectral_norm < 1e-8:
        return 0
    
    return (frobenius_norm**2 / spectral_norm**2).item()

def analyze_transformer_block(block, block_idx):
    """Analyze a single transformer block"""
    metrics = {
        'block_idx': block_idx,
        'attention': {},
        'mlp': {}
    }
    
    # Analyze attention layers
    if hasattr(block, 'self_attn') or hasattr(block, 'attn'):
        attn_layer = getattr(block, 'self_attn', None) or getattr(block, 'attn', None)
        
        # Collect attention weight matrices
        attn_weights = []
        param_names = []
        
        for name, param in attn_layer.named_parameters():
            if 'weight' in name and param.requires_grad:
                attn_weights.append(param.data)
                param_names.append(name)
        
        if attn_weights:
            # Compute metrics for each weight matrix and combine them
            total_fro_norm_sq = 0
            total_params = 0
            all_singular_values = []
            
            for w in attn_weights:
                # Flatten to 2D if needed
                if len(w.shape) > 2:
                    w_2d = w.view(-1, w.shape[-1])
                else:
                    w_2d = w
                
                total_fro_norm_sq += torch.norm(w_2d, 'fro').item() ** 2
                total_params += w_2d.numel()
                
                # Get singular values
                try:
                    _, s, _ = torch.svd(w_2d.float())
                    all_singular_values.extend(s.tolist())
                except:
                    pass
            
            # Combined metrics
            combined_fro_norm = np.sqrt(total_fro_norm_sq)
            
            # For effective rank, use the combined singular values
            if all_singular_values:
                all_singular_values = sorted(all_singular_values, reverse=True)
                s_tensor = torch.tensor(all_singular_values)
                s_normalized = s_tensor / (s_tensor[0] + 1e-8)
                effective_rank = (s_normalized > 1e-3).sum().item()
                spectral_norm = s_tensor[0].item()
                stable_rank = (torch.sum(s_tensor**2) / s_tensor[0]**2).item() if s_tensor[0] > 1e-8 else 0
                condition_number = (s_tensor[0] / s_tensor[-1]).item() if s_tensor[-1] > 1e-8 else float('inf')
            else:
                effective_rank = 0
                spectral_norm = 0
                stable_rank = 0
                condition_number = float('inf')
            
            metrics['attention'] = {
                'frobenius_norm': combined_fro_norm,
                'spectral_norm': spectral_norm,
                'effective_rank': effective_rank,
                'stable_rank': stable_rank,
                'condition_number': condition_number,
                'num_parameters': total_params,
                'param_names': param_names
            }
    
    # Analyze MLP layers
    mlp_attr_names = ['mlp', 'feed_forward', 'ffn']
    mlp_layer = None
    
    for attr_name in mlp_attr_names:
        if hasattr(block, attr_name):
            mlp_layer = getattr(block, attr_name)
            break
    
    if mlp_layer is not None:
        # Collect MLP weight matrices
        mlp_weights = []
        param_names = []
        
        for name, param in mlp_layer.named_parameters():
            if 'weight' in name and param.requires_grad:
                mlp_weights.append(param.data)
                param_names.append(name)
        
        if mlp_weights:
            # Compute metrics for each weight matrix and combine them
            total_fro_norm_sq = 0
            total_params = 0
            all_singular_values = []
            
            for w in mlp_weights:
                # Flatten to 2D if needed
                if len(w.shape) > 2:
                    w_2d = w.view(-1, w.shape[-1])
                else:
                    w_2d = w
                
                total_fro_norm_sq += torch.norm(w_2d, 'fro').item() ** 2
                total_params += w_2d.numel()
                
                # Get singular values
                try:
                    _, s, _ = torch.svd(w_2d.float())
                    all_singular_values.extend(s.tolist())
                except:
                    pass
            
            # Combined metrics
            combined_fro_norm = np.sqrt(total_fro_norm_sq)
            
            # For effective rank, use the combined singular values
            if all_singular_values:
                all_singular_values = sorted(all_singular_values, reverse=True)
                s_tensor = torch.tensor(all_singular_values)
                s_normalized = s_tensor / (s_tensor[0] + 1e-8)
                effective_rank = (s_normalized > 1e-3).sum().item()
                spectral_norm = s_tensor[0].item()
                stable_rank = (torch.sum(s_tensor**2) / s_tensor[0]**2).item() if s_tensor[0] > 1e-8 else 0
                condition_number = (s_tensor[0] / s_tensor[-1]).item() if s_tensor[-1] > 1e-8 else float('inf')
            else:
                effective_rank = 0
                spectral_norm = 0
                stable_rank = 0
                condition_number = float('inf')
            
            metrics['mlp'] = {
                'frobenius_norm': combined_fro_norm,
                'spectral_norm': spectral_norm,
                'effective_rank': effective_rank,
                'stable_rank': stable_rank,
                'condition_number': condition_number,
                'num_parameters': total_params,
                'param_names': param_names
            }
    
    return metrics

def compute_gradient_norms(model, data_loader, device='cuda'):
    """Compute gradient norms per block (requires data)"""
    model.eval()
    gradient_metrics = {}
    
    # Get a sample batch
    try:
        batch = next(iter(data_loader))
        if isinstance(batch, dict):
            input_ids = batch['input_ids'].to(device)
        else:
            input_ids = batch[0].to(device)
        
        # Forward pass
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Extract gradient norms per block
        if hasattr(model, 'transformer'):
            transformer = model.transformer
        elif hasattr(model, 'model'):
            transformer = model.model
        else:
            transformer = model
            
        if hasattr(transformer, 'layers') or hasattr(transformer, 'h'):
            layers = getattr(transformer, 'layers', None) or getattr(transformer, 'h', None)
            
            for block_idx, block in enumerate(layers):
                block_grad_norm = 0.0
                param_count = 0
                
                for param in block.parameters():
                    if param.grad is not None:
                        block_grad_norm += torch.norm(param.grad).item() ** 2
                        param_count += 1
                
                gradient_metrics[f'block_{block_idx}'] = {
                    'gradient_norm': np.sqrt(block_grad_norm),
                    'param_count': param_count
                }
        
        model.zero_grad()  # Clean up
        
    except Exception as e:
        logger.warning(f"Could not compute gradient norms: {e}")
        gradient_metrics = {}
    
    return gradient_metrics

def analyze_model(model_path, tokenizer_path=None, output_path=None, compute_gradients=False):
    """Main analysis function"""
    logger.info(f"Loading model from {model_path}")
    
    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map='auto' if torch.cuda.is_available() else None
        )
        device = next(model.parameters()).device
        logger.info(f"Model loaded on device: {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None
    
    # Get transformer layers
    if hasattr(model, 'transformer'):
        transformer = model.transformer
    elif hasattr(model, 'model'):
        transformer = model.model
    else:
        transformer = model
    
    layers = None
    if hasattr(transformer, 'layers'):
        layers = transformer.layers
    elif hasattr(transformer, 'h'):
        layers = transformer.h
    elif hasattr(transformer, 'blocks'):
        layers = transformer.blocks
    
    if layers is None:
        logger.error("Could not find transformer layers")
        return None
    
    logger.info(f"Found {len(layers)} transformer layers")
    
    # Analyze each block
    results = {
        'model_path': str(model_path),
        'num_blocks': len(layers),
        'blocks': []
    }
    
    for block_idx, block in enumerate(layers):
        logger.info(f"Analyzing block {block_idx}")
        block_metrics = analyze_transformer_block(block, block_idx)
        results['blocks'].append(block_metrics)
    
    # Compute gradient norms if requested
    if compute_gradients:
        logger.info("Computing gradient norms...")
        try:
            # Create simple tokenizer and data for gradient computation
            if tokenizer_path:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Simple text for gradient computation
            text = "The quick brown fox jumps over the lazy dog."
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            
            # Create simple data loader
            class SimpleDataset:
                def __init__(self, inputs):
                    self.inputs = inputs
                
                def __iter__(self):
                    yield self.inputs
            
            data_loader = SimpleDataset(inputs)
            gradient_metrics = compute_gradient_norms(model, data_loader, device)
            results['gradient_norms'] = gradient_metrics
            
        except Exception as e:
            logger.warning(f"Gradient computation failed: {e}")
            results['gradient_norms'] = {}
    
    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    return results

def print_summary(results):
    """Print a summary similar to your FIM output"""
    if not results:
        return
    
    print(f"Model: {Path(results['model_path']).name}, Blocks: {results['num_blocks']}")
    
    for block_data in results['blocks']:
        block_idx = block_data['block_idx']
        
        # Attention metrics
        if 'attention' in block_data and block_data['attention']:
            attn = block_data['attention']
            attn_fro = attn.get('frobenius_norm', 0)
            attn_rank = attn.get('effective_rank', 0)
            attn_cond = attn.get('condition_number', float('inf'))
            print(f"Block {block_idx:2d} | Attn: Fro={attn_fro:10.2f} | Rank={attn_rank:2d} | Cond={attn_cond:8.1f}")
        
        # MLP metrics
        if 'mlp' in block_data and block_data['mlp']:
            mlp = block_data['mlp']
            mlp_fro = mlp.get('frobenius_norm', 0)
            mlp_rank = mlp.get('effective_rank', 0) 
            mlp_cond = mlp.get('condition_number', float('inf'))
            print(f"         | MLP:  Fro={mlp_fro:10.2f} | Rank={mlp_rank:2d} | Cond={mlp_cond:8.1f}")

def main():
    parser = argparse.ArgumentParser(description='Compute baseline layer-wise metrics')
    parser.add_argument('model_path', type=str, help='Path to model')
    parser.add_argument('--tokenizer_path', type=str, help='Path to tokenizer (if different from model)')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    parser.add_argument('--gradients', action='store_true', help='Compute gradient norms (requires data)')
    parser.add_argument('--print_summary', action='store_true', help='Print summary to console')
    
    args = parser.parse_args()
    
    results = analyze_model(
        args.model_path,
        args.tokenizer_path,
        args.output,
        args.gradients
    )
    
    if results and args.print_summary:
        print_summary(results)

if __name__ == '__main__':
    main()