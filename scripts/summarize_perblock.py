import json
import sys
import os

def summarize_perblock(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Header info
    arch = data.get("architecture", "unknown")
    num_blocks = data.get("num_blocks", 0)
    print(f"Model: {arch}, Blocks: {num_blocks}")

    # Process Blocks
    blocks = data.get("blocks", [])
    for block in blocks:
        b_idx = block['block_idx']
        
        # safely get attention top eigenvalue
        attn_top = block.get('attention', {}).get('top', 0.0)
        
        # safely get mlp top eigenvalue
        mlp_top = block.get('mlp', {}).get('top', 0.0)

        print(f"Block {b_idx:2d} | Attn: {attn_top:12.2f} | MLP: {mlp_top:12.2f}")

    # Process Final Output (The part that was crashing)
    output_data = data.get("output", {})
    
    if "eigenvalues" in output_data and len(output_data["eigenvalues"]) > 0:
        out_top = output_data["eigenvalues"][0]
        # You can adjust the formatting below to match your original preference
        print(f"Output   | Top Eigenvalue: {out_top:.2f}")
    elif "error" in output_data:
        print(f"Output   | Error: {output_data['error_type']} (See file for details)")
    else:
        print("Output   | N/A")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summarize_perblock.py <path_to_json>")
    else:
        summarize_perblock(sys.argv[1])