import numpy as np
from scipy.stats import t

# 1. Block counts (n) per architecture
block_counts = {
    "Qwen 3.5 (0.8B)": 24,
    "GPT-2 (124M)": 12,
    "SmolLM Treatment": 30,
    "SmolLM Control A": 30,
    "SmolLM Control B": 30,
    "Llama 3.2 (1B)": 16,
    "Gemma 3 (1B)": 26,
    "Qwen 2.5 (0.5B)": 24
}

# 2. Your precise correlation values (Gen 1 to Gen 5)
correlations = {
    "Qwen 3.5 (0.8B)": [-0.5357, -0.5765, -0.6513, -0.6548, -0.7026],
    "GPT-2 (124M)": [-0.6364, -0.3853, -0.2448, -0.4406, -0.6084],
    "SmolLM Treatment": [-0.4011, -0.4305, -0.4370, -0.5164, -0.4950],
    "SmolLM Control A": [-0.5164, -0.5164, -0.5359, -0.5293, -0.5404],
    "SmolLM Control B": [-0.5022, -0.5337, -0.4932, -0.5266, -0.5141],
    "Llama 3.2 (1B)": [-0.1382, -0.3353, -0.1912, -0.4118, -0.0647],
    "Gemma 3 (1B)": [0.3621, 0.3046, 0.0509, -0.6014, -0.2834],
    "Qwen 2.5 (0.5B)": [0.4478, 0.3617, -0.4470, 0.2848, 0.0635]
}

# 3. P-value calculation function
def calculate_p_value(rho, n):
    if rho == 0.0:
        return 1.0
    # Calculate t-statistic
    t_stat = rho * np.sqrt((n - 2) / (1 - rho**2))
    # Calculate two-tailed p-value
    p_val = 2 * t.sf(abs(t_stat), df=n-2)
    return p_val

# 4. Execute and print the results
print(f"{'Model Architecture':<20} | {'Gen 1':<8} | {'Gen 2':<8} | {'Gen 3':<8} | {'Gen 4':<8} | {'Gen 5':<8}")
print("-" * 70)

for model, rhos in correlations.items():
    n = block_counts[model]
    p_values = [calculate_p_value(r, n) for r in rhos]
    
    # Format p-values to 4 decimal places
    formatted_p_vals = [f"{p:.4f}" for p in p_values]
    
    print(f"{model:<20} | {formatted_p_vals[0]:<8} | {formatted_p_vals[1]:<8} | {formatted_p_vals[2]:<8} | {formatted_p_vals[3]:<8} | {formatted_p_vals[4]:<8}")