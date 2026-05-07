# The Sensitivity-Drift Paradox: Architecture-Dependent FIM-Drift Dynamics in Recursive Self-Distillation

## 1. Executive Summary

This project investigates the weight-space geometry of model collapse during recursive self-distillation. The central finding is the **Sensitivity-Drift Paradox**: blocks with high Fisher sensitivity (high per-block FIM) systematically drift less than low-sensitivity blocks under standard AdamW, producing a negative Spearman correlation between per-block FIM and parameter drift.

The relationship is **architecture-dependent**: sequential transformers (attention and MLP in series) exhibit a negative FIM-drift correlation across all conditions, while parallel transformers (attention and MLP on shared residual simultaneously) exhibit a positive correlation, causing 5–50× faster collapse.

The mechanism is **AdamW's second-moment accumulation (v_t)**: a four-point optimizer ablation (SGD through β₂ ∈ {0.9, 0.999, 0.9999}) demonstrates a dose-response relationship. Crucially, resetting v_t between generations converts sequential models into parallel-like collapse (+0.69** by Gen5) while leaving parallel models unaffected (+0.77** at Gen1, staying positive), establishing that **v_t's cross-generation memory is the source of sequential architecture's self-protective property** and confirming that parallel collapse is architectural rather than memory-dependent.

**Paper title:** *The Sensitivity-Drift Paradox: How Architecture Shapes Collapse in Recursive Self-Distillation*

---

## 2. Model Lineup and Architecture Classification

### 2.1 Included Models

| Model | HuggingFace ID | Blocks | Arch | Role |
|---|---|---|---|---|
| SmolLM2-135M | HuggingFaceTB/SmolLM2-135M | 30 | Sequential | Main treatment + controls |
| GPT-2-117M | gpt2 | 12 | Sequential | Secondary |
| Llama-3.2-1B | meta-llama/Llama-3.2-1B | 16 | Sequential | Secondary |
| Phi-1.5-1.3B | microsoft/phi-1_5 | 24 | **Parallel** | Primary parallel |
| Pythia-1.4B | EleutherAI/pythia-1.4b | 24 | **Parallel** | Primary parallel |
| Falcon-7B | tiiuae/falcon-7b | 32 | **Parallel+MQA** | Mechanistic probe (MQA split) |

### 2.2 Excluded Models

| Model | Reason |
|---|---|
| Qwen2.5-0.5B | Extreme GQA (14 Q / 2 KV heads) — incommensurable per-block FIM |
| Qwen3.5-0.8B | DeltaNet hybrid attention (18/24 blocks linear recurrence) |
| Gemma-3-1b-it | Hybrid sliding-window (22/26 local + 4/26 global) |
| Gemma 4 | All above + Per-Layer Embeddings + shared KV cache + multimodal |

### 2.3 SmolLM Control Conditions

| Condition | Description | Purpose |
|---|---|---|
| Treatment | Recursive synthetic data | Main collapse simulation |
| Control A | Fresh real TinyStories slices each generation | Healthy training baseline |
| Control B | Same fixed TinyStories slice repeated | Isolates recursive collapse from overfitting |

---

## 3. Experimental Protocol

- **Generations:** 5 recursive generations per model
- **Data:** 50k synthetic samples per generation, TinyStories-style, length 256
- **Optimiser:** AdamW, lr=5×10⁻⁵, β₁=0.9, β₂=0.999, weight_decay=0.01
- **FIM computation:** Per-block top eigenvalue via SLQ on Fisher-Vector Products; FIM_b = λ_max_attn + λ_max_MLP
- **Drift:** ‖W_b^(n) − W_b^(0)‖_F / ‖W_b^(0)‖_F
- **Correlation:** Spearman ρ on log₁₀(FIM_b) vs Δ_b per (model, generation) pair

---

## 4. Main Results: FIM-Drift Correlations

| Model | G1 | G2 | G3 | G4 | G5 | Sig | Arch |
|---|---|---|---|---|---|---|---|
| SmolLM Treatment | −0.40* | −0.43* | −0.44* | −0.52** | −0.49** | 5/5 | Sequential |
| SmolLM CtrlA | −0.52** | −0.52** | −0.54** | −0.53** | −0.54** | 5/5 | Sequential |
| SmolLM CtrlB | −0.50** | −0.53** | −0.49** | −0.53** | −0.51** | 5/5 | Sequential |
| GPT-2 | −0.64* | −0.39 | −0.24 | −0.44 | −0.61* | 2/5 | Sequential |
| Llama | −0.14 | −0.34 | −0.19 | −0.41 | −0.06 | 0/5 | Sequential |
| Phi-1.5 | +0.19 | +0.39 | +0.47* | +0.65** | +0.46* | 3/5 | **Parallel** |
| Pythia | +0.75** | +0.81** | +0.72** | +0.56** | +0.84** | 5/5 | **Parallel** |

**Sign test:** 25/25 sequential negative; 10/10 parallel positive. Welch t=−14.24, p=3.81×10⁻⁹, d=5.96.

**Perplexity Gen0→Gen5:** SmolLM +29.5%, GPT-2 +142.7%, Llama +150.5%, Phi-1.5 +598.6%, Pythia +1426.4%, CtrlA −22.7%.

**Falcon-7B (parallel+MQA, 4 gens):** combined ρ near-zero (−0.28, −0.16, −0.32, −0.12) but MQA split reveals mechanism beneath — see Section 5.4.

---

## 5. Mechanism: AdamW v_t and Cross-Generation Memory

### 5.1 AdamW v_t Hypothesis

v_t = β₂ × v_{t-1} + (1−β₂) × g_t²
α_eff = α / (√v_t + ε)

FIM_{b,ii} ≈ E[g_i²] → high-FIM blocks accumulate larger v_t → smaller α_eff → suppressed drift → negative correlation.
Parallel residual streams eliminate gradient flow redirection → invert correlation sign.

### 5.2 Optimizer Ablation (4-point dose-response, SmolLM)

| Condition | G5 ρ | Interpretation |
|---|---|---|
| SGD (no v_t) | +0.70** | No adaptive suppression; high-FIM blocks drift proportionally |
| AdamW β₂=0.9 | +0.46* | Fast v_t decay; weak differential |
| AdamW β₂=0.999 | −0.51** | Standard; stable self-damping |
| AdamW β₂=0.9999 | −0.06 | Over-saturation; all blocks equally suppressed |

### 5.3 v_t Cross-Generation Memory: 2×2 Factorial

Resetting AdamW state at each generation boundary:

| Condition | G1 | G3 | G5 | Verdict |
|---|---|---|---|---|
| SmolLM (Seq), reset | −0.53** | +0.31 | +0.69** | Flips positive |
| Pythia (Par), reset | +0.77** | +0.56** | −0.14 | Stays positive |

Sequential with v_t reset: flips from negative to positive, collapsing like a parallel model. Parallel with v_t reset: stays positive, architecture determines outcome regardless of optimizer memory. The recursive synthetic data loop naturally produces gradient structures driving positive FIM-drift correlation. v_t's cross-generation memory converts this into the negative correlation in sequential architectures. In parallel architectures this conversion is structurally impossible regardless of optimizer memory.

Per-block FIM-inverse gradient clipping (clip threshold ∝ 1/FIM_b) produces the same flip (+0.79** by Gen5), confirming from a different angle.

### 5.4 Falcon-7B MQA Split: Sub-Block Mechanistic Probe

Falcon-7B uses parallel residual stream with Multi-Query Attention (MQA): a single KV projection receives gradient from all 71 query heads simultaneously. Combined FIM-drift correlation is near-zero but splitting the fused QKV projection reveals the mechanism:

| Gen | Combined ρ | Q-only ρ | KV-only ρ |
|---|---|---|---|
| 1 | −0.28 | −0.37* | +0.60** |
| 2 | −0.16 | −0.39* | +0.14 |
| 3 | −0.32 | −0.41* | +0.46** |
| 4 | −0.12 | −0.36* | +0.57** |

KV projection (shared across 71 heads): consistently positive — parallel mechanism operates as predicted. Q projection: consistently negative — disproportionately large v_t from 71 heads suppresses Q drift, mimicking sequential behaviour. The two effects cancel in the combined metric. Falcon's collapse velocity (+254% in 4 gens) confirms the parallel architecture prediction. This demonstrates the v_t theory at sub-block granularity.

---

## 6. Intervention Experiments

| Condition | PPL Δ Gen0→5 | FIM-drift ρ G5 | Why it works/fails |
|---|---|---|---|
| Normal treatment | +29.5% | −0.49** | Baseline |
| Frozen late (20-29) | +15.0% | −0.80 to −0.85 | Budget reduction; self-similar paradox |
| Ortho drift | +21.0% | — | No consistent drift direction (cosine −0.5%) |
| Smart ortho | +29.1% | — | Catastrophic EV explosion (B0 FIM 25,063) |
| EWC λ=500 | +29.3% | ~−0.49 | Stagnation; FIM range too flat (1.4–3.2×) |
| EWC λ=100 | +35.1% | — | Destabilisation valley |
| EWC λ=50 | +36.9% | — | Deepest valley |
| TCE γ=0.9 | +31.0% | −0.28 (0/5 sig) | Correlation weakened but collapse unchanged |
| TCE γ=0.95–0.999 | +33–36% | ~−0.25 (0/5 sig) | Same as γ=0.9, insensitive to gamma |
| SGD ablation | +35.2% | +0.70** | Mechanism probe; removes v_t entirely |
| v_t reset (SmolLM) | +35.2% | +0.69** | Cross-gen memory probe; confirms finding |
| v_t reset (Pythia) | +1789% | stays +ve | Parallel unaffected; confirms 2×2 factorial |
| PB gradient clipping | +36.3% | +0.79** | Confirms v_t memory finding from different angle |
| KL distill λ=0.1 | +11.0% | +0.51** G5 | Partial slowing; KL competes with CE |
| KL distill λ=0.5 | −3.1% | −0.14 (0/5 sig) | Stagnation anchored at Gen0 quality |
| Control A (real data) | −22.7% | −0.54** | Only intervention allowing genuine learning + no collapse |

### EWC vs KL Stagnation: Key Difference

EWC stagnation plateaus at +29.3% degradation: penalty grows with parameter distance × FIM, amplifying v_t for high-FIM blocks.

KL stagnation plateaus at −3% (near Gen0): penalty grows with distributional divergence, self-correcting as collapse progresses and anchoring at Gen0 quality.

---

## 7. Geometric Analyses

### 7.1 Per-Block Relative Critical Sharpness (λc)

λc^b = 2/ηc^b where ηc^b = smallest step along Gen_n drift that increases Gen0 loss.

FIM-λc Spearman correlation sign split:
- Sequential: 19/20 positive (high-FIM blocks have high λc — drift opposes Gen0 geometry)
- Parallel: 9/10 negative (high-FIM blocks have low λc — drift compatible with Gen0 geometry)

One sequential failure: Llama Gen3 (ρ=−0.20, n=16 blocks, low power). One parallel failure: Phi-1.5 Gen5 (ρ=+0.59**, severe collapse dissolves measurable gradient structure).

Welch t=−4.53, p=6.5×10⁻⁴. In sequential models, AdamW suppresses precisely the blocks whose drift would most destabilise pretraining geometry. In parallel models, high-FIM blocks drift freely in flat directions of the Gen0 landscape.

### 7.2 Drift Subspace Dimensionality (SVD of ΔW)

Normalised effective rank 2–13% for both architecture types. No architecture-dependent split. Rules out "unstructured drift" as the velocity explanation.

### 7.3 Drift Direction Cosine Similarity

All models −0.19 to −0.49, 0% positive. Oscillatory collapse: consecutive drift vectors anti-correlate because optimizer resets between generations. Explains orthogonalisation failure.

---

## 8. Reproduction and Usage

**The Sensitivity-Drift Paradox: How Architecture Shapes Collapse in Recursive Self-Distillation**

### Setup
```bash
pip install torch transformers datasets scipy numpy tqdm accelerate sentencepiece
```

### Reproducing the Main Results

**1. Run recursive treatment for a model**
```bash
# SmolLM2-135M (main model)
python scripts/run_treatment_recursive_pb_fim.py

# Other models — change MODEL_ID, MODEL_FOLDER_PREFIX, NUM_BLOCKS at top of script
# GPT-2:   MODEL_ID="gpt2", NUM_BLOCKS=12
# Llama:   MODEL_ID="meta-llama/Llama-3.2-1B", NUM_BLOCKS=16
# Phi-1.5: MODEL_ID="microsoft/phi-1_5", NUM_BLOCKS=24
# Pythia:  MODEL_ID="EleutherAI/pythia-1.4b", NUM_BLOCKS=24
# Falcon:  MODEL_ID="tiiuae/falcon-7b", NUM_BLOCKS=32
```

**2. Compute per-block parameter drift**
```bash
python scripts/parameter_drift.py \
  --base "HuggingFaceTB/SmolLM2-135M" \
  --target models/treatment_gen_1 \
  --out results/drift_gen_1.json

# Falcon MQA split (Q vs KV drift):
python scripts/parameter_drift.py \
  --base "tiiuae/falcon-7b" \
  --target models/falcon-7b_treatment_gen_1 \
  --out results/falcon_drift_split_gen_1.json \
  --split-mqa
```

**3. Evaluate perplexity**
```bash
python scripts/evaluate_all_metrics_master.py
```

### Mechanism Experiments

**Optimizer ablation (β₂ sweep)**
```bash
python scripts/sgd_ablation.py        # SGD baseline
python scripts/beta2_ablation.py      # β₂ ∈ {0.9, 0.999, 0.9999}
```

**v_t cross-generation memory (2×2 factorial)**
```bash
# Sequential model (SmolLM) — flips positive
python scripts/vt_reset_ablation.py

# Parallel model (Pythia) — stays positive
# Change MODEL_ID to EleutherAI/pythia-1.4b and DATA_PREFIX accordingly
python scripts/vt_reset_ablation.py
```

**Per-block gradient clipping**
```bash
python scripts/pbclip_ablation.py
```

### Intervention Experiments
```bash
python scripts/so_what_layer_freeze.py   # Layer freezing
python scripts/so_what_ewc.py            # EWC λ ∈ {50, 100, 500}
python scripts/tce_ablation.py           # Truncated Cross-Entropy
python scripts/tce_sweep.py              # TCE γ sweep
python scripts/kl_distill_ablation.py    # KL distillation from Gen0
```

### Geometric Analyses
```bash
python scripts/critical_sharpness_collapse.py   # Per-block relative critical sharpness
python scripts/drift_subspace_svd.py             # Drift subspace dimensionality
python scripts/drift_cosine_similarity.py        # Drift direction cosine similarity
```

### Directory Structure
```
├── data/          # Synthetic training data per generation per model
├── models/        # Saved checkpoints per generation per condition
├── results/       # FIM JSONs, drift JSONs, summary JSONs, CSV metrics
└── scripts/       # All experiment scripts
```

### Models and Data
Trained model checkpoints and generated datasets are available on HuggingFace: **[https://huggingface.co/droq98/sensitivity-drift-paradox]**

---

## 9. Mathematical Formulations

### 9.1 Per-Block Fisher Information (Empirical FIM)

$$F_b(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \nabla_{\theta_b} \log p(x|\theta) \nabla_{\theta_b} \log p(x|\theta)^T \right]$$

Computed via SLQ on top eigenvalues: $FIM_b = \lambda_{max}(Attn_b) + \lambda_{max}(MLP_b)$.

### 9.2 Relative Parameter Drift

$$\Delta_b^{(t)} = \frac{\|W_b^{(t)} - W_b^{(0)}\|_F}{\|W_b^{(0)}\|_F}$$

### 9.3 AdamW $v_t$ Mechanism

$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\alpha_{\text{eff}} = \frac{\alpha}{\sqrt{v_t} + \varepsilon}$$

High-FIM blocks: large $\|g_t\|$ $\to$ large $v_t$ $\to$ small $\alpha_{\text{eff}}$ $\to$ suppressed drift.

### 9.4 EWC Loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{synth}} + \lambda \sum_b \left[\log_{10}(\text{FIM}_b) \cdot \text{mean}\left((W_b - W_b^{(0)})^2\right)\right]$$

### 9.5 Cosine Similarity of Incremental Drift

$$\delta_n = W_{\text{gen}_n} - W_{\text{gen}_{n-1}}$$
$$\cos(\delta_n, \delta_{n+1}) = \frac{\delta_n \cdot \delta_{n+1}}{\|\delta_n\| \|\delta_{n+1}\|}$$

### 9.6 Relative Critical Sharpness ($\lambda_c$)

$$\lambda_c^b = 2/\eta_c^b$$

Where $\eta_c^b$ is the smallest step along $\delta_n$ that increases Gen0 loss.

### 9.7 Spearman Rank Correlation

$$\rho = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$$

Where $d_i$ is rank difference between $\log_{10}(FIM_b)$ and $\Delta_b$.

---

## 10. Open Experiments

| Experiment | Priority | Estimated time | Notes |
|---|---|---|---|
| β₂ ablation on Phi-1.5 | Medium | ~15 hours 5090 | May give cleaner result than Pythia |
| GPT-J-6B treatment | Low | ~8 hours A100 | 3rd clean parallel MHA family |
| KL distillation λ=1.0 | Low | ~90 min 5090 | Completes lambda sweep |
| Temperature scaling T=1.5, T=2.0 | Low | ~3 hours 5090 | Data-level intervention |
| Selective v_t decay (custom optimizer) | Low | Journal version | — |
| Activation space probing across generations | Low | Journal version | — |

