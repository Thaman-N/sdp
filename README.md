# The Sensitivity-Drift Paradox: Architecture-Dependent FIM-Drift Dynamics in Recursive Self-Distillation

## 1. Executive Summary

This project investigates the weight-space geometry of model collapse during recursive self-distillation — the process of repeatedly training a model on its own synthetic outputs. The central finding is the **Sensitivity-Drift Paradox**: blocks with high Fisher sensitivity (high per-block FIM) systematically drift *less* than low-sensitivity blocks under standard AdamW optimisation, producing a consistent negative Spearman correlation between per-block FIM and parameter drift.

The key discovery is that this relationship is **architecture-dependent**: sequential attention transformers (attention and MLP in series on the residual stream) exhibit a negative FIM-drift correlation across all conditions and all generations, while parallel attention transformers (attention and MLP computed simultaneously on the shared residual) invert this to a positive correlation. This sign difference explains why parallel architectures collapse 5–50× faster under recursive training.

The mechanism is causally grounded in AdamW's second-moment accumulation (v_t): a three-point β₂ ablation (β₂ ∈ {0.9, 0.999, 0.9999}) demonstrates that the correlation sign tracks the v_t accumulation regime. Four weight-level intervention experiments confirm that the FIM-drift bottleneck is a global network-level equilibrium that reconstitutes under any local perturbation. Two geometric analyses (drift subspace dimensionality, drift direction cosine similarity) characterise the collapse geometry and rule out structural alternatives.

**Paper title:** *The Sensitivity-Drift Paradox: Architecture-Dependent FIM-Drift Dynamics in Recursive Self-Distillation*

---

## 2. Model Lineup and Architecture Classification

### 2.1 Included Models

| Model | HuggingFace ID | Blocks | Arch Type | Role |
|---|---|---|---|---|
| SmolLM2-135M | HuggingFaceTB/SmolLM2-135M | 30 | Sequential | Main treatment + controls |
| GPT-2-117M | gpt2 | 12 | Sequential | Secondary sequential |
| Llama-3.2-1B | meta-llama/Llama-3.2-1B | 16 | Sequential | Secondary sequential |
| Phi-1.5-1.3B | microsoft/phi-1_5 | 24 | **Parallel** | Primary parallel |
| Pythia-1.4B | EleutherAI/pythia-1.4b | 24 | **Parallel** | Primary parallel |

**Sequential** = attention and MLP computed in series (standard residual stream: x → attn → add → MLP → add).  
**Parallel** = attention and MLP computed simultaneously on the same residual input (x → [attn + MLP] → add), used in Phi-1.5 and Pythia (use_parallel_residual=True).

### 2.2 Excluded Models and Reasons

| Model | Reason |
|---|---|
| Qwen2.5-0.5B | Underdetermined: extreme GQA (14 Q-heads / 2 KV-heads). Per-block FIM confounds query vs. KV sensitivity — incommensurable block comparisons. |
| Qwen3.5-0.8B | DeltaNet hybrid: 18/24 blocks use linear recurrence attention (not standard softmax). FIM computation assumes standard attention. |
| Gemma-3-1b-it | Hybrid sliding-window: 22/26 blocks use local sliding-window attention (512 tokens), 4/26 use full attention. Two incommensurable attention types within the same block ranking. |
| Gemma 4 (all sizes) | Same hybrid attention confound as Gemma 3, compounded by Per-Layer Embeddings (PLE, second conditioning pathway), shared KV cache (last N layers reuse K/V projections), and multimodal architecture (text + image + audio). All four confounds present simultaneously. |

### 2.3 SmolLM Control Conditions

| Condition | Description | Purpose |
|---|---|---|
| Treatment | Recursive: each generation trained on synthetic output of previous | Main collapse simulation |
| Control A (fresh real) | Trained on unique non-overlapping slices of TinyStories each generation | Establishes healthy training baseline |
| Control B (static real) | Repeated training on same fixed TinyStories slice | Distinguishes recursive collapse from simple overfitting |

---

## 3. Experimental Protocol

- **Generations:** 5 recursive generations per model
- **Data:** 50k synthetic samples per generation (TinyStories-style; sequence length 256)
- **Optimiser:** AdamW, lr=5×10⁻⁵, β₁=0.9, β₂=0.999, weight_decay=0.01
- **FIM computation:** Per-block FIM top eigenvalues via Stochastic Lanczos Quadrature on Fisher-Vector Products. FIM_b = FIM_attn_b + FIM_mlp_b.
- **Drift computation:** Relative Frobenius norm vs. Gen0 anchor: Δ_b = ‖W_b^(n) − W_b^(0)‖_F / ‖W_b^(0)‖_F
- **Correlation:** Spearman ρ on log₁₀(FIM_b) vs Δ_b per generation per model
- **Evaluation:** Perplexity on TinyStories validation set at each generation

---

## 4. The Sensitivity-Drift Paradox: Main Results

### 4.1 Spearman FIM-Drift Correlations

| Model | G1 | G2 | G3 | G4 | G5 | Sig | Arch |
|---|---|---|---|---|---|---|---|
| SmolLM Treatment | −0.40* | −0.43* | −0.44* | −0.52** | −0.49** | 5/5 | Sequential |
| SmolLM CtrlA | −0.52** | −0.52** | −0.54** | −0.53** | −0.54** | 5/5 | Sequential |
| SmolLM CtrlB | −0.50** | −0.53** | −0.49** | −0.53** | −0.51** | 5/5 | Sequential |
| GPT-2 | −0.64* | −0.39 | −0.24 | −0.44 | −0.61* | 2/5 | Sequential |
| Llama | −0.14 | −0.34 | −0.19 | −0.41 | −0.06 | 0/5 | Sequential |
| Phi-1.5 | +0.19 | +0.39 | +0.47* | +0.65** | +0.46* | 3/5 | **Parallel** |
| Pythia | +0.75** | +0.81** | +0.72** | +0.56** | +0.84** | 5/5 | **Parallel** |

*p < 0.05, **p < 0.01

**Sign test:** 25/25 sequential pairs negative; 10/10 parallel pairs positive.  
**Welch t-test:** t = −14.24, p = 3.81×10⁻⁹, Cohen's d = 5.96.

The controls (CtrlA, CtrlB) show the same negative correlation as the treatment despite not collapsing, confirming the paradox is a property of AdamW training dynamics rather than an artefact of synthetic data degradation.

### 4.2 Perplexity Degradation

| Model | Gen0 | Gen5 | Δ |
|---|---|---|---|
| SmolLM Treatment | 6.05 | 7.84 | +29.5% |
| GPT-2 | 11.04 | 26.78 | +142.7% |
| Llama | 5.50 | 13.78 | +150.5% |
| **Phi-1.5** | 3.97 | 27.71 | **+598.6%** |
| **Pythia** | 5.79 | 88.38 | **+1426.4%** |
| SmolLM CtrlA | 6.05 | 4.68 | −22.7% |
| SmolLM CtrlB | 6.05 | 4.69 | −22.4% |

Parallel models collapse 5–50× faster, consistent with the positive FIM-drift correlation creating an unconstrained runaway feedback loop.

### 4.3 Norm and Rank Stability

Across all models and all five generations:
- Weight norms change < 0.5% (pure rotation, not norm growth)
- Effective matrix rank changes < 0.6% under all three rank definitions
- Condition numbers volatile (e.g., GPT-2: +378%) — spectral distortion without rank collapse

Collapse manifests as **high-dimensional weight rotation**, not dimensional compression or norm growth.

---

## 5. Mechanistic Grounding: The β₂ Ablation

### 5.1 The AdamW v_t Hypothesis

The FIM-drift paradox arises from AdamW's second-moment accumulation:

```
v_t = β₂ * v_{t-1} + (1-β₂) * g_t²
α_eff = α / (√v_t + ε)
```

High-FIM blocks generate larger gradient magnitudes, accumulating larger v_t, which reduces their effective learning rate — suppressing their drift. This is precisely the negative correlation. For parallel architectures, simultaneous attention+MLP updates on the shared residual prevent gradient flow redirection, inverting the relationship.

**Prediction:** Reducing β₂ (faster v_t decay) should weaken or flip the correlation; increasing β₂ should strengthen it, then saturate all blocks equally (eliminating the differential) at the extreme.

### 5.2 β₂ Ablation Results (SmolLM, 50k samples, 5 generations)

| β₂ | G1 | G2 | G3 | G4 | G5 | Interpretation |
|---|---|---|---|---|---|---|
| 0.9 (fast decay) | −0.04 | +0.05 | +0.21 | +0.34 | **+0.46*** | Sign flip — v_t decays too fast for differential suppression |
| 0.999 (standard) | −0.20 | −0.35 | **−0.44*** | **−0.47**** | **−0.51**** | Stable negative — standard AdamW operating regime |
| 0.9999 (slow decay) | **−0.34** | **−0.37*** | **−0.44*** | **−0.46*** | −0.06 | Strengthens early (faster v_t build-up), then saturates to zero at Gen5 |

All three conditions behave as theoretically predicted:
- β₂=0.9: sign flip to +0.46* — fast v_t decay eliminates differential suppression between blocks
- β₂=0.999: stable negative to −0.51** — standard AdamW operating regime
- β₂=0.9999: negative appears earlier and stronger than standard (Gen1: −0.34 vs −0.20), then collapses to −0.06 at Gen5 as v_t accumulates so strongly that all blocks reach near-zero effective learning rate, eliminating the differential

The inverted-U pattern (positive at low β₂, negative at standard β₂, zero at high β₂) confirms the causal role of v_t accumulation and defines its operating window.

---

## 6. Intervention Experiments

All experiments reuse the same SmolLM synthetic data as the normal treatment. The goal is to test whether the FIM-drift bottleneck can be broken at the weight level.

| Condition | Gen0→Gen5 PPL | EV crash gen | Block-11 taming | Mechanism |
|---|---|---|---|---|
| Normal | +29.5% | Gen1 (13) | No | Baseline |
| **Frozen late** (blocks 20–29 frozen) | **+15.0%** | Never | No | Parameter budget ↓ |
| Ortho drift | +21.0% | Climbing | No | Destabilises landscape |
| Smart ortho | +29.1% | 428→8061→10 | No | Catastrophic EV explosion |
| EWC λ=500 | +29.3% | Gen3 (10) | Yes (94%) | Stagnation |
| EWC λ=100 | +35.1% | Gen2 (11) | Yes (95%) | Destabilisation valley |
| EWC λ=50 | +36.9% | Gen5 (16) | Yes (95%) | Destabilisation valley |

### 6.1 Key Findings

**Frozen late layers:** The only intervention providing meaningful improvement (+15%), but works purely by reducing the trainable parameter budget. The Spearman correlation reconstitutes in the remaining 20 free blocks (ρ ≈ −0.80 to −0.85), confirming the bottleneck is **self-similar** — AdamW reconstitutes the same gradient hierarchy in whatever trainable parameters remain.

**EWC lambda sweep:** U-shaped perplexity curve. High lambda (500) stagnates training; the EWC gradient dominates v_t, suppressing all blocks uniformly to near-zero drift. Medium lambda (100–50) creates a destabilisation valley where learning is impeded but collapse is not prevented. The structural failure: log₁₀(FIM) weights span only 1.4–3.2× across 30 blocks — insufficient differential contrast for EWC to selectively protect high-sensitivity layers.

**EWC side-finding:** Despite failing on perplexity, EWC progressively tames the structural anomaly blocks (Block 11 MLP: 1527→68 by Gen5 at λ=500). The EWC anchor pulls anomaly blocks toward Gen0 values which themselves decay. This is a secondary geometric effect independent of the perplexity story.

**Ortho interventions:** Fail because in high-dimensional weight space (~330k dimensions per block), consecutive drift vectors share only ~0.5% cosine alignment. The ortho step removes negligible drift while displacing weights substantially — confirmed by the cosine similarity analysis (Section 8).

**Theoretical conclusion:** No weight-level intervention can break the v_t feedback loop because any gradient signal — synthetic loss or regularisation — contributes to v_t accumulation. The loop is broken only by changing the data distribution.

---

## 7. Geometric Analyses

### 7.1 Drift Subspace Dimensionality (SVD of ΔW)

**Method:** For each block, compute ΔW = W_gen5 − W_gen0, run truncated SVD, measure effective rank (% of matrix dimension needed to explain 95% of Frobenius drift variance).

| Model | Arch | Mean EffRank | Normalised (% of dim) | Top1 Var% |
|---|---|---|---|---|
| SmolLM Trt | Sequential | 72.9 | 12.7% | 10.1% |
| SmolLM CtrlA | Sequential | 73.6 | 12.8% | 7.5% |
| GPT-2 | Sequential | 43.6 | 5.7% | 16.7% |
| Llama | Sequential | 133.1 | 6.5% | 27.9% |
| Phi-1.5 | Parallel | 46.2 | 2.3% | 15.4% |
| Pythia | Parallel | 40.8 | 2.0% | 21.8% |

**Result:** Null finding on architecture dependence. Effective rank is dominated by matrix dimension rather than architecture type. After normalising, sequential and parallel models show comparable proportional drift dimensionality (both low single-digit percent of matrix dimension).

**Paper use:** Rules out the "unstructured drift" alternative hypothesis — the architecture-dependence of collapse velocity does not arise from qualitative differences in drift subspace structure.

### 7.2 Drift Direction Cosine Similarity

**Method:** For each model and each block, compute incremental drift vectors δₙ = Wₙ − Wₙ₋₁ and measure cosine similarity between consecutive generation pairs: cos(δ₁, δ₂), cos(δ₂, δ₃), cos(δ₃, δ₄), cos(δ₄, δ₅). Average across weight matrices within each block.

| Model | Arch | Mean Cosine | Early blocks | Late blocks | % Positive |
|---|---|---|---|---|---|
| SmolLM Trt | Sequential | −0.187 | −0.288 | −0.087 | 0% |
| SmolLM CtrlA | Sequential | −0.434 | −0.448 | −0.420 | 0% |
| GPT-2 | Sequential | −0.439 | −0.456 | −0.422 | 0% |
| Llama | Sequential | −0.486 | −0.492 | −0.480 | 0% |
| Phi-1.5 | Parallel | −0.431 | −0.468 | −0.395 | 0% |
| Pythia | Parallel | −0.492 | −0.486 | −0.499 | 0% |

**Result:** All cosine similarities are negative (−0.19 to −0.49). 0% positive blocks. No architecture-dependent split.

**Explanation:** The optimizer state resets between generations. The next generation's synthetic data was generated by a model that drifted in direction δₙ, so its distribution partially reflects that drift. New gradients therefore partially oppose the previous drift direction (oscillatory collapse). Early blocks (high FIM, large gradients) show more negative cosine than late blocks, consistent with stronger v_t-mediated momentum oscillation in high-FIM regions.

**Paper use:** Explains the failure of orthogonalisation interventions — drift direction is anti-correlated between generations, leaving no stable direction to project out. Confirms that architecture type predicts FIM-magnitude of drift but not directional structure.

---

## 8. Directory Structure

```
hessian-spectral-analysis/
├── data/
│   ├── treatment_synthetic_gen_1-5/          # SmolLM recursive synthetic data
│   ├── beta2_ablation/
│   │   ├── shared_gen1_data/                 # Shared Gen1 data for β₂ ablation
│   │   ├── smollm_beta2_0_9_gen2-5/
│   │   ├── smollm_beta2_0_999_gen2-5/
│   │   └── smollm_beta2_0_9999_gen2-5/
├── models/
│   ├── treatment_gen_1-5/                    # SmolLM treatment
│   ├── control_generation_1-5/               # SmolLM CtrlA
│   ├── control_b_gen_1-5/                    # SmolLM CtrlB
│   ├── gpt2_treatment_gen_1-5/
│   ├── llama_treatment_gen_1-5/
│   ├── phi-1_5_treatment_gen_1-5/
│   ├── pythia-1.4b_treatment_gen_1-5/
│   ├── frozen_late_gen_1-5/
│   ├── ortho_drift_gen_1-5/
│   ├── smart_ortho_gen_1-5/
│   ├── ewc_lambda50_gen_1-5/
│   ├── ewc_lambda100_gen_1-5/
│   ├── ewc_lambda500_gen_1-5/
│   └── beta2_ablation/
│       ├── smollm_beta2_0_9_gen1-5/
│       ├── smollm_beta2_0_999_gen1-5/
│       └── smollm_beta2_0_9999_gen1-5/
├── results/
│   ├── summary/
│   │   ├── comprehensive_metrics.csv         # Perplexity/diversity/coherence all conditions
│   │   └── comprehensive_metrics.json
│   ├── [model]_gen_[N]/                      # FIM JSON per model per generation
│   ├── [model]_drift_gen_[N].json            # Drift JSON per model per generation
│   ├── beta2_ablation/
│   │   ├── beta2_summary.json
│   │   └── smollm_beta2_[b2]_gen[N]/
│   ├── drift_subspace/
│   │   ├── svd_results.json
│   │   └── svd_summary.csv
│   └── drift_cosine/
│       ├── cosine_results.json
│       └── cosine_summary.csv
└── scripts/
    ├── perblock_fim.py
    ├── parameter_drift.py
    ├── run_treatment_recursive_pb_fim.py
    ├── evaluate_all_metrics_master.py
    ├── so_what_layer_freeze.py
    ├── so_what_drift_ortho.py
    ├── so_what_smart_ortho.py
    ├── so_what_ewc.py
    ├── beta2_ablation.py
    ├── drift_subspace_svd.py
    └── drift_cosine_similarity.py
```

<!-- ---

## 9. Publication Status

### 9.1 Workshop Targets

| Workshop | Conference | Deadline | Page limit | Format | Status |
|---|---|---|---|---|---|
| Weight-Space Symmetries (WSS) | ICML 2026 | April 30, 2026 | 4 pages | LaTeX, ICML 2026 + custom WSS sty | **Primary target** |
| Mechanistic Interpretability | ICML 2026 | May 8, 2026 | 4 (short) / 8 (long) | LaTeX, ICML or NeurIPS 2026 | **Secondary target** |

Both workshops are non-archival and permit dual submission. Notification by May 15.

WSS sty file: `https://www.weightsymmetry.com/assets/icml2026_weightsymmetry.sty`  
ICML 2026 style: `https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip`

### 9.2 Journal Targets (post-workshop)

| Journal | IF | Fit |
|---|---|---|
| Neural Networks (Elsevier) | ~7.0 | Excellent — mechanistic analysis of training dynamics |
| Neurocomputing (Elsevier) | ~6.0 | Excellent — empirical multi-architecture study |
| TMLR | Rolling | Strong — rewards empirical rigor over novelty |
| TNNLS | ~10.0 | Reach — high math bar, strong if formatted rigorously |

### 9.3 Framing Notes

WSS framing: The architecture-dependent FIM-drift relationship is a statement about how the geometric structure of weight-space (sequential vs. parallel residual stream computation) shapes training dynamics. The result characterises how architectural symmetry structure propagates into the loss landscape under recursive training.

MechInterp framing: FIM-drift correlation is a mechanistic finding about how AdamW's internal v_t state creates a systematic gradient hierarchy across transformer blocks. The β₂ ablation is a causal intervention on a single internal variable. The paper directly characterises model internals to explain training behaviour. -->

---

## 10. Mathematical Formulations

### 10.1 Per-Block Fisher Information (Empirical FIM)

$$F_b(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \nabla_{\theta_b} \log p(x|\theta) \nabla_{\theta_b} \log p(x|\theta)^T \right]$$

Computed via Stochastic Lanczos Quadrature on Fisher-Vector Products. Block-level scalar: FIM_b = top eigenvalue of attn + top eigenvalue of MLP.

### 10.2 Relative Parameter Drift

$$\Delta_b^{(t)} = \frac{\|W_b^{(t)} - W_b^{(0)}\|_F}{\|W_b^{(0)}\|_F}$$

### 10.3 AdamW v_t Mechanism

$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\alpha_{\text{eff}} = \frac{\alpha}{\sqrt{v_t} + \varepsilon} \to 0 \text{ as } v_t \text{ grows}$$

High-FIM blocks: large $\|g_t\|$ → large $v_t$ → small $\alpha_{\text{eff}}$ → suppressed drift → **negative correlation**.  
Parallel architecture: simultaneous attention+MLP on shared residual → no gradient flow redirection → **positive correlation**.

### 10.4 EWC Loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{synth}} + \lambda \sum_b \left[\log_{10}(\text{FIM}_b) \cdot \text{mean}\left((W_b - W_b^{(0)})^2\right)\right]$$

### 10.5 Cosine Similarity of Incremental Drift

$$\delta_n = W_{\text{gen}_n} - W_{\text{gen}_{n-1}}$$
$$\cos(\delta_n, \delta_{n+1}) = \frac{\delta_n \cdot \delta_{n+1}}{\|\delta_n\| \|\delta_{n+1}\|}$$

### 10.6 Spearman Rank Correlation

$$\rho = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$$

where $d_i$ is the rank difference between $\log_{10}(\text{FIM}_b)$ and $\Delta_b$ for block $i$, $n$ = number of blocks.