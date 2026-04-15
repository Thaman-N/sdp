# Record of Peer Review and Framework Iteration

## 1. Core Mechanism: AdamW v_t and the Sensitivity-Drift Paradox

**Initial claim:** The Sensitivity-Drift Paradox is an empirical observation: high-FIM blocks drift less than low-FIM blocks during recursive self-distillation.

- **Criticism:** Purely empirical. No mathematical derivation linking FIM to AdamW behaviour. Claiming a "mechanism" requires derivation.
- **Resolution (Cleared):** Derived the bottleneck from AdamW's update equations. FIM approximates expected squared gradients (F_ii ≈ E[g_i²]), which maps to AdamW's second moment v_t. High-FIM blocks accumulate larger v_t → smaller effective learning rate α_eff = α/(√v_t + ε) → suppressed drift. This is the negative correlation.
- **Further causal proof:** β₂ ablation (Section 5 of README). β₂=0.9 flips the sign to +0.46*; β₂=0.999 gives stable −0.51**; β₂=0.9999 strengthens then saturates to zero (over-accumulation). All three arms behave as the v_t theory predicts — inverted-U pattern confirms causal role and operating window.
- **Status:** **Causally proven.** Mechanistic explanation is theoretically derived and experimentally confirmed.

---

## 2. Architecture Dependence: The Sign Split

**Initial claim:** The negative FIM-drift correlation is a universal property of transformer training under recursive distillation.

- **Criticism:** If it's universal, why does it vary so much across architectures and generations? Claiming universality when the data shows model-by-model variation invites rejection.
- **Resolution (Reframed — major finding):** The correlation is NOT universal. It is architecture-dependent. Sequential attention transformers (attention and MLP in series) show negative correlation; parallel attention transformers (attention and MLP on shared residual input simultaneously) show positive correlation. This is the paper's primary contribution.
  - Sign test: 25/25 sequential negative, 10/10 parallel positive. p = 2.9×10⁻¹¹.
  - Welch t = −14.24, p = 3.81×10⁻⁹, Cohen's d = 5.96.
  - Parallel models collapse 5–50× faster (Phi-1.5: +598%, Pythia: +1426% vs SmolLM: +29.5%).
- **Mechanism for the sign inversion:** In sequential residual streams, high-FIM early blocks accumulate large v_t and are suppressed; gradient flows to later low-FIM blocks → negative correlation. In parallel residual streams, attention and MLP update the same residual simultaneously, preventing the gradient flow redirection mechanism → positive correlation.
- **Status:** **Solid.** The strongest result in the paper.

---

## 3. Old Architecture Theory: RoPE vs. APE (Scrapped)

**Initial claim:** Positional encoding type (RoPE vs APE) determines collapse dynamics.

- **Criticism:** Qwen2.5-0.5B uses RoPE but shows the most extreme Gen1 saturation in the dataset. The theory is broken by a single counterexample.
- **Resolution (Abandoned):** Theory purged entirely. The real architectural variable is sequential vs. parallel attention computation on the residual stream, not positional encoding.
- **Status:** **Failed and scrapped.** Not referenced anywhere in current paper.

---

## 4. Old Taxonomy: Three Empirical Archetypes (Superseded)

**Initial claim:** Models fall into Archetype 1 (Delayed Rebound), Archetype 2 (Immediate Saturation), or Archetype 3 (Oscillatory Instability) based on top eigenvalue trajectory.

- **Criticism:** The taxonomy is post-hoc and describes phenomenology without explaining the mechanism. It cannot predict which archetype a new model will exhibit.
- **Resolution (Superseded):** The archetype taxonomy was largely driven by the now-excluded models (Qwen3.5, Qwen2.5, Gemma3). After model exclusions:
  - Qwen3.5: excluded (DeltaNet hybrid architecture)
  - Qwen2.5: excluded (extreme GQA underdetermination)
  - Gemma3: excluded (hybrid sliding-window attention)
  
  The remaining models naturally split into sequential (negative correlation, moderate collapse) and parallel (positive correlation, fast collapse). The archetype taxonomy is no longer needed — the sequential/parallel classification provides the mechanistic explanation the archetypes lacked.
  
  EV trajectory still informative for describing collapse dynamics within architecture type but is not a primary claim.
- **Status:** **Superseded by the architecture-dependence finding.** EV trajectory data retained for description but not as a predictive framework.

---

## 5. Pseudoreplication in Statistical Proof (Resolved)

**Initial claim:** The paradox is backed by a global pooled correlation ρ = −0.3733 with p = 4.17×10⁻³³ across 960 transformer blocks.

- **Criticism:** Severe pseudoreplication. Pooling all generations assumes 960 independent observations; in reality, Block 0 in Gen2 is autocorrelated with Block 0 in Gen1. The p-value is fabricated by inflated degrees of freedom.
- **Resolution (Cleared):** Global pooled correlation purged. Statistical proof now relies exclusively on per-generation disaggregated reporting. The sign test across all (model, generation) pairs is the primary statistical claim — each pair is genuinely independent given the architecture conditioning.
- **Status:** **Resolved.** Per-generation per-model reporting is the correct unit of analysis.

---

## 6. Intervention Experiments: "No Weight-Level Fix" Claim

**Initial claim:** Weight-level interventions can mitigate collapse.

- **Criticism (self-correction):** If the mechanism is a global network-level equilibrium maintained by AdamW's v_t, then any weight-level intervention that generates gradients (including regularisation) feeds v_t and cannot break the loop. The claim should be the opposite.
- **Resolution:** Four intervention experiments conducted:
  1. **Frozen late layers (+15%):** Only meaningful improvement, but works purely by reducing trainable parameter budget. The correlation reconstitutes (ρ ≈ −0.80 to −0.85) in the remaining free blocks — **self-similar paradox** confirmed. Freezing does not disrupt the mechanism, it just reduces the total capacity available.
  2. **Ortho drift (+21%):** Fails because consecutive drift vectors share ~0.5% cosine alignment in high-dimensional weight space (confirmed by cosine similarity analysis). Ortho removes negligible actual drift while destabilising the landscape.
  3. **Smart ortho (+29.1%):** Combining freeze + ortho compounds gradient pressure into fewer free blocks. Block 0 FIM reaches 25,063 (Gen3). Catastrophic EV explosion (428→8,061). Worst instability of all conditions.
  4. **EWC λ sweep (λ=50,100,500):** U-shaped perplexity curve. High λ stagnates (EWC gradient overwhelms v_t, uniform suppression). Medium λ creates destabilisation valley (35-37%). The structural failure: log₁₀(FIM) weights span only 1.4–3.2× across 30 blocks — insufficient differential contrast for EWC. EWC gradients feed v_t, amplifying rather than counteracting the existing suppression of high-FIM blocks.
- **Current claim:** "No weight-level intervention prevents collapse. The FIM-drift bottleneck is a global network-level equilibrium that reconstitutes under any local perturbation. The only validated mitigation is data-level anchoring (Gerstgrasser et al. 2024)."
- **Status:** **Empirically established, theoretically grounded by v_t argument.**

---

## 7. Model Exclusions: Justification Record

**Issue:** Several models initially included turned out to have architectural confounds that make per-block FIM comparison invalid.

### Qwen2.5-0.5B — Excluded
- **Reason:** Extreme Grouped Query Attention (14 Q-heads / 2 KV-heads). Per-block FIM measures query sensitivity and KV sensitivity in incommensurable ways. The FIM of a block with 14 Q-heads is not comparable to a block where all heads are full attention.
- **Observed symptom:** Oscillatory positive correlations that couldn't be explained by sequential or parallel classification.
- **Status:** Excluded from all correlation analyses. Data archived but not referenced in paper.

### Qwen3.5-0.8B — Excluded
- **Reason:** DeltaNet hybrid architecture. 18 of 24 blocks use linear recurrence attention (not standard softmax). FIM computation via Fisher-Vector Products assumes standard softmax attention. Results from hybrid blocks are incomparable.
- **Observed symptom:** Strong negative correlations in early generations but these reflect the DeltaNet blocks' different sensitivity structure, not the phenomenon we're studying.
- **Status:** Excluded. Data archived.

### Gemma-3-1b-it — Excluded
- **Reason:** Hybrid sliding-window attention. 22/26 blocks use local sliding-window (512 token window); 4/26 use full global attention. Two fundamentally different attention computation regimes within the same block ranking make FIM values incommensurable across blocks.
- **Status:** Excluded. Data archived.

### Gemma 4 (all sizes) — Excluded
- **Reason:** Same hybrid attention as Gemma 3, now compounded by: (1) Per-Layer Embeddings (PLE) — a second conditioning pathway alongside the residual stream in E2B/E4B; (2) Shared KV cache — last N layers reuse K/V projections from earlier layers, meaning those blocks don't compute their own K/V; (3) Multimodal training (text+image+audio) makes text-only recursive setup non-standard. All four confounds present simultaneously.
- **Status:** Excluded. Assessed post-release (April 2026) and found unsuitable.

---

## 8. Geometric Analyses: Two Null Findings (Useful)

### 8.1 Drift Subspace Dimensionality

**Hypothesis:** Sequential models show lower-rank drift (structured rotation) than parallel models (diffuse rotation).

- **Result:** Null. Effective rank is dominated by matrix dimension (larger matrices → more dims). After normalising: sequential ~2–13%, parallel ~2%. No clean architecture split.
- **Use in paper:** Rules out "unstructured drift" as the explanation for architecture-dependent collapse velocity. The mechanism is the FIM-drift correlation, not geometric differences in the drift itself.

### 8.2 Drift Direction Cosine Similarity

**Hypothesis:** Sequential drift is near-zero cosine (random each generation); parallel drift is positive cosine (compounding direction).

- **Result:** Null on architecture dependence. ALL models show negative cosine similarity (−0.19 to −0.49). 0% positive blocks. No architecture split.
- **Explanation:** Oscillatory collapse. Optimizer state resets between generations; next generation's synthetic data partially reflects previous drift; new gradients partially oppose it.
- **Use in paper:** Explains ortho failure — not only is drift low-rank, the direction itself anti-correlates between generations. Also confirms that architecture type predicts FIM-magnitude relationships but not directional structure.

---

## 9. Outstanding Vulnerabilities and Mitigations

| Vulnerability | Severity | Mitigation |
|---|---|---|
| Only 2 parallel model families | Medium | d=5.96, p=3.8×10⁻⁹ — large enough effect that 3rd parallel family would add confidence but isn't necessary. Falcon-7B identified as candidate if needed. |
| β₂ ablation only on SmolLM | Medium | Shows the mechanism on the primary model. Prediction for parallel architectures (β₂ inversion should be absent or weaker) is not tested. Note this as future work. |
| Llama 0/5 significant | Low | Expected — 16 blocks gives insufficient statistical power for Spearman. Sign is correct (negative). Retained as qualitative support, not as statistical proof. |
| GPT-2 only 2/5 significant | Low | Same reason. 12 blocks. Sign is correct in all 5 generations. |
| EWC lambda range limited | Low | Tested λ ∈ {50,100,500}. U-shape confirmed across 3 points. λ<10 not tested — predict approaches normal (+29.5%) as λ→0. Note as open question. |
| SLQ convergence not shown | Low | Standard practice. Appendix should state num_batches, num_eigenvalues, and note that results are stable across multiple runs on SmolLM. |