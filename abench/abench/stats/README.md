# Statistical Testing Framework

## Overview

This module provides a structured framework for running statistical tests in benchmarking pipelines.  
It is organized in two layers:

- **Layer 1**: atomic statistical tests (Mann–Whitney, Wilcoxon paired, Wasserstein, t-test, one-sample tests, etc.), each wrapped in a unified API and optionally normalized (Z-scores).
- **Layer 2**: generic machinery for building populations from arrays and masks (`apply_stat_test`), plus predefined **test schemas** implementing higher-level testing logic (contrast tests, paired subgroup tests, etc.).

The framework is designed to make statistical testing:
- consistent,
- composable,
- repeatable,
- and fully declarative through a registry-based architecture.

This framework try to cleanly separates:
- Low-level tests (Layer 1)  
- Test execution (apply_stat_test) 
- Population selection (apply_mask)  
- High-level logic (schemas)

ensuring *reproducibility, clarity, and rigorous statistical comparisons* in trustworthy AI benchmarking.

---

# 1. Layer 1 – Low-level Statistical Test Wrappers

Layer 1 provides uniform wrappers around statistical tests.  
A test wrapper:

- receives **two populations** (already extracted)  
- performs the statistical test  
- returns a standardized `TwoSampleTestResult`:
  - `statistic` — either raw or normalized  
  - `pvalue` — as provided by SciPy (when applicable)  
  - `extra` — metadata (sample sizes, raw statistic, normalization info, etc.)

Most tests now include a `normalize: bool` parameter.

## 1.1. Normalized versions – philosophy

Some statistical tests produce **statistics that are not directly comparable** across:
- different sample sizes
- different experiments
- different distributions

To enable consistent evaluation and combination of tests (e.g., contrast testing),
normalized variants are provided.

### Why normalization?
Normalization transforms a statistic into:
- a **dimensionless** metric
- often approximating a **Z-score**
- or a **scale-invariant distance** (e.g., Wasserstein normalized by MAD)

This greatly facilitates:
- interpretation  
- comparison across models / attacks / datasets  
- construction of higher higher-order composite metrics  

---

##  1.2. Unified Test Registry

All tests from Layer 1 are registered in a single dictionary:

```
TEST_REGISTRY = {    
  "t": {"fn": student_t_test,
        "arity": 2,
        "paired": False,
        "normalize": False,  
        "description": "Student two-sample t-test assuming equal variances."": ..., "arity": 2, "paired": False, ... },
    ...
    "t1": ..
    ...
}
```

Each entry includes:

- `fn` : the test function  
- `arity` : 1 (one-sample) or 2 (two-sample)  
- `paired` : only for two-sample tests  
- `normalize` : False = Raw test-statistic and True = Normalized test-statistic
- `description` : concise English documentation  

---

## 1.4. Summary Table of All Integrated Tests

### 🔵 Two-sample tests (paired)

| Name | Test | Goal | Notes |
|------|------|------|-------|
| `paired_t` | Paired t-test | Mean difference for matched samples | Parametric |
| `wilcoxon` | Wilcoxon signed-rank | Median/location difference for paired samples | Non-parametric; robust |

---

### 🟠 Two-sample tests (unpaired)

| Name | Test | Goal | Notes |
|------|------|------|-------|
| `t` | Student t-test | Mean difference | Assumes equal variances |
| `welch` | Welch t-test | Mean difference | Robust to unequal variances |
| `mw` | Mann–Whitney U | Location/median difference | Non-parametric; independent samples |
| `ks` | Kolmogorov–Smirnov | Compare full CDFs | Sensitive to any distribution shift |
| `cvm` | Cramér–von Mises | Smooth global distribution comparison | Captures shape differences |
| `ad2` | Anderson–Darling (two-sample) | Distribution equality with tail sensitivity | More powerful in tails |
| `levene` | Levene variance test | Variance equality | Robust under non-normality |

---

### 🟣 One-sample tests

| Name | Test | Goal | Notes |
|------|------|------|-------|
| `t1` | One-sample t-test | Mean vs reference | Parametric |
| `wilcoxon1` | One-sample Wilcoxon | Median vs reference | Non-parametric |
| `shapiro` | Shapiro–Wilk | Normality of distribution | Ideal for small/medium n |
| `ad1` | Anderson–Darling (one-sample) | Goodness-of-fit to target distribution |Tail-sensitive |
| `chi2_var` | Chi-square variance test | Variance vs target | Exact under normality |
| `binom` | Binomial proportion test | Proportion vs target value | Perfect for coverage tests |

---

### Tests with Normalized Variants

| Test Name | Raw Version | Normalized Version | Normalization Rule | Notes |
|-----------|-------------|--------------------|--------------------|-------|
| **Mann–Whitney U** | `mw` | `mw_norm` | Z = (U − μ)/σ | Z-score of rank statistic |
| **Kolmogorov–Smirnov** | `ks` | `ks_norm` | √(n1 n2/(n1+n2)) · D | Scaled KS distance |
| **Levene (variance)** | `levene` | `levene_norm` | Z = Φ⁻¹(1 − p/2) | Directionless Z-score |
| **Student t-test** | `t` | `t_norm` | Z = sign(t)·Φ⁻¹(1 − p/2) | Comparable Z-scale |
| **Welch t-test** | `welch` | `welch_norm` | Same as above | Robust to unequal variances |
| **Wasserstein-1** | `wass` | `wass_norm` | W₁ / MAD | Dimensionless Wasserstein |
| **Wilcoxon (paired)**  | `wilcoxon` | `wilcoxon_norm` | Z = (W − μ)/σ  | Z-score of rank |  



### Comparaison des tests pour analyser A vs B en cross-validation

| Test  | Type d'échantillons | Hypothèse testée | Hypothèses de validité | Ce qu’il détecte | alternative="greater" signifie… | Quand l’utiliser ? |

|---|---|---|---|---|---|---|
| **Wilcoxon signed-rank** | Apparié (1-sample sur d) | median(d) = 0 | Non paramétrique, distribution symétrique| Différence de médiane des per-fold differences   | A > B ↔ d = A−B > 0 | Le test par défaut en CV : robuste, apparié, aucune normalité. |
| **t-test apparié**            | Apparié (2-samples paired)   | mean(A−B) = 0                                     | Différences ~ normales (ou K assez grand)| Différence de moyenne des per-fold differences   | A > B ↔ mean(A−B) > 0                          | Si tu veux tester la différence de moyenne et K≥10–20.          |
| **Wilcoxon paired (2-sample)**| Apparié                      | median(A) = median(B)                             | Non paramétrique, distribution symétrique| Différence de tendance centrale A vs B           | A > B ↔ A_i > B_i en majorité                  | Équivalent à signed-rank si A,B appariés ; version 2-vecteurs.  |
| **Mann–Whitney U (MWU)**      | Non apparié                  | P(A>B) = 0.5                                      | Indépendance, distributions quelconques   | Différence de rangs entre deux populations       | A > B ↔ scores_A stochastically ≥ scores_B     | Il faut s'assurer d'aggréger de statistique de test normalisé Z~N(0,1)  |

📝 Notes d’interprétation
✔️ Pourquoi Wilcoxon signed-rank est le meilleur choix en CV ? :
- Utilise les différences par fold → exploit le couplage naturel des données.
- Non paramétrique → aucune hypothèse forte sur la distribution des performances.
- Fonctionne très bien même avec peu de folds (K ≥ 5).

✔️ Quand utiliser le t-test apparié ?
- Si tu veux analyser les moyennes plutôt que la médiane.
- Si les différences A−B semblent approximativement normales (ou si K est “grand”).
- Plus puissant si les conditions sont remplies, mais moins robuste que Wilcoxon.

✔️ Alternative directionnelle (greater, less, two-sided)

Pour un vecteur de différences d = A − B :

- greater → tester si A > B 𝐻1:𝑑>0
- less → tester si A < B 𝐻1:𝑑<0
- two-sided → tester juste s’il y a une différence, dans un sens ou l’autre

| Test                        | Apparié ? | Z normalisable ? | Agrégation valide ? | Modes recommandés                                      | Notes importantes                                                 |
|-----------------------------|-----------|-------------------|----------------------|--------------------------------------------------------|-------------------------------------------------------------------|
| **Wilcoxon signed-rank**    | ✔️ oui    | ✔️ oui (Z global) | ✔️ OUI               | - mean Z                                               | Test par défaut pour CV (robuste, non-paramétrique).              |
|                             |           |                   |                      | - Wilcoxon signed-rank sur Z                          |                                                                   |
|                             |           |                   |                      | - sign test                                            |                                                                   |
| **t-test apparié**          | ✔️ oui    | ✔️ oui            | ✔️ OUI               | - mean Z (idéal)                                       | Puissant mais dépend d’une approx. normale des différences.       |
|                             |           |                   |                      | - Fisher (sur p-values)                                |                                                                   |
| **Wilcoxon paired (2-sample)** | ✔️ oui | ✔️ oui            | ✔️ OUI               | - mean Z                                               | Équivalent conceptuel au signed-rank mais sous forme 2-échantillons|
|                             |           |                   |                      | - Wilcoxon signed-rank sur Z                          |                                                                   |
| **MWU (Mann–Whitney U)**    | ❌ non    | ✔️ oui            | ❌ NON               | ❌ **Aucun mode d’agrégation recommandé**              | Test non apparié → incompatible avec cross-validation.            |
|                             |           |                   |                      |                                                        | Même Z-normalisé ≠ test correct en CV.                            |

## 1.5. Examples

### 1.5.1 Mann–Whitney U (unpaired)

```python
result = apply_stat_test(
    x1_source=delta_ratio,
    mask1=mask_altered,
    mask2=mask_normal,
    test_name="mw",
    alternative="greater",
)
print(result.statistic, result.pvalue)
```

---

### 1.5.2 Wilcoxon (paired)

```python
result = apply_stat_test(
    x1_source=ratio_control,
    x2_source=ratio_attack,
    mask1=test_mask,
    test_name="wilcoxon",
)
```

---

### 1.5.3 One-sample t-test

```python
result = apply_stat_test(
    x1_source=errors,
    mask1=test_mask,
    test_name="t1",
    mu0=0.0,
)
```

---

### 1.5.4 Coverage test (binomial)

```python
result = apply_stat_test(
    x1_source=inside_interval_bool,
    mask1=test_mask,
    test_name="binom",
    p0=0.95,
)
```

# 1.6. Interpretation of Normalized Statistics

### 1.6.1 Z-score normalization
Normalized statistics approximating a Z-score:

- **Positive large value** → strong evidence populations differ in that direction  
- **Near zero** → distributions are similar  
- **Negative** (for signed tests) → reverse direction effect  

Applies to:
- Mann–Whitney (Z)
- Student/Welch (signed Z)
- Wilcoxon signed-rank (Z)  
- Levene (unsigned Z)

### 1.6.2 Scale-normalized metrics
Metrics normalized by a scale (e.g., MAD):

- **Dimensionless**
- **Robust to outliers**
- **Comparable between experiments**

Applies to:
- Wasserstein normalized
- KS normalized (via √n_eff scaling)

---

# 2. Layer 2 – Schema for statistical test application

Layer 2 provides the intermediate logic connecting low-level statistical tests (Layer 1) with high-level evaluation schemas used in benchmark analysis.  

While Layer 1 defines *what* a statistical test is, Layer 2 defines *how* these tests are applied to concrete experimental populations.

## 2.1 Key Ideas

### 2.1.1 Declarative Access to Statistical Tests  
Layer 2 exposes a unified interface (`apply_test`) that selects and applies any registered test from `TEST_REGISTRY`.  
It abstracts over the differences between:
- one-sample vs two-sample tests,  
- paired vs unpaired tests,  
- raw vs normalized statistics.

This ensures a consistent and predictable behavior across all downstream evaluation mechanisms.

### 2.1.2 Consistent Sample Extraction  
Layer 2 used mask utilities (`apply_mask`, `apply_mask_along_dim`) that convert boolean masks or index lists into properly structured sample subsets.  
This step provides:
- correct and reproducible sample selection,
- compatibility with heterogeneous dataset formats,
- uniform preprocessing before any statistical computation.

### 2.1.3 Foundation for Higher-Level Test Schemas  
Layer 2 deliberately avoids embedding domain-specific logic or experimental semantics.  
Instead, it provides the primitives that higher-level schemas rely on:
- sample selection,
- paired/unpaired alignment,
- central handling of normalization,
- routing to Layer-1 statistical functions.

Schemas such as **MWU Contrast**, **Wilcoxon Subset**, or **Wasserstein Delta Contrast** are thus easy to implement, maintain, and evaluate consistently.

### 2.1.4 Extensibility and Reproducibility  
Because Layer 2 centralizes the invocation of statistical machinery, extending the system is straightforward:
- Adding a new statistical test → extend Layer 1  
- Adding a new evaluation schema → combine Layer 2 primitives  
- Reproducibility is guaranteed by isolating test invocation from experimental logic

This ensures a scalable and maintainable statistical evaluation pipeline.

Schemas implement **high-level statistical testing logic** relying on:

## 2.2 - Apply_test : Low-level interface for statistical test application 

A minimal interface:

```
apply_test(x1, x2=None, test_name="mw_norm", **kwargs)
```

- Delegates to the function registered under `TEST_REGISTRY[test_name]`.
- Supports one-sample or two-sample tests, paired/unpaired.
- Assumes `x1` and `x2` are **already extracted subsets**.

---
### 2.2.1 Responsibilities of apply_test

| Responsibility | Description |
|----------------|-------------|
| Test lookup | Retrieve function & metadata from registry |
| Arity resolution | 1-sample vs 2-sample |
| Pairing logic | Automatic handling of paired vs unpaired tests |
| Error validation | Ensure consistency of sizes, masks, and test type |
| Kwargs forwarding | Pass test-specific args (e.g., `mu0`, `dist`, `alternative`) |

This abstraction ensures consistency and lowers maintenance costs.

---

## 2.3 Test Schema : High-level wrapper or process for test schema application

Beyond applying individual statistical tests, many evaluation scenarios require
**comparing how two populations differ across multiple conditions**, or measuring how
an intervention modifies the relationship between groups.  
Contrast schemas address this need by defining *structured, multi-stage*
statistical evaluations built on top of the core Layer-1 and Layer-2 primitives.

A contrast schema typically operates on:

- a **reference population** (baseline),
- a **comparison population** (e.g., attacked, modified, shifted),
- one or more **group partitions**, and
- a combination of **paired** and **unpaired** statistical operations.

Each schema outputs one or more values that summarize:
1. how strongly the intervention changes the data **within groups** (effect amplitude),  
2. how it alters the **separability between groups**,  
3. and whether these changes are consistent across populations.

This modular design enables the implementation of sophisticated evaluation
strategies—such as MWU-based contrast or Wasserstein-based amplitude contrast—
while maintaining consistency, interpretability, and extensibility across the
entire benchmarking framework.

### 2.3.1 apply_schema

Generic schema performing selection + execution of any test.

**Interpretation**
- Large |statistic| → strong evidence  
- Statistic > 0 → effect in direction of “greater”  
- Statistic near 0 → inconclusive  

---

### 2.3.2 mw_contrast — Two-stage Mann–Whitney contrast

Implements your original benchmark contrast logic.

**Stage 1**
Z-score of:
Δ = (sample_ref − sample_cmp)  
between `group1` and `group2`.

**Stage 2**
Compare how well sample_ref and sample_cmp separately discriminate groups  
(using shared normalization across MWUs).

**Interpretation**
- value_1 > 0 → Δ larger in group1  
- value_2 > 0 → cmp discriminates better than ref  
- values near 0 → no strong effect  

---

### 2.3.3 wilcoxon_paired_subset

Paired Wilcoxon signed-rank Z-score restricted to a mask.

**Interpretation**
- Z >> 0: ref > cmp  
- Z << 0: ref < cmp  
- |Z| small: no difference  

---

### 2.3.4 mw_subset

Z-normalized Mann–Whitney on masked subset.

**Interpretation**
- Z >> 0: ref > cmp  
- Z << 0: ref < cmp  
- |Z| small: similar distributions  

---

### 2.3.4  mw_contrast

The **Wasserstein Delta Contrast Schema** is a distance-based analogue of the MWU Contrast Schema. It is designed to be sensitive to **amplitude differences**, not only rank changes, making it suitable for detecting distributional shifts where magnitude matters.

The Wasserstein Delta Contrast Schema provides:

- **Amplitude sensitivity** (unlike rank-based MWU),
- **Paired inference** at the effect level,
- **Unpaired inference** at the group-separability level,
- **Global normalization** ensuring meaningful comparison between components,
- A structure parallel to the MWU Contrast Schema, but richer for continuous shift detection.

Its integration in Layer 2+ demonstrates the flexibility of the system: high-level experimental logic can be built compositionally from clean, modular statistical primitives.

**Stage 1 — Paired Δ-Contrast**
We compute the per-sample effect:
\[
\Delta = \text{sample\_ref} - \text{sample\_cmp}
\]
and measure how differently this effect is distributed across groups:
\[
\text{value\_1} = 
\frac{ W_1(\Delta_{\text{group1}},\, \Delta_{\text{group2}}) }
     { \text{global\_scale} }
\]
 

**Stage 2 — Change in Group Separability**
We assess how much the separation between groups changes from reference to comparison:

\[
\text{value\_2} =
\frac{ W_1(\text{cmp}_{g1},\text{cmp}_{g2}) 
     - W_1(\text{ref}_{g1},\text{ref}_{g2}) }
     { \text{global\_scale} }
\]

**Interpretation**
- `value_1 ≈ 0` → similar effect amplitude for both groups  
- `value_1 > 0` → stronger effect in group1  
- `value_1 < 0` → stronger effect in group2
- `value_2 > 0` → comparison condition increases separability  
- `value_2 < 0` → comparison condition reduces separability  
- `value_2 ≈ 0` → no meaningful change  
---

## 2.4 Example Usage

### Mann–Whitney contrast

```
run_schema = SCHEMA_REGISTRY["mw_contrast"]["fn"]
value_1, value_2 = run_schema(
    sample_ref=ratio_ref,
    sample_cmp=ratio_cmp,
    mask_group1=mask_altered,
    mask_group2=mask_normal,
)
```

### Wilcoxon subset

```
run_schema = SCHEMA_REGISTRY["wilcoxon_paired_subset"]["fn"]
z = run_schema(
    sample_ref=ref_values,
    sample_cmp=cmp_values,
    mask_group=ctx_mask,
)
```

### Generic schema

```
run_schema = SCHEMA_REGISTRY["basic_stat_test"]["fn"]
res = run_schema(
    x1_source=data,
    test_name="mw_norm",
    mask1=mask_A,
    mask2=mask_B,
)
```

---

# 3. Design Goals

This architecture provides:

- **Separation of concerns**  
  (statistical logic vs application/selection logic)

- **Modularity**  
  (adding tests does not affect application layer)

- **Reliability and clarity**  
  thanks to automatic handling of masks, pairing, and arity

- **Flexibility**  
  easily extended to custom tests, cross-experiment pipelines, and robustness evaluation

- **Reusability**  
  reusable in any benchmarking pipeline dealing with masked data subsets

A robust foundation for trustworthy AI evaluation and uncertainty analysis.
