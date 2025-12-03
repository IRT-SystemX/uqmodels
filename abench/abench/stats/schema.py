# Test intermediate level

from typing import Callable, Optional
from abench.utils import apply_mask
import abench.stats.stats as ABstats 
from abench.stats.stats import TEST_REGISTRY 
from functools import partial

import numpy as np

def apply_test(
    x1,
    *,
    test_name: str,
    x2=None,
    **test_kwargs,
):
    """
    Apply a registered statistical test (one- or two-sample) on already
    selected populations.

    The behavior (one-sample vs two-sample, paired vs unpaired, normalized vs
    raw statistic) is defined in TEST_REGISTRY[test_name].

    Parameters
    ----------
    x1 : array_like
        First input sample (or the only sample for one-sample tests).
    test_name : str
        Name of the test in TEST_REGISTRY.
    x2 : array_like, optional
        Second input sample (for two-sample tests).
    **test_kwargs :
        Additional keyword arguments forwarded to the underlying test function
        (e.g. alternative, normalize flags if not already fixed by partial,
        centering options, etc.).

    Returns
    -------
    SampleTestResult or OneSampleTestResult
        Result object returned by the underlying test function, typically with:
        - statistic: raw or normalized test statistic (depending on registry),
        - pvalue: associated p-value (if defined for the test),
        - extra: metadata (sample sizes, raw statistic, normalization info, ...).

    Raises
    ------
    KeyError
        If test_name is not found in TEST_REGISTRY.
    ValueError
        If the provided samples are inconsistent with the test arity or pairing.
    """
    if test_name not in TEST_REGISTRY:
        raise KeyError(
            f"Unknown test_name '{test_name}'. "
            f"Available: {list(TEST_REGISTRY.keys())}"
        )

    entry = TEST_REGISTRY[test_name]
    fn = entry["fn"]
    arity = entry["arity"]
    paired = entry.get("paired", False)

    x1 = np.asarray(x1)

    # ----- One-sample case -----
    if arity == 1:
        if x2 is not None:
            raise ValueError(
                f"One-sample test '{test_name}' received an unexpected second sample."
            )
        return fn(x1, **test_kwargs)

    # ----- Two-sample case -----
    if x2 is None:
        raise ValueError(f"Two-sample test '{test_name}' requires a second sample x2.")

    x2 = np.asarray(x2)

    if paired and x1.size != x2.size:
        raise ValueError(
            f"Paired test '{test_name}' requires same number of observations, "
            f"got {x1.size} and {x2.size}."
        )

    return fn(x1, x2, **test_kwargs)

def apply_schema(
    x1_source,
    *,
    test_name: str,
    x2_source=None,
    mask1=None,
    mask2=None,
    **test_kwargs,
):
    """
    Build sample(s) using optional masks and apply a registered test.
    One-sample or two-sample behavior comes from TEST_REGISTRY.
    """
    if test_name not in TEST_REGISTRY:
        raise KeyError(f"Unknown test '{test_name}'.")

    entry = TEST_REGISTRY[test_name]
    arity = entry["arity"]
    paired = entry.get("paired", False)

    # --- One-sample ---
    if arity == 1:
        x = apply_mask(x1_source, mask1)
        return apply_test(x, test_name=test_name, **test_kwargs)

    # --- Two-sample ---
    if x2_source is None:
        x2_source = x1_source

    x1 = apply_mask(x1_source, mask1)
    x2 = apply_mask(x2_source, mask2)

    if paired and x1.size != x2.size:
        raise ValueError(f"Paired test '{test_name}' requires equal sample sizes.")

    return apply_test(x1, x2=x2, test_name=test_name, **test_kwargs)

def apply_paired_contrast_mannwhitney_schema(
    sample_ref: np.ndarray,
    sample_cmp: np.ndarray,
    mask_group1: np.ndarray,
    mask_group2: np.ndarray,
):
    """
    Generic two-stage contrast schema based on Mann–Whitney U for paired comparaison
    Warning : it's a rank based test -> unrelevant to catch amplitude, only separability.

    Stage 1: Compare Δ = (sample_ref − sample_cmp) between group1 vs group2
             using a normalized MWU Z-score.

    Stage 2: Compare MWU Z-scores computed separately on sample_ref and
             sample_cmp (group1 vs group2), then return the signed difference,
             indicating how much more (or less) group separability exists in
             sample_cmp compared to sample_ref.

    Returns
    -------
    (value_1, value_2)
        value_1 : Z-score of Δ contrast between groups.
        value_2 : signed difference between group-separation Z-scores of
                  sample_cmp and sample_ref.
    """
    # Safety conversions
    sample_ref = np.asarray(sample_ref)
    sample_cmp = np.asarray(sample_cmp)
    mask_group1 = np.asarray(mask_group1, dtype=bool)
    mask_group2 = np.asarray(mask_group2, dtype=bool)

    # -------------------------
    # Normalization constants
    # -------------------------
    a = mask_group1.sum()
    b = mask_group2.sum()
    mu = (a * b) / 2.0
    sigma = np.sqrt(a * b * (a + b + 1) / 12.0)

    # -------------------------
    # Stage 1: Δ contrast
    # -------------------------
    delta = sample_ref - sample_cmp

    res_delta = apply_test(
        delta[mask_group1],
        x2=delta[mask_group2],
        test_name="mw",
        alternative="greater",
    )
    U_delta = res_delta.extra["U"]
    value_1 = (U_delta - mu) / sigma

    # -------------------------
    # Stage 2: ref and cmp contrasts
    # -------------------------
    # Reference: group2 vs group1
    res_ref = apply_test(
        sample_ref[mask_group2],
        x2=sample_ref[mask_group1],
        test_name="mw",
        alternative="greater",
    )
    U_ref = res_ref.extra["U"]
    z_ref = (U_ref - mu) / sigma

    # Compared sample: group2 vs group1
    res_cmp = apply_test(
        sample_cmp[mask_group2],
        x2=sample_cmp[mask_group1],
        test_name="mw",
        alternative="greater",)
    U_cmp = res_cmp.extra["U"]
    z_cmp = (U_cmp - mu) / sigma

    # Final signed contrast
    value_2 = -(z_ref - z_cmp)
    return value_1, value_2

def apply_paired_contrast_wasserstein_schema(
    sample_ref: np.ndarray,
    sample_cmp: np.ndarray,
    mask_group1: np.ndarray,
    mask_group2: np.ndarray,
    *,
    scale: str = "mad_global",
    test_name_raw: str = "wass",  # raw Wasserstein (non normalisé)
):
    """
    Wasserstein contrast schema analogous to MWU contrast, with global normalization.

    Stage 1 (value_1):
        Paired deltas:
            delta = sample_ref - sample_cmp
        Compare delta distributions between group1 and group2 using Wasserstein-1:
            W_delta = W(delta[group1], delta[group2])
        Then:
            value_1 = W_delta / global_scale

    Stage 2 (value_2):
        Group separability in ref and cmp:
            W_ref = W(sample_ref[group1], sample_ref[group2])
            W_cmp = W(sample_cmp[group1], sample_cmp[group2])
        Then:
            value_2 = (W_cmp - W_ref) / global_scale

    A single global_scale is used for all distances to ensure comparable units.
    """
    sample_ref = np.asarray(sample_ref)
    sample_cmp = np.asarray(sample_cmp)
    mask_group1 = np.asarray(mask_group1, dtype=bool)
    mask_group2 = np.asarray(mask_group2, dtype=bool)

    if sample_ref.shape != sample_cmp.shape:
        raise ValueError(
            f"Paired Wasserstein contrast requires same shape for ref and cmp, "
            f"got {sample_ref.shape} and {sample_cmp.shape}."
        )

    # ------------------------
    # Global scale
    # ------------------------
    if scale == "mad_global":
        global_arr = np.concatenate([sample_ref, sample_cmp])
        med = np.median(global_arr)
        mad = np.median(np.abs(global_arr - med))
        global_scale = mad if mad > 1e-12 else 1.0
    elif scale == "std_global":
        global_arr = np.concatenate([sample_ref, sample_cmp])
        std = np.std(global_arr)
        global_scale = std if std > 1e-12 else 1.0
    else:
        raise ValueError(f"Unknown scale '{scale}'")



    # ============================
    # Stage 1 — Paired Δ-contrast
    # ============================
    delta = sample_ref - sample_cmp

    W_delta = float(apply_test(delta[mask_group1], 
                               x2=delta[mask_group2], 
                               test_name=test_name_raw).statistic)
    value_1 = W_delta / global_scale

    # ============================
    # Stage 2 — Separability contrast
    # ============================

    W_ref = float(apply_test(sample_ref[mask_group1],
                             x2=sample_ref[mask_group2],
                             test_name=test_name_raw).statistic)
    
    W_cmp = float(apply_test(sample_cmp[mask_group1],
                            x2=sample_cmp[mask_group2],
                            test_name=test_name_raw).statistic)
    
    value_2 = (W_cmp - W_ref) / global_scale

    return value_1, value_2

def apply_wilcoxon_schema(
        sample_ref: np.ndarray,
        sample_cmp: np.ndarray,
        mask_group,
        *,
        alternative="greater"):
    """
    wilcoxon_paired schema on a subset of samples.
    Applies a normalized wilcoxon (Z-score) to (sample_ref, sample_cmp)
    restricted to mask_group.
    """
    res = apply_schema(
        sample_ref,
        x2_source=sample_cmp,
        mask1=mask_group,
        mask2=mask_group,
        test_name="wilcoxon_norm",
        alternative=alternative)
    return float(res.statistic)

def apply_mannwhitneyu_schema(
    sample_ref: np.ndarray,
    sample_cmp: np.ndarray,
    mask_group,
    *,
    alternative: str = "greater",
):
    """
    Mann–Whitney U schema on a subset of samples.
    Applies a normalized MWU (Z-score) to (sample_ref, sample_cmp)
    restricted to mask_group.
    """
    # Sélectionne les deux populations avec apply_mask
    x1 = apply_mask(sample_ref, mask_group)
    x2 = apply_mask(sample_cmp, mask_group)

    # MWU normalisé via TEST_REGISTRY ("mw_norm")
    res = apply_test(
        x1,
        x2=x2,
        test_name="mw_norm",
        alternative=alternative,
    )

    # res.statistic = Z-score (normalisé)
    return float(res.statistic)

SCHEMA_REGISTRY = {
    # ================================================================
    # 1) Generic schema: build samples from masks + apply any test
    # ================================================================
    "basic_stat_test": {
        "fn": apply_schema,
        "description": (
            "Generic schema that builds one or two samples using optional masks "
            "and applies a registered statistical test via apply_stat_test."
        ),
        "interpretation": (
            "Interpretation depends entirely on the underlying statistical test. "
            "Generally:\n"
            "- Large |statistic| → strong evidence against H0.\n"
            "- statistic > 0 indicates the direction defined by 'alternative'.\n"
            "- statistic near 0 → inconclusive / no detectable effect."
        ),
        "outputs": "SampleTestResult or OneSampleTestResult",
    },

    # ================================================================
    # 2) Mann–Whitney contrast schema (Δ + ref/cmp contrast)
    # ================================================================
    "mw_contrast": {
        "fn": apply_paired_contrast_mannwhitney_schema,
        "description": (
            "Two-stage Mann–Whitney contrast.\n"
            "value_1: Z-score of Δ = (sample_ref − sample_cmp) between group1 vs group2.\n"
            "value_2: signed contrast between group separability in sample_cmp vs sample_ref.\n"
            "Both use a common normalization, ensuring comparable scales."
        ),
        "interpretation": (
            "value_1 (Δ contrast):\n"
            "- > 0 (large): group1 has higher Δ = (ref − cmp) than group2 → effect present.\n"
            "- < 0 (large): reversed effect.\n"
            "- near 0: no significant shift between groups.\n\n"
            "value_2 (ref vs cmp contrast):\n"
            "- > 0 (large): sample_cmp separates group1/group2 MORE strongly than sample_ref.\n"
            "- < 0 (large): sample_ref separates better.\n"
            "- near 0: both samples discriminate groups similarly (or weakly)."
        ),
        "base_tests": ["mw"],
        "outputs": "(value_1, value_2)",
    },
    "wass_contrast" : {
        "fn": apply_paired_contrast_wasserstein_schema,
        "description": (
            "Wasserstein-based contrast schema analogous to MWU contrast, with global normalization. "
            "Stage 1: Wasserstein-1 distance between per-sample deltas (ref − cmp) across groups. "
            "Stage 2: change in group separability from ref to cmp, measured by Wasserstein-1."
        ),
        "interpretation": (
            "value_1:\n"
            "- Measures how differently the effect (ref − cmp) is distributed between group1 and group2.\n"
            "- ≈ 0: similar effect amplitude in both groups.\n"
            "- > 0 (large): strong differential effect between groups.\n\n"
            "value_2:\n"
            "- Measures how group separability changes from ref to cmp.\n"
            "- > 0: cmp increases the distributional distance between groups (stronger separation).\n"
            "- < 0: cmp reduces the distance between groups (groups become more similar).\n"
            "- ≈ 0: group separability unchanged."
        ),
        "base_tests": ["wass"],
        "outputs": "(value_1, value_2)",
    },

    # ================================================================
    # 3) Paired Wilcoxon schema on a subset
    # ================================================================
    "wilcoxon_paired_subset": {
        "fn": partial(apply_wilcoxon_schema, alternative="greater"),
        "description": (
            "Applies a normalized paired Wilcoxon signed-rank test (Z-score) to "
            "sample_ref vs sample_cmp restricted to mask_group."
        ),
        "interpretation": (
            "Z-score measures how often sample_ref > sample_cmp within the masked subset.\n"
            "- Z >> 0: strong evidence that sample_ref > sample_cmp (directional effect).\n"
            "- Z << 0: strong evidence that sample_ref < sample_cmp.\n"
            "- |Z| small: difference not statistically detectable in this subgroup."
        ),
        "base_tests": ["wilcoxon"],
        "outputs": "scalar_z",
    },

    # ================================================================
    # 4) Mann–Whitney schema on a subset
    # ================================================================
    "mw_subset": {
        "fn": partial(apply_mannwhitneyu_schema, alternative="greater"),
        "description": (
            "Normalized Mann–Whitney U (Z-score) on sample_ref vs sample_cmp restricted "
            "to mask_group. Equivalent to a classical MWU Z-statistic on the masked samples."
        ),
        "interpretation": (
            "Z-score measures whether sample_ref tends to be greater than sample_cmp in "
            "the selected subset.\n"
            "- Z >> 0: sample_ref > sample_cmp (first-order stochastic dominance).\n"
            "- Z << 0: reversed dominance sample_ref < sample_cmp.\n"
            "- |Z| small: distributions considered similar for this subset."
        ),
        "base_tests": ["mw_norm"],
        "outputs": "scalar_z",
    },
}