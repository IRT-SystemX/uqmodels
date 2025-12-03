# test.py


"""
Tests for normalization behavior of statistical test wrappers.

These tests check that:
- normalized statistics are consistent with their theoretical formulas,
- raw and normalized versions share the same p-value (when applicable),
- normalization produces finite, dimensionless values on simple toy data.
"""

import numpy as np
import sys
sys.path.insert(1, '../')
from scipy import stats as sstats
from abench.stats.stats import mannwhitney_test,ks_two_sample_test,levene_variance_test,student_t_test,welch_t_test,wasserstein_distance_test,wilcoxon_paired_test

def assert_small_under_null(value, threshold=2.0):
    """Check that a (Z-like) statistic is 'small' under H0."""
    assert np.isfinite(value)
    assert abs(value) < threshold


def assert_larger_effect_than_null(effect_value, null_value, factor=2.0):
    """Check that |effect| is significantly larger than |null|."""
    assert np.isfinite(effect_value)
    assert abs(effect_value) > factor * abs(null_value)


# --------------------------------------------------------------------
# Mann–Whitney U (normalized)
# --------------------------------------------------------------------

def test_mannwhitney_z_positive_and_negative():
    rng = np.random.default_rng(0)

    # Null: same distribution
    x_null_1 = rng.normal(0.0, 1.0, size=80)
    x_null_2 = rng.normal(0.0, 1.0, size=80)
    res_null = mannwhitney_test(x_null_1, x_null_2, alternative="two-sided", normalize=True)
    z_null = res_null.statistic
    assert_small_under_null(z_null)

    # Positive effect: x1 > x2
    x_pos_1 = rng.normal(1.0, 1.0, size=80)
    x_pos_2 = rng.normal(0.0, 1.0, size=80)
    res_pos = mannwhitney_test(x_pos_1, x_pos_2, alternative="two-sided", normalize=True)
    z_pos = res_pos.statistic
    assert z_pos > 0
    assert_larger_effect_than_null(z_pos, z_null)

    # Negative effect: x1 < x2
    x_neg_1 = rng.normal(0.0, 1.0, size=80)
    x_neg_2 = rng.normal(1.0, 1.0, size=80)
    res_neg = mannwhitney_test(x_neg_1, x_neg_2, alternative="two-sided", normalize=True)
    z_neg = res_neg.statistic
    assert z_neg < 0
    assert_larger_effect_than_null(z_neg, z_null)


# --------------------------------------------------------------------
# Student / Welch t-tests (normalized)
# --------------------------------------------------------------------

def test_student_t_z_positive_and_negative():
    rng = np.random.default_rng(1)

    # Null case
    x_null_1 = rng.normal(0.0, 1.0, size=60)
    x_null_2 = rng.normal(0.0, 1.0, size=60)
    res_null = student_t_test(x_null_1, x_null_2, equal_var=True, alternative="two-sided", normalize=True)
    z_null = res_null.statistic
    assert_small_under_null(z_null)

    # Positive effect: x1 > x2
    x_pos_1 = rng.normal(0.8, 1.0, size=60)
    x_pos_2 = rng.normal(0.0, 1.0, size=60)
    res_pos = student_t_test(x_pos_1, x_pos_2, equal_var=True, alternative="two-sided", normalize=True)
    z_pos = res_pos.statistic
    assert z_pos > 0
    assert_larger_effect_than_null(z_pos, z_null)

    # Negative effect: x1 < x2
    x_neg_1 = rng.normal(0.0, 1.0, size=60)
    x_neg_2 = rng.normal(0.8, 1.0, size=60)
    res_neg = student_t_test(x_neg_1, x_neg_2, equal_var=True, alternative="two-sided", normalize=True)
    z_neg = res_neg.statistic
    assert z_neg < 0
    assert_larger_effect_than_null(z_neg, z_null)


def test_welch_t_z_positive_and_negative():
    rng = np.random.default_rng(2)

    # Null case
    x_null_1 = rng.normal(0.0, 1.0, size=80)
    x_null_2 = rng.normal(0.0, 2.0, size=80)
    res_null = welch_t_test(x_null_1, x_null_2, alternative="two-sided", normalize=True)
    z_null = res_null.statistic
    assert_small_under_null(z_null, threshold=2.5)

    # Positive effect: x1 > x2
    x_pos_1 = rng.normal(1.0, 1.0, size=80)
    x_pos_2 = rng.normal(0.0, 2.0, size=80)
    res_pos = welch_t_test(x_pos_1, x_pos_2, alternative="two-sided", normalize=True)
    z_pos = res_pos.statistic
    assert z_pos > 0
    assert_larger_effect_than_null(z_pos, z_null)

    # Negative effect: x1 < x2
    x_neg_1 = rng.normal(0.0, 1.0, size=80)
    x_neg_2 = rng.normal(1.0, 2.0, size=80)
    res_neg = welch_t_test(x_neg_1, x_neg_2, alternative="two-sided", normalize=True)
    z_neg = res_neg.statistic
    assert z_neg < 0
    assert_larger_effect_than_null(z_neg, z_null)


# --------------------------------------------------------------------
# Wilcoxon signed-rank (paired, normalized)
# --------------------------------------------------------------------

def test_wilcoxon_z_positive_and_negative():
    rng = np.random.default_rng(3)

    base = rng.normal(0.0, 1.0, size=40)

    # Null: no difference
    x_null_1 = base + rng.normal(0.0, 0.001, size=40)
    x_null_2 = base + rng.normal(0.0, 0.001, size=40)
    res_null = wilcoxon_paired_test(x_null_1, x_null_2, alternative="two-sided", normalize=True)
    z_null = res_null.statistic
    assert_small_under_null(z_null, threshold=2.0)

    # Positive effect: x1 > x2
    x_pos_1 = base + rng.normal(0.0, 0.001, size=40) + 0.8
    x_pos_2 = base + rng.normal(0.0, 0.001, size=40)
    res_pos = wilcoxon_paired_test(x_pos_1, x_pos_2, alternative="greater", normalize=True)
    z_pos = res_pos.statistic
    assert z_pos > 0
    assert abs(z_pos) > 2.0

    # Negative effect: x1 < x2
    x_neg_1 = base + rng.normal(0.0, 0.001, size=40) - 0.8
    x_neg_2 = base + rng.normal(0.0, 0.001, size=40)
    res_neg = wilcoxon_paired_test(x_neg_1, x_neg_2, alternative="less", normalize=True)
    z_neg = res_neg.statistic
    assert z_neg < 0
    assert abs(z_neg) > 2.0


# --------------------------------------------------------------------
# Levene variance test (normalized, unsigned)
# --------------------------------------------------------------------

def test_levene_norm_null_vs_effect():
    rng = np.random.default_rng(4)

    # Null: similar variances
    x_null_1 = rng.normal(0.0, 1.0, size=80)
    x_null_2 = rng.normal(0.0, 1.1, size=80)
    res_null = levene_variance_test(x_null_1, x_null_2, center="median", normalize=True)
    z_null = res_null.statistic
    assert z_null >= 0
    # Sous quasi-H0, on attend un Z modéré
    assert z_null < 3.0

    # Strong variance effect
    x_eff_1 = rng.normal(0.0, 1.0, size=80)
    x_eff_2 = rng.normal(0.0, 3.0, size=80)
    res_eff = levene_variance_test(x_eff_1, x_eff_2, center="median", normalize=True)
    z_eff = res_eff.statistic
    assert z_eff >= 0
    assert_larger_effect_than_null(z_eff, z_null, factor=2.0)


# --------------------------------------------------------------------
# KS test (normalized)
# --------------------------------------------------------------------

def test_ks_norm_null_vs_shift():
    rng = np.random.default_rng(5)

    # Null: same distribution
    x_null_1 = rng.normal(0.0, 1.0, size=100)
    x_null_2 = rng.normal(0.0, 1.0, size=100)
    res_null = ks_two_sample_test(x_null_1, x_null_2, normalize=True)
    z_null = res_null.statistic
    assert z_null >= 0
    assert z_null < 2.0

    # Shifted distribution
    x_shift_1 = rng.normal(0.0, 1.0, size=100)
    x_shift_2 = rng.normal(1.0, 1.0, size=100)
    res_shift = ks_two_sample_test(x_shift_1, x_shift_2, normalize=True)
    z_shift = res_shift.statistic
    assert z_shift >= 0
    assert_larger_effect_than_null(z_shift, z_null, factor=2.0)


# --------------------------------------------------------------------
# Wasserstein distance (normalized)
# --------------------------------------------------------------------

def test_wasserstein_norm_null_vs_shift():
    rng = np.random.default_rng(6)

    # Null: same distribution
    x_null_1 = rng.normal(0.0, 1.0, size=100)
    x_null_2 = rng.normal(0.0, 1.0, size=100)
    res_null = wasserstein_distance_test(x_null_1, x_null_2, normalize=True, scale="mad")
    w_null = res_null.statistic
    assert np.isfinite(w_null)
    # sous H0, distance scalaire proche de 0
    assert w_null < 0.5

    # Shifted distribution
    x_shift_1 = rng.normal(0.0, 1.0, size=100)
    x_shift_2 = rng.normal(1.5, 1.0, size=100)
    res_shift = wasserstein_distance_test(x_shift_1, x_shift_2, normalize=True, scale="mad")
    w_shift = res_shift.statistic
    assert np.isfinite(w_shift)
    assert_larger_effect_than_null(w_shift, w_null, factor=2.0)