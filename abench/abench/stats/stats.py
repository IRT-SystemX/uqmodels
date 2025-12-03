# Implementation assisted by AI : #TOCHECK details.
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
from scipy import stats as sstats
from functools import partial

# Test low level
@dataclass
class SampleTestResult:
    """
    Container for two-sample test results.

    Attributes
    ----------
    statistic : float
        Main test statistic (t, U, W, D, etc.).
    pvalue : float
        Associated p-value.
    extra : dict
        Optional extra information (sample sizes, effect size, etc.).
    """
    statistic: float
    pvalue: float
    extra: Dict[str, Any] | None = None

def one_sample_t_test(x: np.ndarray, mu0: float = 0.0,**kwargs) -> SampleTestResult:
    """
    One-sample t-test for mean against a reference value.

    Goal
    ----
    Test whether the mean of x differs from a target value mu0.

    Specifics
    ---------
    - Parametric: assumes approximate normality (or large n).
    """
    x = np.asarray(x)
    test = sstats.ttest_1samp(x, popmean=mu0)
    return SampleTestResult(
        statistic=float(test.statistic),
        pvalue=float(test.pvalue),
        extra={"n": x.size, "mu0": mu0},
    )

def one_sample_wilcoxon(x: np.ndarray, mu0: float = 0.0,**kwargs) -> SampleTestResult:
    """
    One-sample Wilcoxon signed-rank test for median against a reference.

    Goal
    ----
    Non-parametric test of whether the median of x equals mu0.

    Specifics
    ---------
    - Works on signed ranks of (x - mu0).
    - Robust alternative to the one-sample t-test.
    """
    x = np.asarray(x)
    test = sstats.wilcoxon(x - mu0)
    return SampleTestResult(
        statistic=float(test.statistic),
        pvalue=float(test.pvalue),
        extra={"n": x.size, "mu0": mu0},
    )

def anderson_normality_test(x: np.ndarray, dist: str = "norm",**kwargs) -> SampleTestResult:
    """
    Anderson–Darling one-sample goodness-of-fit test.

    Goal
    ----
    Test whether x comes from a specified distribution (default: normal).

    Specifics
    ---------
    - More sensitive in the tails than KS.
    - Provides critical values and significance levels.
    """
    x = np.asarray(x)
    test = sstats.anderson(x, dist=dist)
    # No direct p-value; we expose significance levels in 'extra'.
    return SampleTestResult(
        statistic=float(test.statistic),
        pvalue=np.nan,
        extra={
            "n": x.size,
            "dist": dist,
            "critical_values": test.critical_values,
            "significance_level": test.significance_level,
        },
    )

def chi_square_variance_test(x: np.ndarray, sigma2_0: float,**kwargs) -> SampleTestResult:
    """
    Chi-square test for variance against a target value.

    Goal
    ----
    Test whether Var(x) equals a specified value sigma2_0.

    Specifics
    ---------
    - Exact under normality assumption.
    - Tests (n-1)*s^2 / sigma2_0 against a chi-square distribution.
    """
    x = np.asarray(x)
    n = x.size
    if n < 2:
        return SampleTestResult(statistic=np.nan, pvalue=np.nan,
                                   extra={"n": n, "sigma2_0": sigma2_0, "error": "n<2"})
    s2 = np.var(x, ddof=1)
    chi2 = (n - 1) * s2 / sigma2_0
    p_right = 1 - sstats.chi2.cdf(chi2, df=n - 1)
    p_left = sstats.chi2.cdf(chi2, df=n - 1)
    # Two-sided p-value
    p_two = 2 * min(p_left, p_right)
    return SampleTestResult(
        statistic=float(chi2),
        pvalue=float(p_two),
        extra={"n": n, "sigma2_hat": s2, "sigma2_0": sigma2_0},
    )

def binomial_proportion_test(successes: np.ndarray, p0: float,**kwargs) -> SampleTestResult:
    """
    Binomial test for a single proportion against a target p0.

    Goal
    ----
    Test whether the observed proportion of True values equals p0.

    Specifics
    ---------
    - No distributional assumption beyond Bernoulli trials.
    - Useful for coverage / calibration checks.
    """
    successes = np.asarray(successes).astype(bool)
    k = int(successes.sum())
    n = successes.size
    test = sstats.binomtest(k, n, p=p0)
    return SampleTestResult(
        statistic=float(k),
        pvalue=float(test.pvalue),
        extra={"n": n, "k": k, "p0": p0},
    )

def shapiro_normality_test(
    x: np.ndarray,
    *,
    normalize: bool = False,
    **kwargs) -> SampleTestResult:
    """
    Shapiro–Wilk normality test, with optional p-value-based normalization.

    Goal
    ----
    Test whether a 1D sample x comes from a normal distribution.
    Statistic W is in [0, 1], where lower values indicate departure from normality.

    Normalization
    -------------
    - If normalize=False:
        statistic = W (raw Shapiro–Wilk statistic).
    - If normalize=True:
        statistic = Z = Φ^{-1}(1 - p/2)
        (unsigned Z-like score growing with non-normality).

    Parameters
    ----------
    x : array_like
        Input sample.
    normalize : bool, optional
        Whether to convert W to a Z-like normality deviation score.

    Returns
    -------
    SampleTestResult
        - statistic: W or Z
        - pvalue: p-value from scipy.stats.shapiro
        - extra: dict with W, normalize, n
    """
    x = np.asarray(x)

    if x.size < 3:
        return SampleTestResult(
            statistic=np.nan,
            pvalue=np.nan,
            extra={"n": x.size, "empty_or_too_small": True, "normalize": normalize},
        )

    # SciPy Shapiro–Wilk: W, p
    W, p = sstats.shapiro(x)
    W = float(W)
    p = float(p)

    if not normalize:
        stat = W
    else:
        # Convert p-value -> positive Z-like score
        if p <= 0:
            stat = np.inf
        else:
            stat = sstats.norm.isf(p / 2.0)

    return SampleTestResult(
        statistic=stat,
        pvalue=p,
        extra={
            "W": W,
            "normalize": normalize,
            "n": x.size,
        },
    )
# ========= Parametric mean-comparison tests =========

def student_t_test(
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    equal_var: bool = True,
    alternative: str = "two-sided",
    normalize: bool = False,
    **kwargs) -> SampleTestResult:
    """
    Two-sample Student t-test for mean difference, with optional Z-normalization.

    Goal
    ----
    Test whether the means of two independent populations are equal.

    Normalization
    -------------
    - If normalize=False (default):
        statistic = t (raw t-statistic).
    - If normalize=True:
        statistic = Z, a normal-score derived from the p-value:
            Z = sign(t) * Φ^{-1}(1 - p/2)
        where Φ is the standard normal CDF.
        This yields a signed quantity whose magnitude increases with
        evidence against H0, on a common scale across different df.

    Parameters
    ----------
    x1, x2 : array_like
        Independent samples.
    equal_var : bool, optional
        If True, use pooled-variance t-test (classical Student test).
        If False, use Welch correction for unequal variances.
    alternative : {"two-sided", "less", "greater"}, optional
        Alternative hypothesis. Note: scipy's ttest_ind alternative parameter
        is only available in recent versions; if not available, fallback
        to two-sided.
    normalize : bool, optional
        Whether to convert t into a signed Z-score based on the p-value.

    Returns
    -------
    SampleTestResult
        - statistic: t or Z depending on normalize.
        - pvalue: p-value from scipy.stats.ttest_ind.
        - extra: dict with t, n1, n2, equal_var, alternative, normalize.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    n1, n2 = x1.size, x2.size

    if n1 == 0 or n2 == 0:
        return SampleTestResult(
            statistic=np.nan,
            pvalue=np.nan,
            extra={"n1": n1, "n2": n2, "empty": True, "normalize": normalize},
        )

    # Gestion alternative selon version de SciPy
    try:
        test = sstats.ttest_ind(x1, x2, equal_var=equal_var, alternative=alternative)
        t = float(test.statistic)
        p = float(test.pvalue)
    except TypeError:
        # Ancienne version: pas de 'alternative', on fait two-sided
        test = sstats.ttest_ind(x1, x2, equal_var=equal_var)
        t = float(test.statistic)
        p = float(test.pvalue)
        # approximation pour alternative != "two-sided"
        if alternative == "less":
            # H1: mean_x1 < mean_x2
            if t >= 0:
                p = 1.0 - p / 2.0
            else:
                p = p / 2.0
        elif alternative == "greater":
            # H1: mean_x1 > mean_x2
            if t <= 0:
                p = 1.0 - p / 2.0
            else:
                p = p / 2.0

    if normalize:
        if p <= 0.0:
            Z = np.inf * np.sign(t) if t != 0 else np.inf
        else:
            Z_abs = sstats.norm.isf(p / 2.0)  # Z >= 0
            Z = np.sign(t) * Z_abs
        stat = Z
    else:
        stat = t

    return SampleTestResult(
        statistic=stat,
        pvalue=p,
        extra={
            "t": t,
            "n1": n1,
            "n2": n2,
            "equal_var": equal_var,
            "alternative": alternative,
            "normalize": normalize,
        },
    )

def welch_t_test(
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    alternative: str = "two-sided",
    normalize: bool = False,
    **kwargs) -> SampleTestResult:
    """
    Welch two-sample t-test for unequal variances, with optional Z-normalization.

    Goal
    ----
    Test mean difference between two independent groups when variances may differ.

    Normalization
    -------------
    Same as student_t_test: statistic can be raw t or Z-score derived
    from the p-value.

    Parameters
    ----------
    x1, x2 : array_like
        Independent samples.
    alternative : {"two-sided", "less", "greater"}, optional
        Alternative hypothesis.
    normalize : bool, optional
        Whether to convert t into a signed Z-score based on the p-value.

    Returns
    -------
    SampleTestResult
    """
    return student_t_test(
        x1,
        x2,
        equal_var=False,
        alternative=alternative,
        normalize=normalize,
    )

def wasserstein_distance_test(
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    normalize: bool = False,
    scale: str = "mad",
    eps: float = 1e-12,
    **kwargs) -> SampleTestResult:
    """
    First Wasserstein distance (Earth Mover's Distance) between two samples,
    with optional normalization by a scale parameter.

    Goal
    ----
    Measure the distributional distance between two 1D samples.

    Normalization
    -------------
    - If normalize=False (default):
        statistic = W1, the raw Wasserstein distance.
    - If normalize=True:
        statistic = W1 / s, where s is a scale parameter estimated from
        the pooled sample (x1 ∪ x2). Typical choices:
          * "mad": median absolute deviation (robust, default),
          * "std": standard deviation (ddof=1).

        This makes the distance dimensionless and more comparable across
        experiments with different scales.

    Parameters
    ----------
    x1, x2 : array_like
        Samples (1D).
    normalize : bool, optional
        Whether to divide W1 by a scale parameter s.
    scale : {"mad", "std"}, optional
        Scale estimator used when normalize=True:
          - "mad": robust MAD * 1.4826 (approximate σ under normality),
          - "std": standard deviation of the pooled sample.
    eps : float, optional
        Small constant to avoid division by zero when scale is very small.

    Returns
    -------
    SampleTestResult
        - statistic: W1 or normalized W1, depending on normalize.
        - pvalue: NaN (no canonical test distribution is assumed here).
        - extra: dict with W1, scale_value, n1, n2, normalize, scale.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    n1, n2 = x1.size, x2.size

    if n1 == 0 or n2 == 0:
        return SampleTestResult(
            statistic=np.nan,
            pvalue=np.nan,
            extra={"n1": n1, "n2": n2, "empty": True, "normalize": normalize},
        )

    W1 = float(sstats.wasserstein_distance(x1, x2))

    if not normalize:
        stat = W1
        scale_value = None
    else:
        pooled = np.concatenate([x1, x2])
        if scale == "mad":
            med = np.median(pooled)
            mad = np.median(np.abs(pooled - med))
            scale_value = 1.4826 * mad  # ≈ σ under normality
        elif scale == "std":
            scale_value = np.std(pooled, ddof=1)
        else:
            raise ValueError(f"Unknown scale '{scale}'. Use 'mad' or 'std'.")

        if scale_value < eps:
            stat = np.nan  # distance not meaningful if scale ~ 0
        else:
            stat = W1 / scale_value

    return SampleTestResult(
        statistic=stat,
        pvalue=np.nan,
        extra={
            "W1": W1,
            "scale_value": scale_value,
            "scale": scale,
            "n1": n1,
            "n2": n2,
            "normalize": normalize,
        },
    )

def paired_t_test(x1: np.ndarray, x2: np.ndarray, **kwargs) -> SampleTestResult:
    """
    Paired t-test for mean difference between matched observations.

    Goal
    ----
    Test whether the mean of (x1 - x2) is zero for paired samples.

    Specifics
    ---------
    - Requires x1 and x2 to be aligned one-to-one (same length).
    - Assumes normality of the differences x1 - x2.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    if x1.size != x2.size:
        raise ValueError(f"Paired t-test requires same length, got {x1.size} and {x2.size}.")
    test = sstats.ttest_rel(x1, x2)
    return SampleTestResult(
        statistic=float(test.statistic),
        pvalue=float(test.pvalue),
        extra={"n": x1.size},
    )

# ========= Non-parametric rank-based tests =========

# ============================================================
# Mann–Whitney U (rank-sum) with optional Z-normalization
# ============================================================

def mannwhitney_test(
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    alternative: str = "greater",
    normalize: bool = False,
    **kwargs) -> SampleTestResult:
    """
    Mann–Whitney U test (Wilcoxon rank-sum), with optional Z-normalization.

    Goal
    ----
    Compare two independent samples using a rank-based non-parametric test.

    Normalization
    -------------
    - If normalize=False (default):
        statistic = U (raw Mann–Whitney statistic).
    - If normalize=True:
        statistic = Z = (U - mu) / sigma
        where mu = a*b/2 and sigma^2 = a*b*(a+b+1)/12 under H0.

    Parameters
    ----------
    x1, x2 : array_like
        Independent samples.
    alternative : {"two-sided", "less", "greater"}, optional
        Alternative hypothesis, forwarded to scipy.stats.mannwhitneyu.
    normalize : bool, optional
        Whether to return a Z-normalized statistic instead of raw U.

    Returns
    -------
    SampleTestResult
        - statistic: U or Z, depending on normalize.
        - pvalue: p-value returned by scipy.
        - extra: dict with U, mu, sigma, n1, n2, alternative, normalize.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    a, b = x1.size, x2.size

    if a == 0 or b == 0:
        return SampleTestResult(
            statistic=np.nan,
            pvalue=np.nan,
            extra={"n1": a, "n2": b, "empty": True, "normalize": normalize},
        )

    test = sstats.mannwhitneyu(x1, x2, alternative=alternative)
    try:
        U = float(test.statistic)
        p = float(test.pvalue)
    except AttributeError:  # older SciPy
        U = float(test[0])
        p = float(test[1])

    mu = (a * b) / 2.0
    sigma = np.sqrt((a * b * (a + b + 1)) / 12.0)

    if normalize:
        stat = (U - mu) / sigma
    else:
        stat = U

    return SampleTestResult(
        statistic=stat,
        pvalue=p,
        extra={
            "U": U,
            "mu": mu,
            "sigma": sigma,
            "n1": a,
            "n2": b,
            "alternative": alternative,
            "normalize": normalize,
        },
    )

def wilcoxon_paired_test(
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    alternative: str = "two-sided",
    normalize: bool = False,
    **kwargs) -> SampleTestResult:
    """
    Wilcoxon signed-rank test for paired samples, with optional Z-normalization.

    Goal
    ----
    Non-parametric paired test comparing the distribution of differences x1 - x2
    to zero using signed ranks.

    Normalization
    -------------
    - If normalize=False (default):
        statistic = W (raw Wilcoxon signed-rank statistic, sum of signed ranks).
    - If normalize=True:
        statistic = Z = (W - mu) / sigma
        with
            n  = number of paired observations,
            mu = n(n+1)/4,
            sigma^2 = n(n+1)(2n+1)/24
        under the null hypothesis (no effect), assuming no ties in ranks.

    Parameters
    ----------
    x1, x2 : array_like
        Paired samples (same length). Each x1[i] is matched with x2[i].
    alternative : {"two-sided", "less", "greater"}, optional
        Alternative hypothesis, forwarded to scipy.stats.wilcoxon.
    normalize : bool, optional
        Whether to convert W into a Z-score based on its null distribution.

    Returns
    -------
    TwoSampleTestResult
        - statistic: W or Z depending on normalize.
        - pvalue: p-value from scipy.stats.wilcoxon.
        - extra: dict with W, mu, sigma, n, alternative, normalize.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    if x1.size != x2.size:
        raise ValueError(
            f"Wilcoxon signed-rank test requires paired samples of "
            f"same length, got {x1.size} and {x2.size}."
        )

    n = x1.size
    if n == 0:
        return SampleTestResult(
            statistic=np.nan,
            pvalue=np.nan,
            extra={"n": n, "empty": True, "normalize": normalize},
        )
    
    diff = x1 - x2
    if np.all(diff == 0):
        # Cas dégénéré : toutes les diff = 0 => aucune info contre H0
        W = 0.0
        p = 1.0
        mu = n * (n + 1) / 4.0
        sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        stat = 0.0 if not normalize else (W - mu) / sigma
        return TwoSampleTestResult(
            statistic=stat,
            pvalue=p,
            extra={
                "W": W,
                "mu": mu,
                "sigma": sigma,
                "n": n,
                "alternative": alternative,
                "normalize": normalize,
            },
        )


    test = sstats.wilcoxon(x1, x2, alternative=alternative)

    try:
        W = float(test.statistic)
        p = float(test.pvalue)
    except AttributeError:  # old SciPy API
        W = float(test[0])
        p = float(test[1])

    # Null distribution moments for Wilcoxon W (no ties)
    mu = n * (n + 1) / 4.0
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)

    if normalize:
        stat = (W - mu) / sigma
    else:
        stat = W

    return SampleTestResult(
        statistic=stat,
        pvalue=p,
        extra={
            "W": W,
            "mu": mu,
            "sigma": sigma,
            "n": n,
            "alternative": alternative,
            "normalize": normalize,
        },
    )

# ========= Full-distribution comparison tests =========

# ============================================================
# Kolmogorov–Smirnov two-sample test with optional scaling
# ============================================================

def ks_two_sample_test(
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    normalize: bool = False,
    **kwargs) -> SampleTestResult:
    """
    Kolmogorov–Smirnov two-sample test, with optional normalized statistic.

    Goal
    ----
    Compare the full cumulative distributions of two samples.

    Normalization
    -------------
    - If normalize=False (default):
        statistic = D, the maximum absolute difference between empirical CDFs.
    - If normalize=True:
        statistic = Z_KS = sqrt( (n1 * n2) / (n1 + n2) ) * D
        which is the standard scaling used in asymptotic theory.

    Parameters
    ----------
    x1, x2 : array_like
        Independent samples.
    normalize : bool, optional
        Whether to apply the classical KS scaling to D.

    Returns
    -------
    SampleTestResult
        - statistic: D or scaled D, depending on normalize.
        - pvalue: p-value from scipy.stats.ks_2samp.
        - extra: dict with D, n1, n2, normalize.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    n1, n2 = x1.size, x2.size

    if n1 == 0 or n2 == 0:
        return SampleTestResult(
            statistic=np.nan,
            pvalue=np.nan,
            extra={"n1": n1, "n2": n2, "empty": True, "normalize": normalize},
        )

    test = sstats.ks_2samp(x1, x2, alternative="two-sided", mode="auto")
    try:
        D = float(test.statistic)
        p = float(test.pvalue)
    except AttributeError:
        D = float(test[0])
        p = float(test[1])

    if normalize:
        n_eff = (n1 * n2) / (n1 + n2)
        stat = np.sqrt(n_eff) * D
    else:
        stat = D

    return SampleTestResult(
        statistic=stat,
        pvalue=p,
        extra={"D": D, "n1": n1, "n2": n2, "normalize": normalize},
    )


def cramervonmises_two_sample_test(x1: np.ndarray, x2: np.ndarray,**kwargs) -> SampleTestResult:
    """
    Cramér–von Mises two-sample test.

    Goal
    ----
    Detect differences between two cumulative distributions in a smoother way than KS.

    Specifics
    ---------
    - Non-parametric test based on integrated squared distance between CDFs.
    - Sensitive to global shape differences, not just the maximum deviation.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    test = sstats.cramervonmises_2samp(x1, x2)
    return SampleTestResult(
        statistic=float(test.statistic),
        pvalue=float(test.pvalue),
        extra={"n1": x1.size, "n2": x2.size},
    )

def anderson_darling_two_sample_test(x1: np.ndarray, x2: np.ndarray,**kwargs) -> SampleTestResult:
    """
    Anderson–Darling k-sample test (used here for k=2).

    Goal
    ----
    Compare distributions with extra sensitivity in the tails.

    Specifics
    ---------
    - Generalization of Anderson–Darling to multiple samples.
    - Often more powerful than KS when tail behavior is important.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    test = sstats.anderson_ksamp([x1, x2])
    # anderson_ksamp returns an "approximate" p-value
    return SampleTestResult(
        statistic=float(test.statistic),
        pvalue=float(test.significance_level),
        extra={"n1": x1.size, "n2": x2.size, "critical_values": test.critical_values},
    )

# ============================================================
# Levene variance test with optional Z-like normalization
# ============================================================

def levene_variance_test(
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    center: str = "median",
    normalize: bool = False,
    **kwargs) -> SampleTestResult:
    """
    Levene (or Brown–Forsythe) test for equality of variances,
    with optional Z-like normalization.

    Goal
    ----
    Test whether two groups have equal population variances.

    Normalization
    -------------
    - If normalize=False (default):
        statistic = F, the Levene F-statistic.
    - If normalize=True:
        statistic = Z, derived from the p-value via a normal approximation:
            Z = Φ^{-1}(1 - p/2)
        where Φ is the standard normal CDF.
        This yields a positive quantity increasing with evidence against H0.
        Direction (which variance is larger) is not encoded.

    Parameters
    ----------
    x1, x2 : array_like
        Independent samples.
    center : {"mean", "median", "trimmed"}, optional
        Center used in Levene's test (forwarded to scipy.stats.levene).
    normalize : bool, optional
        Whether to convert the F-statistic into a Z-like score based on p-value.

    Returns
    -------
    SampleTestResult
        - statistic: F or Z, depending on normalize.
        - pvalue: p-value from scipy.stats.levene.
        - extra: dict with F, n1, n2, center, normalize.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    n1, n2 = x1.size, x2.size

    if n1 == 0 or n2 == 0:
        return SampleTestResult(
            statistic=np.nan,
            pvalue=np.nan,
            extra={"n1": n1, "n2": n2, "empty": True, "normalize": normalize},
        )

    test = sstats.levene(x1, x2, center=center)
    try:
        F = float(test.statistic)
        p = float(test.pvalue)
    except AttributeError:
        F = float(test[0])
        p = float(test[1])

    if normalize:
        # Two-sided p-value -> corresponding Z > 0
        # Z = Φ^{-1}(1 - p/2), where Φ is standard normal CDF.
        # We encode only the strength of evidence, not the direction.
        if p <= 0.0:
            # numerical safety: p=0 -> very large Z
            Z = np.inf
        else:
            Z = sstats.norm.isf(p / 2.0)
        stat = Z
    else:
        stat = F

    return SampleTestResult(
        statistic=stat,
        pvalue=p,
        extra={"F": F, "n1": n1, "n2": n2, "center": center, "normalize": normalize},
    )


TEST_REGISTRY = {
    # ======== Two-sample, paired ========
    "paired_t": {
        "fn": paired_t_test,
        "arity": 2,
        "paired": True,
        "normalize": False,  
        "description": "Paired t-test comparing means of matched samples."
    },
    "wilcoxon": {
        "fn":wilcoxon_paired_test,
        "arity": 2,
        "paired": True,
        "normalize": False,  
        "description": "Raw Wilcoxon signed-rank test (paired). Returns the W statistic (sum of signed ranks).",
        "statistic_scale": "raw",
    },

    "wilcoxon_norm": {
        "fn": partial(wilcoxon_paired_test, normalize=True),
        "arity": 2,
        "paired": True,
        "normalize": True,  
        "description": "Normalized Wilcoxon signed-rank test (paired). Uses Z = (W − mu)/sigma under the null hypothesis.",
    },

    # ======== Two-sample, unpaired ========
    "t": {
        "fn": student_t_test,
        "arity": 2,
        "paired": False,
        "normalize": False,  
        "description": "Student two-sample t-test assuming equal variances."
    },
    "t_norm": {
    "fn": partial(student_t_test,normalize=True),
    "arity": 2,
    "paired": False,
    "normalize": True,  
    "description": "Normalized Student t-test (signed Z-score). Converts Student p-value into Z = sign(t) * Φ^{-1}(1 − p/2). Provides a comparable effect metric across sample sizes."
    },

    "welch": {
        "fn": welch_t_test,
        "arity": 2,
        "paired": False,
        "normalize": False,  
        "description": "Welch t-test for unequal variances."
    },

    "welch_norm": {
        "fn": partial(welch_t_test,normalize=True),
        "arity": 2,
        "paired": False,
        "normalize": True,  
        "description": "Normalized Welch t-test (signed Z-score). Converts Welch p-value into Z = sign(t) * Φ^{-1}(1 − p/2). Produces a consistent, comparable effect measure even under unequal variances."
    },

    "mw": {
        "fn": mannwhitney_test,
        "arity": 2,
        "paired": False,
        "normalize": False,   
        "description": "Mann–Whitney U test for independent samples."
    },

    "mw_norm": {
        "fn": partial(mannwhitney_test,normalize=True),
        "arity": 2,
        "paired": False,
        "normalize": True,   
        "description": "Normalized Mann–Whitney U test (Z-score). Uses Z = (U − μ)/σ with μ = ab/2 and σ² = ab(a+b+1)/12. Produces a dimensionless, comparable measure of stochastic dominance."
    },

     "wass": {
        "fn": wasserstein_distance_test,
        "arity": 2,
        "paired": False,
        "normalize": False,  
        "description": "Raw Wasserstein-1 (Earth Mover's Distance) between the two samples. Statistical scale: raw distance depending on feature scale."
    },


    "wass_norm": {
        "fn": partial(wasserstein_distance_test,normalize=True, scale="mad"),
        "arity": 2,
        "paired": False,
        "normalize": True,  
        "description": "Normalized Wasserstein-1 distance. Computes W1 / MAD(pooled) to obtain a robust, dimensionless measure of distributional shift comparable across experiments."
    },
    
    "ks": {
        "fn": ks_two_sample_test,
        "arity": 2,
        "paired": False,
        "normalize": False,  
        "description": "Kolmogorov–Smirnov test comparing two empirical CDFs."
    },

    "ks_norm": {
    "fn": partial(ks_two_sample_test,normalize=True),
    "arity": 2,
    "paired": False,
    "normalize": True,  
    "description": "Normalized Kolmogorov–Smirnov test. Uses scaled statistic Z = sqrt(n1*n2/(n1+n2)) * D, providing a dimensionless metric stable across sample sizes."
    },

    "cvm": {
        "fn": cramervonmises_two_sample_test,
        "arity": 2,
        "paired": False,
        "normalize": False,  
        "description": "Cramér–von Mises two-sample test for distributional differences."
    },

    "ad2": {
        "fn": anderson_darling_two_sample_test,
        "arity": 2,
        "paired": False,
        "normalize": False,  
        "description": "Anderson–Darling k-sample test (k=2) with tail sensitivity."
    },

    "levene": {
        "fn": levene_variance_test,
        "arity": 2,
        "paired": False,
        "normalize": False,  
        "description": "Levene/Brown–Forsythe test for equality of variances."
    },

    "levene_norm": {
        "fn": partial(levene_variance_test,normalize=True),
        "arity": 2,
        "paired": False,
        "normalize": True,  
        "description": "Normalized Levene variance test. Converts the two-sided p-value to a Z-like score: Z = Φ^{-1}(1 − p/2). Produces a positive, scale-free measure of variance inequality."
        },

    # ======== One-sample tests ========
    "t1": {
        "fn": one_sample_t_test,
        "arity": 1,
        "paired": False,
        "normalize": False,  
        "description": "One-sample t-test for mean vs reference value."
    },
    "wilcoxon1": {
        "fn": one_sample_wilcoxon,
        "arity": 1,
        "paired": False,
        "normalize": False,  
        "description": "One-sample Wilcoxon signed-rank test for median vs reference."
    },
    "shapiro": {
        "fn": shapiro_normality_test,
        "arity": 1,
        "paired": False,
        "normalize": False,  
        "description": "Shapiro–Wilk normality test for a single sample."
    },
        "shapiro_norm": {
        "fn": shapiro_normality_test,
        "arity": 1,
        "paired": False,
        "normalize": True,  
        "description": "Normalized Shapiro–Wilk test. Converts p-value into Z = Φ⁻¹(1−p/2), yielding a positive deviation score from normality."
    },
    "ad1": {
        "fn": anderson_normality_test,
        "arity": 1,
        "paired": False,
        "normalize": False,
        "description": "Anderson–Darling one-sample goodness-of-fit test."
    },
    "chi2_var": {
        "fn": chi_square_variance_test,
        "arity": 1,
        "paired": False,
        "normalize": False,
        "description": "Chi-square test for variance vs target value."
    },
    "binom": {
        "fn": binomial_proportion_test,
        "arity": 1,
        "paired": False,
        "normalize": False,
        "description": "Binomial proportion test vs target p0."
    },
}

