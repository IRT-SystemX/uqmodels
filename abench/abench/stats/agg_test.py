import numpy as np
from dataclasses import dataclass
import abench.store.api as api
from typing import Any, Dict, Iterable, Optional
from abench.stats.schema import apply_test
from scipy import stats as sstats
from abench.stats.stats import TEST_REGISTRY

def compute_statistical_test(storing,name_component_1,name_component_2,Exp_plan_1,Exp_plan_2,metric,test_name="mw_norm",alternative="greater"):
    result = []
    arity = TEST_REGISTRY[test_name]['arity']
    for n,((Train_set_1,Test_set_list_1),(Train_set_2,Test_set_list_2)) in enumerate(zip(Exp_plan_1.items(),Exp_plan_2.items())):
        result.append([])
        for nn,(set_name_1,set_name_2) in enumerate(zip(Test_set_list_1,Test_set_list_2)):
            result[n].append([])
            X,y,output,context,metadata=api.get_data_and_output(storing,component_name=name_component_1,trainset_name=Train_set_1,set_name=set_name_1)
            sample_ref = metric.compute(y,output,context,metadata[0]['target_arg'])
    
            X,y,output,context,metadata=api.get_data_and_output(storing,component_name=name_component_2,trainset_name=Train_set_2,set_name=set_name_2)
            sample_cmp = metric.compute(y,output,context,metadata[0]['target_arg'])

            if(type(sample_ref) is list):
                for nnn,(s_ref,s_cmp) in enumerate(zip(sample_ref,sample_cmp)):
                    result[n][nn].append([])
                    if(arity==1):
                        res = apply_test(s_ref-s_cmp,test_name=test_name,alternative=alternative)
                    else:
                        res = apply_test(s_ref,x2=s_cmp,test_name=test_name,alternative=alternative)
                    result[n][nn][nnn].append(res)
            else:
                if(arity==1):
                    res = apply_test(sample_ref-sample_cmp,test_name=test_name,alternative=alternative)
                else:
                    res = apply_test(sample_ref,x2=sample_cmp,test_name=test_name,alternative=alternative)
                result[n][nn].append(res)
    return(result)

@dataclass
class AggregatedTestResult:
    """
    Aggregated result over multiple runs / CV folds.

    statistic : float
        Global statistic (depends on 'method').
    pvalue : float
        Global p-value when defined for the method.
    method : str
        Aggregation method used.
    extra : dict
        Additional info (per-fold statistics, etc.).
    """
    statistic: float
    pvalue: float
    method: str
    extra: Optional[Dict[str, Any]] = None


def aggregate_test_results(
    results: Iterable[Any],
    *,
    method: str = "mean_z",
) -> AggregatedTestResult:
    """
    Aggregate per-fold / per-run test results.

    Parameters
    ----------
    results : iterable of objects
        Each element must expose at least a 'statistic' attribute,
        and optionally a 'pvalue' attribute.
        (e.g. your SampleTestResult / TwoSampleTestResult).
    method : {"mean_z", "wilcoxon", "fisher", "sign"}
        - "mean_z"  : mean Z, converted to a global Z (and p-value).
        - "wilcoxon": Wilcoxon signed-rank on statistics vs 0.
        - "fisher"  : Fisher combination of p-values.
        - "sign"    : sign test (binomial) on sign(statistic).

    Returns
    -------
    AggregatedTestResult
    """
    results = list(results)
    if len(results) == 0:
        return AggregatedTestResult(
            statistic=np.nan,
            pvalue=np.nan,
            method=method,
            extra={"empty": True},
        )

    # Extract stats & p-values
    stats = np.array([float(r.statistic) for r in results], dtype=float)
    pvals = np.array(
        [
            getattr(r, "pvalue", np.nan)
            for r in results
        ],
        dtype=float,
    )

    # Mask valid entries depending on method
    mask_valid = np.isfinite(stats)
    if method == "fisher":
        mask_valid &= np.isfinite(pvals) & (pvals > 0) & (pvals <= 1)

    stats = stats[mask_valid]
    pvals = pvals[mask_valid]
    k = stats.size

    if k == 0:
        return AggregatedTestResult(
            statistic=np.nan,
            pvalue=np.nan,
            method=method,
            extra={"all_invalid": True},
        )

    # -------------------------
    # Method: mean_z
    # -------------------------
    if method == "mean_z":
        # Under H0: each stat ~ N(0,1), independent
        z_mean = np.mean(stats)
        # sd(mean) = 1/sqrt(k), so global Z:
        z_global = z_mean * np.sqrt(k)
        p = 2.0 * sstats.norm.sf(abs(z_global))
        return AggregatedTestResult(
            statistic=z_global,
            pvalue=p,
            method=method,
            extra={"per_fold_stats": stats.tolist()},
        )

    # -------------------------
    # Method: wilcoxon (signed-rank vs 0)
    # -------------------------
    if method == "wilcoxon":
        try:
            test = sstats.wilcoxon(stats, alternative="two-sided")
            # On peut laisser la stat brute (W) comme statistic globale
            return AggregatedTestResult(
                statistic=float(test.statistic),
                pvalue=float(test.pvalue),
                method=method,
                extra={"per_fold_stats": stats.tolist()},
            )
        except ValueError:
            # Cas dégénéré (tous les stats == 0, etc.)
            return AggregatedTestResult(
                statistic=0.0,
                pvalue=1.0,
                method=method,
                extra={"per_fold_stats": stats.tolist(), "degenerate": True},
            )

    # -------------------------
    # Method: fisher (combine p-values)
    # -------------------------
    if method == "fisher":
        # X = -2 ∑ ln(p_i) ~ chi2_{2k}
        X = -2.0 * np.sum(np.log(pvals))
        df = 2 * k
        p = 1.0 - sstats.chi2.cdf(X, df=df)
        return AggregatedTestResult(
            statistic=float(X),
            pvalue=float(p),
            method=method,
            extra={"per_fold_pvalues": pvals.tolist(), "df": df},
        )

    # -------------------------
    # Method: sign test on statistics
    # -------------------------
    if method == "sign":
        n_pos = int(np.sum(stats > 0))
        n_tot = int(k)
        # binom test H0: P(stat > 0) = 0.5
        bt = sstats.binomtest(n_pos, n_tot, p=0.5, alternative="two-sided")
        # On peut reporter une "statistic" = proportion de signes positifs
        prop_pos = n_pos / n_tot
        return AggregatedTestResult(
            statistic=prop_pos,
            pvalue=float(bt.pvalue),
            method=method,
            extra={
                "n_pos": n_pos,
                "n_tot": n_tot,
                "per_fold_stats": stats.tolist(),
            },
        )

    # -------------------------
    # Unknown method
    # -------------------------
    raise ValueError(
        f"Unknown aggregation method '{method}'. "
        f"Available: 'mean_z', 'wilcoxon', 'fisher', 'sign'."
    )