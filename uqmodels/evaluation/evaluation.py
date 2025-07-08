import numpy as np

import uqmodels.evaluation.base_metrics as metrics
from uqmodels.utils import compute_born

# Evaluation anaylsis :


def var_explain(y, pred, var_A, var_E, set_):
    for n in list(set(set_)):
        flag_set = set_ == n
        if set_.sum() == 0:
            print(n, "%data", 0)
            return [0, 0, 0, 0, 0, 0]
        w = flag_set.mean()
        y_ = y[flag_set]
        y_b = np.mean(y[flag_set])
        y_h = pred[flag_set]
        va = var_A[flag_set]
        ve = var_E[flag_set]
        r = y_ - y_h
        y_ - y_b
        vloc_unexplain = w * y_.var()
        if y_.var() < 10e-5:
            vloc_unexplain = 10e-5
        vloc_explain = w * np.power(y_b - y.mean(), 2).mean()

        vtot = np.power(y - y.mean(), 2).mean()
        array = np.array(
            [
                np.power(y_h - y_b, 2).mean() + (2 * r * (y_h - y_b)).mean(),
                np.power(r, 2).mean(),
                va.mean(),
                ve.mean(),
            ]
        )

        array_bis = np.array(
            [
                (vloc_unexplain + vloc_explain) / w,
                vloc_explain / w,
                np.power(y_h - y_b, 2).mean() + (2 * r * (y_h - y_b)).mean(),
                np.power(r, 2).mean(),
                va.mean(),
                ve.mean(),
            ]
        )
    print(
        n,
        "%data",
        np.round(w * 100, 1),
        "Explain_ctx :",
        np.round(array * w / (vloc_unexplain), 3),
        "Explain_glob :",
        np.round(array_bis * w / vtot, 3),
    )
    return ()


def evaluate(y, output, list_metrics, list_sets=None, context=None, verbose=False):
    list_perf = []
    if list_sets is None:
        list_sets = [np.ones(len(y)) == 1]
    for metrics_ in list_metrics:
        list_perf.append(metrics_.compute(y, output, list_sets, context))
        if verbose:
            print(metrics_.name, list_perf[-1])

    return list_perf


def compute_perf(Y_, pred, sigma, test):
    # Compare mutli-performance for MSE et UQ coverage compare to naive baseline
    perf_baseline = []
    perf_baseline1 = []
    perf_modele = []
    pref_cov = []
    pref_sharp = []
    for i in range(pred.shape[1]):
        std_min = Y_[test, i].std() + 0.000001
        env_bot, env_top = compute_born(pred, sigma, 0.1)

        perf_baseline.append(
            metrics.mean_absolute_error(Y_[test, i], np.roll(Y_[test, i], 1)) / std_min
        )
        mean_MA = (
            np.roll(Y_[test, i], -1)
            + np.roll(Y_[test, i], -2)
            + np.roll(Y_[test, i], -3)
            + np.roll(Y_[test, i], -4)
        ) / 4

        perf_baseline1.append(
            metrics.mean_absolute_error(Y_[test, i], mean_MA) / std_min
        )
        perf_modele.append(
            metrics.mean_absolute_error(Y_[test, i], pred[test, i]) / std_min
        )
        pref_cov.append(
            np.round(
                metrics.average_coverage(
                    Y_[test, i], env_bot[test, i], env_top[test, i]
                ),
                2,
            )
        )
        pref_sharp.append(
            np.round(
                metrics.sharpness(env_bot[test, i], env_top[test, i])
                / Y_[test, i].std(),
                2,
            )
        )

    print("temoin0_mae_r", np.round(np.array(perf_baseline), 2).tolist())
    print("temoin1_mae_r", np.round(np.array(perf_baseline1), 2).tolist())
    print("mae_r", np.round(np.array(perf_modele), 2).tolist())
    print("cover", pref_cov)
    print("sharp", pref_sharp)
