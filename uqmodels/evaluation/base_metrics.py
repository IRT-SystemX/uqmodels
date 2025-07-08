"""
Metrics module for UQ method evaluation.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# @TODO Add meta class for automatic metric evaluation on the benchmark

# Base intermediare_metrics


def mae(y_true, y_pred):
    return mean_absolute_error(y_pred, y_true)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_pred, y_true))


def q_loss(y, pred, per):
    x = y - pred
    return ((per - 1.0) * x * (x < 0) + per * x * (x >= 0)).mean()


def quantile_loss(y, y_pred_lower, y_pred_upper, alpha):
    q_loss_bot = q_loss(y, y_pred_lower, (1 - alpha) / 2)
    q_loss_top = q_loss(y, y_pred_upper, 1 - ((1 - alpha) / 2))
    return q_loss_bot + q_loss_top


def NLL_loss(y, pred, sigma):
    -np.log(sigma) - 0.5 * np.log(2 * np.pi) - (y - pred) ** 2 / (2 * (sigma**2))


def perf_pred(y_pred, y):
    return mean_absolute_error(y_pred, y), rmse(y_pred, y)


def average_coverage(y_true, y_pred_lower, y_pred_upper):
    return ((y_true >= y_pred_lower) & (y_true <= y_pred_upper)).mean()


def ace(y_true, y_pred_lower, y_pred_upper, alpha):
    cov = average_coverage(y_true, y_pred_lower, y_pred_upper)
    return cov - (1 - alpha)


def sharpness(y_pred_lower, y_pred_upper):
    return (np.abs(y_pred_upper - y_pred_lower)).mean()


def sharpness2(y_pred_lower, y_pred_upper):
    return np.sqrt(np.power(y_pred_upper - y_pred_lower, 2)).mean()


def interval_score(y_true, y_pred_lower, y_pred_upper, alpha):
    return (
        -2 * alpha * (y_pred_upper - y_pred_lower)
        - 4 * (y_pred_lower - y_true) * (y_true < y_pred_lower)
        - 4 * (y_true - y_pred_upper) * (y_pred_upper < y_true)
    ).mean()


def print_real_metrics_meta(res):
    res_mean = np.array(res).mean(axis=0)
    res_std = np.array(res).std(axis=0)
    if res_mean.shape[0] == 3:
        name_list = ["TRAIN", "CAL", "TEST"]
    else:
        name_list = ["TRAIN", "TEST"]
    for n, flag in enumerate(name_list):
        str_1 = "pred_mae = {mae:.3f}±{std_mae:.3f}, pred_mse = {mse:.3f}±{std_mse:.3f}"
        print(
            name_list[n],
            "        ",
            str_1.format(
                mae=res_mean[n][0],
                mse=res_mean[n][1],
                std_mae=res_std[n][0],
                std_mse=res_std[n][1],
            ),
        )
        str_2 = "Diff_cov= {dif_ace:.1f}%±{std_dif_ace:.1f}, Q_loss = {qloss:.3f}±{std_qloss:.3f}"
        print(
            str_2.format(
                dif_ace=res_mean[n][2],
                std_dif_ace=res_std[n][2],
                std_qloss=res_std[n][5],
                qloss=res_mean[n][5],
            )
        )
        str_3 = "Sharpness = {sharp:.3f}±{std_sharp:.3f}, Sharpness² = {sharp2:.3f}±{std_sharp:.3f}"
        print(
            str_3.format(
                sharp=res_mean[n][3],
                std_sharp=res_std[n][3],
                sharp2=res_mean[n][4],
                std_sharp2=res_std[n][3],
            )
        )
    return (np.round(res_mean, 3), np.round(res_std, 3))


def real_metrics(
    y, pred_, bot, top, train, test, alpha=0.90, train_fit=None, verbose=0
):
    res = []

    flag_list = [train, test]
    if not isinstance(train_fit, type(None)):
        if train_fit.sum() != train.sum():
            train_cal = np.copy(train)
            train_cal[train_fit] = False
            flag_list = [train_fit, train_cal, test]

    for n, flag in enumerate(flag_list):
        a = [-1, -1]
        if pred_ is not None:
            a = perf_pred(pred_[flag], y[flag])
        b = 100 * np.abs(ace(y[flag], bot[flag], top[flag], alpha))
        c = sharpness(bot[flag], top[flag])
        d = sharpness2(bot[flag], top[flag])
        e = quantile_loss(y[flag], bot[flag], top[flag], alpha)
        res.append(list((a[0], a[1], b, c, d, e)))
    return res
