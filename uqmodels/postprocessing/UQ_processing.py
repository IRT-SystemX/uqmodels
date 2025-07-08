####################################################################
# Ensemble of processing aim to process UQmesures into Intermediate quantity or into final UQ-KPI

from copy import deepcopy

import numpy as np
import scipy

import uqmodels.utils as ut
from uqmodels.utils import apply_middledim_reduction, cut


def check_y_vs_pred_and_UQ_shape(y, pred, UQ=None):
    """Check if y, pred and UQ have compatible shape

    Args:
        y (_type_): _description_
        pred (_type_): _description_
        UQ (_type_): _description_

    Returns:
        y,pred,UQ : Reshaped if needed
    """
    if y.shape != pred.shape:
        print("warning y shape :", y.shape, "pred shape", pred.shape)
        if (len(y.shape) == (1 + len(pred.shape))) and (y.shape[-1] == 1):
            if UQ is not None:
                UQ = np.expand_dims(UQ, -1)
            return (y, np.expand_dims(pred, -1), UQ)

        elif ((len(y.shape) + 1) == (len(pred.shape))) and (pred.shape[-1] == 1):
            if UQ is not None:
                UQ = UQ[..., 0]
            return (y, pred[..., 0], UQ)
    else:
        return (y, pred, UQ)


def check_UQ(UQ, type_UQ, type_UQ_params):
    """Check if UQ and type_UQ are compatible.
    Args:
        UQ (_type_): UQ object
        type_UQ (_type_): type_UQ specification
    """

    if type_UQ in ["res_2var", "2var", "var_A&E"]:
        if not isinstance(UQ, np.ndarray) and (len(UQ) < 2):
            raise ValueError(
                type_UQ + " UQ should contains at least 2 dimension not " + str(len(UQ))
            )

    elif type_UQ in ["quantile", "res_quantile"]:
        len_UQ = len(type_UQ_params["list_alpha"])
        if UQ.shape[0] != len_UQ:
            raise ValueError(
                type_UQ
                + " UQ should contains len(UQ)=="
                + len_UQ
                + " not "
                + str(len(UQ))
            )

    elif type_UQ in ["var", "res_var"]:
        if not isinstance(UQ, np.ndarray):
            raise ValueError("type(UQ) shoud be np.ndarray not " + type(UQ))
    elif type_UQ in ["None"]:
        pass
    else:
        raise ValueError(
            type_UQ
            + " not in ['var','res_var', 'res_2var', '2var', 'var_A&E', 'quantile', 'res_quantile']"
        )


def get_extremum_var(
    var, min_cut=0, max_cut=1, var_min=0, var_max=None, factor=1, mode_multidim=True
):
    dim = np.arange(len(var.shape))
    # Mode multivariate
    if mode_multidim:
        dim = np.arange(len(var.shape))
        var_min_ = np.maximum(
            np.quantile(var, min_cut, axis=dim[:-1]) / factor, var_min
        )
        var_max_ = np.quantile(var, max_cut, axis=dim[:-1]) * factor
    else:
        var_min_ = np.maximum(np.quantile(var, min_cut) / factor, var_min)
        var_max_ = np.quantile(var, max_cut) * factor

    if var_max is not None:
        var_max_ = np.minimum(var_max_, var_max)

    return (var_min_, var_max_)


def get_nominal_disentangled_UQ_ratio(UQ, q_var=1, q_Eratio=2):
    """Compute nominal_disentangled_UQ_ration form set of data and k_var_e

    Args:
        UQ (_type_): _description_
        k_var_e (_type_): nominal considerate ratio (high-pass filters)

    Returns:
        ndUQ_ratio: nominal disentangled_UQ_ratio
    """
    if q_Eratio is None:
        q_Eratio = 2

    var_a, var_e = UQ
    var_ratio = var_e / np.power(var_a, q_var)
    if q_Eratio < 1:  # Empirique quantile
        ndUQ_ratio = np.quantile(var_ratio, 1 - q_Eratio)
    else:  # N-sigma quantile
        ndUQ_ratio = np.mean(var_ratio) + q_Eratio * np.std(
            cut(var_ratio, 0.005, 0.995)
        )
    return ndUQ_ratio


def split_var_dUQ(
    UQ,
    q_var=1,
    q_var_e=1,
    ndUQ_ratio=None,
    extremum_var_TOT=(None, None),
    E_cut_in_var_nominal=True,
    A_res_in_var_atypic=False,
):
    """split var_A&E into var_nominal & var_atypique based threshold

    Args:
        UQ (_type_): _description_
        q_var (int, optional): _description_. Defaults to 1.
        k_var_e (int, optional): _description_. Defaults to 1.
        q_var_e (int, optional): _description_. Defaults to 1.
        ndUQ_ratio (int, optional): _description_. Defaults to 0.
        extremum_var_TOT (tuple, optional): _description_. Defaults to (None, None).

    Returns:
        _type_: _description_
    """
    if ndUQ_ratio is None:
        ndUQ_ratio = 0

    var_a, var_e = UQ
    ratio_EA = var_e / var_a
    ratio_EA_res = np.maximum(ratio_EA - ndUQ_ratio, 0)
    if E_cut_in_var_nominal:
        var_e_cut = (ratio_EA - ratio_EA_res) * var_a
        var_a = var_a + var_e_cut

    var_e = norm_var(
        var_e * var_a, min_cut=0, max_cut=1, var_min=0, var_max=None, q_var=q_var_e
    )

    # Part of high values link to abnormal espitemic unconfidence
    var_e_res = norm_var(
        ratio_EA_res * var_a,
        min_cut=0,
        max_cut=1,
        var_min=0,
        var_max=None,
        q_var=q_var_e,
    )

    var_a = norm_var(var_a, q_var=q_var)
    # Part of low A_values link to nominal aleatoric uncertainty
    var_a_cut = ut.threshold(
        var_a, min_val=extremum_var_TOT[0], max_val=extremum_var_TOT[1]
    )

    # Part of low E_values link to nominal epistemic uncertainty (nominal extrapolation cost)
    var_e_cut = var_e - var_e_res  # equal to (ratio_EA - ratio_EA_res) * var_a

    # Part of high A_values link to abnormal aleatoric uncertainty estimation
    var_a_res = var_a - var_a_cut

    if A_res_in_var_atypic:
        var_e_res = var_e_res + var_a_res

    return (var_a_cut, var_e_res)


def get_extremum_var_TOT_and_ndUQ_ratio(
    UQ,
    type_UQ,
    pred=None,
    y=None,
    type_UQ_params=None,
    min_cut=0,
    max_cut=1,
    var_min=0,
    var_max=None,
    factor=2,
    q_var=1,
    q_Eratio=2,
    mode_multidim=True,
    E_cut_in_var_nominal=True,
    A_res_in_var_atypic=False,
    **kwargs_process_TOT,
):
    """Estimate parameters use to seperate from var_A & var_E theirs
        Respective nominal part (Affected in TOT,ATYPIC) et atypical part
        using empirical threeshold

    Args:
        UQ (_type_): _description_
        type_UQ (_type_, optional): _description_. Defaults to None.
        pred (_type_, optional): _description_. Defaults to None.
        y (_type_, optional): _description_. Defaults to None.
        type_UQ_params (_type_, optional): _description_. Defaults to None.
        min_cut (int, optional): _description_. Defaults to 0.
        max_cut (int, optional): _description_. Defaults to 1.
        var_min (int, optional): _description_. Defaults to 0.
        var_max (_type_, optional): _description_. Defaults to None.
        factor (int, optional): _description_. Defaults to 2.
        q_var (int, optional): _description_. Defaults to 1.
        q_Eratio (int, optional): _description_. Defaults to 2.
        mode_multidim (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    sigma_TOT, sigma_E = process_UQmeasure_to_TOT_and_E_sigma(
        UQ,
        type_UQ=type_UQ,
        pred=None,
        y=None,
        type_UQ_params=None,
        var_min=0,
        var_max=None,
        min_cut=0,
        max_cut=1,
        extremum_var_TOT=(var_min, var_max),
        **kwargs_process_TOT,
    )
    var_E = np.power(sigma_E, 2)
    var_TOT = np.power(sigma_TOT, 2)
    extremum_var_TOT = get_extremum_var(
        var_TOT,
        min_cut=min_cut,
        max_cut=max_cut,
        var_min=var_min,
        var_max=var_max,
        factor=factor,
        mode_multidim=mode_multidim,
    )

    UQ_bis = (var_TOT, var_E)
    UQ_bis = split_var_dUQ(
        UQ,
        q_var=1,
        q_var_e=1,
        ndUQ_ratio=None,
        E_cut_in_var_nominal=E_cut_in_var_nominal,
        A_res_in_var_atypic=A_res_in_var_atypic,
        extremum_var_TOT=extremum_var_TOT,
    )

    ndUQ_ratio = get_nominal_disentangled_UQ_ratio(UQ_bis, q_var, q_Eratio)

    return (extremum_var_TOT, ndUQ_ratio)


def norm_var(var, min_cut=0, max_cut=1, var_min=0, var_max=None, q_var=1, epsilon=1e-8):
    """variance normalisation
    Args:
        var (np.array): variance to normalise
        min_cut (float): Bottom extremun percentile : [0,1].
        max_cut (float): Bottom upper percentile: [0,1].
        var_min (float, optional): minimal variance assumption. Defaults to 0.
        q_var (float): Power coefficent

    Returns:
        var: normalised variance
    """
    var = deepcopy(var)
    var_min, var_max = get_extremum_var(var, min_cut, max_cut, var_min, var_max)

    var = ut.threshold(var, min_val=var_min + epsilon, max_val=var_max + 2 * epsilon)
    if q_var != 1:
        var = np.power(var, q_var)
    return var


def renormalise_UQ(UQ, type_UQ, scaler=None, var_min=0, var_max=None):
    """UQmeasure to normalised

    Args:
        UQ (np.array): UQmeasure provided by an UQEstimator
        type_UQ (_type_): nature of the UQeeasure
        rescale_val (np.array): rescale variance term (may be obtain by scikilearn Standard normalizer)
        var_min (float, optional): minimal variance assumption. Defaults to 0.

    Returns:
        UQ: normalised UQmeasure
    """
    escale_val = 1
    if scaler is not None:
        rescale_val = scaler.var_

    if type_UQ in ["res_var", "var", "var_A&E"]:
        UQ = UQ * rescale_val
        if var_min > 0:
            UQ = np.maximum(UQ, var_min)
        if var_max is not None:
            UQ = np.minimum(UQ, var_max)

    elif type_UQ in ["res_2var", "2var"]:
        UQ = np.array([UQ[0] * rescale_val, UQ[1] * rescale_val])
        if var_min > 0:
            UQ = np.maximum(UQ, var_min)
        if var_max is not None:
            UQ = np.minimum(UQ, var_max)

    elif type_UQ in ["quantile", "res_quantile"]:
        inv_trans = scaler.inverse_transform
        UQ = np.array([inv_trans(Q) for Q in UQ])

    else:
        print("renormalise_UQ :", type_UQ, "not_covered")

    return UQ


def process_UQmeasure_to_sigma(
    UQ,
    type_UQ,
    pred=None,
    y=None,
    type_UQ_params=None,
    var_min=0,
    var_max=None,
    min_cut=0,
    max_cut=1,
    q_var=1,
    reduc_filter=None,
):
    """Process UQ measure into sigma (sqrt(varirance)) according prediction, UQmeasure and observation.

    Args:
        UQ (np.array or list): UQmeasure obtain from UQEstimator
        type_UQ (_type_): Type UQ that the nature of UQmeasure
        pred (np.array): prediction provide by a predictor or an UQEstimator
        y (np.array): Targets/Observation
        type_UQ_params : additional parameters link to type paradigm (ex : alpha for quantile)
        min_cut (_type_): Bottom extremun percentile.
        max_cut (_type_): Bottom upper percentile.
        q_var (_type_): Power coefficent
        sigma_min (float, optional): Minimum values of UQ considered.

    Returns:
        sigma : standards deviation estimation link to ML-uncertainty
    """
    check_UQ(UQ, type_UQ, type_UQ_params)

    # Case sigma
    if type_UQ in ["res_var", "var"]:
        sigma = np.sqrt(
            norm_var(
                UQ,
                min_cut=min_cut,
                max_cut=max_cut,
                var_min=var_min,
                var_max=var_max,
                q_var=q_var,
            )
        )

    # Case 2 sigma
    elif type_UQ in ["res_2var", "2var"]:
        sigma_bot = np.sqrt(
            norm_var(
                UQ[0],
                min_cut=min_cut,
                max_cut=max_cut,
                var_min=var_min,
                var_max=var_max,
                q_var=q_var,
            )
        )
        sigma_top = np.sqrt(
            norm_var(
                UQ[1],
                min_cut=min_cut,
                max_cut=max_cut,
                var_min=var_min,
                var_max=var_max,
                q_var=q_var,
            )
        )
        sigma = np.concatenate([sigma_bot[None, :], sigma_top[None, :]])

    elif type_UQ == "var_A&E":
        sigma = np.sqrt(
            norm_var(
                UQ[0] + UQ[1],
                min_cut=min_cut,
                max_cut=max_cut,
                var_min=var_min,
                var_max=var_max,
                q_var=q_var,
            )
        )

    elif type_UQ in ["quantile", "res_quantile"]:
        print("Warning quantile renormalisation based on gaussian assumption")
        sigma = []
        for n, param_alpha in type_UQ_params:
            current_sigma = scipy.stats.norm.ppf(param_alpha, 0, 1)
            sigma.append(np.abs((pred - UQ[n])) / current_sigma)
        var = np.power(np.array(sigma).mean(axis=0), 2)
        sigma = np.sqrt(norm_var(var, min_cut, max_cut, var_min, q_var=q_var))

    sigma = apply_middledim_reduction(sigma, reduc_filter)

    return sigma


def process_UQmeasure_to_TOT_and_E_sigma(
    UQ,
    type_UQ,
    pred=None,
    y=None,
    type_UQ_params=None,
    var_min=0,
    var_max=None,
    min_cut=0,
    max_cut=1,
    q_var=1,
    q_var_e=1,
    k_var_e=1,
    q_Eratio=None,
    ndUQ_ratio=None,
    extremum_var_TOT=(None, None),
    reduc_filter=None,
    roll=0,
    **kwargs,
):
    """Process UQ measure into sigma_tot & sigma_E (sqrt(varirance)) according prediction, UQmeasure and observation.

    Args:
        UQ (np.array or list): UQmeasure obtain from UQEstimator
        type_UQ (_type_): Type UQ that the nature of UQmeasure
        pred (np.array): prediction provide by a predictor or an UQEstimator
        y (np.array): Targets/Observation
        type_UQ_params : additional parameters link to type paradigm (ex : alpha for quantile)
        min_cut (_type_): Bottom extremun percentile.
        max_cut (_type_): Bottom upper percentile.
        q_var (_type_): Power coefficent
        sigma_min (float, optional): Minimum values of UQ considered.

    Returns:
        sigma : standards deviation estimation link to ML-uncertainty
    """

    check_UQ(UQ, type_UQ, type_UQ_params)
    # Case sigma

    # Naive disentanglement:
    # - Hypothesis estimation of var_E uppon estimation lead to entangled var_E that include amont of var_A in var_E
    # - Epistemics contains 'nominal espitemics quantity

    # Consider a ratio of k_var_e nominal epistemics to swith towarsd E var.
    if (
        (ndUQ_ratio is None)
        & (extremum_var_TOT[0] is None)
        & (extremum_var_TOT[1] is None)
    ):
        # print('Compute ndUQ_ratio from provided data')
        extremum_var_TOT, ndUQ_ratio = get_extremum_var_TOT_and_ndUQ_ratio(
            UQ,
            type_UQ=type_UQ,
            min_cut=min_cut,
            max_cut=max_cut,
            var_min=0,
            var_max=None,
            factor=2,
            q_var=q_var,
            q_Eratio=q_Eratio,
            mode_multidim=True,
        )

    var_nominal, var_atypic = split_var_dUQ(
        UQ,
        q_var=q_var,
        q_var_e=q_var_e,
        ndUQ_ratio=ndUQ_ratio,
        extremum_var_TOT=extremum_var_TOT,
    )

    sigma_nominal = np.sqrt(var_nominal)
    sigma_atypic = k_var_e * np.sqrt(var_atypic)

    sigma_nominal = apply_middledim_reduction(sigma_nominal, reduc_filter, roll=roll)

    sigma_atypic = apply_middledim_reduction(sigma_atypic, reduc_filter, roll=roll)
    return (sigma_nominal, sigma_atypic)


def process_UQmeasure_to_residu(
    UQ,
    type_UQ,
    pred,
    y,
    type_UQ_params=None,
    d=2,
    min_cut=0,
    max_cut=1,
    q_var=1,
    var_min=0,
    var_max=None,
    with_born=False,
    k_var_e=0,
    q_var_e=None,
    q_Eratio=None,
    extremum_var_TOT=(None, None),
    ndUQ_ratio=None,
    reduc_filter=None,
    roll=0,
    debug=False,
):
    """Process UQ measure to residu according prediction, UQmeasure and observation.

    Args:
        UQ (np.array or list): UQ measure obtain from UQ-estimator
        type_UQ (_type_): Type UQ that caracterise UQ measre
        pred (np.array): prediction
        y (np.array): Targets/Observation
        type_UQ_params : additional parameters link to type paradigm (ex : alpha for quantile)
        min_cut (_type_): Bottom extremun percentile.
        max_cut (_type_): Bottom upper percentile.
        min_cut (_type_): Bottom extremun percentile.
        max_cut (_type_): Bottom upper percentile.
        q_var (_type_): Power coefficent
        sigma_min (float, optional): Minimum values of UQ considered.

    Returns:
        res : residu
    """
    check_UQ(UQ, type_UQ, type_UQ_params)

    if q_var_e is None:
        q_var_e = q_var

    if k_var_e is None:
        k_var_e = 0

    res_norm = y - pred

    if type_UQ in ["var", "res_var", "var_A&E"]:
        if type_UQ in ["var", "res_var"]:
            sigma = np.sqrt(
                norm_var(
                    UQ,
                    min_cut=min_cut,
                    max_cut=max_cut,
                    var_min=var_min,
                    var_max=var_max,
                    q_var=q_var,
                )
            )
            E_penalisation = 0

        elif type_UQ == "var_A&E":
            sigma, E_penalisation = process_UQmeasure_to_TOT_and_E_sigma(
                UQ,
                type_UQ=type_UQ,
                pred=pred,
                y=y,
                min_cut=min_cut,
                max_cut=max_cut,
                var_min=var_min,
                var_max=var_max,
                q_var=q_var,
                q_var_e=q_var_e,
                k_var_e=k_var_e,
                q_Eratio=q_Eratio,
                extremum_var_TOT=extremum_var_TOT,
                ndUQ_ratio=ndUQ_ratio,
            )

            if debug:
                print(
                    "params : q_var_e:",
                    q_var_e,
                    "k_var_e:",
                    k_var_e,
                    "q_Eratio:",
                    q_Eratio,
                    "extremum_var_TOT:",
                    extremum_var_TOT,
                    "ndUQ_ratio:",
                    ndUQ_ratio,
                )

                print(
                    "d1",
                    np.mean(np.mean(res_norm, axis=0), axis=0),
                    np.mean(np.mean(E_penalisation, axis=0), axis=0),
                )

        sign_res = np.sign(res_norm)
        # In normalised residu, add epistemic set_off equal to part of epistemic overhang (E/tot > (k_var))

        sigma = sigma + ut.EPSILON
        if debug:
            print(
                "Start build res raw :",
                np.abs(res_norm).mean(),
                "Mean_Escore :",
                E_penalisation.mean(),
                "Mean_Sigma :",
                sigma.mean(),
            )

        res_norm = (np.abs(res_norm) + E_penalisation) / (sigma)
        res_norm = sign_res * np.power(res_norm, d)

        if with_born:
            born_bot = (pred + E_penalisation) - (sigma)
            born_top = (pred - E_penalisation) + (sigma)

            born_bot = np.minimum(born_bot, pred)
            born_top = np.maximum(born_top, pred)

    elif type_UQ in ["2var", "res_2var", "quantile", "res_quantile"]:
        res_norm = y - pred
        sign_res = np.sign(res_norm)
        if type_UQ in ["2var", "res_2var"]:
            print("Naïf approach mean of 2 sigma")
            UQ_bot = np.sqrt(norm_var(UQ[0], min_cut, max_cut, var_min, q_var))
            UQ_top = np.sqrt(norm_var(UQ[1], min_cut, max_cut, var_min, q_var))
        elif type_UQ == "quantile":
            UQ_bot = pred - cut(UQ[0], min_cut, max_cut)
            UQ_top = cut(UQ[1], min_cut, max_cut) - pred
        elif type_UQ == "res_quantile":
            UQ_bot = cut(UQ[0], min_cut, max_cut)
            UQ_top = cut(UQ[1], min_cut, max_cut)

        reshape_marker = False
        if len(y.shape) == 1:
            res_norm = res_norm[:, None]
            if with_born:
                born_bot = born_bot[:, None]
                born_top = born_top[:, None]

            reshape_marker = True

        for dim in range(y.shape[1]):
            mask = res_norm[:, dim] > 0
            sigma = UQ_bot[mask, dim]
            E_penalisation = 0
            res_norm[mask, dim] = (res_norm[mask, dim] + E_penalisation) / (sigma)

            res_norm = sign_res * np.power(res_norm, d)

            if with_born:
                born_bot[mask, dim] = born_bot[mask, dim] + E_penalisation - (sigma)
                born_top[mask, dim] = born_top[mask, dim] - E_penalisation + (sigma)

            sigma = np.power(UQ_top[~mask, dim], q_var)
            res_norm[~mask, dim] = res_norm[~mask, dim] / (sigma)
            res_norm = np.sign(res_norm) * np.power(res_norm, d)
            if with_born:
                born_bot[~mask, dim] = born_bot[~mask, dim] + E_penalisation + (sigma)
                born_top[~mask, dim] = born_top[~mask, dim] - E_penalisation + (sigma)

        if reshape_marker:
            res_norm = res_norm[:, 0]

    if debug:
        print("Mid build res norm:", np.abs(res_norm).mean(), reduc_filter, roll)

    res_norm = apply_middledim_reduction(res_norm, reduc_filter, roll=roll)

    if debug:
        print("End build res norm :", np.abs(res_norm).mean())

    if with_born:
        return res_norm, (born_bot, born_top)
    else:
        return res_norm


def process_UQmeasure_to_quantile(
    UQ,
    type_UQ,
    pred,
    y=None,
    type_UQ_params=None,
    alpha=0.05,
    var_min=0,
    var_max=None,
    min_cut=0,
    max_cut=1,
    q_var=1,
    reduc_filter=None,
    roll=0,
):
    """Process UQ measure into gaussian_quantile according prediction, UQmeasure and observation
    and a alpha quantile lvl

    Args:
        UQ (np.array or list): UQmeasure obtain from UQEstimator
        type_UQ (_type_): Type UQ that the nature of UQmeasure
        pred (np.array): prediction provide by a predictor or an UQEstimator
        y (np.array): Targets/Observation
        type_UQ_params : additional parameters link to type paradigm (ex : alpha for quantile)
        min_cut (_type_): Bottom extremun percentile.
        max_cut (_type_): Bottom upper percentile.
        q_var (_type_): Power coefficent
        sigma_min (float, optional): Minimum values of UQ considered.

    Returns:
        gaussian_quantile : alpha quantile lvl based on gaussian assumption
    """
    check_UQ(UQ, type_UQ, type_UQ_params)

    if type_UQ in ["quantile", "res_quantile"]:
        if type_UQ_params is None:
            print(
                "Missing type_UQ_params : cannot manipulate quantile without alpha parameters"
            )
        elif alpha in type_UQ_params["list_alpha"]:
            print("recover quantile")
            idx = type_UQ_params["list_alpha"].index(alpha)
            if type_UQ == "quantile":
                gaussian_quantile = UQ[idx]

            elif type_UQ == "res_quantile":
                idx = type_UQ_params["list_alpha"].index(alpha)
                gaussian_quantile = pred + UQ[idx]

    elif type_UQ in ["res_2var", "2var"]:
        sigma = process_UQmeasure_to_sigma(
            UQ,
            type_UQ,
            pred,
            y,
            type_UQ_params,
            var_min=var_min,
            var_max=var_max,
            min_cut=min_cut,
            max_cut=max_cut,
            q_var=q_var,
        )
        y_shape = pred.shape

        if alpha > 0.5:
            sigma = sigma[1].reshape(y_shape)
        else:
            sigma = sigma[0].reshape(y_shape)
        gaussian_quantile = pred + scipy.stats.norm.ppf(alpha, 0, sigma)

    elif type_UQ in ["var_A&E", "res_var", "var"]:
        sigma = process_UQmeasure_to_sigma(
            UQ,
            type_UQ,
            pred,
            y,
            type_UQ_params,
            var_min=var_min,
            var_max=var_max,
            min_cut=min_cut,
            max_cut=max_cut,
            q_var=q_var,
        )
        gaussian_quantile = pred + scipy.stats.norm.ppf(alpha, 0, sigma)

    else:
        raise ValueError(type_UQ + " not covered")

    gaussian_quantile = apply_middledim_reduction(
        gaussian_quantile, reduc_filter, roll=roll
    )
    return gaussian_quantile


def process_UQmeasure_to_Epistemicscore(
    UQ,
    type_UQ,
    pred=None,
    y=None,
    type_UQ_params=None,
    var_min=0,
    var_max=None,
    min_cut=0,
    max_cut=1,
    q_var=1,
    q_var_e=None,
    k_var_e=1,
    mode="relative_likelihood",
    reduc_filter=None,
    roll=0,
    **kwargs,
):
    """Process UQ measure to epistemicvalues according UQmeasure

    Args:
        UQ (np.array or list): UQ measure obtain from UQ-estimator
        type_UQ (_type_): Type UQ that caracterise UQ measre
        pred (np.array): prediction unused
        y (np.array): Targets/Observation unused
        type_UQ_params : additional parameters unused
    Returns:
        Eval : Epistemic unconfidence score.
    """
    check_UQ(UQ, type_UQ, type_UQ_params)

    if q_var_e is None:
        q_var_e = q_var

    var_A = norm_var(
        UQ[0], min_cut=min_cut, max_cut=max_cut, var_min=var_min, q_var=q_var
    )

    var_E = norm_var(
        UQ[1], min_cut=0, max_cut=1, var_min=0, var_max=var_max, q_var=q_var_e
    )

    if type_UQ == "var_A&E":
        if mode == "relative_likelihood":
            Eval = np.exp(-0.5 * np.log(1 + (var_A / var_E)))

        if mode == "relative_variance_part":
            Eval = np.sqrt(var_E / (var_A + var_E))

            Eval = np.maximum(Eval - (1 - k_var_e), 0)

    else:
        print(
            "process_UQmeasure_to_Epistemicscore : ",
            type_UQ,
            " not covered : only cover var_A&E",
        )
        raise ValueError

    Eval = apply_middledim_reduction(Eval, reduc_filter, roll=roll)
    return Eval


def fit_PI(
    UQ,
    type_UQ,
    pred,
    y=None,
    list_alpha=[0.025, 0.975],
    type_UQ_params=None,
    reduc_filter=None,
    **kwargs,
):
    """Function that estimate PIs parameters according to normal assumption a list of quantile parametrized
    by list alpha from the UQmeasure

    Args:
        pred (_type_): Mean prediction
        UQ (_type_): UQmeasure
        type_UQ (_type_):  UQmeasure hypothesis
        list_alpha (list, optional): Quantile values. Defaults to [0.025, 0.975].
        type_UQ_params (_type_, optional): UQmeasure params. Defaults to None.
        PIs_params (_type_, optional): PIs_params : May use externally fit_PIs to obtain it.
            If None, fit_PIs is used internally.

    Raises:
        ValueError: type_UQ not covered

    Returns:
        list_PIs : List of computed quantiles
    """
    PIs_params_ = None
    return PIs_params_


def compute_PI(
    UQ,
    type_UQ,
    pred,
    y=None,
    type_UQ_params=None,
    list_alpha=[0.025, 0.975],
    var_min=0,
    var_max=None,
    min_cut=0,
    max_cut=1,
    q_var=1,
    params_=None,
    reduc_filter=None,
    **kwargs,
):
    """Compute according to normal assumption a list of quantile parametrized by list alpha from the UQmeasre

    Args:
        pred (_type_): Mean prediction
        UQ (_type_): UQmeasure
        type_UQ (_type_):  UQmeasure hypothesis
        list_alpha (list, optional): Quantile values. Defaults to [0.025, 0.975].
        type_UQ_params (_type_, optional): UQmeasure params. Defaults to None.
        params_ (_type_, optional): params : May use externally fit_PIs to obtain it.
            If None, fit_PIs is used internally.

    Raises:
        ValueError: type_UQ not covered

    Returns:
        list_PIs : List of computed quantiles
        params : Params provided or computed
    """

    if params_ is None:
        params_ = fit_PI(
            UQ,
            type_UQ,
            pred,
            y,
            list_alpha,
            type_UQ_params=type_UQ_params,
            reduc_filter=reduc_filter,
        )

    list_PIs = []
    for alpha in list_alpha:
        quantile_env = process_UQmeasure_to_quantile(
            UQ,
            type_UQ,
            pred,
            y=None,
            type_UQ_params=type_UQ_params,
            alpha=alpha,
            var_min=var_min,
            var_max=var_max,
            min_cut=min_cut,
            max_cut=max_cut,
            q_var=q_var,
            reduc_filter=reduc_filter,
        )

        list_PIs.append(quantile_env)

    if len(list_PIs) == 1:
        list_PIs = list_PIs[0]
    return list_PIs, params_


def fit_Epistemic_score(
    UQ,
    type_UQ,
    pred=None,
    y=None,
    type_UQ_params=None,
    list_percent=[0.50, 0.80, 0.95, 0.98, 0.995, 1],
    var_min=0,
    var_max=None,
    min_cut=0.1,
    max_cut=0.97,
    q_var=1,
    q_Eratio=3,
    mode="score",
    reduc_filter=None,
    **kwargs,
):
    """Function that estimate parameters link to Epistemic_score_normalisation based on quantile lvl

    Args:
        pred (_type_): Mean prediction
        UQ (_type_): UQmeasure
        type_UQ (_type_):  UQmeasure hypothesis
        list_percent (list, optional): Quantile values to normalise.
            Defaults to [0.50, 0.80, 0.90, 0.95, 0.975, 0.99, 0.999].
        type_UQ_params (_type_, optional): UQmeasure params. Defaults to None.

    returns:
       params : Params needed by compute_Epistemic_score
    """
    if type_UQ != "var_A&E":
        raise ("type_UQ should be 'var_A&E' not " + type_UQ)

    extremum_var_TOT, ndUQ_ratio = get_extremum_var_TOT_and_ndUQ_ratio(
        UQ,
        type_UQ=type_UQ,
        min_cut=min_cut,
        max_cut=max_cut,
        var_min=var_min,
        var_max=var_max,
        factor=2,
        q_var=q_var,
        q_Eratio=q_Eratio,
        mode_multidim=True,
    )

    # Do not apply reduc_filter in an intermedaire function.
    sigma_nominal, sigma_atypique = process_UQmeasure_to_TOT_and_E_sigma(
        UQ=UQ,
        type_UQ=type_UQ,
        pred=pred,
        y=y,
        var_min=var_min,
        var_max=var_max,
        min_cut=min_cut,
        max_cut=max_cut,
        q_var=q_var,
        type_UQ_params=type_UQ_params,
        q_Eratio=q_Eratio,
        q_var_e=1,
        k_var_e=1,
        ndUQ_ratio=ndUQ_ratio,
        extremum_var_TOT=extremum_var_TOT,
        reduc_filter=None,
    )
    UQ = np.power(sigma_nominal, 2), np.power(sigma_atypique, 2)

    Epistemic_score = process_UQmeasure_to_Epistemicscore(
        UQ=UQ,
        type_UQ=type_UQ,
        pred=pred,
        y=y,
        var_min=0,
        var_max=None,
        min_cut=0,
        max_cut=1,
        q_var=q_var,
        type_UQ_params=type_UQ_params,
        reduc_filter=reduc_filter,
    )

    list_q_val = []
    for n, i in enumerate(list_percent):
        list_q_val.append(np.quantile(Epistemic_score, i, axis=0))

    params_ = list_q_val, list_percent, ndUQ_ratio, extremum_var_TOT
    return params_


def compute_Epistemic_score(
    UQ,
    type_UQ,
    pred=None,
    y=None,
    type_UQ_params=None,
    list_percent=[0.80, 0.90, 0.99, 0.999, 1],
    var_min=0,
    var_max=None,
    min_cut=0,
    max_cut=1,
    q_var=1,
    q_Eratio=3,
    mode="levels",
    params_=None,
    reduc_filter=None,
    **kwargs,
):
    """Function that compute Epistemic_score_lvl from (predictor) & UQEstimor outputs &
    fitted parameters provided by fit_Epistemic_score

    Args:
        pred (_type_): Mean prediction
        UQ (_type_): UQmeasure
        type_UQ (_type_):  UQmeasure hypothesis
        list_percent (list, optional): Quantile values. Defaults to [0.025, 0.975].
        type_UQ_params (_type_, optional): UQmeasure params. Defaults to None.
        params_ (_type_, optional): May use externally fit_Epistemic_score to obtain it. If None, internal called to fit

    Returns:
        Epistemic_scorelvl : Epistemic_score_lvl : Quantile class values of Epistemic score
        params : Parameters provided or computed
    """
    if type_UQ != "var_A&E":
        raise ("type_UQ should be 'var_A&E' not " + type_UQ)

    if params_ is None:
        params_ = fit_Epistemic_score(
            UQ=UQ,
            type_UQ=type_UQ,
            pred=pred,
            y=y,
            list_percent=list_percent,
            var_min=var_min,
            var_max=var_max,
            min_cut=min_cut,
            max_cut=max_cut,
            q_var=q_var,
            reduc_filter=None,
            **kwargs,
        )

    list_q_val, list_percent, ndUQ_ratio, extremum_var_TOT = params_

    # Do not apply reduc_filter in an intermedaire function.
    sigma, sigma_E = process_UQmeasure_to_TOT_and_E_sigma(
        UQ=UQ,
        type_UQ=type_UQ,
        pred=pred,
        y=y,
        var_min=var_min,
        var_max=var_max,
        min_cut=min_cut,
        max_cut=max_cut,
        q_var=q_var,
        type_UQ_params=type_UQ_params,
        q_Eratio=q_Eratio,
        q_var_e=1,
        k_var_e=1,
        ndUQ_ratio=ndUQ_ratio,
        extremum_var_TOT=extremum_var_TOT,
        reduc_filter=None,
    )

    UQ = np.power(sigma, 2), np.power(sigma_E, 2)

    Epistemic_score = process_UQmeasure_to_Epistemicscore(
        UQ=UQ,
        type_UQ=type_UQ,
        pred=pred,
        y=y,
        var_min=0,
        var_max=None,
        min_cut=0,
        max_cut=1,
        q_var=q_var,
        type_UQ_params=type_UQ_params,
        reduc_filter=reduc_filter,
    )

    Epistemic_scorelvl = np.zeros(Epistemic_score.shape)

    if mode == "score":
        return Epistemic_score, params_

    else:
        for q_val in list_q_val:
            if len(pred.shape) == 1:
                Epistemic_scorelvl += Epistemic_score > q_val
            else:
                for d in range(Epistemic_scorelvl.shape[1]):
                    Epistemic_scorelvl[:, d] += Epistemic_score[:, d] > q_val[d]
        return Epistemic_scorelvl, params_
