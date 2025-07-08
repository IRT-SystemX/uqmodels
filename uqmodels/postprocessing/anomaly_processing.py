####################################################################
# Ensemble of processing aim to process UQmesures into Intermediate quantity or into Anom-KPI


import numpy as np
import scipy
from sklearn.covariance import EmpiricalCovariance

import uqmodels.postprocessing.UQ_processing as UQ_proc
from uqmodels.utils import apply_conv, apply_middledim_reduction


def score_seuil(s, per_seuil=0.995, local=True):
    """Thresholding of extrem values by percentile by dimension if local = True or globally else.

    Args:
        s (_type_): _description_
        per_seuil (float, optional): _description_. Defaults to 0.995.
        local (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """

    if local:
        p_s = np.quantile(np.abs(s), per_seuil, axis=0)
    else:
        p_s = np.quantile(np.abs(s), per_seuil)

    # Seuillage

    s = np.sign(s) * np.minimum(np.abs(s), p_s)

    return s


# Convolution of the signal "s" with a filter "filt"


def fit_calibrate(
    residuals,
    ctx_mask=None,
    beta=0.05,
    type_norm=None,
    empiric_rescale=False,
    deg=1,
    d=1,
    debug=False,
):
    """Function that estimate calibrate parameters form residu link to some hypothesis

    Args:
        s (_type_): score
        ctx_mask : mask for ctx calib
        beta (_type_): Anomaly target ratio
        type_norm (_type_, optional): assymption
        empiric_rescale (bool, optional): _description_. Defaults to True.
        deg (int, optional): _description_. Defaults to 1.

    Type_norm :
        "Nsigma_local" : Fexible anomalie thresold (N-sgima) by dimension | Hypothesis :
            Anomaly distributed homegeneously by dimension.
        "Nsigma_global" : Fexible anomalie thresold (N-sgima) global :  homogeneous anomaly distribution by dimension.
        "Chi2": Theorical Chi2 correction
        "quantiles_local" : rigid anomalie thresold (quantiles) by dimension :
            heterogeneos anomaly distrubtion by dimension.
        "quantiles_global" : (Default) rigid anomalie thresold (quantiles) global :
            homogeneous anomaly distribution by dimension.
        "None" : No normalisation

    Returns:
        params : (list_norm_val : calibration coeficient for each ctx_values
                   list_ctx_id : list of ctx_values)
    """
    if ctx_mask is None:
        ctx_mask = np.zeros(len(residuals))
    list_ctx_id = np.sort(list(set(ctx_mask)))

    if debug:
        print(
            "fit_calib Start",
            (residuals < -1).mean(),
            (residuals > 1).mean(),
            "type_norm",
            type_norm,
        )

    list_norm_val = []
    for v_ctx in list_ctx_id:
        mask = ctx_mask == v_ctx
        # Fexible anomalie thresold (N-sgima) by dimension | Hypothesis : Anomaly
        # distributed homegeneously by dimension.
        if type_norm == "Nsigma_local":
            f_corr = 1
            if empiric_rescale:
                f_corr = residuals[mask].std(axis=0)
            threshold = np.power(scipy.stats.norm.ppf(1 - beta, 0, 1), 2)

        # Fexible anomalie thresold (N-sgima) global :  heterogenous anomaly distribution by dimension.
        elif type_norm == "Nsigma_global":
            f_corr = 1
            if empiric_rescale:
                f_corr = residuals[mask].std()

            threshold = np.power(scipy.stats.norm.ppf(1 - beta, 0, 1), 2)

        # Theorical Chi2 correction
        elif type_norm == "Chi2":
            f_corr = 1
            threshold = np.power(scipy.stats.chi2.ppf(1 - beta, deg), d / 2)

        # Chebyshev's inequality
        elif type_norm == "chebyshev_local":
            f_corr = residuals[mask].std(axis=0)
            threshold = 1 / np.sqrt(beta)

        # Chebyshev's inequality
        elif type_norm == "chebyshev_global":
            f_corr = residuals[mask].std()
            threshold = 1 / np.sqrt(beta)

        # Cantelli's inequality
        elif type_norm == "Cantelli_local":
            f_corr = residuals[mask].std(axis=0)
            threshold = np.sqrt((1 - beta) / beta)

        # Cantelli's  inequality
        elif type_norm == "Cantelli_global":
            f_corr = residuals[mask].std()
            threshold = np.sqrt((1 - beta) / beta)

        # rigid anomalie thresold (quantiles) by dimension :  heterogeneos anomaly distrubtion by dimension.
        elif type_norm == "quantiles_local":
            f_corr = 1
            threshold = np.quantile(np.abs(residuals[mask]), 1 - beta, axis=0)

        elif type_norm == "quantiles_global":
            f_corr = 1
            threshold = np.quantile(np.abs(residuals[mask]), 1 - beta)

        else:
            f_corr = 1
            threshold = 1

        list_norm_val.append(f_corr * threshold)

        if debug:
            print(
                "debug fit_calibrate : Val_ctx",
                v_ctx,
                "target",
                beta,
                "v_mean",
                np.round(np.abs(residuals).mean(), 3),
                "emp",
                (np.abs(residuals[mask]) >= list_norm_val).mean(),
                list_norm_val,
                type_norm,
            )

    params_ = list_norm_val, list_ctx_id

    return params_


def compute_calibrate(
    residuals,
    ctx_mask=None,
    beta=0.05,
    type_norm=None,
    empiric_rescale=True,
    deg=1,
    params_=None,
    mode="score",
    debug=False,
):
    """Function that apply calibration on residu based on some hypothesis

    Args:
        res (_type_): residu to calibrate
        ctx_mask (None): mask for ctx calib
        beta (_type_): Anomaly target ratio
        type_norm (_type_, optional): assymption
        empiric_rescale (bool, optional): _description_. Defaults to True.
        deg (int, optional): Power to apply to the residu. Defaults to 1.

    type_norm assymption :
        "Nsigma_local" : Fexible anomalie thresold (N-sgima) by dimension | Hypothesis :
            Anomaly distributed homegeneously by dimension.
        "Nsigma_global" : Fexible anomalie thresold (N-sgima) global :  homogeneous anomaly distribution by dimension.
        "Chi2": Theorical Chi2 correction
        "quantiles_local" : rigid anomalie thresold (quantiles) by dimension :
            heterogeneos anomaly distrubtion by dimension.
        "quantiles_global" : (Default) rigid anomalie thresold (quantiles) global :
            homogeneous anomaly distribution by dimension.
        "None" : No normalisation


    Returns:
         res : calibrated residu
         params: Params provided or computed
    """
    if ctx_mask is None:
        ctx_mask = np.zeros(len(residuals))

    if params_ is None:
        params_ = fit_calibrate(
            residuals,
            ctx_mask=ctx_mask,
            beta=beta,
            type_norm=type_norm,
            empiric_rescale=empiric_rescale,
            deg=deg,
            debug=debug,
        )

    list_norm_val, list_ctx_id = params_

    if mode == "score":
        if debug:
            print(
                "compute_calib Start",
                (np.abs(residuals) > 1).mean(),
                "type_norm",
                type_norm,
                "res_mean",
                np.abs(residuals).mean(),
                list_norm_val,
                (np.abs(residuals) > list_norm_val[0]).mean(),
            )

    for n, ctx_val in enumerate(list_ctx_id):
        mask = ctx_mask == ctx_val
        residuals[mask].shape, list_norm_val[n].shape
        if mode == "score":

            residuals[mask] = residuals[mask] / list_norm_val[n]

        if mode == "born":
            residuals[mask] = residuals[mask] * list_norm_val[n]

    if mode == "score":
        if debug:
            print(
                "compute_calib out",
                (np.abs(residuals) > 1).mean(),
                "type_norm",
                type_norm,
                "res_mean",
                np.abs(residuals).mean(),
            )

    return residuals, params_


def fit_born_calibrated(
    UQ,
    type_UQ,
    pred,
    y,
    type_UQ_params=None,
    beta=0.1,
    type_res="res",
    ctx_mask=None,
    var_min=0,
    min_cut=0,
    max_cut=0,
    q_var=1,
    empiric_rescale=True,
):
    """!!!!Depreciated !!! Estimate calibration parameters in order to calibrate born from UQ measure, pred, observation and target

    Args:
        UQ (np.array or list): UQmeasure obtain from UQEstimator
        type_UQ (_type_): Type UQ that the nature of UQmeasure
        pred (np.array): prediction provide by a predictor or an UQEstimator
        y (np.array): Targets/Observation
        type_UQ_params : additional parameters link to type paradigm (ex : alpha for quantile)
        beta: target miss-coverage of the borns
        min_cut (_type_): Bottom extremun percentile.
        max_cut (_type_): Bottom upper percentile.
        q_var (_type_): Power coefficent
        sigma_min (float, optional): Minimum values of UQ considered.

    Returns:
        params = (list_norm_val, : list of calibration term for each ctx_values,
                list_ctx_id : list of ctx_values)
    """
    if type_res == "res":
        residuals = y - pred

    if type_res == "w_res":
        residuals = UQ_proc.process_UQmeasure_to_residu(
            UQ,
            type_UQ,
            pred,
            y,
            type_UQ_params=type_UQ_params,
            min_cut=min_cut,
            max_cut=max_cut,
            var_min=var_min,
            q_var=q_var,
        )

    elif type_res == "cqr":
        y_pred_lower = UQ_proc.process_UQmeasure_to_quantile(
            UQ,
            type_UQ,
            pred,
            y=None,
            type_UQ_params=type_UQ_params,
            alpha=beta / 2,
            var_min=0,
            min_cut=0,
            max_cut=1,
            q_var=1,
        )

        y_pred_upper = UQ_proc.process_UQmeasure_to_quantile(
            UQ,
            type_UQ,
            pred,
            y=None,
            type_UQ_params=type_UQ_params,
            alpha=1 - beta / 2,
            var_min=0,
            min_cut=0,
            max_cut=1,
            q_var=1,
        )

        residuals = np.maximum(y_pred_lower - y, y - y_pred_upper) * -(
            y_pred_lower - y
        ) > (y - y_pred_upper)

    elif type_res == "no_calib":
        return ([1], None)

    else:
        raise ValueError("type_res" + type_res + " not covered")

    list_norm_val, list_ctx_id = fit_calibrate(
        residuals,
        ctx_mask=ctx_mask,
        beta=beta,
        type_norm="quantiles_local",
        empiric_rescale=empiric_rescale,
        deg=1,
    )

    # To do asymetrical : calibration distinct for negative and positive residuals
    # Add symetric or asymetric parameters mode
    # -> to implement in calibrate_residuals
    # list_norm_val_neg&pos, list_ctx_id = calibrate_residuals(residuals,
    # ctx_mask=None, alpha=alpha, type_norm="quantiles_local",
    # empiric_rescale=True, deg=1)
    params_ = (list_norm_val, list_ctx_id)
    return params_


def compute_born_calibrated(
    UQ,
    type_UQ,
    pred,
    y=None,
    type_UQ_params=None,
    beta=0.1,
    type_res="res",
    ctx_mask=None,
    params_=None,
):
    """!!!!Depreciated !!! Compute_UQ_calibration from UQ measure, pred, observation and target

    Args:
        UQ (np.array or list): UQmeasure obtain from UQEstimator
        type_UQ (str): Type UQ that the nature of UQmeasure
        pred (np.array): prediction provide by a predictor or an UQEstimator
        y (np.array): Targets/Observation
        type_UQ_params (_type_, optional): additional parameters link to type paradigm (ex : alpha for quantile)
        beta (float, optional): target miss-coverage of the borns
        type_res :
            "no_calib : No calibration
            "res" : Calibration based on mean residuals
            "w_res" : Calibration based on weigthed mean residuals
            "cqr" : Calibration based on quantile residuals
        ctx_mask (_type_, optional): _description_. Defaults to None.
        params (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
         (y_pred_lower, y_pred_upper), params : (upper & lower calibrated bound) & compute params
    """
    if ctx_mask is None:
        ctx_mask = np.zeros(len(y))

    if params_ is None:
        params_ = fit_born_calibrated(
            UQ,
            type_UQ,
            pred,
            y,
            type_UQ_params,
            beta=beta,
            type_res=type_res,
            ctx_mask=ctx_mask,
        )

    list_norm_val, list_ctx_id = params_

    if type_res == "res":
        UQ = pred * 0 + 1
        type_UQ = "var"

    y_pred_lower = UQ_proc.process_UQmeasure_to_quantile(
        UQ,
        type_UQ,
        pred,
        y=None,
        type_UQ_params=type_UQ_params,
        alpha=beta / 2,
        var_min=0,
        min_cut=0,
        max_cut=1,
        q_var=1,
    )

    y_pred_upper = UQ_proc.process_UQmeasure_to_quantile(
        UQ,
        type_UQ,
        pred,
        y=None,
        type_UQ_params=type_UQ_params,
        alpha=1 - beta / 2,
        var_min=0,
        min_cut=0,
        max_cut=1,
        q_var=1,
    )

    for n, ctx_val in list_ctx_id:
        mask = ctx_mask == ctx_val
        if type_res in ["res", "w_res", "no_calib"]:
            y_pred_lower[mask] = (
                (y_pred_lower[mask] - pred[mask]) * list_norm_val[n]
            ) + pred[mask]
            y_pred_upper[mask] = (
                (y_pred_upper[mask] - pred[mask]) * list_norm_val[n]
            ) + pred[mask]

        elif type_res in ["cqr"]:
            y_pred_lower[mask] = y_pred_lower[mask] - list_norm_val[n]
            y_pred_upper[mask] = y_pred_upper[mask] + list_norm_val[n]
        else:
            raise ValueError(
                "compute_UQ_calibration : type_res : " + type_res + " Not covered"
            )
        # Hold assymetrical val

    return (y_pred_lower, y_pred_upper), params_


# FONCTION PRINCIPALE : CONSTRUCTION DU SCORE.
# res = Résidu source
# v_mean = biais source
# v_std = variance source.
# Train = mask lié à l'échantillon de test (Pour pouvoir adapater le seuil entre l'échantillon train et test)

# Paramètre :
# Per_nom : ratio de donnée normale (local) à priori (par défault = 95%)
# beta : Target anomaly (par défault = 0.05)
# Per_seuil = seuillage percentage maximum sur le score d'anomalie.f
# min_cut = Valeur pour la coupe basse de la variance et du biais.100
# max_cut = Valeur pour la coupe haute de la variance et du biais.
# d = Mise en forme puissance du score final.
# Type_Norm :
# - True = Constante de normalisation selon une hypothèse de distribution d'anomalie homgène par dimension.
# - False = Constante de normalisation selon une hypothèse de distribution d'anomalie hétérogène par dimension.
# filt = filtre de convolution a appliquées.
# Q paramètre q  lié à l'hypthèse de distrubution des anomalies par contexte :
#             q < 1 : Anomalie lié au contexte de forte variance.
#             q = 1 : Anomalie distribué de manière homogène par contexte
#             q > 1 : Anomalie lié au contexte de faible variance.
# Dim = dimension de la série.


def fit_anom_score(
    UQ,
    type_UQ,
    pred,
    y,
    type_UQ_params=None,
    ctx_mask=None,
    beta=0.01,
    per_seuil=0.9995,
    min_cut=0.01,
    max_cut=0.97,
    var_min=0,
    d=2,
    q_var=1,
    k_var_e=1,
    q_var_e=1,
    q_Eratio=3,
    type_norm="quantiles_local",
    empiric_rescale=False,
    global_median_normalization=False,
    filt=None,
    reduc_filter=None,
    roll=0,
    debug=False,
    **kwargs
):
    """Fit parameters link to anomaly score calibration using Prediction and UQ measure.

    Args:
        UQ (UQ-measure): UQ measure from UQ estimateurs
        type_UQ (str, optional): _description_. Defaults to "sigma".
        pred (array): prediction
        y (array): Real values
        type_UQ_params (str, optional): additional information about UQ. Defaults to None.
        ctx_mask : mask for ctx_calib
        beta (float, optional): Anom target Defaults to 0.05.
        per_seuil (float, optional): Threshold Defaults to 0.995.
        min_cut (float, optional): extremum cut Defaults to 0.005.
        max_cut (float, optional): extremum cut Defaults to 0.995.
        var_min (float, optional): Threshold of minimum var, Default to 0
        d (int, optional): residual power Defaults to 2.
        type_norm (str, optional): Type of normalisation see  Defaults to "quantiles_local".
        global_median_normalization (bool,optional): Normalisation of residu by each step
            by the median residu of all dimension.
        filt (list, optional): _description_. Defaults to [0, 1, 0].
        q_var (int, optional): _description_. Defaults to 1.

    Returns:
        params : parameters to calibrate anomaly score
    """
    # print('debug', UQ.shape, type_UQ, pred.shape,
    #      y.shape, reduc_filter, filt, roll)

    y, pred, UQ = UQ_proc.check_y_vs_pred_and_UQ_shape(y, pred, UQ)

    ndUQ_ratio, extremum_var_TOT = None, (None, None)
    if type_UQ == "var_A&E":
        extremum_var_TOT, ndUQ_ratio = UQ_proc.get_extremum_var_TOT_and_ndUQ_ratio(
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

    anom_score = UQ_proc.process_UQmeasure_to_residu(
        UQ,
        type_UQ,
        pred,
        y=y,
        type_UQ_params=type_UQ_params,
        d=d,
        min_cut=min_cut,
        max_cut=max_cut,
        q_var=q_var,
        var_min=var_min,
        k_var_e=k_var_e,
        q_var_e=q_var_e,
        q_Eratio=q_Eratio,
        ndUQ_ratio=ndUQ_ratio,
        extremum_var_TOT=extremum_var_TOT,
        reduc_filter=reduc_filter,
        roll=roll,
        debug=debug,
    )

    if global_median_normalization:
        global_norm = np.quantile(anom_score, 0.5, axis=1)[:, None]
        anom_score = anom_score - global_norm

    # Dim reduction :

    # Seuillage des résidu normalisé

    anom_score = apply_conv(anom_score, filt)
    anom_score = score_seuil(anom_score, per_seuil, True)

    # Normalisation lié au seuil d'anomalie (en fonction de type_norm)
    calib_params_ = fit_calibrate(
        anom_score,
        ctx_mask=ctx_mask,
        beta=beta,
        type_norm=type_norm,
        empiric_rescale=empiric_rescale,
        d=d,
        debug=debug,
    )

    return ndUQ_ratio, extremum_var_TOT, calib_params_


def compute_anom_score(
    UQ,
    type_UQ,
    pred,
    y,
    type_UQ_params=None,
    ctx_mask=None,
    beta=0.005,
    per_seuil=0.9995,
    min_cut=0.01,
    max_cut=0.97,
    var_min=0,
    var_max=None,
    d=2,
    type_norm="quantiles_local",
    empiric_rescale=False,
    global_median_normalization=False,
    filt=None,
    reduc_filter=None,
    q_var=1,
    q_var_e=1,
    q_Eratio=3,
    k_var_e=1,
    with_born=False,
    params_=None,
    debug=False,
    roll=0,
    **kwargs
):
    """Compute contextual deviation score from observation, Prediction and UQ measure.
        Then apply normalise considering threeshold based on beta miss_covered target

    Args:
        UQ (UQ-measure): UQ measure from UQ estimateurs
        type_UQ (str, optional): _description_. Defaults to "sigma".
        pred (array): prediction
        y (array): Real values
        type_UQ_params (str, optional): _description_. Defaults to "sigma".
        beta (float, optional): Anom target Defaults to 0.05.
        per_seuil (float, optional): Threshold Defaults to 0.995.
        min_cut (float, optional): extremum cut Defaults to 0.005.
        max_cut (float, optional): extremum cut Defaults to 0.995.
        var_min (float, optional): Threshold of minimum var, Default to 0
        d (int, optional): residual power Defaults to 2.
        type_norm (str, optional): Type of normalisation see norm Defaults to "quantiles_local".
        empiric_rescale (bool,optional): Force empiric rescale based on train percentile
        global_median_normalization (bool,optional): Normalisation of residu by each step
            by the median residu of all dimension.
        filt (list, optional): fitler applied to the score . Defaults to [0, 1, 0].
        q_var (int, optional): power coefficent apply to the score. Defaults to 1.
        params : params provide by fit function. Defaults to None imply internal call to fit

    Type_norm :
        "Nsigma_local" : Fexible anomalie thresold (N-sgima) by dimension | Hypothesis :
            Anomaly distributed homegeneously by dimension.
        "Nsigma_global" : Fexible anomalie thresold (N-sgima) global :  homogeneous anomaly distribution by dimension.
        "Chi2": Theorical Chi2 correction
        "quantiles_local" : rigid anomalie thresold (quantiles) by dimension :
            heterogeneos anomaly distrubtion by dimension.
        "quantiles_global" : (Default) rigid anomalie thresold (quantiles) global :
            homogeneous anomaly distribution by dimension.
        "None" : No normalisation

    Returns:
        anom_score: 2D anomaly score matrix
        params : parameters provided or computed
    """

    y, pred, UQ = UQ_proc.check_y_vs_pred_and_UQ_shape(y, pred, UQ)

    # IF not fit_params :
    if params_ is None:
        params_ = fit_anom_score(
            UQ,
            type_UQ,
            pred,
            y,
            type_UQ_params=type_UQ_params,
            ctx_mask=ctx_mask,
            beta=beta,
            per_seuil=per_seuil,
            min_cut=min_cut,
            max_cut=max_cut,
            d=d,
            q_var=q_var,
            k_var_e=k_var_e,
            q_var_e=q_var_e,
            q_Eratio=q_Eratio,
            type_norm=type_norm,
            var_min=var_min,
            global_median_normalization=global_median_normalization,
            filt=filt,
            reduc_filter=reduc_filter,
            roll=roll,
            **kwargs
        )

    ndUQ_ratio, extremum_var_TOT, calib_params_ = params_

    # Résidu normalisé
    anom_score = UQ_proc.process_UQmeasure_to_residu(
        UQ,
        type_UQ,
        pred,
        y=y,
        type_UQ_params=type_UQ_params,
        d=d,
        min_cut=min_cut,
        max_cut=max_cut,
        q_var=q_var,
        var_min=var_min,
        k_var_e=k_var_e,
        q_var_e=q_var_e,
        q_Eratio=q_Eratio,
        ndUQ_ratio=ndUQ_ratio,
        extremum_var_TOT=extremum_var_TOT,
        with_born=with_born,
        reduc_filter=reduc_filter,
        roll=roll,
        debug=debug,
    )

    if with_born:
        anom_score, born = anom_score

    global_norm = 0
    if global_median_normalization:
        global_norm = np.quantile(anom_score, 0.5, axis=-1)[:, None]
        anom_score = anom_score - global_norm

    if with_born:
        if type_UQ == "var_A&E":
            sigma, E_penalisation = UQ_proc.process_UQmeasure_to_TOT_and_E_sigma(
                UQ,
                type_UQ,
                pred,
                y=y,
                type_UQ_params=type_UQ_params,
                min_cut=min_cut,
                max_cut=max_cut,
                var_min=var_min,
                var_max=var_max,
                q_var=q_var,
                q_var_e=q_var_e,
                k_var_e=k_var_e,
                ndUQ_ratio=ndUQ_ratio,
                extremum_var_TOT=extremum_var_TOT,
                reduc_filter=reduc_filter,
                roll=roll,
            )
        else:
            sigma = UQ_proc.process_UQmeasure_to_sigma(
                UQ,
                type_UQ,
                pred,
                y=y,
                type_UQ_params=type_UQ_params,
                min_cut=min_cut,
                max_cut=max_cut,
                q_var=q_var,
                var_min=var_min,
                reduc_filter=reduc_filter,
            )

            E_penalisation = 0
        anom_padding = global_norm * (sigma)

    else:
        anom_padding = 0
        E_penalisation = 0

    # Reduc middle dim for 3D object (ex : multi-step prediction)
    if debug:
        print("res_val post res", np.abs(anom_score).mean())
    # Apply temp Convolution

    anom_score = apply_conv(anom_score, filt)

    # Seuillage des résidu normalisé
    # -> Warning : difference between anom & born.
    anom_score = score_seuil(anom_score, per_seuil, True)

    if debug:
        print("Start", (anom_score < -1).mean(), (anom_score > 1).mean(), beta)
        if with_born:
            print(
                "Start",
                ((born[0] + anom_padding) > y).mean(),
                ((born[1] + anom_padding) < y).mean(),
            )

    # IF not fit_params :

    # Normalisation lié au seuil d'anomalie (en fonction de type_norm)

    anom_score, _ = compute_calibrate(
        anom_score,
        ctx_mask=ctx_mask,
        beta=beta,
        type_norm=type_norm,
        empiric_rescale=empiric_rescale,
        deg=1,
        params_=calib_params_,
        debug=debug,
    )

    anom_score = np.sign(anom_score) * np.power(np.abs(anom_score), d)

    if debug:
        print("End", (anom_score < -1).mean(), (anom_score > 1).mean())

    if with_born:
        pred = apply_middledim_reduction(pred, reduc_filter=reduc_filter, roll=roll)

        res_born = np.power(sigma, d)
        res_born, _ = compute_calibrate(
            res_born,
            ctx_mask=ctx_mask,
            beta=beta,
            type_norm=type_norm,
            empiric_rescale=empiric_rescale,
            deg=1,
            params_=calib_params_,
            mode="born",
        )
        res_born_bot = pred + np.minimum(
            0, -np.power(res_born, 1 / d) + anom_padding + E_penalisation
        )

        res_born_top = pred + np.maximum(
            0, np.power(res_born, 1 / d) + anom_padding - E_penalisation
        )

        born = res_born_bot, res_born_top

        if debug:
            print(
                "Anom_born",
                ((y < born[0]) | (y > born[1])).mean(axis=0),
                "Anom_res",
                (np.abs(anom_score) > 1).mean(axis=0),
            )
        return (anom_score, born), params_
    else:
        return anom_score, params_


def fit_score_fusion(
    score,
    ctx_mask=None,
    type_fusion="mahalanobis",
    type_norm="quantiles_global",
    beta=0.05,
    per_seuil=0.995,
    filt=[0, 1, 0],
    fusion_reduc_filter=None,
    fusion_debug=False,
    **kwargs
):
    # middle dim reduction
    score = UQ_proc.apply_middledim_reduction(score, fusion_reduc_filter)

    # Temporal convolution
    score = score_seuil(apply_conv(score, filt), per_seuil, True)
    if type_fusion == "mahalanobis":
        # Mahalanobis pour synthese en score global.
        cov_estimator = EmpiricalCovariance(assume_centered=True).fit(score)
        anom_score = cov_estimator.mahalanobis(score)
        # Seuillage du score aggrégées
    else:
        cov_estimator = None
        anom_score = np.abs(score).mean(axis=1)

    if fusion_debug:
        print("Start_fit", (anom_score < -1).mean(), (anom_score > 1).mean())

    anom_score = score_seuil(anom_score, per_seuil, True).reshape((-1, 1))

    params_calibrate_ = fit_calibrate(
        anom_score, ctx_mask, beta, type_norm, empiric_rescale=True, debug=fusion_debug
    )

    return (params_calibrate_, cov_estimator)


def compute_score_fusion(
    score,
    ctx_mask=None,
    type_fusion="mahalanobis",
    type_norm="quantiles_global",
    beta=0.05,
    per_seuil=0.995,
    filt=[0, 1, 0],
    d=1,
    fusion_reduc_filter=None,
    params_=None,
    fusion_debug=False,
    **kwargs
):
    """Compute and aggregate and 2D anomaly score matrix into an aggregated 1D

    Args:
        score (_type_): _description_
        ctx_mask (_type_): _description_
        type_fusion (str, optional): mahalanobis or mean. Defaults to "mahalanobis".
        type_norm (str, optional): _description_. Defaults to "quantiles_global".
        beta (float, optional): anom ratio target. Defaults to 0.95.
        per_seuil (float, optional): _description_. Defaults to 0.995.
        filt (list, optional): _description_. Defaults to [0, 1, 0].
        d (int, optional): _description_. Defaults to 2.
        list_norm_val (_type_, optional): _description_. Defaults to None.
        list_ctx_val (_type_, optional): _description_. Defaults to None.
        params_: params provided by fit function. Defaults to None imply interal call to fit

    Returns:
        score_agg: 1D anomaly score matrix
    """
    if params_ is None:
        params_ = fit_score_fusion(
            score,
            ctx_mask,
            type_fusion=type_fusion,
            type_norm=type_norm,
            beta=beta,
            per_seuil=per_seuil,
            filt=filt,
        )

    params_calibrate_, cov_estimator = params_

    # Middledim reduction
    score = UQ_proc.apply_middledim_reduction(score, fusion_reduc_filter)

    # Temporal convolution
    score = score_seuil(apply_conv(score, filt), per_seuil, True)

    if fusion_debug:
        print("Start_compute", (score < -1).mean(), (score > 1).mean())

    if type_fusion == "mahalanobis":
        # Mahalanobis pour synthese en score global.
        score_agg = cov_estimator.mahalanobis(score)
        # Seuillage du score aggrégées
    else:  # type_fusion == "mean":
        score_agg = np.abs(score).mean(axis=1)

    score_agg = score_seuil(score_agg, per_seuil, True).reshape((-1, 1))
    score_agg, _ = compute_calibrate(
        score_agg,
        ctx_mask,
        beta,
        type_norm,
        params_=params_calibrate_,
        debug=fusion_debug,
    )

    # Résidu quadratique.
    score_agg = np.sign(score_agg) * np.power(score_agg, d)

    if fusion_debug:
        print("End compute", (score < -1).mean(), (score > 1).mean())

    return score_agg, params_
