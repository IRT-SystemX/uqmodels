"""
Data preprocessing module.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from uqmodels.preprocessing.tsfresh_utils import (
    select_tsfresh_params, fit_tsfresh_feature_engeenering,
    compute_tsfresh_feature_engeenering)
from uqmodels.preprocessing.preprocessing import (
    downscale_series, upscale_series,
    select_features_from_FI, select_data_and_context)
from uqmodels.utils import base_cos_freq, convolute_1D

# import uqmodels.test as UQ_test


def normalise_panda(dataframe, mode, scaler=None):
    """Apply normalisation on a dataframe

    Args:
        dataframe (_type_): _description_
        mode (_type_): _description_

    Returns:
        _type_: _description_
    """
    if mode == "fit":
        scaler = StandardScaler()
        scaler.fit(dataframe.values)
        return scaler
    if mode == "fit_transform":
        scaler = StandardScaler()
        values = scaler.fit_transform(dataframe.values)
        dataframe = pd.DataFrame(
            values, columns=dataframe.columns, index=dataframe.index
        )
        return (dataframe, scaler)
    if mode == "transform":
        values = scaler.fit_transform(dataframe.values)
        dataframe = pd.DataFrame(
            values, columns=dataframe.columns, index=dataframe.index
        )
        return dataframe

# Target selection from data :


def select_data(data, context=None, ind_data=None, **kwargs):
    """Select data from ind_data indice array

    Args:
        data (ndarray): data
        ind_data (ind_array, optional): selected data. Defaults to None : all dim are pick

    Returns:
        data_selected : Ndarray that contains np.concatenation of all selected features
    """
    data_selected = select_data_and_context(
        data, context=None, ind_data=ind_data, ind_context=None
    )
    return data_selected


# ----------------------------------------- #
# PCA transformation form data & context

def fit_pca(
    data,
    context=None,
    n_components=3,
    data_lag=1,
    ind_data=None,
    ind_context=None,
    **kwargs
):
    """Fit&Compute for PCA features generation: fit PCA from selected data & context.

    Args:
        data (ndarray): data
        context (ndarray, optional): context_data. Defaults to None.
        n_components (int, optional): n_components of pca. Defaults to 3.
        ind_data (ind_array, optional): selected data.
        Defaults to None : all dim are pick
        ind_context (ind_array, optional): seletected data context. Defaults to None : all dim are pick
            if there is context
    """
    data = np.roll(data, data_lag, axis=0)
    data_to_reduce = select_data_and_context(
        data=data, context=context, ind_data=ind_data, ind_context=ind_context, **kwargs
    )
    PCA_model = PCA(n_components=n_components)
    PCA_model.fit(data_to_reduce)
    return PCA_model


def compute_pca(
    data,
    context=None,
    n_components=3,
    data_lag=1,
    ind_data=None,
    ind_context=None,
    params_=None,
    **kwargs
):
    """Fit&Compute for PCA features generation:
    compute PCA from selected data & context and params which contains fitted pca.
        if params is none call fit_pca to get a fitted PCA_model

    Args:
        data (ndarray): data
        context (ndarray, optional): context_data. Defaults to None.
        n_components (int, optional): n_components of pca. Defaults to 3.
        ind_data (ind_array, optional): selected data. Defaults to None : all dim are pick
        ind_context (ind_array, optional): seletected data context.
        Defaults to None : all dim are pick if there is context

    Return:
        data_reduced,PCA_model

    """
    if params_ is None:
        PCA_model = fit_pca(
            data,
            context=context,
            n_components=n_components,
            data_lag=data_lag,
            ind_data=ind_data,
            ind_context=ind_context,
            **kwargs
        )
    else:
        PCA_model = params_

    data = np.roll(data, data_lag, axis=0)
    data_to_reduce = select_data_and_context(
        data=data, context=context, ind_data=ind_data, ind_context=ind_context
    )
    data_reduced = PCA_model.transform(data_to_reduce)
    return data_reduced, PCA_model


def automatic_periods_detection(array):
    autocorr = np.argmax(pd.Series.autocorr(array) + 4)  # N_ech periodicity
    return autocorr


def fit_compute_periods(
    data,
    context=None,
    ind_data=None,
    ind_context=None,
    periodicities=[1],
    freqs=[1],
    params_=None,
    **kwargs
):
    """Turn step_scale context array into cos/sin periodic features

    Args:
        context (_type_): context_data
        ind_context (_type_): ind of step_scale
        modularity (_type_): modularity of data
        freq (list, optional): frequence of sin/cos. Defaults to [1].
    """
    step_scale = select_data_and_context(
        data=data, context=context, ind_data=None, ind_context=ind_context, **kwargs
    )

    list_features = []
    for period in periodicities:
        features_tmp = base_cos_freq((step_scale % period) / period, freqs)
        list_features.append(features_tmp)

    list_features = np.concatenate(list_features, axis=1)

    return (list_features, params_)


def fit_compute_lag_values(
    data,
    context=None,
    ind_data=None,
    ind_context=None,
    derivs=[0],
    windows=[1],
    lag=[0],
    delay=0,
    params=None,
    **kwargs
):
    """Turn step_scale context array into cos/sin periodic features

    Args:
        context (_type_): context_data
        ind_context (_type_): ind of step_scale
        modularity (_type_): modularity of data
        freq (list, optional): frequence of sin/cos. Defaults to [1].
    """
    selected_data = select_data_and_context(
        data=data, context=context, ind_data=ind_data, ind_context=ind_context, **kwargs
    )
    MA_derivate, _ = fit_compute_MA_derivate(
        selected_data, derivs=derivs, windows=windows
    )
    lag_features, _ = fit_compute_lag(MA_derivate, lag=lag, delay=delay)
    return (lag_features, params)


def fit_compute_MA_derivate(
    data,
    context=None,
    ind_data=None,
    ind_context=None,
    windows=[1],
    lags=[0],
    derivs=[0],
    params=None,
    **kwargs
):
    """Compute a MA-values of the window last values, then apply lags, then derivates and returns values.
    Apply a 1-lag by default
    """

    data = select_data_and_context(
        data=data, context=context, ind_data=ind_data, ind_context=ind_context, **kwargs
    )

    dim = data.shape
    features = []
    for w in windows:
        conv_array = np.roll(data, w - 1, axis=0)
        if w > 1:
            filter_ = np.concatenate([np.ones(w) * 1 / w])
            lag_array = np.roll(data, w - 1, axis=0)
            conv_array = convolute_1D(lag_array, filter_)

        for lag in lags:
            lag_array = conv_array
            if lag != 0:
                lag_array = np.roll(conv_array, lag, axis=0)
            for deriv in derivs:
                deriv_array = lag_array
                if deriv != 0:
                    deriv_array = np.diff(
                        lag_array, deriv, prepend=np.ones((1, dim[1])) * data[0], axis=0
                    )
                features.append(deriv_array)

    deriv_features = np.stack(features).reshape(-1, dim[0], dim[1])
    deriv_features = np.swapaxes(deriv_features, 2, 1).reshape(-1, dim[0]).T
    return deriv_features, params


def fit_compute_lag(
    data,
    context=None,
    lag=[0],
    delay=0,
    ind_data=None,
    ind_context=None,
    params=None,
    **kwargs
):
    """Create lag features from a numerical array
    Args:
        Y (float array): Target to extract lag-feature
        lag (int, optional): Lag number. Defaults to 3.
        delay (int, optional): Delay before 1 lag feature. Defaults to 0.
    """

    data = select_data_and_context(
        data=data, context=context, ind_data=ind_data, ind_context=ind_context, **kwargs
    )

    dim = data.shape
    new_features_list = []
    new_features_name = []
    for i in np.array(lag) + delay:
        Y_cur = np.roll(data, i, axis=0)
        if i > 0:
            Y_cur[0:i] = 0
        for g in range(dim[1]):
            new_features_list.append(Y_cur[:, g])
            new_features_name.append("lag_" + str(i) + "_dim:" + str(g))
    new_features_name = np.array(new_features_name)
    return np.array(new_features_list).T, params


def fit_FE_by_estimator(
    data,
    context,
    ind_data=None,
    ind_context=None,
    estimator=None,
    estimator_params=dict(),
    data_lag=[1],
    **kwargs
):
    data = pd.DataFrame(np.roll(data.values, data_lag, axis=0), columns=data.columns)
    data = select_data_and_context(
        data=data, context=context, ind_data=ind_data, ind_context=ind_context, **kwargs
    )
    estimator = estimator(**estimator_params)
    estimator.fit(data)
    return estimator


def compute_FE_by_estimator(
    data,
    context,
    ind_data=None,
    ind_context=None,
    estimator=None,
    estimator_params=dict(),
    data_lag=[1],
    params_=None,
    **kwargs
):
    data = pd.DataFrame(np.roll(data.values, data_lag, axis=0), columns=data.columns)

    data = select_data_and_context(
        data=data, context=context, ind_data=ind_data, ind_context=ind_context, **kwargs
    )

    if params_ is None:
        params_ = fit_FE_by_estimator(
            data, context, ind_data, ind_context, estimator, **kwargs
        )
    estimator = params_
    feature = estimator.transform(data)
    return (feature, estimator)


def fit_feature_engeenering(data, context=None, dict_FE_params=dict(), **kwargs):
    if "resample_data_params" in dict_FE_params.keys():
        resample_data_params = dict_FE_params["resample_data_params"]
        if resample_data_params["type"] == "upscale":
            data = upscale_series(data, **resample_data_params)

        elif resample_data_params["type"] == "downscale":
            data = downscale_series(data, **resample_data_params)

    if "resample_context_params" in dict_FE_params.keys():
        resample_context_params = dict_FE_params["resample_context_params"]
        if resample_data_params["type"] == "upscale":
            context = upscale_series(context, **resample_context_params)

        elif resample_data_params["type"] == "downscale":
            context = downscale_series(context, **resample_context_params)

    list_features = []
    if "raw_selection" in dict_FE_params.keys():
        if not isinstance(dict_FE_params["raw_selection"], list):
            dict_FE_params["raw_selection"] = [dict_FE_params["raw_selection"]]

        for dict_params in dict_FE_params["raw_selection"]:
            feature_tmp = select_data_and_context(
                data=data, context=context, **dict_params
            )
            list_features.append(feature_tmp)

    if "periods" in dict_FE_params.keys():
        if not isinstance(dict_FE_params["periods"], list):
            dict_FE_params["periods"] = [dict_FE_params["periods"]]
        for dict_params in dict_FE_params["periods"]:
            feature_tmp, _ = fit_compute_periods(data, context=context, **dict_params)
            dict_params["params_"] = None
            list_features.append(feature_tmp)

    if "MA_derivate" in dict_FE_params.keys():
        if not isinstance(dict_FE_params["MA_derivate"], list):
            dict_FE_params["MA_derivate"] = [dict_FE_params["MA_derivate"]]
        for dict_params in dict_FE_params["MA_derivate"]:
            feature_tmp, _ = fit_compute_MA_derivate(
                data, context=context, **dict_params
            )
            dict_params["params_"] = None
            list_features.append(feature_tmp)
            print(feature_tmp.shape)

    if "FE_by_estimator" in dict_FE_params.keys():
        if not isinstance(dict_FE_params["FE_by_estimator"], list):
            dict_FE_params["FE_by_estimator"] = [dict_FE_params["FE_by_estimator"]]

        for dict_params in dict_FE_params["FE_by_estimator"]:
            params_ = fit_FE_by_estimator(data, context=context, **dict_params)
            dict_params["params_"] = params_
            feature_tmp, _ = compute_FE_by_estimator(data, context, **dict_params)
            list_features.append(feature_tmp)

    if "MV_features" in dict_FE_params.keys():
        if not isinstance(dict_FE_params["MV_features"], list):
            dict_FE_params["MV_features"] = [dict_FE_params["MV_features"]]

        for dict_params in dict_FE_params["MV_features"]:
            params_ = fit_MV_features(data, context=context, **dict_params)
            dict_params["params_"] = params_
            feature_tmp, _ = compute_MV_features(data, context, **dict_params)
            list_features.append(feature_tmp)

    if "ctx_features" in dict_FE_params.keys():
        if not isinstance(dict_FE_params["ctx_features"], list):
            dict_FE_params["ctx_features"] = [dict_FE_params["ctx_features"]]

        for dict_params in dict_FE_params["ctx_features"]:
            params_ = fit_ctx_features(data, context=context, **dict_params)
            dict_params["params_"] = params_
            feature_tmp, _ = compute_ctx_features(data, context, **dict_params)
            list_features.append(feature_tmp)

    if "ts_fresh" in dict_FE_params.keys():
        if not isinstance(dict_FE_params["ts_fresh"], list):
            dict_FE_params["ts_fresh"] = [dict_FE_params["ts_fresh"]]

        for dict_params in dict_FE_params["ts_fresh"]:
            ts_fresh_params_ = fit_tsfresh_feature_engeenering(
                data, context=context, **dict_params
            )
            dict_params["params_"] = ts_fresh_params_
            dict_params["step"] = 1
            (feature_tmp, _), _ = compute_tsfresh_feature_engeenering(
                data, context=context, **dict_params
            )
            list_features.append(feature_tmp)

    X = np.concatenate(list_features, axis=1)
    if "selection" in dict_FE_params.keys():
        dict_FE_params["selection"]["params_"] = select_features_from_FI(
            X, data, **dict_FE_params["selection"]
        )
    return dict_FE_params


def compute_feature_engeenering(
    data, context=None, dict_FE_params=dict(), params_=None
):
    if "resample_data_params" in dict_FE_params.keys():
        resample_data_params = dict_FE_params["resample_data_params"]
        if resample_data_params["type"] == "upscale":
            data = upscale_series(data, **resample_data_params)

        elif resample_data_params["type"] == "downscale":
            data = downscale_series(data, **resample_data_params)

    if "resample_context_params" in dict_FE_params.keys():
        resample_context_params = dict_FE_params["resample_context_params"]
        if resample_data_params["type"] == "upscale":
            context = upscale_series(context, **resample_context_params)

        elif resample_data_params["type"] == "downscale":
            context = downscale_series(context, **resample_context_params)

    if params_ is None:
        params_ = fit_feature_engeenering(data, context, dict_FE_params=dict_FE_params)
    dict_FE_params = params_

    list_features = []
    if "raw_selection" in dict_FE_params.keys():
        for dict_params in dict_FE_params["raw_selection"]:
            list_features.append(
                select_data_and_context(data=data, context=context, **dict_params)
            )

    if "periods" in dict_FE_params.keys():
        for dict_params in dict_FE_params["periods"]:
            feature_tmp, _ = fit_compute_periods(data, context=context, **dict_params)
            list_features.append(feature_tmp)

    if "MA_derivate" in dict_FE_params.keys():
        for dict_params in dict_FE_params["MA_derivate"]:
            feature_tmp, _ = fit_compute_MA_derivate(
                data, context=context, **dict_params
            )
            list_features.append(feature_tmp)

    if "MV_features" in dict_FE_params.keys():
        for dict_params in dict_FE_params["MV_features"]:
            feature_tmp, _ = compute_MV_features(data, context=context, **dict_params)
            list_features.append(feature_tmp)

    if "ctx_features" in dict_FE_params.keys():
        for dict_params in dict_FE_params["ctx_features"]:
            feature_tmp, _ = compute_ctx_features(data, context=context, **dict_params)
            list_features.append(feature_tmp)

    if "ts_fresh" in dict_FE_params.keys():
        for dict_params in dict_FE_params["ts_fresh"]:
            (feature_tmp, _), _ = compute_tsfresh_feature_engeenering(
                data, context=context, **dict_params
            )
            list_features.append(feature_tmp)

    X = np.concatenate(list_features, axis=1)
    if "selection" in dict_FE_params.keys():
        mask = dict_FE_params["selection"]["params_"]
        X = X[:, mask]

    return X, dict_FE_params


def get_FE_params(delta=None):
    """Provide defaults parameters for features engenering

    Args:
        delta (_type_, optional): resample step parameters
    """

    dict_FE = {
        "raw_selection": {"ind_context": [3]},
        "fit_compute_MA_derivate": {
            "windows": [1, 10, 60],
            "lags": [1],
            "derivs": [0, 1, 10],
        },
        "periods": {"ind_context": [0], "periodicities": [10, 100], "freqs": [1, 2]},
        "ts_fresh": {
            "window": 20,
            "step": 5,
            "ts_fresh_params": select_tsfresh_params(["mean", "cwt"]),
        },
    }
    if delta is not None:
        dict_FE["resample_data_params"] = {
            "type": "downscale",
            "delta": delta,
            "mode": "mean",
        }
        dict_FE["resample_context_params"] = {
            "type": "downscale",
            "delta": delta,
            "mode": "first",
        }
    return dict_FE


def fit_MV_features(
    data,
    context,
    ind_data=None,
    ind_context=None,
    focus=None,
    n_components=3,
    n_neighboor=4,
    lags=[0],
    derivs=[0],
    windows=[1],
    **kwargs
):
    """Naive FE function : Fit function to select features having stronger correlation to targets,
    plus compute PCA synthesis of them

    Args:
        data (_type_): _description_
        context (_type_): _description_
        ind_data (_type_, optional): _description_. Defaults to None.
        ind_context (_type_, optional): _description_. Defaults to None.
        focus (_type_, optional): _description_. Defaults to None.
        n_components (int, optional): _description_. Defaults to 3.
        n_neighboor (int, optional): _description_. Defaults to 4.
        lags (list, optional): _description_. Defaults to [0].
        derivs (list, optional): _description_. Defaults to [0].
        windows (list, optional): _description_. Defaults to [1].

    Returns:
        _type_: _description_
    """
    series = select_data(data, context, ind_data=ind_data, ind_context=ind_context)
    ind_focus = ind_data.index(focus)

    order = None
    if n_neighboor > 0:
        print(series.shape)
        corr_matrice = np.corrcoef(series, rowvar=False)
        order = np.argsort(corr_matrice[ind_focus])[::-1][1:]

    estimator = None
    scaler = None
    if n_components > 0:
        estimator = PCA(n_components=n_components)
        scaler = StandardScaler()
        series = scaler.fit_transform(series)
        estimator.fit(series)

    return order, scaler, estimator


def compute_MV_features(
    data,
    context,
    ind_data=None,
    ind_context=None,
    focus=None,
    n_components=3,
    n_neighboor=4,
    lags=[0],
    derivs=[0],
    windows=[1],
    params_=None,
    **kwargs
):
    """Naive FE function : Fit function to select features having stronger correlation to targets,
    plus compute PCA synthesis of them

    Args:
        data (_type_): _description_
        context (_type_): _description_
        ind_data (_type_, optional): _description_. Defaults to None.
        ind_context (_type_, optional): _description_. Defaults to None.
        focus (_type_, optional): _description_. Defaults to None.
        n_components (int, optional): _description_. Defaults to 3.
        n_neighboor (int, optional): _description_. Defaults to 4.
        lags (list, optional): _description_. Defaults to [0].
        derivs (list, optional): _description_. Defaults to [0].
        windows (list, optional): _description_. Defaults to [1].
        params_ (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if params_ is None:
        params_ = fit_MV_features(
            data,
            context,
            ind_data=ind_data,
            ind_context=ind_context,
            focus=focus,
            n_components=n_components,
        )

    order, scaler, estimator = params_

    series = select_data(data, context, ind_data=ind_data, ind_context=ind_context)

    list_series = []
    if n_neighboor > 0:
        list_series.append(series[:, order[:n_neighboor]])

    if n_components > 0:
        series = scaler.transform(series)
        pca_series = estimator.transform(series)
        list_series.append(pca_series)

    if len(list_series) == 1:
        list_series = list_series[0]
    else:
        list_series = np.concatenate(list_series, axis=1)

    list_series, _ = fit_compute_MA_derivate(
        list_series, derivs=derivs, lags=lags, windows=windows
    )

    return (list_series, params_)


def fit_ctx_features(
    data, context, ind_data=None, ind_context=None, n_components=3, lags=[0], **kwargs
):
    """Produce contextual information by apply a PCA on ctx_measure + nan_series if provided

    Args:
        list_channels (_type_): ctx_sources to synthesize 2D (times, features) array
        nan_series (_type_, optional): nan_series : capteurs issues localisation.
        list_target_channels (list, optional): Defaults to [0].

    Returns:
        X_ctx
    """

    selected_data = select_data(data, context, ind_data=ind_data, ind_context=context)

    scaler = StandardScaler()
    selected_data = scaler.fit_transform(selected_data)
    estimator = PCA(n_components)
    estimator.fit_transform(selected_data)
    return scaler, estimator


def compute_ctx_features(
    data,
    context,
    ind_data=None,
    ind_context=None,
    n_components=3,
    lag=0,
    params_=None,
    **kwargs
):
    """Produce contextual information by apply a PCA on ctx_measure + nan_series if provided

    Args:
        list_channels (_type_): ctx_sources to synthesize 2D (times, features) array
        nan_series (_type_, optional): nan_series : capteurs issues localisation.
        list_target_channels (list, optional): Defaults to [0].

    Returns:
        X_ctx
    """
    if params_ is None:
        params_ = fit_ctx_features(
            data,
            context,
            ind_data=ind_data,
            ind_context=ind_context,
            n_components=n_components,
        )

    scaler, estimator = params_

    selected_data = select_data(data, context, ind_data=ind_data, ind_context=context)

    selected_data = scaler.transform(selected_data)
    selected_data = estimator.transform(selected_data)

    selected_data, _ = fit_compute_lag(selected_data, lag=[lag])

    return selected_data, params_
