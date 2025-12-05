"""
Utility module ensuring safe CPU-only usage of tsfresh.

Because tsfresh relies on numba, which may try to load CUDA runtime
libraries (libcudart.so) and fail in environments without a full CUDA
toolkit, this module forces NUMBA_DISABLE_CUDA=1 by default to prevent
CUDA initialization, unless UQMODELS_ENABLE_NUMBA_CUDA=1 is explicitly set.
"""

import os
import warnings
import numpy as np
import pandas as pd

from uqmodels.preprocessing.preprocessing import (
    select_features_from_FI,
    select_data_and_context,
    build_window_representation)

from uqmodels.utils import mask_corr_feature_target

_TSFRESH_IMPORTED = False
_tsfresh = None
_EfficientFCParameters = None


def _ensure_numba_cpu_only():
    """Désactive Numba CUDA par défaut, sauf si l'utilisateur a explicitement opté pour le contraire."""
    if os.environ.get("UQMODELS_ENABLE_NUMBA_CUDA", "0") == "1":
        # L'utilisateur sait ce qu'il fait, on ne touche pas à NUMBA_DISABLE_CUDA
        return

    if "NUMBA_DISABLE_CUDA" not in os.environ:
        os.environ["NUMBA_DISABLE_CUDA"] = "1"
        warnings.warn(
            "uqmodels a désactivé Numba CUDA (NUMBA_DISABLE_CUDA=1) pour éviter "
            "des conflits CUDA lors de l'utilisation de tsfresh. "
            "Définissez UQMODELS_ENABLE_NUMBA_CUDA=1 avant d'importer uqmodels "
            "si vous souhaitez gérer Numba+CUDA vous-même.",
            RuntimeWarning,
            stacklevel=2,
        )


def _get_tsfresh():
    """
    Import tsfresh safely (CPU-only by default) and raise a clear error
    if the library is not installed.
    """
    global _TSFRESH_IMPORTED, _tsfresh, _EfficientFCParameters

    if not _TSFRESH_IMPORTED:
        _ensure_numba_cpu_only()
        try:
            import tsfresh
            from tsfresh.feature_extraction import (
                EfficientFCParameters as _EFP,
            )
        except ImportError as exc:
            raise ImportError(
                "tsfresh is required for feature extraction but is not installed.\n"
                "Please install it via:\n\n"
                "    pip install tsfresh==0.21.1\n\n"
                "or include it in your project's dependencies."
            ) from exc

        _tsfresh = tsfresh
        _EfficientFCParameters = _EFP
        _TSFRESH_IMPORTED = True

    return _tsfresh, _EfficientFCParameters


def select_tsfresh_params(
    list_keys=["variance", "skewness", "fft", "cwt", "fourrier", "mean" "trend"]
):
    tsfresh, EfficientFCParameters = _get_tsfresh()
    dict_params = dict()
    for key in list_keys:
        for k in EfficientFCParameters().keys():
            if key in k:
                dict_params[k] = EfficientFCParameters()[k]
    return dict_params


def fit_tsfresh_feature_engeenering(
    data,
    context=None,
    window=10,
    step=None,
    ts_fresh_params=None,
    ind_data=None,
    ind_context=None,
    **kwargs
):
    tsfresh, EfficientFCParameters = _get_tsfresh()

    data = select_data_and_context(
        data=data, context=context, ind_data=ind_data, ind_context=ind_context
    )

    if step is None:
        step = window

    df_ts, y_target = build_window_representation(data, step, window)

    if ts_fresh_params is None:
        ts_fresh_params = EfficientFCParameters()

    X_extracted = tsfresh.extract_features(
        df_ts, ts_fresh_params, column_id="id", column_sort="time"
    )

    filter_ = np.isnan(X_extracted).sum(axis=0) == 0
    X_extracted = X_extracted.loc[:, filter_]

    filter_ = mask_corr_feature_target(X_extracted, y_target)
    X_extracted = X_extracted.loc[:, filter_]

    X_selected = tsfresh.feature_selection.select_features(X_extracted, y_target[:, 0])

    for i in range(data.shape[1] - 1):
        X_selected_tmp = tsfresh.feature_selection.select_features(
            X_extracted, y_target[:, i + 1]
        )

        mask = [
            col_tmp not in X_selected.columns.values.tolist()
            for col_tmp in X_selected_tmp.columns.values
        ]
        X_selected = pd.concat([X_selected, X_selected_tmp.loc[:, mask]], axis=1)

    mask_feature_selection = select_features_from_FI(X_selected.values, y_target)
    name_feature_selected = X_selected.columns[mask_feature_selection].values

    kind_to_fc_parameters = tsfresh.feature_extraction.settings.from_columns(
        name_feature_selected
    )
    return kind_to_fc_parameters


def compute_tsfresh_feature_engeenering(
    data,
    context=None,
    window=10,
    step=10,
    ind_data=None,
    ind_context=None,
    params_=None,
    **kwargs
):
    tsfresh, _ = _get_tsfresh()

    data = select_data_and_context(
        data=data, context=context, ind_data=ind_data, ind_context=ind_context
    )

    if step is None:
        step = window

    if params_ is None:
        params_ = fit_tsfresh_feature_engeenering(
            data=data,
            context=context,
            window=window,
            step=step,
            **kwargs,
        )

    df_ts, y_target = build_window_representation(data, step, window)
    X_extracted = tsfresh.extract_features(
        df_ts,
        kind_to_fc_parameters=params_,
        column_id="id",
        column_sort="time",
        disable_progressbar=True,
    )
    return (X_extracted.values, y_target), params_
