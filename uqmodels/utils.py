import math
import sys
from copy import deepcopy
from typing import Iterable

import numpy as np
import scipy
import tensorflow as tf
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

EPSILON = sys.float_info.min  # small value to avoid underflow


def add_random_state(random_state, values):
    """hold addition with possible None values

    Args:
        random_state (int or None): Random state
        values (int): Values to add

    Returns:
        random_state: int or None
    """
    if random_state is None:
        return None
    else:
        return random_state + values


def generate_random_state(random_state=None):
    """Drawn random number is random_state is None, else return random_state

    Args:
        random_state (int or None, optional): random_state. Defaults to None.

    Returns:
        random_state: random_state int
    """
    if random_state is None:
        random_state = np.random.randint(0, 10000000)
    return random_state


def norm_signal(y):
    return (y - y.mean()) / y.std()


def cum_sum_consecutive_zero(array):
    """Count consecutive 0
         1 0 1 0 0 0 -> 0 1 0 1 2 3

    Args:
        array (_type_): array of consecutive 0

    Returns:
        _type_: _description_
    """
    len_ = len(array)
    inds = np.nonzero(array)[0]
    consecutive_count = np.zeros(len_)

    if not inds[0] == 0:
        consecutive_count[: inds[0]] += 1
        inds = np.concatenate([[0], inds])

    flag = np.ones(len_)
    cpt = 1
    flag[inds] = 0
    while flag.sum() > 0:
        inds = inds + 1
        if inds[-1] > len_ - 1:
            inds = inds[:-1]
        inds = inds[flag[inds] != 0]
        consecutive_count[inds] += cpt
        flag[inds] = 0
        cpt += 1
    return consecutive_count


def threshold(array, min_val=None, max_val=None):
    """Threshold an array with min_val as lower bound and max_val as upper bound
        Can hold mutlidimensional threshold on last dimension
    Args:
        array (_type_): array nd to threeshold
        min_val (val, array or): _description_
        max_val (_type_): _description_
    """
    if min_val is not None:
        array = np.maximum(array, min_val)
    if max_val is not None:
        array = np.minimum(array, max_val)
    return array


def corr_matrix_array(m, a):
    """
    Parameters
    ----------
    a: numpy array
    v: true val

    Returns
    -------
    c: numpy array
       correlation coefficients of v against matrix m
    """
    a = a.reshape(-1, 1)

    mean_t = np.mean(m, axis=0)
    std_t = np.std(m, axis=0)

    mean_i = np.mean(a, axis=0)
    std_i = np.std(a, axis=0)

    mean_xy = np.mean(m * a, axis=0)

    c = (mean_xy - mean_i * mean_t) / (std_i * std_t)
    return c


def coefficients_spreaded(X):
    coefficients = np.array(
        [
            1,
            2,
            3,
            5,
            8,
            10,
            15,
            20,
            25,
            33,
            50,
            100,
            250,
            500,
            1000,
            2500,
            5000,
            10000,
            25000,
            50000,
            100000,
        ]
    )
    mask = coefficients < X
    return coefficients[mask]


def Extract_dict(dictionaire, str_keys):
    list_extract = []
    for str_name in str_keys:
        list_extract.append(dictionaire[str_name])

    if len(list_extract) == 1:
        list_extract = list_extract[0]
    return list_extract


def sizeof_fmt(num, suffix="B"):
    """by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified"""
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix}"


def track_memory(string):
    print("Tracking_memory : " + string)
    for name, size in sorted(
        ((name, sys.getsizeof(value)) for name, value in list(globals().items())),
        key=lambda x: -x[1],
    )[:10]:
        print(f"{name:>30}: {sizeof_fmt(size):>8}")


# Track memorry leakage/


def propagate(bool_array, n_prop=1, inv=False, sym=False):
    # Propagate True of a boolean flag to previous and following index
    for _ in range(n_prop):
        if inv or sym:
            bool_array = bool_array | np.roll(bool_array, -1)
        elif np.invert(inv) or sym:
            bool_array = bool_array | np.roll(bool_array, 1)
    return bool_array


def expand_flag(flag):
    # Propagate True of a boolean flag to previous and folloxing index
    return (np.roll(flag, 1) + np.roll(flag, 0) + np.roll(flag, -1)) > 0


def get_coeff(alpha):
    return scipy.stats.norm.ppf(alpha, 0, 1)


def flatten(y):
    if y is not None:
        if (len(y.shape) == 2) & (y.shape[1] == 1):
            y = y[:, 0]
    return y


# Folding function


def get_fold_nstep(size_window, size_subseq, padding):
    # Compute the number of step according to size of window, size of subseq and padding.
    return int(np.floor((size_window - size_subseq) / (padding))) + 1


# To do update : Hold padding to limit train redunduncy
def stack_and_roll(array, horizon, lag=0, seq_idx=None, step=1):
    """_summary_

    Args:
        array (_type_): array 2D to stack and roll
        horizon (_type_): depth
        lag (int, optional): _description_. Defaults to 0.
        seq_idx (_type_, optional): _description_. Defaults to None.
        step (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    # Perform a stack "horizon" time an array and roll in temporal stairs.
    # a n+1 dimensional array with value rolled k times for the kth line
    shape = (1, horizon, 1)
    new_array = np.tile(array[:, None], shape)

    if seq_idx is None:
        for i in np.arange(horizon):
            lag_current = horizon - lag - i - 1
            new_array[:, i] = np.roll(new_array[:, i], lag_current, axis=0)

        # Erase procedure
        for i in np.arange(horizon):
            lag_current = horizon - lag - i - 1
            if lag_current > 0:
                new_array[i, :lag_current] = 0

        # by blocks :

    else:
        _, idx = np.unique(seq_idx, return_index=True)
        for j in seq_idx[np.sort(idx)]:
            flag = seq_idx == j
            list_idx = np.nonzero(flag)[0]
            for i in np.arange(horizon):
                lag_current = horizon - lag - i - 1
                new_array[flag, i] = np.roll(new_array[flag, i], lag_current, axis=0)
                # A optimize : get first non zeros values.
            for i in np.arange(horizon):
                lag_current = horizon - lag - i - 1
                new_array[list_idx[i], :lag_current] = 0
    if step > 1:
        mask = (np.arange(len(array)) % step) == step - 1
        new_array = new_array[mask]
    return new_array


def stack_and_roll_layer(inputs, size_window, size_subseq, padding, name=""):
    # slide_tensor = []
    n_step = get_fold_nstep(size_window, size_subseq, padding)
    # Implementation numpy
    # if False:
    #     for i in range(n_step):
    #         slide_tensor.append(
    #             inputs[:, (i * padding) : (i * padding) + size_subseq, :][:, None]
    #         )
    #     return Lambda(lambda x: K.concatenate(x, axis=1), name=name + "_rollstack")(
    #         slide_tensor
    #     )
    x = tf.map_fn(
        lambda i: inputs[:, (i * padding): (i * padding) + size_subseq, :],
        tf.range(n_step),
        fn_output_signature=tf.float32,
    )
    x = tf.transpose(x, [1, 0, 2, 3])
    return x


def apply_middledim_reduction(ndarray, reduc_filter=None, roll=0):
    """Apply middledim using ponderate mean reduc_filter weigth or do nothing

    Args:
        ndarray (np.array): object to reduce
        reduc_filter (np.array): ponderate weigth for reduction

    Return:
        reduced_ndarray: reduced object
    """

    ndarray = deepcopy(ndarray)
    if reduc_filter is None:
        return ndarray

    reduc_filter = np.array(reduc_filter)

    dim_g = ndarray.shape[-1]
    list_reduced_ndarray_by_dim = []
    # Pour chaque dimension de s
    for i in range(dim_g):
        # Incompatibility test :
        if len(reduc_filter) != ndarray.shape[1]:
            print(
                "error expect size of reduc filter : ",
                str(ndarray.shape[1]),
                "current size :",
                str(len(reduc_filter)),
            )
            break

        filt = reduc_filter / reduc_filter.sum()
        ndarray[:, :, i] = ndarray[:, :, i] * filt
        if roll:
            reduced_ndarray = ndarray[:, 0, i]
            for j in range(ndarray.shape[1] - 1):
                reduced_ndarray += np.roll(ndarray[:, j + 1, i], (j + 1) * roll, axis=0)
        else:
            ndarray[:, :, i] = ndarray[:, :, i] * filt
            reduced_ndarray = ndarray[:, :, i].sum(axis=1)

        list_reduced_ndarray_by_dim.append(reduced_ndarray[:, None])
    reduced_ndarray = np.concatenate(list_reduced_ndarray_by_dim, axis=1)
    return reduced_ndarray


def apply_mask(list_or_array_or_none, mask, axis=0, mode="bool_array"):
    """Select subpart of an array/list_of_array/tupple using a mask and np.take function.
        If list_or_array_or_none is array, then direclty apply selection on it.
        else if it is a Tupple or list of array apply it on array in the list/tupple structure

    Args:
        list_or_array_or_none (_type_): ND array (list or tupple or ND.array)
        mask (_type_): Mask as boolean array or indices array.
        axis (int, optional): axis on sub array to apply. Defaults to 1.
        mode (str, optional):if 'bool_array', turn bool_array_mask into an indice_array. Defaults to 'bool_array'.

    Returns:
        (List or Tuple or array or Ndarray) : Sub_selected list of array or Ndarray
    """

    if mode == "bool_array":
        indices = np.arange(len(mask))[mask]

    if type(list_or_array_or_none) in [list, tuple]:
        return [np.take(array, indices, axis=axis) for array in list_or_array_or_none]
    if list_or_array_or_none is None:
        return None
    else:
        return np.take(list_or_array_or_none, indices, axis=axis)


def base_cos_freq(array, freq=[2]):
    """Transform 1D modulo features [1, N] in cyclic (cos,sin) features for given freq"""
    features = []
    for i in freq:
        features.append(np.cos(i * math.pi * array))
        features.append(np.sin(i * math.pi * array))
    return np.concatenate(features).reshape(len(freq) * 2, -1).T


# Preprocessing function


def apply_conv(score, filt=None, reduc_filter=None):
    """Apply naivly by dimension a convolution to s using filt as filter

    Args:
        s (np.array): score to 1D convolute
        filt (np.array): filter to apply

    Returns:
        s: convoluted score
    """
    if filt is None:
        return score

    if len(score.shape) == 3:
        dim_g = score.shape[-1]
        # Pour chaque dimension de s
        for t in range(score.shape[1]):
            for i in range(score.shape[2]):
                score[:, t, i] = np.convolve(score[:, t, i], filt, mode="same")
    else:
        dim_g = score.shape[-1]
        # Pour chaque dimension de s
        for i in range(dim_g):
            score[:, i] = np.convolve(score[:, i], filt, mode="same")
    return score


def cut(values, cut_min, cut_max):
    """Apply percentile minimal and maximal threeshold

    Args:
        values (_type_): values to cut
        cut_min (_type_): percentile of the minimam cut threeshold
        cut_max (_type_): percentile of the maximal cut threeshold

    Returns:
        _type_: _description_
    """

    if values is None:
        return None

    vpmin = np.quantile(values, cut_min, axis=0)
    vpmax = np.quantile(values, cut_max, axis=0)
    values = np.minimum(values, vpmax)
    values = np.maximum(values, vpmin)
    return values


def format_data(self, X, y, fit=False, mode=None, flag_inverse=False):
    """Feature and target Formatting"""
    if self.rescale:
        flag_reshape = False
        if y is not None:
            if len(y.shape) == 1:
                flag_reshape = True
                y = y[:, None]

        if fit:
            scalerX = StandardScaler(with_mean=True, with_std=True)
            X = scalerX.fit_transform(X)
            scalerY = StandardScaler(with_mean=True, with_std=True)
            y = scalerY.fit_transform(y)
            self.scaler = [scalerX, scalerY]

        elif not flag_inverse:
            if X is not None:
                X = self.scaler[0].transform(X)
            if y is not None:
                y = self.scaler[1].transform(y)
        else:
            X_transformer = self.scaler[0].inverse_transform
            Y_transformer = self.scaler[1].inverse_transform
            if X is not None and not len(X) == 0:
                X = X_transformer(X)

            if y is not None and not len(y) == 0:
                sigma = np.sqrt(self.scaler[1].var_)
                if (mode == "sigma") | (mode == "var") | (mode == "var_A&E"):
                    y = y * sigma

                elif mode == "2sigma":
                    y_reshape = np.moveaxis(y, -1, 0)
                    if len(y.shape) == 3:
                        y = np.concatenate(
                            [
                                np.expand_dims(i, 0)
                                for i in [y_reshape[0] * sigma, y_reshape[1] * sigma]
                            ],
                            axis=0,
                        )
                    else:
                        y = np.concatenate(
                            [
                                y_reshape[0] * sigma,
                                y_reshape[1] * sigma,
                            ],
                            axis=-1,
                        )
                elif mode == "quantile":
                    y_bot = Y_transformer(y[:, 0])
                    y_top = Y_transformer(y[:, 1])
                    y = np.concatenate([y_bot, y_top], axis=1)

                else:
                    y = Y_transformer(y)

        if flag_reshape:
            y = y[:, 0]
    return (X, y)


# PIs basics computation functionalities


def compute_born(y_pred, sigma, alpha, mode="sigma"):
    """Compute y_upper and y_lower boundary from gaussian hypothesis (sigma or 2sigma)

    Args:
        y_pred (array) : Mean prediction
        sigma (array) : Variance estimation
        alpha (float) : Misscoverage ratio
        mode (str) : Distribution hypothesis
        (sigma : gaussian residual hypothesis, 2sigma : gaussian positive and negative residual hypothesis)

    Returns:
       (y_lower,y_upper): Lower and upper bondary of Predictive interval
    """
    # Case sigma
    if mode == "sigma":
        y_lower = y_pred + scipy.stats.norm.ppf((alpha / 2), 0, sigma)
        y_upper = y_pred + scipy.stats.norm.ppf((1 - (alpha / 2)), 0, sigma)
    # Case 2 sigma
    elif mode == "2sigma":
        sigma = np.moveaxis(sigma, -1, 0)
        y_lower = y_pred + scipy.stats.norm.ppf((alpha / 2), 0, sigma[0])
        y_upper = y_pred + scipy.stats.norm.ppf((1 - (alpha / 2)), 0, sigma[1])
    else:
        raise NotImplementedError("This mode is not yet implemented.")
    return y_lower, y_upper


# Gaussian mixture quantile estimation :
def mixture_quantile(pred, var_A, quantiles, n_jobs=5):
    def aux_mixture_quantile(pred, var_A, quantiles):
        list_q = []
        n_data = pred.shape[1]
        n_mixture = pred.shape[0]
        for n in range(n_data):
            mean_law = scipy.stats.norm(
                pred[:, n, 0].mean(), np.sqrt(var_A[:, n, 0].mean())
            )
            xmin = mean_law.ppf(0.0000001)
            xmax = mean_law.ppf(0.9999999)
            scale = np.arange(xmin, xmax, (xmax - xmin) / 300)
            Mixture_cdf = np.zeros(len(scale))
            for i in range(n_mixture):
                cur_law = scipy.stats.norm(pred[i, n, 0], np.sqrt(var_A[i, n, 0]))
                Mixture_cdf += cur_law.cdf(scale) / n_mixture
            q_val = []
            for q in quantiles:
                q_val.append(scale[np.abs(Mixture_cdf - q).argmin()])
            list_q.append(q_val)
        return np.array(list_q)

    list_q = []
    n_data = pred.shape[1]
    parallel_partition = np.array_split(np.arange(n_data), 3)
    # Split inputs of auxillar parralel tree statistics extraction
    parallel_input = []
    for partition in parallel_partition:
        parallel_input.append((pred[:, partition], var_A[:, partition], quantiles))
    list_q = Parallel(n_jobs=n_jobs)(
        delayed(aux_mixture_quantile)(*inputs) for inputs in parallel_input
    )
    return np.concatenate(list_q, axis=0)


# Scikit tuning function


def aux_tuning(
    model,
    X,
    Y,
    params=None,
    score="neg_mean_squared_error",
    n_esti=100,
    folds=4,
    random_state=None,
):
    """Random_search with sequential k-split

    Args:
        model (scikit model): Estimator
        X ([type]): Features
        Y ([type]): Target
        params ([type], optional): parameter_grid. Defaults to None.
        score (str, optional): score. Defaults to 'neg_mean_squared_error'.
        n_esti (int, optional): Number of grid try . Defaults to 100.
        folds (int, optional): Number of sequential fold. Defaults to 4.
        verbose (int, optional): [description]. Defaults to 0.
        random_state (bool) : handle experimental random using seeds.
    """
    if isinstance(params, type(None)):
        return model
    tscv = TimeSeriesSplit(n_splits=folds)
    random_search = RandomizedSearchCV(
        model,
        param_distributions=params,
        n_iter=n_esti,
        scoring=score,
        n_jobs=8,
        cv=tscv.split(X),
        verbose=1,
        random_state=random_state,
    )
    Y = np.squeeze(Y)
    random_search.fit(X, Y)
    return random_search.best_estimator_


def agg_list(list_: Iterable):
    try:
        return np.concatenate(list_, axis=0)
    except ValueError:
        return None


def agg_func(list_: Iterable):
    try:
        return np.mean(list_, axis=0)
    except TypeError:
        return None


class GenericCalibrator:
    def __init__(self, type_res="res", mode="symetric", name=None, alpha=0.1):
        """Generic calibrator implementing several calibratio

        Args:
            type_res (str, optional): type of score
                "no_calib : No calibration
                "res" : Calibration based on mean residuals
                "w_res" : Calibration based on weigthed mean residuals
                "cqr" : Calibration based on quantile residuals

            mode (str, optional):
                if "symetric" : symetric calibration ()
                else perform  calibration independently on positive and negative residuals.
            name (_type_, optional): _description_. Defaults to None.
            alpha (float, optional): _description_. Defaults to 0.1.
        """
        super().__init__()
        self._residuals = None
        self.name = name
        if name is None:
            self.name = type_res + "_" + mode
        self.mode = mode
        self.alpha = alpha
        self.type_res = type_res

        if mode == "symetric":
            self.fcorr = 1
        else:
            self.fcorr_lower = 1
            self.fcorr_upper = 1

    def estimate(
        self, y_true, y_pred, y_pred_lower, y_pred_upper, sigma_pred, **kwargs
    ):
        flag_res_lower = np.zeros(y_true.shape)
        flag_res_upper = np.zeros(y_true.shape)

        if (self.type_res == "res") | (self.type_res == "w_res"):
            if self.type_res == "res":
                residuals = y_true - y_pred
                sigma_pred = y_true * 0 + 1

            if self.type_res == "w_res":
                # sigma_pred = np.maximum(y_pred_upper - y_pred_lower, EPSILON) / 2
                residuals = y_true - y_pred

                if len(residuals.shape) == 1:
                    flag_res_lower = residuals <= 0
                    flag_res_upper = residuals >= 0

            else:
                flag_res_lower = np.concatenate(
                    [
                        np.expand_dims(residuals[:, i] <= 0, -1)
                        for i in range(0, y_true.shape[1])
                    ]
                )
                flag_res_upper = np.concatenate(
                    [
                        np.expand_dims(residuals[:, i] >= 0, -1)
                        for i in range(0, y_true.shape[1])
                    ]
                )

            if y_pred.shape != sigma_pred.shape:
                sigma_pred = np.moveaxis(sigma_pred, -1, 0)
                residuals[flag_res_lower] = (
                    np.abs(residuals)[flag_res_lower] / (sigma_pred[0])[flag_res_lower]
                )
                residuals[flag_res_upper] = (
                    np.abs(residuals)[flag_res_upper] / (sigma_pred[1])[flag_res_upper]
                )
            else:
                residuals = np.abs(residuals) / (sigma_pred)

        elif self.type_res == "cqr":
            residuals = np.maximum(y_pred_lower - y_true, y_true - y_pred_upper)
            flag_res_lower = (y_pred_lower - y_true) >= (y_true - y_pred_upper)
            flag_res_upper = (y_pred_lower - y_true) <= (y_true - y_pred_upper)

        elif self.type_res == "no_calib":
            return

        else:
            print("Unknown type_res")
            return

        if self.mode == "symetric":
            self.fcorr = np.quantile(
                residuals, (1 - self.alpha) * (1 + 1 / len(residuals))
            )

        else:
            self.fcorr_lower = np.quantile(
                residuals[flag_res_lower],
                np.minimum((1 - self.alpha) * (1 + 1 / flag_res_lower.sum()), 1),
            )

            self.fcorr_upper = np.quantile(
                residuals[flag_res_upper],
                np.minimum((1 - self.alpha) * (1 + 1 / flag_res_upper.sum()), 1),
            )
        return

    def calibrate(self, y_pred, y_pred_lower, y_pred_upper, sigma_pred, **kwargs):
        if self.type_res == "res":
            sigma_pred = y_pred * 0 + 1

        if self.mode == "symetric":
            fcorr_lower = self.fcorr
            fcorr_upper = self.fcorr
        else:
            fcorr_lower = self.fcorr_lower
            fcorr_upper = self.fcorr_upper

        if self.type_res in ["res", "w_res", "no_calib"]:
            if y_pred.shape != sigma_pred.shape:
                sigma_pred = np.moveaxis(sigma_pred, -1, 0)
                y_pred_lower = y_pred - sigma_pred[0] * fcorr_lower
                y_pred_upper = y_pred + sigma_pred[1] * fcorr_upper
            else:
                y_pred_lower = y_pred - sigma_pred * fcorr_lower
                y_pred_upper = y_pred + sigma_pred * fcorr_upper

        elif self.type_res in ["cqr"]:
            y_pred_upper = y_pred_upper + fcorr_upper
            y_pred_lower = y_pred_lower - fcorr_lower
        else:
            print("Unknown type_res")
            return

        return y_pred_lower, y_pred_upper


def dimensional_reduction(data, mode="umap", reducer=None, fit=True, **kwargs):
    if "n_components" not in kwargs.keys():
        kwargs["n_components"] = 3

    if reducer is not None:
        pass

    else:
        if mode == "umap":
            if "n_neighbors" not in kwargs:
                kwargs["n_neighbors"] = 50
            if "min_dist" not in kwargs:
                kwargs["min_dist"] = 0.01
            # reducer = umap.UMAP(**kwargs)

        elif mode == "tsne":
            reducer = TSNE(**kwargs)

        elif mode == "pca":
            reducer = PCA(**kwargs)

    embedding = reducer.fit_transform(data)
    return (embedding, reducer)


# Moving average function for Lag feature extraction


def fit_compute_lag(Y, lag=[1, 2, 3], delay=0):
    """Create lag features from a numerical array
    Args:
        Y (float array): Target to extract lag-feature
        lag (int, optional): Lag number. Defaults to 3.
        delay (int, optional): Delay before 1 lag feature. Defaults to 0.
    """
    dim = Y.shape
    new_features_list = []
    new_features_name = []
    for i in np.array(lag) + delay:
        Y_cur = np.roll(Y, i, axis=0)
        if i > 0:
            Y_cur[0:i] = 0
        for g in range(dim[1]):
            new_features_list.append(Y_cur[:, g])
            new_features_name.append("lag_" + str(i) + "_dim:" + str(g))
    new_features_name = np.array(new_features_name)
    return (np.array(new_features_list).T, new_features_name)


def autocorr(x):
    "Compute autocorrelation of x"
    result = np.correlate(x, x, mode="full")
    return result[int(result.size / 2):]


def convolute_1D(array, filter=None):
    """Convolution by dimension for 1 od 2D array using np.convolve

    Args:
        array (_type_): array_to_convolve
        filter (_type_, optional): convolution fitler. Defaults to None.

    Raises:
        ValueError: dimension error

    Returns:
        array: array_convoluted
    """
    if filter is None:
        return array
    else:
        array_convoluted = []
        if len(array.shape) == 1:
            array_convoluted = np.convolve(array, filter, "same")
        elif len(array.shape) == 2:
            for i in range(array.shape[1]):
                array_convoluted.append(
                    np.convolve(array[:, i], filter, "same")[:, None]
                )
            array_convoluted = np.concatenate(array_convoluted, axis=1)
        else:
            print("Inapropriate size of array : hold 1 or 2d array")
            raise ValueError
        return array_convoluted
