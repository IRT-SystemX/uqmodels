"""
Data preprocessing module.
"""

import copy

import numpy as np
import pandas as pd
import sklearn
from scipy.interpolate import interp1d
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import TimeSeriesSplit

import uqmodels.preprocessing.structure as pre_struc
import uqmodels.utils as ut

# Target selection based on corrcoef


def compute_corr_and_filter(data):
    mat_corr = np.corrcoef(data.T)
    signal_to_drop = np.isnan(mat_corr[0])
    mat_corr[np.isnan(mat_corr)] = 0
    return (mat_corr, signal_to_drop)


def get_k_composants(mat_corr, n_cible):
    d = mat_corr.shape[0]
    corr_flow = []
    for i in range(100):
        corr_flow.append(
            connected_components(
                np.maximum(0, np.abs(mat_corr) - i / 100) != 0, connection="strong"
            )[1]
        )
    corr_flow = np.concatenate(corr_flow).reshape(100, d)
    step = np.argmax(np.max(corr_flow, axis=1) == n_cible)
    list_components = []
    for component in range(n_cible):
        mask = corr_flow[step] == component
        list_components.append(np.arange(d)[mask])
    return list_components


def select_best_representant(mat_corr, list_components):
    best_signals = []
    for i, ind_components in enumerate(list_components):
        ind_best = np.argmax(mat_corr[ind_components][:, ind_components].sum(axis=1))
        best_signals.append(ind_components[ind_best])
    return best_signals


def select_signal(data, n_cible=None):
    if n_cible is None:
        n_cible = data.shape[1]

    mat_corr, signal_to_drop = compute_corr_and_filter(data)
    raw_ind = np.arange(data.shape[1])
    restricted_mat_corr = mat_corr[~signal_to_drop][:, ~signal_to_drop]
    list_restricted_components = get_k_composants(restricted_mat_corr, n_cible)
    best_restricted_signals = select_best_representant(
        restricted_mat_corr, list_restricted_components
    )
    best_signals = raw_ind[~signal_to_drop][best_restricted_signals]

    print(np.max(mat_corr[best_signals], axis=0).sum() / data.shape[1])
    return (best_signals, signal_to_drop)


#


def check_is_pd_date(date):
    if isinstance(date, str):
        try:
            date = pd.to_datetime(date)
        except BaseException:
            pass
    return date


# Base replacement function


def locate_near_index_values(index_scale, index_val):
    if (index_val > index_scale[-1]) or (index_val < index_scale[0]):
        return None
    else:
        ind = np.abs(index_scale - index_val).argmin()
        if (index_scale[ind] - index_val) > 0:
            return ind - 1
        else:
            return ind


def get_event_into_series(
    list_events, index_scale, n_type_event, dtype="datetime64[s]"
):
    """Locate flag erros of sensors in regular time_refenrencial"""
    nan_series = np.zeros((len(index_scale), n_type_event + 1))
    index_scale = index_scale.astype(dtype).astype(float)
    for n, dict_events in enumerate(list_events):
        for index_event in list(dict_events.keys()):
            type_anom = dict_events[index_event]
            try:
                start_event, end_event = index_event
                start_index = (
                    pd.to_datetime(start_event)
                    .to_datetime64()
                    .astype(dtype)
                    .astype(float)
                )

                end_index = (
                    pd.to_datetime(end_event)
                    .to_datetime64()
                    .astype(dtype)
                    .astype(float)
                )

                if (start_index is None) & (end_index is None):
                    pass
                else:
                    if start_index is None:
                        start_index = 0

                    if end_index is None:
                        end_index = -1

                    nan_series[start_index:end_index, type_anom] += 1

            except BaseException:
                index_event = (
                    pd.to_datetime(index_event)
                    .to_datetime64()
                    .astype(dtype)
                    .astype(float)
                )
                ind = locate_near_index_values(index_scale, index_event)
                nan_series[ind, type_anom] += 1

    return nan_series


def extract_sensors_errors(series, type_sensor_error=[]):
    """Extract list of non floating values

    Args:
        series (_type_): series of sensor_values
        type_sensor_error (list, optional): list of others errors.
    """

    for i in list(set(series)):
        try:
            float(i)
        except ValueError as e:
            e = str(e).split("'")[1]
            if e not in type_sensor_error:
                type_sensor_error.append(e)
    return type_sensor_error


def handle_nan(y):
    "Replace nan values by last values"
    nan_flag = np.isnan(y)
    last_non_nan = np.copy(y)
    for j in np.arange(len(y))[nan_flag]:
        if j < 1:
            last_non_nan[j] = np.nanmean(y, axis=0)
        else:
            last_non_nan[j] = y[j - 1]
        v = last_non_nan[j]
        y[j] = v
    return y


def interpolate(
    x,
    y,
    xnew=None,
    time_structure=None,
    type_interpolation="linear",
    fill_values=None,
    moving_average=False,
):
    """Drop nan values & perform 'interpolation' interpolation from [x,y]  to [xnew,ynew]
        if xnew is none, compute xnew from time_structure

        if moving_average=True perform "interpolate moving average" using int(len(xnew)/len(x))= M
        in order to perform mean of M interpolated point evenly distributed for each step.

    Args:
        x (array): X_axis
        y (array): Y_axis (values)
        xnew (array): new X_axis
        moving_average (bool, optional): Perform moving average 'interpolation'.

    Returns:
        ynew: new interpolated Y_axis
    """
    if x is None:
        x = np.arange(len(y))

    if fill_values is None:
        fill_values = np.nanmean(y, axis=0)

    only_fill_nan = False
    if (time_structure is None) and (xnew is None):
        only_fill_nan = True
        xnew = x

    precision = 0
    if xnew is None:
        # Interpolation de la série avec prise en compte de l'échantillonage théorique selon la time structure
        if time_structure is None:
            print("time_structure is None")
            raise ValueError

        delta = time_structure.get("delta", default_value=1)
        # If precision and frequence are provided, else default_value
        precision = time_structure.get("precision", default_value=0)
        frequence = time_structure.get("frequence", default_value=1)

        new_time_step = delta * frequence

        xnew = pre_struc.get_regular_step_scale(
            delta=new_time_step, range_=(x[-1] - x[0]), time_offset=x[0]
        )

    # Ignore Nan values
    nan_flag = np.isnan(y)

    # ration of moding average
    ynew = np.zeros(len(xnew))
    f = interp1d(
        x[~nan_flag], y[~nan_flag], type_interpolation, fill_value="extrapolate"
    )

    if only_fill_nan:
        ynew = y
        ynew[nan_flag] = f(xnew[nan_flag])

    else:
        ynew = np.zeros(len(xnew))

        # ratio of moving average
        ratio = 1
        if moving_average:
            ratio = int(len(x) / len(xnew))
            if ratio < 1:
                ratio = 1

        for i in range(ratio):
            ynew += f(xnew - (i * xnew[0]) / ratio) / ratio

        if precision > 0:
            ynew = np.round(ynew, -int(np.floor(np.log10(precision))))

    return xnew, ynew


def add_row(df, date_pivot, mode="first"):
    """Add first or last np.Nan row to df with date_pivot as index values.

    Args:
        df (_type_): dataframe
        date_pivot (_type_): index
        mode (str, optional): 'first' or 'last'. Defaults to 'first'.

    Returns:
        df: dataframe augmented with one row
    """
    new_row = copy.deepcopy(df.iloc[0:1])
    for columns in new_row.columns:
        new_row.loc[new_row.index[0], columns] = np.NaN
    new_row = new_row.rename({new_row.index[0]: date_pivot})

    if mode == "first":
        df = pd.concat([new_row, df])

    if mode == "last":
        df = pd.concat([df, new_row])

    return df


def remove_rows(df, date_pivot, mode="first"):
    """Remove rows smaller/greated than date_pivot. then add apply add_row

    Args:
        df (_type_): dataframe
        date_pivot (_type_): index_pivot
        mode (str, optional): 'first' or 'last'. Defaults to 'first'.

    Returns:
        df: dataframe which removed values and a new bondary row
    """
    if mode == "first":
        if df.index[0] > date_pivot:
            pass
        else:
            df = df[df.index > date_pivot]
    if mode == "last":
        if df.index[-1] < date_pivot:
            pass
        else:
            df = df[df.index < date_pivot]
    df = add_row(df, date_pivot, mode=mode)
    return df


def df_selection(df, start_date=None, end_date=None):
    """Format dataframe to obtain a new version that start at start_date and finish and end_date

    Args:
        df (_type_): dataframe
        start_date (_type_, optional): strat_date or None. Defaults to None: do nothnig
        end_date (_type_, optional): end_date or None. Defaults to None: do nothnig

    Returns:
        dataframe: Time formated dataframe
    """

    if start_date is not None:
        start_date = check_is_pd_date(start_date)
        df = remove_rows(df, start_date, mode="first")
    if end_date is not None:
        end_date = check_is_pd_date(end_date)
        df = remove_rows(df, end_date, mode="last")
    return df


def upscale_series(
    dataframe,
    delta,
    offset=None,
    start_date=None,
    end_date=None,
    mode="time",
    max_time_jump=10,
    replace_val=None,
    **kwargs
):
    """Upsample series using pandas interpolation function

    Args:
        dataframe (_type_): data to resample
        delta (_type_): Timedelta
        offset (str, optional): _description_. Defaults to '-1ms'.
        origin (str, optional): _description_. Defaults to 'start_day'.
        mode (str, optional): _description_. Defaults to 'time'.
        max_time_jump (int, optional): _description_. Defaults to 10.
        replace_val (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # Generate upsampling with NaN values

    start_date = pre_struc.check_date(start_date)
    end_date = pre_struc.check_date(end_date)
    delta = pre_struc.check_delta(delta)

    origin_date = start_date
    if delta is not None:
        origin_date = origin_date - delta

    dataframe = df_selection(dataframe, start_date=origin_date, end_date=end_date)

    # Fill nan using interpolation
    if mode == "previous":
        upscale_dataframe = dataframe.resample(
            pd.Timedelta(delta), origin=pd.to_datetime(start_date), offset=offset
        ).ffill(limit=max_time_jump)
    else:
        upscale_dataframe = (
            dataframe.resample(
                pd.Timedelta(delta), origin=pd.to_datetime(start_date), offset=offset
            )
            .mean()
            .interpolate(mode, limit=max_time_jump)
        )

    if max_time_jump is not None:
        mask = np.isnan(upscale_dataframe.values)
        # Replace using by min values
        if replace_val is None:
            replace_val = dataframe.min(axis=0).values - 0.1 * np.abs(
                dataframe.min(axis=0).values
            )

        # Replace using default values
        if isinstance(replace_val, np.ndarray):
            for i, val in enumerate(replace_val):
                upscale_dataframe.iloc[mask[:, i], i] = val

        else:
            upscale_dataframe[mask] = replace_val

    return upscale_dataframe[1:]


def downscale_series(
    dataframe,
    delta,
    offset="-1ms",
    start_date=None,
    end_date=None,
    mode="mean",
    dtype="datetime64[s]",
    **kwargs
):
    end_date = pre_struc.check_date(end_date, dtype)
    start_date = pre_struc.check_date(start_date, dtype)
    delta = pre_struc.check_delta(delta, dtype)

    dataframe = df_selection(
        dataframe, start_date=pd.to_datetime(start_date - delta), end_date=end_date
    )

    downscale = dataframe.resample(
        pd.Timedelta(delta), origin=pd.to_datetime(start_date), offset=offset
    )

    if hasattr(downscale, mode):
        func_ = getattr(downscale, mode)
        downscale_dataframe = func_().interpolate("time")
    else:
        raise (ValueError, "mode:", mode, "not handle")
    downscale_dataframe.index -= pd.Timedelta(offset)
    return downscale_dataframe


def rolling_statistics(
    data, delta, step=None, reduc_functions=["mean"], reduc_names=["mean"], **kwargs
):
    """Compute rollling_statistics from dataframe

    Args:
        data (pd.DataFrame): dataframe (times,sources)
        delta (int or timedelta64): size of rolling window
        step (int):  Evaluate the window at every ``step`` result
        reduc_functions (_type_): str of pandas window function (fast) or custom set->stat function (slow)
        reduc_names (_type_): name stored in stat_dataframe
        time_mask (_type_, optional): time_mask. Defaults to None.
        **kwargs: others paramaters provide to DataFrame.rolling

    Returns:
        _type_: _description_
    """

    new_data = []
    colums_transformation = []
    reduc_functions = reduc_functions.copy()
    reduc_names = reduc_names.copy()

    columns = data.columns
    if not isinstance(delta, int):
        delta = pd.Timedelta(pre_struc.check_delta(delta, "datetime64[ns]"))
        data.index = data.index.astype("datetime64[ns]")

    flag_extremum = False
    if "extremum" in reduc_functions:
        flag_extremum = True
        ind_extremum = reduc_functions.index("extremum")
        reduc_functions[ind_extremum] = "max"
        reduc_functions.append("min")
        reduc_names.append("min")

    for n, reduc_func in enumerate(reduc_functions):
        colums_transformation.append(reduc_names[n])
        if isinstance(reduc_func, str):
            if reduc_func == "cov_to_median":
                # Hold unimplementation of step
                df_mean = data.rolling(delta, step=step, min_periods=1, **kwargs).mean()
                tmp_df = df_mean.rolling(20, min_periods=1).cov(df_mean.median(axis=1))
                tmp_df = tmp_df.fillna(0)
            elif reduc_func == "corr_to_median":
                # Hold unimplementation of step
                df_mean = data.rolling(delta, step=step, min_periods=1, **kwargs).mean()
                tmp_df = df_mean.rolling(20).corr(df_mean.median(axis=1))
                tmp_df = tmp_df.fillna(0)
            else:
                rolling = data.rolling(delta, step=step, min_periods=1, **kwargs)
                func_ = getattr(rolling, reduc_func)
                tmp_df = func_()

        else:
            rolling = data.rolling(delta, step=step, min_periods=1, **kwargs)
            tmp_df = rolling.apply(reduc_func)
        tmp_df.columns = [reduc_names[n] + "_" + str(i) for i in tmp_df.columns]
        new_data.append(tmp_df)

    data = pd.concat(new_data, axis=1)
    if flag_extremum:
        for column in columns:
            column = str(column)
            # Max is already named extremum
            data["extremum_" + column] = np.maximum(
                np.abs(data["min_" + column].values-data["mean_" + column].values),
                np.abs(data["extremum_" + column].values-data["mean_" + column].values),
            )
            data.drop(["min_" + column], axis=1, inplace=True)

    return data


# Base measure


def entropy(y, set_val, v_bins=100):
    "Compute naive entropy score of y by tokenize values with max of v_bins"
    flag_nan = np.isnan(y)
    y[flag_nan] = -0.1
    if len(y) == 0:
        return 0
    else:
        labels = np.digitize(
            y,
            bins=np.sort(
                [
                    np.quantile(y, i / set_val)
                    for i in np.arange(0, min(set_val, v_bins))
                ]
            ),
        )
        labels = labels.astype(int)

        value, counts = np.unique(labels, return_counts=True)
        count = np.zeros(set_val + 1) + 1
        for n, i in enumerate(value):
            count[i] = counts[n]
        norm_counts = count / count.sum()
        return -(norm_counts * np.log(norm_counts)).sum()


# Preprocessing function:


def df_interpolation_and_fusion(list_df, target_index_scale, dtype="datetime64[s]"):
    """Interpolation of all sources on a same temporal referencial

    Args:
        list_df (list of 2D array): List of dataframe
        target_index_scale (_type_): Indice of sensors
        dtype:

    Returns:
        interpolated_data: List of interpolated array
    """
    interpolated_data = []
    list_columns = []
    if type(target_index_scale):
        target_index_scale = target_index_scale.astype(dtype).astype(float)

    for df in list_df:
        df_index = df.index.values.astype(dtype).astype(float)
        list_columns.append(list(df.columns))
        if len(df_index) == len(target_index_scale):
            if (df_index - target_index_scale).sum() == 0:
                interpolated_data.append(df.values)
        else:
            new_channels = np.stack(
                [
                    interpolate(
                        x=df_index,
                        y=channels,
                        xnew=target_index_scale,
                        moving_average=True,
                    )[1]
                    for channels in df.values.T
                ]
            ).T

            interpolated_data.append(new_channels)

    interpolated_data = np.swapaxes(interpolated_data, 0, 1).reshape(
        len(target_index_scale), -1
    )

    interpolated_data = pd.DataFrame(
        interpolated_data,
        columns=np.concatenate(list_columns),
        index=target_index_scale.astype(dtype),
    )

    return interpolated_data


# Map reduce statistics (may be computationally suboptimal)


def Past_Moving_window_mapping(array, deta, window_size=None):
    # Past moving mapping using yield: Unefficient for large data !
    if window_size is None:
        window_size = deta

    for i in range(int(np.floor(len(array) / deta))):
        yield array[max(0, i * deta - window_size) : i * deta + 1]


def identity(x, **kwargs):
    return x


def map_reduce(data, map_=identity, map_paramaters={}, reduce=identity):
    mapping = Regular_Moving_window_mapping(data, **map_paramaters)
    reduced = np.array(list(map(reduce, mapping)))
    return reduced


def auto_corr_reduce(set_):
    mean = np.mean(set_)
    var = np.var(set_)
    set_ = set_ - mean
    acorr = np.correlate(set_, set_, "full")[len(set_) - 1 :]
    acorr = acorr / var / len(set_)
    return acorr


def mean_reduce(set_):
    mu = np.nanmean(set_, axis=0)
    std = np.nanstd(set_, axis=0)
    carac = np.concatenate([mu[:, None], std[:, None]], axis=1)
    return carac


def last_reduce(set_):
    return set_[-1]


def corrcoef_reduce(set_):
    return np.corrcoef(set_.T)


def fft_reduce(set_):
    fft = np.fft.fft(set_.T)
    energy = np.abs(fft)[:, 0 : int(len(set_) / 2)] / len(set_)
    phase = np.angle(fft)[:, 0 : int(len(set_) / 2)]
    carac = np.concatenate([energy, phase], axis=1)
    return carac


def Regular_Moving_window_mapping(array, deta, window_size, mode="left", **kwargs):
    if window_size is None:
        window_size = deta
    for i in range(int(np.floor(len(array) / deta))):
        if mode == "left":
            yield array[i * deta : i * deta + window_size]
        elif mode == "center":
            size_bot = int(np.ceil(window_size / 2))
            size_top = int(np.floor(window_size / 2))
            yield array[i * deta - size_bot : i * deta + size_top]

        elif mode == "right":
            yield array[i * deta - (window_size - 1) : i * deta + 1]

        else:
            raise ("error mode")


# Splitter


def identity_split(X_fit, y_fit, X_calib, y_calib):
    """Identity splitter that wraps an already existing data assignment"""

    def iterable_split(X, y):
        """X and y are placeholders."""
        iterable = [(X_fit, y_fit, X_calib, y_calib)]
        return iterable

    return iterable_split


def random_split(ratio):
    """Random splitter that assign samples given a ratio"""

    def iterable_split(X, y):
        rng = np.random.RandomState(0)
        fit_sample = rng.rand(len(X)) > ratio
        cal_sample = np.invert(fit_sample)
        return [(X[fit_sample], y[fit_sample], X[cal_sample], y[cal_sample])]

    return iterable_split


def kfold_random_split(K, random_state=None):
    """Splitter that randomly assign data into K folds"""
    kfold = sklearn.model_selection.KFold(K, shuffle=True, random_state=random_state)

    def iterable_split(X, y):
        iterable = []
        for fit, calib in kfold.split(X):
            iterable.append((X[fit], y[fit], X[calib], y[calib]))
        return iterable

    return iterable_split


# Encapsulated data from stored dict file:


class splitter:
    """Generic data-set provider (Iterable)"""

    def __init__(self, X_split):
        self.X_split = X_split

    def split(self, X):
        def cv_split(X_split, i):
            train = np.arange(len(X))[X_split < i]
            test = np.arange(len(X))[X_split == i]
            return (train, test)

        return [
            cv_split(self.X_split, i) for i in range(1, 1 + int(self.X_split.max()))
        ]


# Encapsulated data from array:


def dataset_generator_from_array(
    X,
    y,
    context=None,
    objective=None,
    sk_split=TimeSeriesSplit(5),
    repetition=1,
    remove_from_train=None,
    attack_name="",
    cv_list_name=None,
):
    """Produce data_generator (iterable [X, y, X_split, context, objective, name]) from arrays

    Args:
        X (array): Inputs.
        y (array or None): Targets.
        context (array or None): Additional information.
        objective (array or None): Ground truth (Unsupervised task).
        sk_split (split strategy): Sklearn split strategy."""

    def select_or_none(array, sample):
        if array is None:
            return None
        elif isinstance(array, str):
            return array
        else:
            return array[sample]

    if remove_from_train is None:
        remove_from_train = np.zeros(len(X))

    dataset_generator = []
    for n_repet in np.arange(repetition):
        cpt = 0
        if n_repet == 0:
            str_repet = ""
        else:
            str_repet = "_bis" + str(n_repet)

        for train_index, test_index in sk_split.split(X):
            X_split = np.zeros(len(X))
            X_split[train_index] = 1
            X_split[(remove_from_train == 1) & (X_split == 1)] = -1

            sample_cv = sorted(np.concatenate([train_index, test_index]))
            cv_name = "cv_" + str(cpt) + attack_name + str_repet
            if cv_list_name:
                cv_name = cv_list_name[cpt] + attack_name + str_repet
            dataset_generator.append(
                [
                    select_or_none(e, sample_cv)
                    for e in [X, y, X_split, context, objective, cv_name]
                ]
            )
            cpt += 1
    return dataset_generator


####################################################################
# Raw data structure regularisation & statistics synthesize.


def raw_analysis(raw_series, time_structure):
    source = raw_series.columns[0]
    x = raw_series.index
    y = raw_series[source].values
    frequence = time_structure.get("frequence", 1)
    dtype = time_structure.get("dtype", "datetime64[s]")
    n_nan = np.isnan(y).sum()
    range_ = (
        (x.values[-1] - x.values[0])
        .astype(dtype.replace("datetime", "timedelta"))
        .astype(float)
    )

    if len(y) > 0:
        n_obs = len(y)
        ratio_comp = np.round(n_obs / (frequence * range_), 4)

    else:
        n_val, v_entropy, n_obs, n_nan, ratio_comp = 0, 0, 0, 0, 0

    n_val = len(set(y))
    v_entropy = entropy(y, n_val)
    time_structure.set("n_obs", n_obs)
    time_structure.set("ratio_comp", ratio_comp)
    time_structure.set("n_val", n_val)
    time_structure.set("v_entropy", v_entropy)
    print(n_val, v_entropy, n_obs, n_nan, ratio_comp)


####################################################################
# Auxilliaire Preprocessor function


def process_raw_source(self, data, query, structure):
    time_structure = structure

    raw_series, dict_errors = data

    type_sensor_error = time_structure.get("type_sensor_error")
    time_structure.set("list_error", dict_errors)
    time_structure.set("type_sensor_error", type_sensor_error)
    time_structure.set("dict_errors", dict_errors)
    raw_analysis(raw_series, time_structure)
    return (raw_series, time_structure)


def process_irregular_data(self, data, query, structure):
    """Apply interpolation & statistics extraction on data using query parameters
    with metadata stored in structure ['start_date','end_date','delta']
    of structure are used to specificy the start, the end and the statistics_step_synthesis.
    ['window_size','begin_by_interpolation] of query are used to specify
    the final step (delta*window_size) and if there is a pre-interpolation step.


    Args:
        data (_type_): _description_
        query (_type_): _description_
        structure (_type_): _description_

    Returns:
        _type_: _description_
    """

    time_structure = structure
    begin_by_interpolation = False
    begin_by_interpolation = time_structure.get("begin_by_interpolation", False)
    window_size = time_structure.get("window_size", 1)

    if len(data) == 2:
        raw_series, raw_time_structure = data
        dict_errors = raw_time_structure.get("dict_errors")
    else:
        raw_series = data
        dict_errors = {}

    time_structure.set("dict_errors", dict_errors)

    # Specify rolling statistics
    delta = time_structure.get("delta", 1)
    dtype = time_structure.get("dtype", "datetime64[s]")

    stats = time_structure.get("stats", ["mean", "std", "extremum", "count"])

    # Statistics processing experte kwdowledge based:
    frequence = time_structure.get("frequence", 1)
    precision = time_structure.get("precision", 1)

    # Interpolation specification
    interpolation = time_structure.get("interpolation", "previous")
    max_time_jump = time_structure.get("max_time_jump", 10)
    replace_val = time_structure.get("replace_val", None)

    # Regular interpolation on rolling statistics:
    start_date = time_structure.get("start_date", None)
    end_date = time_structure.get("end_date", None)

    if begin_by_interpolation:
        raw_series = regular_stat_series = upscale_series(
            raw_series,
            delta * frequence,
            start_date=start_date,
            end_date=end_date,
            x_time_jump=max_time_jump,
            replace_val=replace_val,
            mode=interpolation,
        )

    new_delta = pre_struc.check_delta(delta * window_size, dtype=dtype)

    # Irregular rolling statistics
    regular_stat_series = rolling_statistics(
        raw_series, delta=new_delta, step=None, reduc_functions=stats, reduc_names=stats
    )

    columns = regular_stat_series.columns
    flag_extremum = ["extremum" in column for column in columns]
    flag_count = ["count" in column for column in columns]
    flag_std = ["std" in column for column in columns]

    # Normalisation of count statistics
    for col_name in columns[flag_std]:
        std_value = regular_stat_series[col_name].values
        std_value[np.isnan(std_value)] = 0.000001
        regular_stat_series[col_name] = np.maximum(precision, std_value)
    # Normalisation of count statistics
    for col_name in columns[flag_extremum]:
        regular_stat_series[col_name] = np.log(
            0.0001 + regular_stat_series[col_name].values
        )

    # Include consecutive non-observation in statistics
    for col_name in columns[flag_count]:
        consecutive_no_obs = (
            -np.log10(
                0.5
                + np.maximum(
                    ut.cum_sum_consecutive_zero(
                        regular_stat_series[col_name].values >= 1
                    )
                    - 1,
                    0,
                )
            )
            / 3
        )

        regular_stat_series[col_name] = (
            (regular_stat_series[col_name].values - consecutive_no_obs)
            / frequence
            * new_delta.astype(float)
        )

    regular_stat_series = upscale_series(
        regular_stat_series,
        new_delta,
        start_date=start_date,
        end_date=end_date,
        max_time_jump=max_time_jump,
        replace_val=replace_val,
        mode=interpolation,
    )

    return (regular_stat_series, time_structure)


####################################################################
# Auxilliaire Preprocessor function


def process_label(
    label_df, sources_selection, start_date, end_date, delta=1, dtype="datetime64[s]"
):
    """Process anom label dataframe with (start: datetime64[s], end: datetime64[s],source)
    Into a ground truth matrix with a regular step scale of delta that start at start_date & end at end_date
    """
    source_anom = [i for i in label_df["source"]]
    flag_selection = [i in sources_selection for i in source_anom]

    step_begin_anoms = (
        label_df[flag_selection]["start"].astype(dtype).astype(int).values
    )
    step_end_anoms = label_df[flag_selection]["end"].astype(dtype).astype(int).values

    step_begin = pre_struc.date_to_step(start_date)
    step_end = pre_struc.date_to_step(end_date)

    step_scale = pre_struc.get_regular_step_scale(
        delta, step_end - step_begin, step_begin
    )

    label_anom = np.zeros((len(step_scale), len(sources_selection)))
    for i, name in enumerate(sources_selection):
        mask = (step_scale > step_begin_anoms[i]) & (step_scale < step_end_anoms[i])
        label_anom[mask, i] += 1
    date_index = pre_struc.step_to_date(
        step_scale, delta=delta, date_init=start_date, dtype=dtype
    )
    label_df = pd.DataFrame(data=label_anom, index=date_index)

    return label_df
