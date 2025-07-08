"""
Specification of structure object representing operation knowledge about specific data structure.
"""

import datetime as dt

import jsonpickle
import numpy as np
import pandas as pd

date_init_default = "1970-01-01 00:00:00.000000"


def get_unit(dtype):
    return np.datetime_data(np.datetime64("0").astype(dtype=dtype))[0]


def check_delta(delta, dtype="datetime64[s]"):
    unit = get_unit(dtype)

    if delta is None:
        return None

    elif hasattr(delta, "to_timedelta64"):  # Hold pd.Timestamp
        delta = delta.to_timedelta64().astype("timedelta64[" + unit + "]")

    elif isinstance(delta, np.timedelta64):
        delta = delta.astype("timedelta64[" + unit + "]")

    elif isinstance(delta, dt.timedelta):
        delta = np.timedelta64(delta.seconds, "s").astype("timedelta64[" + unit + "]")
    else:
        try:
            delta = np.timedelta64(delta, unit)
        except BaseException:
            try:
                delta = np.timedelta64(pd.Timedelta(delta), unit)
            except BaseException:
                ValueError("delta :", delta, "not recognized")
    return delta


def check_date(date, dtype="datetime64[s]"):
    unit = get_unit(dtype)

    if date is None:
        return None

    elif hasattr(date, "to_datetime64"):  # Hold pd.Timestamp
        date = date.to_timedelta64().astype("datetime64[" + unit + "]")

    elif isinstance(date, np.datetime64):
        date = date.astype("datetime64[" + unit + "]")
    else:
        try:
            date = np.datetime64(date).astype("datetime64[" + unit + "]")
        except BaseException:
            ValueError("date :", date, "not recognized")

    return date


class Structure:
    # Data specification :
    # Exemple 1 : Regular time structure specifying temporal interaction.
    # Exemple 2 : Space interaction structure.
    def __init__(self, name, **kwargs):
        self.name = name
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def set(self, key, obj):
        """self[key] = obj using setattr function

        Args:
            key (str): key of attribute
            obj (obj): attribute to store
        """
        setattr(self, key, obj)

    def toJSON(self):
        return jsonpickle.encode(self)

    def get_structure(self, str_key, **kargs):
        return self

    def get(self, keys, default_value=None, **kwarg):
        """get list of obj related to keys (or obj relate to key if not list) return default values if key not found

        Args:
            keys (str or list of str): key or list of ker
            default_value (_type_, optional): default values if key not found. Defaults to None.

        Returns:
            objs : list of obj or a obj
        """

        # Get attributes form keys.
        # Handle list of attributes keys and returs list of attributes
        # Warning remove None if not found
        list_keys = self.__dict__.keys()

        not_list = False
        if not isinstance(keys, list):
            not_list = True
            keys = [keys]

        list_obj = []
        for key in keys:
            if key in list_keys:
                list_obj.append(getattr(self, key))
            else:
                list_obj.append(default_value)

        if not_list:
            list_obj = list_obj[0]

        return list_obj


# Regular temporal representation ############################


def regular_date_scale(start, end=None, periods=None, delta=1, dtype="datetime64[s]"):
    """Create regular date scale of dtype using pd.date_range starting at start date,
    and ending a end date or start + range * freq

    Args:
        start (str or date): start date
        end (str or date or None, optional): end date. Defaults to None : use start + range*freq
        periods (int, optional): number of period. Defaults to 1000.
        delta (int or timedelta, optional): delta of scale.
        dtype (str, optional): dtype. Defaults to "datetime64[s]".

    Returns:
        _type_: _description_
    """

    delta = pd.Timedelta(check_delta(delta, dtype))
    start = check_date(start, dtype)
    end = check_date(end, dtype)

    date_scale = pd.date_range(
        start=start, end=end, periods=periods, freq=delta
    ).astype(dtype=dtype)

    return date_scale


def str_to_date(str, dtype="datetime64[s]"):
    return check_date(str, dtype)


def step_to_date(step, delta=1, dtype="datetime64[s]", date_init=None):
    """Transform float_step or float_step_array into a date using datetime64[s] format and delta + d_init information
        date format : "%Y-%m-%d %H:%M:%S.%f" deeping about precision.
        date = (step/delta-d_init).astype(dtype).tostr()
    Args:
        step (float or np.array(float)): float representing step
        delta (int, optional): delta between two step. Defaults to 1.
        dtype (str, optional): dtype of date. Defaults to 'datetime64[s]'.
        date_init (str, optional): str_date of first step. Defaults to None.

    Returns:
        date or np.array(date): date that can be cast as float using date.astype(str)
    """
    delta = check_delta(delta, dtype).astype(float)
    date_init = check_date(date_init, dtype).astype(float)

    if (not type(step) == np.array) & (not type(step) == np.ndarray):
        step = np.array(step)

    step_init = 0
    if date_init is not None:
        step_init = date_init

    return (step * delta + step_init).astype(dtype)


def date_to_step(date, delta=1, dtype="datetime64[s]", date_init=None):
    """Transform date or date_array into a step using datetime64[s] format and delta + d_init information
        date format : "%Y-%m-%d %H:%M:%S.%f" deeping about precision.
        step = (date).astype(dtype).tofloat * delta + (date_init).astype(dtype).to_float

    Args:
        date (date or np.array(date)): datetime64 or str_date format : "%Y-%m-%d %H:%M:%S.%f"
        delta (int, optional): delta between two step. Defaults to 1.
        dtype (str, optional): dtype of date. Defaults to 'datetime64[s]'.
        date_init (str, optional): str_date of first step. Defaults to None.

    Returns:
        step or np.array(step): step in float representation
    """

    date = check_date(date, dtype).astype(float)
    delta = check_delta(delta, dtype).astype(float)
    step = date / delta

    step_init = 0
    if date_init is not None:
        step_init = check_date(date_init, dtype).astype(float) / delta

    return step - step_init


def get_regular_step_scale(delta, range_temp, time_offset=0, **kwarg):
    """Generate regular step_scale with delta :
    Args:
        delta (int): size of unitary delta between windows
        range_temp (int): temporal range
        padding (int): Initial_state
        mode (str): linespace or arange

    Returns:
        step_scale: Numeric regular time scale
    """
    delta = check_delta(delta).astype(int)

    step_scale = np.arange(time_offset, time_offset + range_temp, delta)
    return np.round(step_scale, 2)


def get_step_mask(step, step_min, step_max, out_of_mask=True):
    """Compute mask of step_scale array from time boundary

    Args:
        time (array): step_scale
        x_min (float): Minimal considered step
        x_max (float): Maximal considered steps
        out_of_mask (bool, optional): if true incorporate the previous and the next out of bondary step.

    Returns:
        _type_: _description_
    """
    mask_step = np.array(((step >= step_min) & (step <= step_max)))
    if (mask_step.sum() > 0) & out_of_mask:
        min_ = max(np.arange(len(mask_step))[mask_step].min(), 2)
        max_ = min(np.arange(len(mask_step))[mask_step].max(), len(step) - 2)
        if (min_ > 2) & (max_ < (len(mask_step) - 2)):
            (
                mask_step[min_ - 2],
                mask_step[min_ - 1],
                mask_step[max_ + 1],
                mask_step[max_ + 2],
            ) = (True, True, True, True)
    else:
        min_ = max(np.arange(len(mask_step))[mask_step].min(), 1)
        max_ = min(np.arange(len(mask_step))[mask_step].max(), len(step) - 1)
        if max_ < (len(mask_step) - 2):
            mask_step[min_ - 1] = True
            mask_step[max_ + 1] = True
    return mask_step


def get_date_mask(
    date,
    date_min,
    date_max,
    out_of_mask=True,
    delta=1,
    dtype="datetime64[s]",
    date_init=None,
):
    date = check_date(date, dtype)
    step = date_to_step(date, dtype=dtype, date_init=date_init)
    step_min = date_to_step(date_min, dtype=dtype, date_init=date_init)
    step_max = date_to_step(date_max, dtype=dtype, date_init=date_init)
    return get_step_mask(step, step_min, step_max, out_of_mask=out_of_mask)


def window_expansion(step, n_expend=5, delta=1):
    new_step = np.repeat(step, n_expend).astype(float)
    for i in range(n_expend):
        mask = (np.arange(len(step)) % n_expend) == i
    new_step[mask] += (n_expend * delta / 2) - delta * i
    return new_step


def time_selection(x, y, x_min, x_max, out_of_mask, mode="step"):
    if mode == "step":
        mask = get_date_mask(x, x_min, x_max, out_of_mask)

    if mode == "date":
        mask = get_date_mask(x, x_min, x_max, out_of_mask)

    return (x[mask], y[mask])


def regular_representation(list_output, list_delta, delta_target, dim_t=0):
    """Resample list of ndarray using np.repeat according to time representation parameters of each source

    Args:
        list_output (_type_): list of models output for each source
        list_step_scale (_type_): list of times parameters for each source

    Returns:
        list_output with same length (using duplication)
    """
    list_new_output = []
    for output, delta in zip(list_output, list_delta):
        list_new_output.append(np.repeat(output, delta, axis=dim_t))
    return list_new_output


class Irregular_time(Structure):
    # Specification of Irregular time series strustures:
    def __init__(
        self,
        name,
        start_date,
        date_init=date_init_default,
        dtype="datetime64[s]",
        **kwargs
    ):
        super().__init__(
            name=name, dtype=dtype, start_date=start_date, date_init=date_init, **kwargs
        )

    def get_date(self, step):
        return step_to_date(
            step, dtype=self.dtype, delta=self.start_date, date_init=self.date_init
        )

    def get_step(self, date):
        return date_to_step(
            date, dtype=self.dtype, delta=self.start_date, date_init=self.date_init
        )


class Regular_time(Structure):
    # Specification of data knowledge relative to a interaction in data :
    # Exemple 1 : Regular time structure specyfying temporal interaction.
    # Exemple 2 : Space interaction structure.
    def __init__(
        self,
        name,
        start_date,
        delta=np.timedelta64(1, "s"),
        window_size=None,
        date_init=date_init_default,
        dtype="datetime64[s]",
        **kargs
    ):
        if window_size is None:
            window_size = delta

        super().__init__(
            name=name,
            dtype=dtype,
            delta=delta,
            window_size=window_size,
            start_date=start_date,
            **kargs
        )

    def get_date(self, step):
        return step_to_date(
            step, delta=self.delta, dtype=self.dtype, date_init=self.date_init
        )

    def get_step(self, date):
        return date_to_step(
            date, delta=self.delta, dtype=self.dtype, date_init=self.date_init
        )

    def get_step_scale(self, start_date, end_date):
        """Generate step_scale using specification
        Returns:
            step_scale : Numeric time regular array"""

        step_begin = self.get_step(start_date)
        step_end = self.get_step(end_date)

        delta = self.get("delta")

        step_scale = get_regular_step_scale(
            delta, range_temp=step_begin - step_end, time_offset=step_begin
        )
        return np.round(step_scale, 2)


# Multi_source_structure ############################


class Multi_source(Structure):
    def __init__(self, regular_sub_structure=True, name="Multi_sources", **kwargs):
        list_structure = []
        for ind, source in enumerate(kwargs["sources"]):
            dict_time_structure = {"name": source}
            for key in kwargs.keys():
                values = kwargs[key]
                if type(values) in [list, np.ndarray]:
                    if len(values) == len(kwargs["sources"]):
                        values = values[ind]
                dict_time_structure[key] = values

            # Put meta data in Irregular_time_structure object
            if not (regular_sub_structure):
                sub_structure = Irregular_time(**dict_time_structure)

            # Put meta data in Regular_time_structure object
            else:
                sub_structure = Regular_time(**dict_time_structure)

            list_structure.append(sub_structure)

        super().__init__(
            name,
            list_key_source=np.arange(len(kwargs["sources"])),
            list_structure_source=list_structure,
            **kwargs
        )

    def get_structure(self, str_key):
        try:
            ind_key = list(self.get("sources")).index(str_key)
            return self.get("list_structure_source")[ind_key]
        except BaseException:
            return self

    def get(self, keys, default_value=None, query=dict()):
        if "source" in query.keys():
            return self.get_structure(query["source"]).get(
                keys, default_value=default_value
            )
        else:
            return super().get(keys, default_value=None)
