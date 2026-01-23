import pandas as pd
import numpy as np
from typing import Iterable, Optional, Union, Literal

ContinuousEncoding = Literal["none", "cyclical"]

def datetime_index_features_descriptor(
    df: pd.DataFrame,
    *,
    tz: Optional[str] = None,
    # categorical outputs
    hour_scheme: str = "dayparts",
    include_daypart: bool = True,
    include_day_type: bool = True,
    include_weekday_name: bool = False,
    include_month_name: bool = False,
    include_season: bool = True,
    special_days: Optional[Iterable[Union[str, pd.Timestamp, np.datetime64]]] = None,
    special_day_label: str = "special",
    # continuous / continuity-aware outputs
    continuous_encoding: ContinuousEncoding = "none",
    add_hour_cyc: bool = True,
    add_dow_cyc: bool = True,
    add_doy_cyc: bool = True,
    add_month_cyc: bool = False,
    prefix: str = "dt_",
) -> pd.DataFrame:
    """
    Build categorical + optional continuity-aware (cyclical) datetime features from a DataFrame datetime index.

    Notes
    -----
    Set `include_daypart=False` and/or `include_day_type=False` to avoid creating
    `dt_daypart` and/or `dt_day_type`.
    """
    out = df.copy()

    # Ensure datetime index
    idx = out.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, errors="raise")
        idx = pd.DatetimeIndex(idx)

    # Timezone handling
    if tz is not None:
        if idx.tz is None:
            idx = idx.tz_localize(tz)
        else:
            idx = idx.tz_convert(tz)

    # ---------------- categorical features ----------------
    hour = idx.hour

    if hour_scheme == "hour":
        out[f"{prefix}hour"] = pd.Categorical(hour.astype(str))

    elif hour_scheme == "business":
        # Only create hour_type if requested via include_daypart (keeps surface minimal)
        if include_daypart:
            out[f"{prefix}hour_type"] = pd.Categorical(
                np.where((hour >= 9) & (hour < 18), "business", "offhours")
            )

    elif hour_scheme == "dayparts":
        if include_daypart:
            bins = pd.cut(
                hour, bins=[-1, 5, 11, 17, 23],
                labels=["night", "morning", "afternoon", "evening"]
            )
            out[f"{prefix}daypart"] = pd.Categorical(bins.astype(str))

    else:
        raise ValueError(f"Unknown hour_scheme={hour_scheme!r}")

    if include_day_type:
        is_weekend = idx.dayofweek >= 5
        out[f"{prefix}day_type"] = pd.Categorical(np.where(is_weekend, "weekend", "weekday"))

    if include_weekday_name:
        out[f"{prefix}weekday_name"] = pd.Categorical(idx.day_name())

    if include_month_name:
        out[f"{prefix}month_name"] = pd.Categorical(idx.month_name())

    if include_season:
        m = idx.month
        season = np.select(
            [m.isin([12, 1, 2]), m.isin([3, 4, 5]), m.isin([6, 7, 8]), m.isin([9, 10, 11])],
            ["winter", "spring", "summer", "autumn"],
            default="unknown",
        )
        out[f"{prefix}season"] = pd.Categorical(season)

    if special_days is not None:
        special = pd.to_datetime(list(special_days)).normalize()
        idx_norm = pd.DatetimeIndex(idx).normalize()
        is_special = idx_norm.isin(special)
        out[f"{prefix}special_day"] = pd.Categorical(
            np.where(is_special, special_day_label, "regular")
        )

    # ---------------- continuity-aware (cyclical) features ----------------
    if continuous_encoding == "cyclical":
        idx_naive = pd.DatetimeIndex(idx).tz_convert(None) if idx.tz is not None else idx

        def _add_cyclical(name: str, values: np.ndarray, period: float) -> None:
            angle = 2.0 * np.pi * (values.astype(float) / period)
            out[f"{prefix}{name}_sin"] = np.sin(angle)
            out[f"{prefix}{name}_cos"] = np.cos(angle)

        if add_hour_cyc:
            _add_cyclical("hour", idx_naive.hour.to_numpy(), period=24.0)
        if add_dow_cyc:
            _add_cyclical("dow", idx_naive.dayofweek.to_numpy(), period=7.0)
        if add_doy_cyc:
            doy = idx_naive.dayofyear.to_numpy() - 1
            years = idx_naive.year.to_numpy()
            is_leap = ((years % 4 == 0) & ((years % 100 != 0) | (years % 400 == 0)))
            period = 366.0 if np.any(is_leap) else 365.0
            _add_cyclical("doy", doy, period=period)
        if add_month_cyc:
            _add_cyclical("month", (idx_naive.month.to_numpy() - 1), period=12.0)

    elif continuous_encoding != "none":
        raise ValueError(f"Unknown continuous_encoding={continuous_encoding!r}")

    return out