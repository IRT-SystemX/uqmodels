import pandas as pd
import numpy as np

#### Code related to dataframe sampling

def _dtype_agg_dict(
    df_idx: pd.DataFrame,
    *,
    float_agg: str = "mean",
    int_agg: str = "first",
    str_agg: str = "first",
    other_agg: str = "first",
) -> dict:
    """
    Build a dtype-aware aggregation dict for pandas .agg().
    """
    float_cols = df_idx.select_dtypes(include=["float"]).columns
    int_cols = df_idx.select_dtypes(include=["int", "Int64", "UInt64"]).columns
    str_cols = df_idx.select_dtypes(include=["object", "string"]).columns

    agg = {
        **{c: float_agg for c in float_cols},
        **{c: int_agg for c in int_cols},
        **{c: str_agg for c in str_cols},
    }
    other_cols = [c for c in df_idx.columns if c not in agg]
    agg.update({c: other_agg for c in other_cols})
    return agg


def _apply_sampling(
    df: pd.DataFrame,
    *,
    seed: int,
    sampling: int = 1,
    sampling_cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Backward compatible sampling.

    - If sampling_cfg is None:
        offset = seed % sampling; take df.iloc[offset::sampling]
    - Else sampling_cfg dict supports:
        method: "iloc" | "time_resample" (default "iloc")
        sampling: int (stride or minutes; default `sampling`)

      For method="iloc":
        offset: "seed_mod" | int (default "seed_mod")

      For method="time_resample":
        ts_col: str (default "timestamp")
        agg: dict|str (optional). If omitted, uses dtype-aware aggregation below:
        float_agg: str (default "mean")
        int_agg: str (default "first")
        str_agg: str (default "first")
        other_agg: str (default "first")
    """
    if sampling_cfg is None:
        stride = sampling
        if stride < 1:
            raise ValueError("sampling must be >= 1")
        if stride == 1:
            return df.reset_index(drop=True)
        offset = seed % stride
        return df.iloc[offset::stride].reset_index(drop=True)

    if not isinstance(sampling_cfg, dict):
        raise TypeError("sampling_cfg must be a dict or None.")

    method = sampling_cfg.get("method", "iloc")
    stride = int(sampling_cfg.get("sampling", sampling))
    if stride < 1:
        raise ValueError("sampling must be >= 1")

    if method == "iloc":
        off = sampling_cfg.get("offset", "seed_mod")
        offset = (seed % stride) if off == "seed_mod" else (int(off) % stride)
        return df.iloc[offset::stride].reset_index(drop=True) if stride > 1 else df.reset_index(drop=True)

    if method != "time_resample":
        raise ValueError(f"Unknown method: {method!r}")

    # --- time_resample
    ts_col = sampling_cfg.get("ts_col", "timestamp")

    # Ensure DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        df_idx = df.sort_index()
    else:
        if ts_col not in df.columns:
            raise KeyError(f"{ts_col} not found and df has no DatetimeIndex.")
        df_idx = df.copy()
        df_idx[ts_col] = pd.to_datetime(df_idx[ts_col])
        df_idx = df_idx.sort_values(ts_col).set_index(ts_col)

    rule = sampling_cfg.get("rule", f"{stride}min")

    # Aggregation: either explicit agg, or dtype-aware (like _resample_df)
    agg = sampling_cfg.get("agg", None)
    if agg is None:
        agg = _dtype_agg_dict(
            df_idx,
            float_agg=sampling_cfg.get("float_agg", "mean"),
            int_agg=sampling_cfg.get("int_agg", "first"),
            str_agg=sampling_cfg.get("str_agg", "first"),
            other_agg=sampling_cfg.get("other_agg", "first"),
        )

    out = df_idx.resample(rule).agg(agg)
    return out.reset_index().rename(columns={"index": ts_col})