from functools import partial
from typing import Any, Callable, Dict, Mapping, Sequence, Optional, Dict, Literal, Iterable, Hashable
from pathlib import Path
from abench.store.store import read,write
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
import re
import os


def explore_csv_hierarchy(root_dir, depth_name_list=None,allowed_ext=('.csv')):
    """
    Recursively explore a directory tree and list all CSV files
    along with their hierarchical structure.

    Parameters
    ----------
    root_dir : str
        Root directory to start the recursive search.
    depth_name_list : list[str] | None
        Optional list of column names for hierarchy levels.
        If None, levels are named as 'level_0', 'level_1', etc.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing one row per CSV file with:
        - one column per directory level,
        - 'filename' for the file name,
        - 'path' for the absolute file path.
    """
    data = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Filter out hidden directories (in-place to affect os.walk traversal)
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]

        # Filter out hidden files
        filenames = [f for f in filenames if not f.startswith('.')]
        filenames = [f for f in filenames if "metadata" not in f]
        for file in filenames:
            if file.endswith(allowed_ext):
                full_path = os.path.join(dirpath, file)
                relative_path = os.path.relpath(full_path, root_dir)
                parts = relative_path.split(os.sep)
                
                if depth_name_list is None:
                    row = {f'level_{i}': part for i, part in enumerate(parts[:-1])}
                else:
                    row = {depth_name_list[i]: part for i, part in enumerate(parts[:-1])}
                row['filename'] = parts[-1]
                row['path'] = full_path
                data.append(row)
    df = pd.DataFrame(data)
    # Optional: sort columns (levels first, then filename/path)
    return df

def load_csv(storing,keys):
    return(read(storing,keys))
    
def save_csv(storing,keys):
    return(write(storing,keys))

def filter_metadata(metadata_df, constraint_selection_list=None, constraint_rejection_list=None):
    df_filtered = metadata_df.copy()

    # Apply inclusion constraints
    if constraint_selection_list:
        for col, allowed_values in constraint_selection_list:
            df_filtered = df_filtered[df_filtered[col].isin(allowed_values)]

    # Apply exclusion constraints
    if constraint_rejection_list:
        for col, rejected_values in constraint_rejection_list:
            df_filtered = df_filtered[~df_filtered[col].isin(rejected_values)]
    return df_filtered

def filter_paths_from_metadata(metadata_df, constraint_selection_list=None, constraint_rejection_list=None):
    df_filtered = filter_metadata(metadata_df, constraint_selection_list=constraint_selection_list, constraint_rejection_list=constraint_rejection_list)
    return df_filtered['path'].tolist()


def enrich_with_descriptors(
    df: pd.DataFrame,
    macro_descriptors: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]] = {},
    group_descriptors: Mapping[str, Callable[[pd.DataFrame], float]] = {},
    *,
    Id_group: Optional[str] = "metaId",
    time_col: str = "frame",
    time_in_index: bool = False,
    quantize: Optional[Mapping[str, Sequence[Any]]] = None,
    cat_prefix: str = "cat_",
) -> pd.DataFrame:
    """
    Compute optional macro descriptors (row-wise / dataframe-wise transforms) and per-group descriptors,
    then broadcast per-group descriptor values back to the original rows.

    This function supports:
    - Standard case: `Id_group` and `time_col` are regular columns.
    - No grouping: set `Id_group=None` to treat the whole dataframe as a single group.
    - Time stored in index: set `time_in_index=True` to sort by the index instead of `time_col`.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    macro_descriptors : Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]]
        Dict of dataframe transforms. Each function receives a dataframe and must return a dataframe
        (typically with additional columns).
    group_descriptors : Mapping[str, Callable[[pd.DataFrame], float]]
        Dict of {output_column_name: function}. Each function receives a group sub-dataframe (sorted in time)
        and returns a scalar descriptor.
    Id_group : str | None
        Column name used to define groups (trajectories). If None, a single global group is used.
    time_col : str
        Column name containing time information when `time_in_index=False`.
    time_in_index : bool
        If True, time is taken from the dataframe index for sorting.
    quantize : Optional[Mapping[str, Sequence[Any]]]
        Optional categorization spec per descriptor. This is passed to your `categorize_series`.
        (Kept as `Any` because your function supports different (mode, spec) formats.)
    cat_prefix : str
        Prefix for generated categorical columns.

    Returns
    -------
    pd.DataFrame
        Enriched dataframe (original rows) with descriptor columns (and optional categorical columns).
    """
    # --- Apply macro descriptors (dataframe transforms)
    for _, func in macro_descriptors.items():
        df = func(df)

    # --- Validate / decide sorting keys
    if time_in_index:
        # Ensure index is sortable and stable
        df_sorted = df.sort_index(kind="mergesort")
    else:
        if time_col not in df.columns:
            raise KeyError(
                f"time_col={time_col!r} not found in df.columns. "
                "Either provide a valid time_col or set time_in_index=True."
            )
        if Id_group is None:
            df_sorted = df.sort_values([time_col], kind="mergesort")
        else:
            if Id_group not in df.columns:
                raise KeyError(
                    f"Id_group={Id_group!r} not found in df.columns. "
                    "Either provide a valid Id_group or set Id_group=None."
                )
            df_sorted = df.sort_values([Id_group, time_col], kind="mergesort")

    # --- Prepare descriptors
    prepared: Dict[str, Callable[[pd.DataFrame], float]] = dict(group_descriptors)

    # --- Compute group descriptors
    results: Dict[Any, Dict[str, float]] = {}

    if Id_group is None:
        # Single global group
        sub = df_sorted
        row: Dict[str, float] = {}
        for name, fn in prepared.items():
            try:
                row[name] = float(fn(sub))
            except Exception:
                row[name] = np.nan
        results["__all__"] = row

        summary = pd.DataFrame.from_dict(results, orient="index")
        # Broadcast to all rows
        enriched = df.copy()
        for col in summary.columns:
            enriched[col] = summary.loc["__all__", col]

    else:
        # Standard grouped case
        for gId, sub in df_sorted.groupby(Id_group, sort=False, as_index=True, group_keys=False):
            row = {}
            for name, fn in prepared.items():
                try:
                    row[name] = float(fn(sub))
                except Exception:
                    row[name] = np.nan
            results[gId] = row

        summary = pd.DataFrame.from_dict(results, orient="index")
        enriched = df.merge(summary, left_on=Id_group, right_index=True, how="left")

    # --- Optional quantization (your existing behavior)
    if quantize:
        for col, spec in quantize.items():
            if col in enriched.columns:
                # Your current code expects (mode, spec). Keep compatibility:
                # - if user provides tuple (mode, spec)
                # - else assume legacy spec is already in desired format
                if isinstance(spec, tuple) and len(spec) == 2:
                    mode, qspec = spec
                else:
                    # fallback (legacy): interpret as quantiles with default mode
                    mode, qspec = "quantile", spec
                name, out = categorize_series(
                    enriched[col], qspec, mode=mode, cat_prefix=cat_prefix, colname=col
                )
                enriched[name] = out

    return enriched

def categorize_series(
    s: pd.Series,
    spec: Optional[Sequence[float | Hashable]] = None,   # new generic name
    *,
    mode: Literal["quantile", "boundary", "vocab","pattern"] = "quantile",
    cat_prefix: str,
    colname: str,
    include_lowest: bool = True,
    right: bool = True,
    expand_outside: bool = True
    ) -> tuple[str, pd.Series]:
    """
    Categorize a pandas Series into ordinal integer classes (1..K).

    Modes
    -----
    - "quantile": `spec` is quantiles in [0,1] → uses pd.qcut(..., duplicates="drop")
    - "boundary": `spec` is numeric boundaries → uses pd.cut(..., include_lowest, right)
                  if expand_outside=True, extend to (-inf, +inf)
    - "vocab":   `spec` is a sequence of *unique tokens* → map token -> 1..K
                 Unseen tokens / missing -> <NA>

    Returns
    -------
    (output_name, categorized_series[Int64])
    """

    n = len(s)
    name = f"{cat_prefix}{colname}"
    na_series = pd.Series(pd.array([pd.NA] * n, dtype="Int64"), index=s.index)

    if mode == "quantile":
        q = np.asarray(spec, dtype=float)
        q = np.clip(q, 0.0, 1.0)
        # enforce strict monotonicity
        q = q[np.concatenate(([True], np.diff(q) > 0))]
        if q.size < 2:
            return name, na_series
        try:
            cats = pd.qcut(s, q=q, labels=False, duplicates="drop")
            out = (cats.astype(float) + 1).astype("Int64")
        except ValueError:
            out = na_series

    elif mode == "boundary":
        edges = np.asarray(spec, dtype=float)
        # strictly increasing
        edges = edges[np.concatenate(([True], np.diff(edges) > 0))]
        if edges.size < 2:
            return name, na_series
        if expand_outside:
            if np.isfinite(edges[0]):   edges = np.concatenate(([-np.inf], edges))
            if np.isfinite(edges[-1]):  edges = np.concatenate((edges, [np.inf]))
        cats = pd.cut(s, bins=edges, labels=False, include_lowest=include_lowest, right=right)
        out = (cats.astype(float) + 1).astype("Int64")

    elif mode == "vocab":
        # --- token -> id mapping (1..K), preserving order & uniqueness --- #
        # remove missing in spec and keep first occurrence only
        seen, vocab = set(), []
        for tok in (spec or []):
            if tok is pd.NA or tok is None:  # skip missing tokens in spec
                continue
            if tok not in seen:
                seen.add(tok)
                vocab.append(tok)
        if len(vocab) == 0:
            return name, na_series

        mapping = {tok: i + 1 for i, tok in enumerate(vocab)}  # 1-based classes
        # map tokens in s → integers; unknowns/missing → <NA>
        out = s.map(mapping).astype("Int64")
    elif mode == "pattern":
        # `spec` must be a dict {pattern: weight}
        if not isinstance(spec, Mapping) or not spec:
            raise TypeError('spec shoulh be Mapping')
        # Case-insensitive, substring match, count max once per pattern
        flags = re.IGNORECASE

        compiled = []
        for pat, w in spec.items():
            p = re.escape(str(pat))          # literal substring
            creg = re.compile(p, flags=flags)
            compiled.append((creg, float(w)))

        s_str = s.astype("string")
        total = pd.Series(0.0, index=s.index)

        for creg, w in compiled:
            m = s_str.str.contains(creg, na=False)    # True if present at least once
            total = total.add(m.astype(float) * w, fill_value=0.0)

        total = total.mask(s_str.isna(), other=np.nan)

        # If sums are integer-valued, return Int64; else Float64
        if all(isinstance(v, (int,float)) and float(v).is_integer() for v in spec.values()) \
        and (total.dropna() % 1 == 0).all():
            out = total.round().astype("Int64")
        else:
            out = total.astype("Float64")
    else:
        raise ValueError("`mode` must be 'quantile', 'boundary', or 'vocab'.")

    return name, out

def apply_perturbations(
    df: pd.DataFrame,
    macro_perturbation: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]] = {},
    group_perturbation: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]] = {},
    *,
    Id_col: str = "metaId",
    time_col: str = "frame"
) -> pd.DataFrame:

    # --- keep original order ---
    _order = df.index

    # --- global perturbations ---
    for name, func in macro_perturbation.items():
        df = func(df)

    if not group_perturbation:
        # re-sort in original order anyway
        return df.loc[_order]

    # --- groupwise perturbations ---
    dfs = []
    df_sorted = df.sort_values([Id_col, time_col], kind="mergesort")

    for gId, sub in df_sorted.groupby(Id_col, sort=False, group_keys=False):
        sub_pert = sub.copy()
        for name, fn in group_perturbation.items():
            try:
                sub_pert = fn(sub_pert)
            except Exception as e:
                raise RuntimeError(f"Perturbator {name} failed on group {gId}") from e
        dfs.append(sub_pert)

    out = pd.concat(dfs, axis=0)

    # --- restore original input order ---
    return out.loc[_order]


def augment_csvs_with_metadata(
    metadata_df: pd.DataFrame,
    columns_to_add=[],
    insert_at='right',     # 'left' -> prepend, 'right' -> append
    enrich_params=None,
) -> pd.DataFrame:
    """
    For each row in `metadata_df` (with columns at least ['dataset','set','filename','path']),
    open the CSV at 'path', add the requested metadata columns to every row, and rewrite the CSV.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Must contain at least the columns: ['dataset', 'set', 'filename', 'path'].
    columns_to_add : tuple/list of str
        Metadata column names to copy from `metadata_df` into each CSV (one scalar per file).
    insert_at : {'left', 'right'}
        Where to place the new columns in the CSV (prepend or append).

    Returns
    -------
    pd.DataFrame
        A report with columns: ['path','status','rows','cols_before','cols_after','message'].
    """
    required = {'dataset', 'set', 'filename', 'path'}
    missing = required - set(metadata_df.columns)
    if missing:
        raise ValueError(f"metadata_df missing required columns: {sorted(missing)}")


    for _, row in metadata_df.iterrows():
        path = row['path']
        
        # Read CSV
        storing, key = os.path.split(path)

        df = read(storing,[key])
        n_rows = len(df)

        # Build the metadata columns (scalar values repeated for all rows)
        meta_values = {col: row[col] for col in columns_to_add if col in metadata_df.columns}

        # Insert columns
        if insert_at == 'left':
            # Prepend: create a new DataFrame with meta first
            meta_df = pd.DataFrame({k: [v]*n_rows for k, v in meta_values.items()})
            df_out = pd.concat([meta_df.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
        elif insert_at == 'right':
            # Append: assign columns directly (pandas will broadcast)
            for k, v in meta_values.items():
                df[k] = v
            df_out = df
        else:
            raise ValueError("insert_at must be 'left' or 'right'")
        # Overwrite CSV
        df_out = df_out.loc[:, ~df.columns.str.contains("^Unnamed")]
        if(enrich_params is not None):
            df_out = enrich_with_descriptors(df_out,**enrich_params)
        df_out.to_csv(path, index=False, lineterminator='\n')

def stratified_train_val_split(df, strat_cols, alpha=0.8, random_state=42):
    """
    Sépare un DataFrame en train et validation avec une stratification multi-colonnes.

    Paramètres
    ----------
    df : pd.DataFrame
        Le dataframe complet.
    strat_cols : list[str]
        Les colonnes utilisées pour la stratification (ex: ['col1', 'col2']).
    alpha : float
        Proportion d'exemples dans le train (entre 0 et 1).
    random_state : int
        Graine pour la reproductibilité.

    Retour
    ------
    df_train : pd.DataFrame
    df_val : pd.DataFrame
    """

    # On crée une colonne "strat_key" combinant les colonnes
    strat_key = df[strat_cols].astype(str).agg('_'.join, axis=1)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=alpha,
        random_state=random_state
    )

    # split
    for train_idx, val_idx in splitter.split(df, strat_key):
        df_train = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()

    return df_train, df_val
