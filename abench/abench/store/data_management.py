from functools import partial
from typing import Any, Callable, Dict, Mapping, Sequence, Optional, Dict, Literal, Iterable, Hashable
from pathlib import Path
from abench.store.store import read,write
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
import re
import os

def load_csv(storing,keys):
    return(read(storing,keys))
    
def save_csv(storing,keys):
    return(write(storing,keys))



def explore_csv_hierarchy(root_dir, depth_name_list=None,allowed_ext=('.csv','.parquet')):
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

def apply_caraterisation_across_datasets(load_fn, carac_fn, metadata=None, path=None):
    """
    Apply a characterization function across multiple datasets discovered from metadata or a filesystem path.

    This utility iterates over a collection of dataset descriptors (one per file), loads each dataset
    using `load_fn`, computes characterization outputs using `carac_fn`, prints a per-file summary,
    and returns a dictionary mapping filenames to characterization results.

    Parameters
    ----------
    load_fn : Callable[[str], Any]
        Function that loads a dataset from a file path. It is called as:`df = load_fn(file_path)`.
        Typical usage returns a pandas DataFrame, but any object accepted by `carac_fn` is valid.
    carac_fn : Callable[[Any], Any]
        Characterization function applied to the loaded dataset. It is called as:
        `carac_fn(df)` and can return any serializable object (scalar, dict, tuple, etc.).
    metadata : pandas.DataFrame, optional
        Table describing the datasets to iterate over. Must contain at least the columns:
        - 'path': str, path to the dataset file
        - 'filename': str, identifier used as the result key
        If None, metadata is inferred from `path` via `explore_csv_hierarchy(path)`.
    path : str or pathlib.Path, optional
        Root directory used to infer metadata when `metadata` is None.
        If both `metadata` and `path` are None, the function raises a ValueError.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping each dataset filename (`metadata['filename']`) to the result returned by `carac_fn`.

    Raises
    ------
    ValueError
        If both `metadata` and `path` are None.
    KeyError
        If `metadata` does not contain required columns ('path', 'filename').

    Notes
    -----
    - `carac_fn` is called twice per dataset in the current implementation
      (once for printing, once for storing). If `carac_fn` is expensive or stochastic,
      consider caching the result per dataset.
    - This function prints progress to stdout for quick interactive exploration.

    Examples
    --------
    >>> def load_csv(p): return pd.read_csv(p, parse_dates=["timestamp"]).set_index("timestamp")
    >>> def carac(df): return {"n_rows": len(df), "n_cols": df.shape[1]}
    >>> out = apply_caraterisation_across_datasets(load_csv, carac, path="data/fcu/")
    >>> list(out.keys())[:3]
    ['scenario_01.csv', 'scenario_02.csv', 'scenario_03.csv']
    """
    if metadata is None:
        if path is None:
            raise(ValueError('metadata and path cannot be both None'))
        else:
            metadata = explore_csv_hierarchy(path)

    res_dict = {}
    for n,i in metadata.iterrows():
        df = load_fn(i['path'])
        print(i['filename'],carac_fn(df))
        res_dict[i['filename']]=carac_fn(df)
    return(res_dict)
    

# =============================================================================
# Descriptor Enrichment Pipeline
# =============================================================================
# Design philosophy:
# - Orchestration: enrich_with_descriptors coordinates specialized operations
#   without implementing their internal logic.
# - Separation of concerns: macro transforms, group-level computations, and
#   quantization are handled by dedicated functions.
# - Fail-fast execution: errors are propagated with contextual information
#   (descriptor, group, operation) rather than silently suppressed.
# - Data integrity: preserve row identity, validate outputs, and prevent
#   unintended column overwrites or silent data loss.
# - Modularity: keep operations independently testable and extensible while
#   maintaining a stable public API.
#
# Principle: invalid enrichment must fail explicitly rather than produce
# silently corrupted datasets.
# =============================================================================

def enrich_with_descriptors(
    df: pd.DataFrame,
    macro_descriptors: Mapping | None = None,
    group_descriptors: Mapping | None = None,
    *,
    Id_group: str | None = "metaId",
    time_col: str = "frame",
    time_in_index: bool = False,
    quantize: Mapping | None = None,
    cat_prefix: str = "cat_",
) -> pd.DataFrame:
    """Orchestrate macro, group, and categorical descriptors."""

    if not df.columns.is_unique:
        raise ValueError("Input dataframe has duplicate column names")

    result = _apply_macro_descriptors(
        df.copy(), macro_descriptors or {}
    )

    result = _apply_group_descriptors(
        result,
        group_descriptors or {},
        Id_group=Id_group,
        time_col=time_col,
        time_in_index=time_in_index,
    )

    result = _apply_quantization(
        result,
        quantize or {},
        cat_prefix=cat_prefix,
    )

    return result

def _apply_macro_descriptors(
    df: pd.DataFrame,
    descriptors: Mapping[str, Callable],
) -> pd.DataFrame:
    """Apply dataframe transforms while preserving row identity."""

    for name, func in descriptors.items():
        previous_index = df.index.copy()

        try:
            result = func(df)
        except Exception as exc:
            raise RuntimeError(
                f"Macro descriptor {name!r} failed"
            ) from exc

        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"Macro descriptor {name!r} must return a DataFrame"
            )

        if not result.index.equals(previous_index):
            raise ValueError(
                f"Macro descriptor {name!r} changed the row index"
            )

        if not result.columns.is_unique:
            raise ValueError(
                f"Macro descriptor {name!r} produced duplicate columns"
            )

        df = result

    return df

def _apply_group_descriptors(
    df: pd.DataFrame,
    descriptors: Mapping[str, Callable],
    *,
    Id_group: str | None,
    time_col: str,
    time_in_index: bool,
) -> pd.DataFrame:
    """Compute descriptors per time-ordered group and broadcast values."""

    if not descriptors:
        return df

    # Validate output names
    overlap = set(descriptors).intersection(df.columns)
    if overlap:
        raise ValueError(
            f"Group descriptor columns already exist: {sorted(overlap)}"
        )

    # Validate grouping
    if Id_group is not None:
        if Id_group not in df.columns:
            raise KeyError(f"Missing group column: {Id_group!r}")

        if df[Id_group].isna().any():
            raise ValueError(
                f"Group column {Id_group!r} contains null identifiers"
            )

    # Sort observations
    if time_in_index:
        if df.index.hasnans:
            raise ValueError("Time index contains missing values")
        ordered = df.sort_index(kind="mergesort")
    else:
        if time_col not in df.columns:
            raise KeyError(f"Missing time column: {time_col!r}")

        if df[time_col].isna().any():
            raise ValueError(
                f"Time column {time_col!r} contains missing values"
            )

        sort_keys = (
            [time_col]
            if Id_group is None
            else [Id_group, time_col]
        )
        ordered = df.sort_values(sort_keys, kind="mergesort")

    # Prepare groups and results
    groups = (
        [(None, ordered)]
        if Id_group is None
        else ordered.groupby(Id_group, sort=False, observed=True)
    )

    results = {name: {} for name in descriptors}

    # Compute scalar descriptors
    for group_id, sub in groups:
        for name, func in descriptors.items():
            try:
                value = float(func(sub))

                if np.isinf(value):
                    raise ValueError("Infinite descriptor value")

            except Exception as exc:
                raise RuntimeError(
                    f"Group descriptor {name!r} failed "
                    f"for group {group_id!r}"
                ) from exc

            results[name][group_id] = value

    # Broadcast results to original rows
    for name, values in results.items():
        if Id_group is None:
            df[name] = values.get(None, np.nan)
        else:
            df[name] = df[Id_group].map(values)

    return df

def _apply_quantization(
    df: pd.DataFrame,
    quantize: Mapping[str, Any],
    *,
    cat_prefix: str,
) -> pd.DataFrame:
    """Apply categorical encoding to selected columns."""

    for col, spec in quantize.items():
        if col not in df.columns:
            raise KeyError(f"Cannot categorize missing column {col!r}")

        if (
            isinstance(spec, tuple)
            and len(spec) == 2
            and isinstance(spec[0], str)
        ):
            mode, qspec = spec
        else:
            mode, qspec = "quantile", spec

        try:
            name, values = categorize_series(
                df[col],
                qspec,
                mode=mode,
                cat_prefix=cat_prefix,
                colname=col,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Categorization failed for column {col!r} "
                f"(mode={mode!r})"
            ) from exc

        if name in df.columns:
            raise ValueError(
                f"Categorization would overwrite column {name!r}"
            )

        if (
            not isinstance(values, pd.Series)
            or not values.index.equals(df.index)
        ):
            raise ValueError(
                f"Categorization of {col!r} changed the row index"
            )

        df[name] = values

    return df

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

################################################################################################

def apply_perturbations(
    df: pd.DataFrame,
    macro_perturbation: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]] = {},
    group_perturbation: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]] = {},
    *,
    Id_col: str = "metaId",
    time_col: str = "frame"
) -> pd.DataFrame:

    # --- keep original order ---
    df = df.reset_index()
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
        print(path)
        
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
        df_out = df_out.loc[:, ~df_out.columns.astype(str).str.match(r"^Unnamed")]

        if enrich_params is not None:
            try:
                df_out = enrich_with_descriptors(df_out, **enrich_params)
            except Exception as exc:
                raise RuntimeError(
                    f"Enrichment failed for file {path!r}"
                ) from exc
        write(storing, [key], df_out)

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
