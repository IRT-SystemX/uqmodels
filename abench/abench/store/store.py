"""
==============================================================
Data Storage Utilities
==============================================================

This module provides a unified interface for reading and writing data,
both in-memory (nested Python dictionaries) and on-disk (filesystem trees).

It includes:
- Functions to explore and filter dataset hierarchies (CSV discovery, metadata filtering).
- Generic read/write utilities supporting pluggable I/O functions
  (e.g., JSON, CSV, pickle, or custom formats).
- Tools for augmenting datasets with metadata and maintaining consistent structure.

Goal:
Enable flexible, backend-agnostic storage management so the same logic
can operate seamlessly in memory or on persistent filesystems.
"""

import os, pickle, joblib
import pandas as pd
from pathlib import Path

# -------------------------------------------------------------------
#  BAS NIVEAU : write / read d'un seul fichier
# -------------------------------------------------------------------

def _is_sklearn(obj) -> bool:
    """Detect sklearn-like objects (very lightweight heuristic)."""
    try:
        from sklearn.base import BaseEstimator
        return isinstance(obj, BaseEstimator)
    except Exception:
        return False

def _parquet_engine() -> str | None:
    """Prefer pyarrow > fastparquet; fallback to pandas default."""
    try: import pyarrow; return "pyarrow"
    except Exception:
        try: import fastparquet; return "fastparquet"
        except Exception: return None

def _choose_format(value, suffix: str | None):
    """
    Decide (kind, suffix). Precedence:
      1) explicit valid suffix wins
      2) sklearn -> joblib; DataFrame -> parquet; else pickle
    """
    if suffix:
        s = suffix.lower()
        if s == ".joblib": return "joblib", ".joblib"
        if s in (".p", ".pickle"): return "pickle", ".p"
        if s == ".parquet": return "parquet", ".parquet"
        if s == ".csv": return "csv", ".csv"
        # unknown -> fallback to type inference
    if _is_sklearn(value): return "joblib", ".joblib"
    if isinstance(value, pd.DataFrame): return "parquet", ".parquet"
    return "pickle", ".p"

def _candidates(base: Path):
    """Try these when no suffix is provided."""
    return [
        base.with_suffix(".joblib"),
        base.with_suffix(".p"),
        base.with_suffix(".pickle"),
        base.with_suffix(".parquet"),
        base.with_suffix(".csv"),
    ]

# ------------------------- write function ------------------------ #

def write_function(value, filename):
    """
    Write `value` to disk:
      sklearn -> .joblib
      DataFrame -> .parquet
      else -> .p
    If filename already has a known suffix, respect it.
    """
    path = Path(filename)
    kind, suf = _choose_format(value, path.suffix if path.suffix else None)
    if path.suffix.lower() != suf:
        path = path.with_suffix(suf)
    path.parent.mkdir(parents=True, exist_ok=True)

    if kind == "joblib":
        joblib.dump(value, path)
    elif kind == "pickle":
        with open(path, "wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif kind == "parquet":
        if not isinstance(value, pd.DataFrame):
            raise TypeError("Parquet writing requires a pandas DataFrame.")
        value.to_parquet(path, engine=_parquet_engine(), index=False)
    elif kind == "csv":
        if not isinstance(value, pd.DataFrame):
            raise TypeError("CSV writing requires a pandas DataFrame.")
        value.to_csv(path, index=False)
    return ()


# -------------------------- read function ------------------------ #

def read_function(filename):
    """
    Read from one of: .joblib | .p/.pickle | .parquet | .csv.
    If no suffix given, search in priority order: joblib > pickle > parquet > csv.
    If `filename` is a directory, return the path (as str).
    """
    path = Path(filename)

    # directory case
    if path.is_dir():
        return str(path)

    # explicit extension path
    if path.suffix:
        s = path.suffix.lower()
        if s == ".joblib":
            return joblib.load(path)
        if s in (".p", ".pickle"):
            with open(path, "rb") as f: return pickle.load(f)
        if s == ".parquet":
            return pd.read_parquet(path, engine=_parquet_engine())
        if s == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported extension: {path.suffix}")

    # auto-detect
    for cand in _candidates(path):
        if cand.is_file():
            return read_function(cand)
    print(FileNotFoundError(f"No file found for {path} "
                            "(tried .joblib/.p/.parquet/.csv)"))

# -------------------------------------------------------------------
#  HAUT NIVEAU : arborescence dict / disque
# -------------------------------------------------------------------

def write(storing, keys, values, write_function=write_function):
    """
    Write data into a nested dictionary or directory tree.

    Parameters
    ----------
    storing : dict | str
        Target: nested dict (in-memory) or root directory path (on disk).
    keys : list[str]
        Hierarchical keys or subdirectories ending with a filename.
    values : Any
        Value to store or write.
    write_function : callable
        Function handling the actual write operation to disk:
        e.g., write_function(values, Path(filepath)).
    """
    if isinstance(storing, dict):
        sub = storing
        for k in keys[:-1]:
            sub = sub.setdefault(k, {})
        sub[keys[-1]] = values
    elif isinstance(storing, str):
        full_path = os.path.join(storing, *keys[:-1])
        os.makedirs(full_path, exist_ok=True, mode=0o777)
        filepath = Path(os.path.join(full_path, keys[-1]))
        write_function(values, filepath)
    else:
        raise TypeError("storing must be dict or str path")

def read(storing, keys, read_function=read_function):
    """
    Read data from a nested dictionary or file system.

    Parameters
    ----------
    storing : dict | str
        Source: nested dict or root directory path.
    keys : list[str]
        Hierarchical keys or subdirectories leading to the target.
    read_function : callable
        Function handling the actual file read:
        e.g., read_function(Path(filepath)) -> object.

    Returns
    -------
    Any
        Retrieved value or file content.
    """
    if isinstance(storing, dict):
        sub = storing
        for k in keys:
            sub = sub.get(k)
            if sub is None:
                return None
        return sub
    elif isinstance(storing, str):
        filepath = Path(os.path.join(storing, *keys))
        return read_function(filepath)
    else:
        raise TypeError("storing must be dict or str path")


#def dict_to_folders(storing_dict):
#    return ()
