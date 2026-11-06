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
import tempfile
import warnings
from typing import Literal

import os, pickle, joblib
import pandas as pd
from pathlib import Path

# -------------------------------------------------------------------
#  BAS NIVEAU : write / read d'un seul fichier
# -------------------------------------------------------------------

_FORMATS = {
    ".joblib": "joblib",
    ".p": "pickle",
    ".pickle": "pickle",
    ".parquet": "parquet",
    ".csv": "csv"}

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

def _choose_format(Path: Path | None) -> str:
    """Resolve the storage format from an explicit file extension."""

    if not Path.suffix:
        raise ValueError(f"File extension is required for disk storage {str(Path)}")

    suffix = Path.suffix.lower()

    if suffix not in _FORMATS:
        raise ValueError(f"Unsupported file extension: {str(Path)}")

    return _FORMATS[suffix]

def _candidates(base: Path):
    """Try these when no suffix is provided."""
    return [
        base.with_suffix(".joblib"),
        base.with_suffix(".p"),
        base.with_suffix(".pickle"),
        base.with_suffix(".parquet"),
        base.with_suffix(".csv"),
    ]


def _serialize(value, path: Path, kind: str) -> None:
    """Serialize an object using the selected storage format."""

    if kind == "joblib":
        joblib.dump(value, path)

    elif kind == "pickle":
        with path.open("wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)

    elif kind in ("csv", "parquet"):
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"{kind} requires a pandas DataFrame")

        if not value.columns.is_unique:
            raise ValueError("Cannot serialize duplicate column names")

        if kind == "csv":
            value.to_csv(path, index=False, lineterminator="\n")

        else:
            value.to_parquet(
                path,
                engine=_parquet_engine() or "auto",
                index=False,
            )

    else:
        raise ValueError(f"Unsupported serialization format: {kind}")

# ------------------------ Parquet Validation ----------------------#
            
def _validate_written(value, path: Path, kind: str, *,  validation: str = "schema") -> None:
    """Validate serialized file structure and dataframe dimensions."""

    if not path.is_file() or path.stat().st_size == 0:
        raise IOError(f"Empty or missing output file: {path}")

    if kind not in ("csv", "parquet"):
        return

    expected_rows = len(value)
    expected_cols = [str(c) for c in value.columns]

    if kind == "csv":
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            columns = next(reader, [])
            rows = 0

            for row in reader:
                if len(row) != len(columns):
                    raise IOError("Invalid CSV row width")
                rows += 1

    elif kind == "parquet":
        if validation == "full":
            verified = pd.read_parquet(
                path, engine=_parquet_engine() or "auto"
            )
            rows = len(verified)
            columns = list(verified.columns)

        elif _parquet_engine() == "pyarrow":
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(path)
            rows = parquet.metadata.num_rows
            columns = parquet.schema_arrow.names

        else:
            verified = pd.read_parquet(
                path, engine=_parquet_engine() or "auto"
            )
            rows = len(verified)
            columns = list(verified.columns)

    if rows != expected_rows:
        raise IOError(
            f"Row count mismatch: {rows} != {expected_rows}"
        )

    if columns != expected_cols:
        raise IOError(
            f"Column schema mismatch: {columns} != {expected_cols}"
        )

# ------------------------- write function ------------------------ #

def write_function(
    value,
    filename,
    *,
    validation: Literal["none", "schema", "full"] = "schema",
) -> None:
    """
    Atomically serialize an object to disk.

    The target is replaced only after successful serialization
    and optional validation.
    """
    if validation not in {"none", "schema", "full"}:
        raise ValueError(f"Invalid validation mode: {validation}")

    path = Path(filename)
    kind = _choose_format(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    # Temporary file in the same filesystem
    fd, tmp = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=path.suffix,
        dir=path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp)

    try:
        _serialize(value, tmp_path, kind)

        if validation != "none":
            _validate_written(
                value,
                tmp_path,
                kind,
                validation=validation,
            )

        # Atomic commit
        os.replace(tmp_path, path)

    except Exception as exc:
        raise RuntimeError(
            f"Storage write failed: {path}\n"
            f"Format: {kind} | Validation: {validation}\n"
            f"Cause: {type(exc).__name__}: {exc}"
        ) from exc

    finally:
        tmp_path.unlink(missing_ok=True)

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

def _check_overwrite(
    exists: bool,
    target: str,
    policy: Literal["warn", "raise", "allow"],
) -> None:
    """Apply the overwrite policy to an existing storage entry."""

    if policy not in {"warn", "raise", "allow"}:
        raise ValueError(f"Invalid overwrite policy: {policy}")

    if not exists:
        return

    if policy == "raise":
        raise FileExistsError(f"Storage entry already exists: {target}")

    if policy == "warn":
        warnings.warn(
            f"Overwriting existing storage entry: {target}",
            RuntimeWarning,
            stacklevel=3,
        )

def write(
    storing,
    keys,
    values,
    write_function=write_function,
    *,
    overwrite: Literal["warn", "raise", "allow"] = "warn",
):
    """
    Write data to a nested dictionary or filesystem.

    Overwrite policies:
        warn  : Warn before replacing existing data (default).
        raise : Prevent replacement.
        allow : Replace silently.
    """
    if not keys:
        raise ValueError("keys cannot be empty")

    if overwrite not in {"warn", "raise", "allow"}:
        raise ValueError(f"Invalid overwrite policy: {overwrite}")

    if isinstance(storing, dict):
        sub = storing

        for key in keys[:-1]:
            sub = sub.setdefault(key, {})
            if not isinstance(sub, dict):
                raise TypeError(
                    f"Intermediate storage entry {key!r} is not a dict"
                )

        key = keys[-1]
        _check_overwrite(
            key in sub,
            str(keys),
            overwrite,
        )
        sub[key] = values

    elif isinstance(storing, (str, os.PathLike)):
        filepath = Path(storing).joinpath(*keys)

        # Resolve the actual output extension before checking existence
        kind = _choose_format(filepath)
        target = (
            filepath if filepath.suffix
            else filepath.with_suffix(kind)
        )

        _check_overwrite(
            target.exists(),
            str(target),
            overwrite,
        )

        target.parent.mkdir(parents=True, exist_ok=True)
        write_function(values, target)

    else:
        raise TypeError("storing must be dict or filesystem path")

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
