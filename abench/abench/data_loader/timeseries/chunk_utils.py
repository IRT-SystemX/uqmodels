"""
chunk_io.py
===========

• create_chunk_df(paths, chunk_size, offset_before=0, offset_after=0)
        → DataFrame listant chaque CHUNK (cœur) + nb de lignes du fichier.

• read_chunk(row, columns=None,
             offset_before=0, offset_after=0)
        → lit un *window* de taille chunk_size + offset_before + offset_after,
          en s’adaptant automatiquement pour le 1er / dernier chunk.
"""
from __future__ import annotations
import os, itertools, gzip, bz2
import pandas as pd
import pyarrow.parquet as pq

# ────────────────────────── low-level row counters ─────────────────────────
def _csv_rows(path: str, buf: int = 1 << 20) -> int:
    opener = open
    if path.endswith(".gz"):   opener = gzip.open
    elif path.endswith(".bz2"): opener = bz2.open
    with opener(path, "rb") as f:
        return max(0, sum(b.count(b"\n") for b in iter(lambda: f.read(buf), b"")) - 1)

def _row_count(path: str) -> int:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pq.ParquetFile(path).metadata.num_rows
    if ext.startswith(".csv"):
        return _csv_rows(path)
    raise ValueError(f"Unsupported file type: {path}")

# ──────────────────────────── public helpers ───────────────────────────────
def create_chunk_df(paths: list[str],
                    chunk_size: int,
                    offset_before: int = 0,
                    offset_after: int = 0) -> pd.DataFrame:
    """
    Build an index DataFrame with one line per *core* chunk.

    Returned columns:
        path | ind_begin | ind_end | file_rows

    ind_begin / ind_end are 0-based indices **inside the file**
    and do *not* include offsets.

    Offsets are given here only to remember how many extra rows the
    training loop will request, but they are not applied yet.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    rows = []
    for p in paths:
        n = _row_count(p)
        for lo in range(0, n, chunk_size):
            hi = min(lo + chunk_size, n)
            rows.append((p, lo, hi, n))
    df = pd.DataFrame(rows,
                      columns=["path", "ind_begin", "ind_end", "file_rows"])
    df.attrs.update(offset_before=offset_before, offset_after=offset_after,
                    chunk_size=chunk_size)
    return df


def read_chunk(row: pd.Series | dict,
               columns: list[str] | None = None,
               offset_before: int = 0,
               offset_after: int = 0) -> pd.DataFrame:
    """
    Load a *window* that spans:

        size = (row["ind_end"] - row["ind_begin"])
              + offset_before + offset_after

    The window is slid if necessary so it remains inside file bounds
    and keeps the requested total size whenever possible.
    """
    path      = row["path"]
    core_lo   = int(row["ind_begin"])
    core_hi   = int(row["ind_end"])
    file_rows = int(row.get("file_rows", _row_count(path)))

    # Compute requested window
    want_len  = (core_hi - core_lo) + offset_before + offset_after
    start     = core_lo - offset_before
    end       = core_hi + offset_after

    # Clip / shift if we overflow
    if start < 0:                # beginning overflow → shift forward
        end   = min(file_rows, end - start)
        start = 0
    if end > file_rows:          # tail overflow → shift backward
        delta = end - file_rows
        start = max(0, start - delta)
        end   = file_rows

    # If the file is smaller than requested length, we return full file
    if end - start < want_len and (end - start) < file_rows:
        # 1) try extend backward
        add = min(start, want_len - (end - start))
        start -= add
        # 2) try extend forward
        add2 = min(file_rows - end, want_len - (end - start))
        end += add2

    # Final guard
    start, end = max(0, start), min(file_rows, end)
    n_rows     = end - start

    # ---------- Parquet ----------
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        pf = pq.ParquetFile(path)
        rg_sizes = [pf.metadata.row_group(i).num_rows
                    for i in range(pf.metadata.num_row_groups)]
        cs = list(itertools.accumulate([0] + rg_sizes))
        want_rg = [i for i, (lo, hi) in enumerate(zip(cs, cs[1:]))
                   if hi > start and lo < end]
        table = pf.read_row_groups(want_rg, columns=columns)
        table = table.slice(start - cs[want_rg[0]], n_rows)
        return table.to_pandas(use_threads=True)

    # ---------- CSV (plain / gz / bz2) ----------
    if ext.startswith(".csv"):
        try:
            import pyarrow.csv as pc
            tbl = pc.read_csv(
                    path,
                    read_options=pc.ReadOptions(skip_rows=start),
                    parse_options=pc.ParseOptions(newlines_in_values=False)
                  )
            if columns:
                tbl = tbl.select(columns)
            return tbl.slice(0, n_rows).to_pandas(use_threads=True)
        except ImportError:
            skip = range(1, start + 1) if start else None
            return pd.read_csv(path, usecols=columns, skiprows=skip, nrows=n_rows)

    raise ValueError(f"Unsupported file type: {path}")

def compute_chunk_info(window_size, horizon_start, prediction_number, y_step,sample_stride=1, seq_per_chunk=1024):
    if(seq_per_chunk is None):
        seq_per_chunk = 1000000000000
    past_len   = window_size
    future_len = horizon_start + (prediction_number - 1) * y_step
    offset_b   = past_len
    offset_a   = future_len
    chunk_size = (seq_per_chunk - 1) * sample_stride + 1
    total_rows = offset_b + chunk_size + offset_a
    return dict(chunk_size=chunk_size,
                offset_before=offset_b,
                offset_after=offset_a,
                rows_loaded_per_call=total_rows)