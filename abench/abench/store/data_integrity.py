"""
High-level architecture
=======================

Overview
--------
This module implements a modular CSV integrity audit designed for ML benchmark
pre-checks. Given an index DataFrame listing CSV file paths, it produces:
  (1) per-file profiles,
  (2) a global reference profile,
  (3) per-file comparisons against the reference,
  (4) actionable diagnostics (macro OK/KO + per-column OK/KO).

Open source like tools that may be use / replace this layers :

- great-expectations : great-expectations 1.11.3 https://github.com/deepchecks/deepchecks
- deepchecks : https://github.com/deepchecks/deepchecks
- whylogs : https://github.com/whylabs/whylogs
- TensorFlow Data Validation : https://www.tensorflow.org/tfx/guide/tfdv?hl=fr
- panderas : https://pandera.readthedocs.io/en/stable/

Core building blocks
--------------------

1) AuditContext
   `AuditContext` is the global configuration object. It centralizes:
   - computation parameters (e.g., histogram bins, top-K categories, CSV read kwargs),
   - decision parameters (e.g., alpha, divergence thresholds, tolerance factors),
   - protocol policies (e.g., how to define "required" columns, whether extra columns are allowed).

   Key idea: criteria should not hardcode constants. All tuning knobs live in `ctx`,
   making the pipeline configurable without changing the criteria code.

2) Criterion (plugin interface)
   A `Criterion` encapsulates one integrity check (columns, dtypes, missingness,
   distributions, ...). Each criterion follows the same 4-step contract:

   - profile(df, ctx) -> dict
       Build the per-file (local) profile for this criterion
       (e.g., dtype map, NaN ratios, histograms, category frequencies).

   - aggregate(list_of_profiles, ctx) -> dict
       Combine all local profiles into a criterion-specific global reference
       (e.g., majority dtypes, robust NaN bounds, reference distributions).

   - compare(local_profile, reference_profile, ctx) -> dict
       Compare a file against the reference and return raw discrepancy measures
       and/or statistical test outputs (e.g., missing columns, dtype mismatches,
       p-values, JS divergence).

   - diagnose(compare_out, ctx) -> dict
       Convert raw measures into an actionable diagnostic:
       - macro status: OK/KO
       - micro status: per-column OK/KO (when relevant)
       - details: scores, p-values, bounds used for decision, etc.

   Key idea: each criterion is self-contained and composable. Adding or removing
   a check is as simple as adding/removing a plugin instance from the auditor.

3) CSVIntegrityAuditor (runner/orchestrator)
   `CSVIntegrityAuditor` orchestrates the full pipeline in three passes:

   Pass A — Profiling:
     For each CSV, load it and run `criterion.profile(...)` for all criteria.

   Pass B — Reference building:
     For each criterion, run `criterion.aggregate(...)` over all collected profiles.

   Pass C — Comparison + Diagnosis:
     For each file and criterion, run `compare(...)` then `diagnose(...)`,
     and compute a per-file `global_status` (KO if at least one criterion is KO).

   Key idea: the runner is criterion-agnostic. It only sequences the standardized
   steps and assembles the structured report (profile → reference → compare → diagnose).

Benefits
--------
- Modularity: criteria are plug-ins.
- Traceability: the report keeps intermediate artifacts for inspection.
- Configurability: behavior is driven by `AuditContext`.
- Evolvability: reference strategies (golden set, robust aggregation, clustering)
  can be introduced by changing only criterion-level `aggregate/compare` logic.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import math
import numpy as np
import pandas as pd
import scipy.stats as st


# ----------------------------- utilities ---------------------------------- #

def _safe_read_csv(path: str, *, read_kwargs: Optional[dict] = None) -> pd.DataFrame:
    """Read a CSV file with user-provided kwargs; raise enriched errors."""
    read_kwargs = read_kwargs or {}
    try:
        return pd.read_csv(path, **read_kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {path}. Error: {type(e).__name__}: {e}") from e


def _normalize_dtype(dtype: Any) -> str:
    """Normalize pandas dtype into a stable label."""
    dt = str(dtype)
    if "datetime" in dt:
        return "datetime"
    if dt.startswith(("int", "Int")):
        return "int"
    if dt.startswith(("float", "Float")):
        return "float"
    if dt in ("bool", "boolean"):
        return "bool"
    if dt in ("string",):
        return "string"
    return "object"


def _logical_kind(norm_dtype: str) -> str:
    """Map normalized dtype to logical kind: numeric/categorical/datetime."""
    if norm_dtype in ("int", "float"):
        return "numeric"
    if norm_dtype == "datetime":
        return "datetime"
    return "categorical"


def _nan_ratio(s: pd.Series) -> float:
    """Return NaN ratio in [0, 1]."""
    if len(s) == 0:
        return float("nan")
    return float(s.isna().mean())


def _hist_numeric(values: np.ndarray, n_bins: int) -> Dict[str, Any]:
    """Histogram summary for numeric values (excluding NaNs)."""
    if values.size == 0:
        return {"kind": "numeric", "n": 0, "bins": None, "counts": None}
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        uniq = float(vmin) if np.isfinite(vmin) else None
        return {"kind": "numeric", "n": int(values.size), "degenerate": True, "value": uniq}
    counts, edges = np.histogram(values, bins=n_bins, range=(vmin, vmax))
    return {
        "kind": "numeric",
        "n": int(values.size),
        "min": vmin,
        "max": vmax,
        "bins": edges.tolist(),
        "counts": counts.astype(int).tolist(),
    }


def _freq_discrete(values: pd.Series, top_k: int = 50) -> Dict[str, Any]:
    """Frequency summary for categorical values (excluding NaNs), top_k + __OTHER__."""
    if len(values) == 0:
        return {"kind": "categorical", "n": 0, "freq": {}}
    vc = values.value_counts(dropna=True)
    if len(vc) <= top_k:
        return {"kind": "categorical", "n": int(vc.sum()), "freq": {str(k): int(v) for k, v in vc.items()}}
    head = vc.iloc[:top_k]
    tail_sum = int(vc.iloc[top_k:].sum())
    freq = {str(k): int(v) for k, v in head.items()}
    freq["__OTHER__"] = tail_sum
    return {"kind": "categorical", "n": int(vc.sum()), "freq": freq, "top_k": top_k}


def _ztest_proportions(p1: float, n1: int, p2: float, n2: int) -> Dict[str, Any]:
    """Two-proportion z-test (approx), returns z and pvalue (SciPy required)."""
    if n1 <= 0 or n2 <= 0:
        return {"z": None, "pvalue": None, "note": "invalid sample sizes"}

    x1 = p1 * n1
    x2 = p2 * n2
    p_pool = (x1 + x2) / (n1 + n2)

    denom = math.sqrt(max(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2), 1e-12))
    z = (p1 - p2) / denom
    pval = float(2 * (1 - st.norm.cdf(abs(z))))
    return {"z": float(z), "pvalue": pval, "p1": float(p1), "p2": float(p2), "n1": int(n1), "n2": int(n2)}


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen–Shannon divergence for discrete distributions (base e)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), eps)
    q = q / max(q.sum(), eps)
    m = 0.5 * (p + q)

    def _kl(a, b):
        a = np.clip(a, eps, 1.0)
        b = np.clip(b, eps, 1.0)
        return float(np.sum(a * np.log(a / b)))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


# ----------------------------- criterion API ------------------------------ #

class Criterion:
    """Plugin interface for a single integrity criterion."""
    name: str

    def profile(self, df: pd.DataFrame, ctx: "AuditContext") -> Dict[str, Any]:
        raise NotImplementedError

    def aggregate(self, profiles: List[Dict[str, Any]], ctx: "AuditContext") -> Dict[str, Any]:
        raise NotImplementedError

    def compare(self, local_profile: Dict[str, Any], reference_profile: Dict[str, Any], ctx: "AuditContext") -> Dict[str, Any]:
        raise NotImplementedError

    def diagnose(self, compare_out: Dict[str, Any], ctx: "AuditContext") -> Dict[str, Any]:
        raise NotImplementedError


# ----------------------------- context/config ----------------------------- #

@dataclass
class AuditContext:
    """Global configuration and thresholds for the audit."""
    n_bins: int = 10
    top_k_categories: int = 50
    alpha: float = 0.01
    max_js_div: float = 0.1
    allow_extra_columns: bool = True
    enforce_column_order: bool = False
    # IMPORTANT: columns become "required" if present in at least this fraction of files
    required_presence_ratio: float = 0.75
    missingness_iqr_k: float = 3.0  # tolerance factor
    read_csv_kwargs: Dict[str, Any] = field(default_factory=dict)


# ----------------------------- criteria ----------------------------------- #

class ColumnSchemaCriterion(Criterion):
    name = "columns"

    def profile(self, df: pd.DataFrame, ctx: AuditContext) -> Dict[str, Any]:
        cols = list(df.columns)
        return {"columns": cols, "n_cols": int(len(cols))}

    def aggregate(self, profiles: List[Dict[str, Any]], ctx: AuditContext) -> Dict[str, Any]:
        all_cols = [p["columns"] for p in profiles]
        n = len(all_cols)

        # Count presence per column across files
        presence: Dict[str, int] = {}
        for cols in all_cols:
            for c in set(cols):
                presence[c] = presence.get(c, 0) + 1

        threshold = int(math.ceil(ctx.required_presence_ratio * max(n, 1)))
        required = sorted([c for c, k in presence.items() if k >= threshold])
        known = sorted(list(presence.keys()))
        ref_order = profiles[0]["columns"] if profiles else []
        return {"required_columns": required, "known_columns": known, "ref_order": ref_order, "presence": presence}

    def compare(self, local_profile: Dict[str, Any], reference_profile: Dict[str, Any], ctx: AuditContext) -> Dict[str, Any]:
        local = set(local_profile["columns"])
        required = set(reference_profile["required_columns"])
        known = set(reference_profile["known_columns"])

        missing = sorted(list(required - local))
        extra = sorted(list(local - known))

        order_ok = True
        if ctx.enforce_column_order:
            ref_order = reference_profile.get("ref_order", [])
            local_cols = [c for c in local_profile["columns"] if c in required]
            ref_cols = [c for c in ref_order if c in required]
            order_ok = (local_cols == ref_cols)

        return {"missing_required": missing, "extra_unknown": extra, "order_ok": bool(order_ok)}

    def diagnose(self, compare_out: Dict[str, Any], ctx: AuditContext) -> Dict[str, Any]:
        missing = compare_out["missing_required"]
        extra = compare_out["extra_unknown"]
        order_ok = compare_out["order_ok"]

        status = "OK"
        if missing:
            status = "KO"
        if (not ctx.allow_extra_columns) and extra:
            status = "KO"
        if ctx.enforce_column_order and (not order_ok):
            status = "KO"

        return {"status": status, "details": compare_out}

class DTypeCriterion(Criterion):
    name = "dtypes"

    def profile(self, df: pd.DataFrame, ctx: AuditContext) -> Dict[str, Any]:
        dtype_map = {c: _normalize_dtype(df[c].dtype) for c in df.columns}
        kind_map = {c: _logical_kind(dtype_map[c]) for c in df.columns}
        return {"dtype": dtype_map, "kind": kind_map}

    def aggregate(self, profiles: List[Dict[str, Any]], ctx: AuditContext) -> Dict[str, Any]:
        votes: Dict[str, Dict[str, int]] = {}
        for p in profiles:
            for c, dt in p["dtype"].items():
                votes.setdefault(c, {})
                votes[c][dt] = votes[c].get(dt, 0) + 1

        ref_dtype = {}
        for c, v in votes.items():
            max_count = max(v.values())
            bests = [dt for dt, cnt in v.items() if cnt == max_count]
            if "float" in bests:
                ref = "float"
            elif "int" in bests:
                ref = "int"
            else:
                ref = bests[0]
            ref_dtype[c] = ref

        return {"ref_dtype": ref_dtype, "dtype_support": votes}

    def compare(self, local_profile: Dict[str, Any], reference_profile: Dict[str, Any], ctx: AuditContext) -> Dict[str, Any]:
        local = local_profile["dtype"]
        ref = reference_profile["ref_dtype"]
        per_col = {}

        for c, ref_dt in ref.items():
            loc_dt = local.get(c, None)
            ok = (loc_dt == ref_dt)
            compatible = ok or ((loc_dt in ("int", "float")) and (ref_dt in ("int", "float")))
            per_col[c] = {"local": loc_dt, "ref": ref_dt, "ok_strict": bool(ok), "ok_compatible": bool(compatible)}

        return {"per_column": per_col}

    def diagnose(self, compare_out: Dict[str, Any], ctx: AuditContext) -> Dict[str, Any]:
        per_col = compare_out["per_column"]
        micro = {c: ("OK" if v["ok_compatible"] else "KO") for c, v in per_col.items()}
        status = "OK" if all(x == "OK" for x in micro.values()) else "KO"
        return {"status": status, "per_column": micro, "details": per_col}

class MissingnessCriterion(Criterion):
    name = "missingness"

    def profile(self, df: pd.DataFrame, ctx: AuditContext) -> Dict[str, Any]:
        ratios = {c: float(df[c].isna().mean()) for c in df.columns}
        return {"nan_ratio": ratios, "n_rows": int(len(df))}

    def aggregate(self, profiles: List[Dict[str, Any]], ctx: AuditContext) -> Dict[str, Any]:
        cols = sorted({c for p in profiles for c in p["nan_ratio"].keys()})
        ref = {}

        for c in cols:
            vals = [p["nan_ratio"][c] for p in profiles if c in p["nan_ratio"] and np.isfinite(p["nan_ratio"][c])]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            q25 = float(np.percentile(arr, 25))
            q75 = float(np.percentile(arr, 75))
            iqr = float(q75 - q25)
            median = float(np.median(arr))

            low = q25 - ctx.missingness_iqr_k * iqr
            high = q75 + ctx.missingness_iqr_k * iqr
            # clamp to [0,1]
            low = float(max(0.0, low))
            high = float(min(1.0, high))

            ref[c] = {"median": median, "q25": q25, "q75": q75, "iqr": iqr, "low": low, "high": high}

        return {"ref_nan_ratio": ref}

    def compare(self, local_profile: Dict[str, Any], reference_profile: Dict[str, Any], ctx: AuditContext) -> Dict[str, Any]:
        local = local_profile["nan_ratio"]
        ref = reference_profile["ref_nan_ratio"]

        per_col = {}
        for c, r in ref.items():
            p_local = local.get(c, float("nan"))
            if not np.isfinite(p_local):
                per_col[c] = {"ok": False, "reason": "missing column or invalid ratio"}
                continue

            ok = (r["low"] <= p_local <= r["high"])
            per_col[c] = {
                "p_local": float(p_local),
                "ref_median": float(r["median"]),
                "low": float(r["low"]),
                "high": float(r["high"]),
                "ok": bool(ok),
            }

        return {"per_column": per_col}

    def diagnose(self, compare_out: Dict[str, Any], ctx: AuditContext) -> Dict[str, Any]:
        per_col = compare_out["per_column"]
        micro = {}
        for c, v in per_col.items():
            if "reason" in v:
                micro[c] = "KO"
            else:
                micro[c] = "OK" if bool(v["ok"]) else "KO"

        status = "OK" if all(x == "OK" for x in micro.values()) else "KO"
        return {"status": status, "per_column": micro, "details": per_col}

class DistributionCriterion(Criterion):
    name = "distribution"

    def profile(self, df: pd.DataFrame, ctx: AuditContext) -> Dict[str, Any]:
        out = {}
        dtypes = {c: _normalize_dtype(df[c].dtype) for c in df.columns}
        for c in df.columns:
            kind = _logical_kind(dtypes[c])
            s = df[c].dropna()

            if kind == "numeric":
                arr = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
                out[c] = _hist_numeric(arr, ctx.n_bins)
            else:
                out[c] = _freq_discrete(s.astype("object"), top_k=ctx.top_k_categories)

        return {"dist": out}

    def aggregate(self, profiles: List[Dict[str, Any]], ctx: AuditContext) -> Dict[str, Any]:
        ref = {}
        col_items: Dict[str, List[Dict[str, Any]]] = {}
        for p in profiles:
            for c, d in p["dist"].items():
                col_items.setdefault(c, []).append(d)

        for c, items in col_items.items():
            kinds = [it.get("kind") for it in items]
            kind = max(set(kinds), key=kinds.count)

            if kind == "numeric":
                mins = [it.get("min") for it in items if it.get("bins") is not None and "min" in it]
                maxs = [it.get("max") for it in items if it.get("bins") is not None and "max" in it]
                if not mins or not maxs:
                    ref[c] = {"kind": "numeric", "note": "insufficient data"}
                    continue

                gmin, gmax = float(np.min(mins)), float(np.max(maxs))
                edges = np.linspace(gmin, gmax, ctx.n_bins + 1)
                agg_counts = np.zeros(ctx.n_bins, dtype=float)

                for it in items:
                    if it.get("bins") is None or it.get("counts") is None:
                        continue
                    local_edges = np.asarray(it["bins"], dtype=float)
                    local_counts = np.asarray(it["counts"], dtype=float)
                    if local_edges.size != ctx.n_bins + 1 or local_counts.size != ctx.n_bins:
                        continue
                    mids = 0.5 * (local_edges[:-1] + local_edges[1:])
                    idx = np.clip(np.digitize(mids, edges) - 1, 0, ctx.n_bins - 1)
                    for j, k in enumerate(idx):
                        agg_counts[k] += local_counts[j]

                ref[c] = {"kind": "numeric", "bins": edges.tolist(), "counts": agg_counts.tolist(), "min": gmin, "max": gmax}
            else:
                agg = {}
                total = 0
                for it in items:
                    if it.get("kind") != "categorical":
                        continue
                    freq = it.get("freq", {})
                    for k, v in freq.items():
                        agg[k] = agg.get(k, 0) + int(v)
                        total += int(v)
                ref[c] = {"kind": "categorical", "freq": agg, "n": int(total)}

        return {"ref_dist": ref}

    def compare(self, local_profile: Dict[str, Any], reference_profile: Dict[str, Any], ctx: AuditContext) -> Dict[str, Any]:
        local = local_profile["dist"]
        ref = reference_profile["ref_dist"]

        per_col = {}
        for c, refd in ref.items():
            locd = local.get(c)
            if locd is None:
                per_col[c] = {"ok": False, "reason": "missing column"}
                continue

            if refd.get("kind") == "numeric" and locd.get("kind") == "numeric":
                if any(refd.get(k) is None for k in ("counts", "bins")) or any(locd.get(k) is None for k in ("counts", "bins")):
                    per_col[c] = {"ok": False, "reason": "insufficient histogram data"}
                    continue

                ref_counts = np.asarray(refd["counts"], dtype=float)
                ref_edges = np.asarray(refd["bins"], dtype=float)

                loc_edges = np.asarray(locd["bins"], dtype=float)
                loc_counts = np.asarray(locd["counts"], dtype=float)

                mids = 0.5 * (loc_edges[:-1] + loc_edges[1:])
                idx = np.clip(np.digitize(mids, ref_edges) - 1, 0, ctx.n_bins - 1)
                aligned = np.zeros(ctx.n_bins, dtype=float)
                for j, k in enumerate(idx):
                    aligned[k] += loc_counts[j]

                js = _js_divergence(aligned, ref_counts)
                per_col[c] = {"metric": "JS", "js_div": float(js), "ok": (js <= ctx.max_js_div)}
            else:
                ref_freq = refd.get("freq", {})
                loc_freq = locd.get("freq", {})
                keys = sorted(set(ref_freq.keys()) | set(loc_freq.keys()))
                p = np.asarray([loc_freq.get(k, 0) for k in keys], dtype=float)
                q = np.asarray([ref_freq.get(k, 0) for k in keys], dtype=float)
                js = _js_divergence(p, q)
                per_col[c] = {"metric": "JS", "js_div": float(js), "ok": (js <= ctx.max_js_div), "n_keys": int(len(keys))}

        return {"per_column": per_col}

    def diagnose(self, compare_out: Dict[str, Any], ctx: AuditContext) -> Dict[str, Any]:
        per_col = compare_out["per_column"]
        micro = {c: ("KO" if "reason" in v else ("OK" if bool(v.get("ok")) else "KO")) for c, v in per_col.items()}
        status = "OK" if all(x == "OK" for x in micro.values()) else "KO"
        return {"status": status, "per_column": micro, "details": per_col}


# ----------------------------- runner/orchestrator ------------------------ #

@dataclass
class CSVIntegrityAuditor:
    """Orchestrates CSV integrity auditing using a set of Criterion plugins."""
    criteria: List[Criterion]
    ctx: AuditContext = field(default_factory=AuditContext)

    def run(self, index_df: pd.DataFrame, *, path_col: str = "path", key_col: Optional[str] = None) -> Dict[str, Any]:
        """Run the full audit pipeline and return a structured report dict."""
        if path_col not in index_df.columns:
            raise ValueError(f"index_df must contain column '{path_col}'.")

        file_keys = []
        for i, row in index_df.iterrows():
            k = str(row[key_col]) if (key_col is not None and key_col in index_df.columns) else str(i)
            file_keys.append(k)

        report: Dict[str, Any] = {
            "meta": {
                "n_files": int(len(index_df)),
                "path_col": path_col,
                "key_col": key_col,
                "ctx": dict(self.ctx.__dict__),
                "criteria": [c.name for c in self.criteria],
            },
            "files": {},
            "reference": {"profile": {}, "meta": {}},
            "summary": {},
        }

        local_profiles: Dict[str, Dict[str, Dict[str, Any]]] = {c.name: {} for c in self.criteria}

        # Pass A: per-file profiling
        for k, (_, row) in zip(file_keys, index_df.iterrows()):
            path = str(row[path_col])
            entry = {"path": path, "profile": {}, "compare": {}, "diagnostic": {}}
            report["files"][k] = entry

            try:
                df = _safe_read_csv(path, read_kwargs=self.ctx.read_csv_kwargs)
                entry["meta"] = {"n_rows": int(len(df)), "n_cols": int(df.shape[1])}
                for crit in self.criteria:
                    p = crit.profile(df, self.ctx)
                    entry["profile"][crit.name] = p
                    local_profiles[crit.name][k] = p
            except Exception as e:
                entry["error"] = f"{type(e).__name__}: {e}"

        # Pass B: reference aggregation
        ref_profile = {}
        for crit in self.criteria:
            crit_profiles = [p for p in local_profiles[crit.name].values()]
            ref_profile[crit.name] = crit.aggregate(crit_profiles, self.ctx)
        report["reference"]["profile"] = ref_profile

        # Pass C: compare + diagnose
        for k, entry in report["files"].items():
            if "error" in entry:
                entry["diagnostic"] = {"global_status": "KO", "reason": "read_error", "error": entry["error"]}
                continue

            global_ok = True
            for crit in self.criteria:
                lp = entry["profile"].get(crit.name, {})
                rp = report["reference"]["profile"].get(crit.name, {})
                comp = crit.compare(lp, rp, self.ctx)
                diag = crit.diagnose(comp, self.ctx)

                entry["compare"][crit.name] = comp
                entry["diagnostic"][crit.name] = diag
                if diag.get("status") != "OK":
                    global_ok = False

            entry["diagnostic"]["global_status"] = "OK" if global_ok else "KO"

        report["summary"] = {
            "n_ok": int(sum(1 for f in report["files"].values() if f.get("diagnostic", {}).get("global_status") == "OK")),
            "n_ko": int(sum(1 for f in report["files"].values() if f.get("diagnostic", {}).get("global_status") == "KO")),
        }
        return report


def build_default_auditor(ctx: Optional[AuditContext] = None) -> CSVIntegrityAuditor:
    """Build an auditor with the 4 core criteria."""
    ctx = ctx or AuditContext()
    return CSVIntegrityAuditor(
        criteria=[ColumnSchemaCriterion(), DTypeCriterion(), MissingnessCriterion(), DistributionCriterion()],
        ctx=ctx,
    )


