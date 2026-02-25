"""
diffusion_reducer.py
====================

Reducer layer for the modular time-series diffusion framework.

This module implements post-run reduction operators that transform a
mask×seed grid of TrajectoryResult into aggregated statistics
(mean/variance, hierarchical decompositions, anomaly maps, etc.).

Design principles
-----------------
- Pure functions: reducers do not depend on the diffusion model instance.
- Canonical input structure:
      groups : List[List[TrajectoryResult]]  (mask × seed grid)
- Flat views are derived locally when needed.
- All external information (schedule tensors, reference signals, sweep metadata)
  is provided via a structured `ctx` dictionary injected by `run()`.

Contract
--------
Each reducer follows the unified signature:

    reduce_<name>(spec: Dict[str, Any], inputs: ReducerInputs) -> Dict[str, tf.Tensor]

where:
    - spec: declarative reducer configuration
    - inputs.groups: canonical mask×seed grid
    - inputs.ctx: structured context (e.g. ctx["schedule"], ctx["data"], ctx["sweep"])

Reducers must:
    - Access trajectory data exclusively via TrajectoryResult.resolve(path)
    - Never access model internals directly
    - Return a dictionary of tensors (no side effects)

Registry
--------
REDUCER_REGISTRY maps a reducer "type" (string) to its implementation.
The dispatcher in `run()` selects and executes reducers using this registry.

Notes
-----
- Hierarchical reducers (two-stage) implement the law of total variance.
- Schedule-aware reducers (e.g., x0 variance approximation) rely on
  ctx["schedule"] tensors injected by the runner.
- Anomaly reducers rely on ctx["data"]["x_ref"] and masking artifacts.

"""


from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Mapping, Callable, Union,  List,Literal, Sequence

import tensorflow as tf
from uqmodels.modelization.TF_estimator.ts_diffusion.diffusion_dataclass import TrajectoryResult

@dataclass(frozen=True)
class ReducerInputs:
    groups: List[List[TrajectoryResult]]   # canonical M×S grid (mask × seed)
    ctx: Dict[str, Any]                    # external context (x_ref, ids, ...)


ReducerFn = Callable[[Mapping[str, Any], ReducerInputs], Dict[str, tf.Tensor]]

def aggregate_masks_decompose(
    group_stats: List[Dict[str, tf.Tensor]],
    *,
    ddof: int = 0,
) -> Dict[str, tf.Tensor]:
    """
    Aggregate per-mask group statistics with a two-level decomposition.

    Inputs
    ------
    group_stats:
        List of dicts with keys:
            - "mean": tf.Tensor (...)
            - "var" : tf.Tensor (...)

    Returns
    -------
    {
        "mean":      E_mask[mean_m],
        "var_intra": E_mask[var_m],
        "var_inter": Var_mask(mean_m),
        "var_total": var_intra + var_inter
    }

    Notes
    -----
    This implements the law of total variance for a discrete mask variable.
    ddof is applied to var_inter (variance across masks).
    """
    if len(group_stats) == 0:
        raise ValueError("aggregate_masks_decompose requires non-empty group_stats.")

    means = tf.stack([tf.cast(d["mean"], tf.float32) for d in group_stats], axis=0)  # (M,...)
    vars_ = tf.stack([tf.cast(d["var"], tf.float32) for d in group_stats], axis=0)   # (M,...)

    mean = tf.reduce_mean(means, axis=0)
    var_intra = tf.reduce_mean(vars_, axis=0)
    var_inter = tf.math.reduce_variance(means, axis=0)

    if ddof != 0:
        m = tf.cast(tf.shape(means)[0], tf.float32)
        var_inter = var_inter * (m / tf.maximum(m - float(ddof), 1.0))

    var_total = var_intra + var_inter
    return {
        "mean": mean,
        "var_intra": var_intra,
        "var_inter": var_inter,
        "var_total": var_total,
    }

# ==================================================================
# 4) Reducer Helper Anomalie
# ==================================================================

def temporal_nearest_observed_distance(mask: tf.Tensor) -> tf.Tensor:
    """
    Distance (in timesteps) to the nearest observed point along time.

    Parameters
    ----------
    mask : tf.Tensor
        Observation mask (B,T,C) or (1,T,C). 1=observed, 0=holdout.

    Returns
    -------
    tf.Tensor
        Distance map (B,T,C), float32.
    """
    mask = tf.cast(mask, tf.float32)
    B = tf.shape(mask)[0]
    T = tf.shape(mask)[1]

    inf = tf.cast(T + 1, tf.float32)

    # -------------------------
    # Left-to-right distances
    # -------------------------
    init = tf.where(mask[:, 0, :] > 0.0, 0.0, inf)  # (B,C)

    def scan_lr(prev, m_t):
        # prev: (B,C), m_t: (B,C)
        d = tf.where(m_t > 0.0, 0.0, prev + 1.0)
        return tf.minimum(d, inf)

    elems_lr = tf.transpose(mask[:, 1:, :], [1, 0, 2])  # (T-1,B,C)
    lr_rest = tf.scan(scan_lr, elems_lr, initializer=init)  # (T-1,B,C)
    lr = tf.concat([init[None, ...], lr_rest], axis=0)  # (T,B,C)
    lr = tf.transpose(lr, [1, 0, 2])  # (B,T,C)

    # -------------------------
    # Right-to-left distances
    # -------------------------
    mask_rev = tf.reverse(mask, axis=[1])  # (B,T,C)
    init_r = tf.where(mask_rev[:, 0, :] > 0.0, 0.0, inf)  # (B,C)

    elems_rr = tf.transpose(mask_rev[:, 1:, :], [1, 0, 2])  # (T-1,B,C)
    rr_rest = tf.scan(scan_lr, elems_rr, initializer=init_r)  # (T-1,B,C)
    rr = tf.concat([init_r[None, ...], rr_rest], axis=0)  # (T,B,C)
    rr = tf.transpose(rr, [1, 0, 2])  # (B,T,C)
    rr = tf.reverse(rr, axis=[1])  # back to original time
    return tf.minimum(lr, rr)

def distance_weight(d: tf.Tensor, *, mode: str, tau: float) -> tf.Tensor:
    """
    Map distances to weights in [0,1].

    mode: "exp" or "inv"
    """
    d = tf.cast(d, tf.float32)
    tau_t = tf.cast(tau, tf.float32)
    mode = str(mode).lower().strip()

    if mode == "exp":
        return tf.exp(-d / (tau_t + 1e-12))
    if mode == "inv":
        return 1.0 / (1.0 + d / (tau_t + 1e-12))
    raise ValueError("weight_mode must be 'exp' or 'inv'.")

def aggregation_weight_map(mask: tf.Tensor, cfg: Optional[Dict[str, Any]]) -> tf.Tensor:
    """
    Compute per-point aggregation weights for a given mask.

    Parameters
    ----------
    mask:
        Mask tensor broadcastable to (B,T,C), typically in {0,1} or [0,1].
    cfg:
        Weight configuration dict. If None or disabled, returns ones.

    Returns
    -------
    tf.Tensor
        Weight map with same shape as mask (float32).
    """
    mask_f = tf.cast(mask, tf.float32)

    if not cfg or not bool(cfg.get("enabled", False)):
        return tf.ones_like(mask_f)

    kind = str(cfg.get("kind", "temporal_nearest_observed")).lower().strip()
    if kind != "temporal_nearest_observed":
        raise ValueError(
            f"Unknown weight kind: {kind!r}. Supported: 'temporal_nearest_observed'."
        )

    mode = str(cfg.get("mode", "exp")).lower().strip()
    tau = float(cfg.get("tau", 5.0))

    d = temporal_nearest_observed_distance(mask_f)
    return distance_weight(d, mode=mode, tau=tau)


def residual_map(x_hat: tf.Tensor, x_ref: tf.Tensor, kind: str) -> tf.Tensor:
    """
    Pointwise residual map.

    Parameters
    ----------
    x_hat, x_ref:
        Tensors broadcastable to the same shape (typically (B,T,C)).
    kind:
        "abs" | "sq"

    Returns
    -------
    tf.Tensor
        Residual map (float32).
    """
    x_hat = tf.cast(x_hat, tf.float32)
    x_ref = tf.cast(x_ref, tf.float32)

    kind = str(kind).lower().strip()
    if kind == "abs":
        return tf.abs(x_hat - x_ref)
    if kind == "sq":
        return tf.square(x_hat - x_ref)
    raise ValueError(f"residual_kind must be 'abs' or 'sq', got {kind!r}.")

def reduce_stack_mean_var(spec: Dict[str, Any], inputs: ReducerInputs) -> Dict[str, tf.Tensor]:
    field = str(spec["field"])
    exp = bool(spec.get("exp", False))
    ddof = int(spec.get("ddof", 0))
    out_prefix = spec.get("out_prefix", None)

    xs = tf.stack([r.resolve(field) for r in inputs.runs], axis=0)
    if exp:
        xs = tf.exp(xs)

    mean = tf.reduce_mean(xs, axis=0)
    var = tf.math.reduce_variance(xs, axis=0)

    if ddof != 0:
        n = tf.cast(tf.shape(xs)[0], tf.float32)
        var = var * (n / tf.maximum(n - float(ddof), 1.0))

    p = out_prefix if out_prefix else field.replace(".", "_")
    return {f"{p}_mean": mean, f"{p}_var": var}

def reduce_group_mean_var(
    spec: Dict[str, Any],
    inputs: ReducerInputs,
) -> Dict[str, tf.Tensor]:
    """
    Reduce each mask-group across seeds and return per-group mean/var.

    Spec
    ----
    {
        "type": "group_mean_var",
        "name": "...",
        "field": "x_hat" | "collect.xxx",
        "exp": bool (optional),
        "ddof": int (optional),
    }

    Behavior
    --------
    For each group (same mask), compute mean/variance across seeds.
    Returns stacked tensors of shape (M, ...), where M = number of groups.
    """
    field = str(spec["field"])
    exp = bool(spec.get("exp", False))
    ddof = int(spec.get("ddof", 0))

    groups = inputs.groups
    if len(groups) == 0:
        raise ValueError("group_mean_var requires non-empty groups.")

    means = []
    vars_ = []

    for group in groups:
        if len(group) == 0:
            raise ValueError("group_mean_var found an empty group.")

        xs = tf.stack([r.resolve(field) for r in group], axis=0)  # (K,...)

        if exp:
            xs = tf.exp(xs)

        mean = tf.reduce_mean(xs, axis=0)
        var = tf.math.reduce_variance(xs, axis=0)

        if ddof != 0:
            n = tf.cast(tf.shape(xs)[0], tf.float32)
            var = var * (n / tf.maximum(n - float(ddof), 1.0))

        means.append(mean)
        vars_.append(var)

    # Stack across groups (M,...)
    mean_stack = tf.stack(means, axis=0)
    var_stack = tf.stack(vars_, axis=0)

    return {
        "mean": mean_stack,
        "var": var_stack,
    }

def reduce_x0_var_approx(
    spec: Dict[str, Any],
    inputs: ReducerInputs,
) -> Dict[str, tf.Tensor]:
    """
    Approximate Var(x0) by propagating eps-space variance through the DDPM x0 formula.

    Spec
    ----
    {
        "type": "x0_var_approx",
        "name": "...",
        "field_log_var": "collect.eps_log_var" (optional),
        "field_t": "collect.t" (optional),
        "ddof": 0 (optional),
        "out_prefix": "x0_var_approx" (optional),
        "eps": 1e-8 (optional)
    }

    Required ctx
    ------------
    ctx["alpha_bars"]: 1D tensor of length T with alpha_bar[t].

    Returns
    -------
    { "<out_prefix>_mean": (B,T,C), "<out_prefix>_var": (B,T,C) }
    """
    runs = [r for g in inputs.groups for r in g]
    if len(runs) == 0:
        raise ValueError("x0_var_approx requires at least one run.")

    field_log_var = str(spec.get("field_log_var", "collect.eps_log_var"))
    field_t = str(spec.get("field_t", "collect.t"))
    ddof = int(spec.get("ddof", 0))
    out_prefix = str(spec.get("out_prefix", "x0_var_approx"))
    eps = float(spec.get("eps", 1e-8))

    schedule = inputs.ctx.get("schedule", None)
    if schedule is None or "alpha_bars" not in schedule:
        raise ValueError("x0_var_approx requires ctx['schedule']['alpha_bars'].")
    alpha_bars_1d = tf.cast(schedule["alpha_bars"], tf.float32)

    x0_vars = []
    for r in runs:
        t_last = r.resolve(field_t)           # expected shape (B,) or broadcastable
        log_var_eps = r.resolve(field_log_var)

        var_eps = tf.exp(tf.cast(log_var_eps, tf.float32))

        # Gather alpha_bar(t) and broadcast to (B,1,1)
        alpha_bar_t = _extract_t(alpha_bars_1d, tf.cast(t_last, tf.int32))
        one_minus_alpha_bar_t = 1.0 - alpha_bar_t

        scale2 = one_minus_alpha_bar_t / (alpha_bar_t + eps)
        x0_var = scale2 * var_eps
        x0_vars.append(x0_var)

    xs = tf.stack(x0_vars, axis=0)  # (N,B,T,C)
    mean = tf.reduce_mean(xs, axis=0)
    var = tf.math.reduce_variance(xs, axis=0)

    if ddof != 0:
        n = tf.cast(tf.shape(xs)[0], tf.float32)
        var = var * (n / tf.maximum(n - float(ddof), 1.0))

    return {f"{out_prefix}_mean": mean, f"{out_prefix}_var": var}

def reduce_two_stage(
    spec: Dict[str, Any],
    inputs: ReducerInputs,
) -> Dict[str, tf.Tensor]:
    """
    Two-stage (hierarchical) reducer over a mask×seed grid.

    Spec
    ----
    {
        "type": "two_stage",
        "name": "...",
        "field": "x_hat" | "collect.xxx",
        "exp": bool (optional),
        "ddof_intra": int (optional),
        "ddof_inter": int (optional),
        "out_prefix": str (optional),
    }

    Behavior
    --------
    Stage 1 (intra): for each mask-group m, reduce across seeds:
        mean_m, var_m
    Stage 2 (inter): aggregate across masks:
        var_intra = E_m[var_m]
        var_inter = Var_m(mean_m)
        var_total = var_intra + var_inter
    """
    groups = inputs.groups
    if len(groups) == 0:
        raise ValueError("two_stage requires non-empty groups.")
    if any(len(g) == 0 for g in groups):
        raise ValueError("two_stage found an empty group (mask with zero runs).")

    field = str(spec["field"])
    exp = bool(spec.get("exp", False))
    ddof_intra = int(spec.get("ddof_intra", 0))
    ddof_inter = int(spec.get("ddof_inter", 0))
    out_prefix = spec.get("out_prefix", None)

    # ---- Stage 1: per-group stats over seeds ----
    group_stats: List[Dict[str, tf.Tensor]] = []
    for g in groups:
        xs = tf.stack([gk.resolve(field) for gk in g], axis=0)  # (S,...)
        if exp:
            xs = tf.exp(xs)

        mean_m = tf.reduce_mean(xs, axis=0)
        var_m = tf.math.reduce_variance(xs, axis=0)

        if ddof_intra != 0:
            n = tf.cast(tf.shape(xs)[0], tf.float32)
            var_m = var_m * (n / tf.maximum(n - float(ddof_intra), 1.0))

        group_stats.append({"mean": mean_m, "var": var_m})

    # ---- Stage 2: decomposition across masks ----
    # Expected signature:
    #   aggregate_masks_decompose(group_stats, ddof=ddof_inter) -> dict with
    #   {"mean", "var_intra", "var_inter", "var_total"}
    agg = aggregate_masks_decompose(group_stats, ddof=ddof_inter)

    p = str(out_prefix) if out_prefix else field.replace(".", "_")
    return {
        f"{p}_mean": agg["mean"],
        f"{p}_var_intra": agg["var_intra"],
        f"{p}_var_inter": agg["var_inter"],
        f"{p}_var_total": agg["var_total"],
    }

def reduce_anomaly_score(
    spec: Dict[str, Any],
    inputs: ReducerInputs,
) -> Dict[str, tf.Tensor]:
    """
    Anomaly score map based on reconstruction residuals under conditional masking.

    Spec
    ----
    {
        "type": "anomaly_score",
        "name": "...",
        "field_pred": "x_hat" | "collect.x0_pred" | "collect.<key>" (optional),
        "residual_kind": "abs" | "sq" (optional),
        "agg": "mean" | "max" (optional),
        "weight_cfg": dict | None (optional),
        "seed_reduce": "mean" | "none" (optional),
        "out_prefix": "anom" (optional),
    }

    Required ctx
    ------------
    ctx["data"]["x_ref"]: reference signal (B,T,C)

    Returns
    -------
    { "<out_prefix>_score": tf.Tensor (B,T,C) }
    """
    groups = inputs.groups
    if len(groups) == 0:
        raise ValueError("anomaly_score requires non-empty groups.")

    data = inputs.ctx.get("data", None)
    if data is None or "x_ref" not in data:
        raise ValueError("anomaly_score requires ctx['data']['x_ref'].")

    x_ref = tf.cast(data["x_ref"], tf.float32)

    field_pred = str(spec.get("field_pred", "x_hat"))
    residual_kind = str(spec.get("residual_kind", "abs")).lower().strip()
    agg = str(spec.get("agg", "mean")).lower().strip()
    weight_cfg = spec.get("weight_cfg", None)
    seed_reduce = str(spec.get("seed_reduce", "mean")).lower().strip()
    out_prefix = str(spec.get("out_prefix", "anom"))

    if agg not in {"mean", "max"}:
        raise ValueError("anomaly_score: agg must be 'mean' or 'max'.")
    if seed_reduce not in {"mean", "none"}:
        raise ValueError("anomaly_score: seed_reduce must be 'mean' or 'none'.")
    if residual_kind not in {"abs", "sq"}:
        raise ValueError("anomaly_score: residual_kind must be 'abs' or 'sq'.")

    per_mask = []

    for group in groups:
        if len(group) == 0:
            raise ValueError("anomaly_score found an empty group.")

        # (A) retrieve predictions for each seed
        preds = []
        for r in group:
            preds.append(tf.cast(r.resolve(field_pred), tf.float32))
        preds = tf.stack(preds, axis=0)  # (S,B,T,C)

        # (B) reduce across seeds (within same mask)
        if seed_reduce == "mean":
            pred = tf.reduce_mean(preds, axis=0)  # (B,T,C)
        else:
            pred = preds[0]  # deterministic pick

        # (C) mask (shared across seeds)
        mask = group[0].mask
        if mask is None:
            raise ValueError("anomaly_score requires TrajectoryResult.mask (mask sweep).")
        mask = tf.cast(mask, tf.float32)
        holdout = 1.0 - mask

        # (D) optional weights based on mask geometry
        weight = aggregation_weight_map(mask, weight_cfg)  # (B,T,C) or ones
        weight = tf.cast(weight, tf.float32)

        # (E) residual and score only on holdout
        resid = residual_map(pred, x_ref, residual_kind)   # (B,T,C)
        score = resid * holdout * weight

        if agg == "mean":
            denom = holdout * weight
            per_mask.append((score, denom))
        else:
            per_mask.append(score)

    # ---- aggregate across masks ----
    if agg == "mean":
        num = tf.add_n([sd[0] for sd in per_mask])
        den = tf.add_n([sd[1] for sd in per_mask])
        score_final = num / (den + 1e-12)
    else:
        stacked = tf.stack(per_mask, axis=0)  # (M,B,T,C)
        score_final = tf.reduce_max(stacked, axis=0)

    return {f"{out_prefix}_score": score_final}

REDUCER_REGISTRY: Dict[str, ReducerFn] = {
"stack_mean_var": reduce_stack_mean_var,
"group_mean_var": reduce_group_mean_var,
"x0_var_approx" : reduce_x0_var_approx,
"two_stage": reduce_two_stage,
"anomaly_score": reduce_anomaly_score}