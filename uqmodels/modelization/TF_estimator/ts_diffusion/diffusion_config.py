"""
diffusion_config.py
===================

Declarative configuration presets for the time-series diffusion framework.

This module provides convenience builders that return RunConfig instances for
common usage patterns (predict, multi-sampling, multi-masking, combined sweeps)
and attaches reducer specifications consumed by BaseDiffusionModel.run(cfg).

Notes
-----
- Reducers are declarative specs and are executed by the runner via a registry
  (see diffusion_reducer.py). Reducers are model-agnostic and operate on the
  canonical mask×seed grid of TrajectoryResult.
- Some reducers require runner-injected context, e.g.:
    * x0_var_approx expects ctx["schedule"]["alpha_bars"].
    * anomaly_score expects ctx["data"]["x_ref"].
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from uqmodels.modelization.TF_estimator.ts_diffusion.diffusion_dataclass import (RunConfig,CollectSpec,SweepSpec)


def get_preset_config(
    name: str,
    *,
    num_steps: int = 30,
    projection: str = "hard",
    seed: Optional[int] = None,
    n_seeds: int = 20,
    n_masks: int = 20,
    collect_keys: Optional[Sequence[str]] = None,
    # Reducers are declarative specs applied by run(cfg)
    reducers: Optional[Sequence[dict]] = None,
    # Convenience toggles (can be combined with `reducers`)
    reduce_x_hat: bool = False,
    reduce_uq5b_space: Optional[str] = None,   # "target" | "eps"
    reduce_x0_var_approx: bool = False,
    # Choose reduction style when sweeps are present
    reduce_style: str = "flat",                # "flat" | "two_stage"
    ) -> RunConfig:
    """
    Return a pre-specified RunConfig preset.

    Presets
    -------
    - "predict":                 1 run, no sweeps
    - "multisample":             seed sweep only
    - "multimasking":            mask sweep only
    - "combined"/"multimasking_multisample":  mask × seed sweeps
    - "uq5b_target":             collects target-space log_var head
    - "uq5b_eps":                collects eps-space log_var head
    - "uq5b_x0_approx"/"x0_var_approx": collects eps_log_var + t for x0 variance approximation

    Reducers
    --------
    - reduce_style="flat":      uses type="stack_mean_var" (global aggregation over all runs)
    - reduce_style="two_stage": uses type="two_stage" (intra seeds then inter masks)
    """
    name = str(name).lower().strip()
    reduce_style = str(reduce_style).lower().strip()
    if reduce_style not in {"flat", "two_stage"}:
        raise ValueError(f"reduce_style must be 'flat' or 'two_stage', got {reduce_style!r}.")

    def _collect(keys: Sequence[str]) -> CollectSpec:
        return CollectSpec(enabled=True, keys=list(keys), reduce="last")

    def _base_cfg(seed_sweep: SweepSpec, mask_sweep: SweepSpec, cspec: Optional[CollectSpec]) -> RunConfig:
        cfg = RunConfig(
            num_steps=int(num_steps),
            projection=projection,
            seed=seed,
            seed_sweep=seed_sweep,
            mask_sweep=mask_sweep,
            collect_spec=cspec,
        )
        cfg.reducers = list(reducers) if reducers is not None else []
        return cfg

    # -------------------------
    # Preset selection
    # -------------------------
    if name == "predict":
        cfg = _base_cfg(
            seed_sweep=SweepSpec(n=1, mode="none"),
            mask_sweep=SweepSpec(n=1, mode="none"),
            cspec=_collect(collect_keys) if collect_keys else None,
        )

    elif name == "multisample":
        cfg = _base_cfg(
            seed_sweep=SweepSpec(n=int(n_seeds), mode="offset"),
            mask_sweep=SweepSpec(n=1, mode="none"),
            cspec=_collect(collect_keys) if collect_keys else None,
        )

    elif name == "multimasking":
        cfg = _base_cfg(
            seed_sweep=SweepSpec(n=1, mode="none"),
            mask_sweep=SweepSpec(n=int(n_masks), mode="offset"),
            cspec=_collect(collect_keys) if collect_keys else None,
        )

    elif name in {"multimasking_multisample", "combined"}:
        cfg = _base_cfg(
            seed_sweep=SweepSpec(n=int(n_seeds), mode="offset"),
            mask_sweep=SweepSpec(n=int(n_masks), mode="offset"),
            cspec=_collect(collect_keys) if collect_keys else None,
        )

    elif name == "uq5b_target":
        cfg = _base_cfg(
            seed_sweep=SweepSpec(n=1, mode="none"),
            mask_sweep=SweepSpec(n=1, mode="none"),
            cspec=_collect(["pred_log_var_target"]),
        )

    elif name == "uq5b_eps":
        cfg = _base_cfg(
            seed_sweep=SweepSpec(n=1, mode="none"),
            mask_sweep=SweepSpec(n=1, mode="none"),
            cspec=_collect(["eps_log_var"]),
        )

    elif name in {"uq5b_x0_approx", "x0_var_approx"}:
        cfg = _base_cfg(
            seed_sweep=SweepSpec(n=1, mode="none"),
            mask_sweep=SweepSpec(n=1, mode="none"),
            cspec=_collect(["eps_log_var", "t"]),
        )

    else:
        raise ValueError(f"Unknown preset config name: {name!r}")

    # -------------------------
    # Helpers: ensure collection keys
    # -------------------------
    def _ensure_collect_key(k: str) -> None:
        if cfg.collect_spec is None:
            cfg.collect_spec = _collect([k])
            return
        if not cfg.collect_spec.enabled:
            cfg.collect_spec.enabled = True
        if cfg.collect_spec.keys is None:
            return  # None means "collect all"
        if k not in cfg.collect_spec.keys:
            cfg.collect_spec.keys = list(cfg.collect_spec.keys) + [k]

    # -------------------------
    # Reducer spec builders (match diffusion_reducer registry)
    # -------------------------
    def _mk_reducer_stack(field: str, *, exp: bool = False, name_: Optional[str] = None) -> dict:
        return {
            "name": name_ if name_ else field.replace(".", "_"),
            "type": "stack_mean_var",
            "field": field,
            "exp": exp,
        }

    def _mk_reducer_two_stage(field: str, *, exp: bool = False, name_: Optional[str] = None) -> dict:
        return {
            "name": name_ if name_ else field.replace(".", "_") + "_decomp",
            "type": "two_stage",
            "field": field,
            "exp": exp,
            # optional ddof knobs can be passed by caller via `reducers=...`
        }

    def _auto_reducer(field: str, *, exp: bool = False, name_: Optional[str] = None) -> dict:
        return _mk_reducer_two_stage(field, exp=exp, name_=name_) if (reduce_style == "two_stage") else _mk_reducer_stack(field, exp=exp, name_=name_)

    # -------------------------
    # Convenience reducer toggles
    # -------------------------

    # 1) Reconstruction aggregation
    if reduce_x_hat:
        cfg.reducers.append(_auto_reducer("x_hat", exp=False, name_="x_hat"))

    # 2) 5B variance aggregation (target/eps)
    if reduce_uq5b_space is not None:
        sp = str(reduce_uq5b_space).lower().strip()
        if sp not in {"target", "eps"}:
            raise ValueError(f"reduce_uq5b_space must be 'target' or 'eps', got {reduce_uq5b_space!r}.")
        if sp == "target":
            _ensure_collect_key("pred_log_var_target")
            cfg.reducers.append(
                _auto_reducer(
                    "collect.pred_log_var_target",
                    exp=True,  # log_var -> var
                    name_="uq5b_target_var",
                )
            )
        else:
            _ensure_collect_key("eps_log_var")
            cfg.reducers.append(
                _auto_reducer(
                    "collect.eps_log_var",
                    exp=True,  # log_var -> var
                    name_="uq5b_eps_var",
                )
            )

    # 3) x0 variance approximation (schedule-aware reducer)
    # Requires collection of eps_log_var and t; and runner-injected ctx["schedule"]["alpha_bars"].
    if reduce_x0_var_approx:
        _ensure_collect_key("eps_log_var")
        _ensure_collect_key("t")
        cfg.reducers.append(
            {
                "name": "x0_var_approx",
                "type": "x0_var_approx",
                "field_log_var": "collect.eps_log_var",
                "field_t": "collect.t",
            }
        )

    return cfg


def get_anomaly_preset(name: str, *, seed: Optional[int] = None) -> RunConfig:
    """
    Presets for anomaly scoring under conditional masking.

    Notes
    -----
    anomaly_score reducer expects runner-injected ctx["data"]["x_ref"] and uses
    TrajectoryResult.mask (no extra collect keys required).
    """
    name = str(name).lower().strip()

    if name == "mc_holdout_mean":
        cfg = get_preset_config(
            "multimasking",
            n_masks=40,
            num_steps=50,
            projection="hard",
            seed=seed,
            reducers=[],
        )
        cfg.reducers.append(
            {
                "name": "anom",
                "type": "anomaly_score",
                "field_pred": "x_hat",
                "residual_kind": "abs",
                "agg": "mean",
                "weight_cfg": {
                    "enabled": True,
                    "kind": "temporal_nearest_observed",
                    "mode": "exp",
                    "tau": 5.0,
                },
            }
        )
        return cfg

    if name == "mc_holdout_max":
        cfg = get_preset_config(
            "multimasking",
            n_masks=40,
            num_steps=50,
            projection="hard",
            seed=seed,
            reducers=[],
        )
        cfg.reducers.append(
            {
                "name": "anom",
                "type": "anomaly_score",
                "field_pred": "x_hat",
                "residual_kind": "abs",
                "agg": "max",
                "weight_cfg": {"enabled": False},
            }
        )
        return cfg

    raise ValueError(f"Unknown anomaly preset: {name!r}")