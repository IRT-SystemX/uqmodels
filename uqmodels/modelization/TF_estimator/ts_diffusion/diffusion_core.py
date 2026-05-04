"""
diffusion_core.py
=================

Core orchestration and model interface for the time-series diffusion framework.

This module belongs to a stratified diffusion framework with a single
dependency direction: primitives -> dynamics -> orchestration -> reduction.

Core model wrapper and orchestration layer for the time-series diffusion framework.

-----------------
This module sits at the orchestration level and coordinates:

- diffusion_noise.py
    Forward noise models (white and structured noise).

- diffusion_masking.py
    Conditioning and masking primitives.

- diffusion_schedule.py
    Diffusion mathematics and transitions (q_sample, p_sample_ddpm, reverse chain).

- diffusion_dataclass.py
    Shared structural objects (TrajectoryResult, ReducerInputs, RunResult).

- diffusion_reducer.py
    Post-run reducers operating on the canonical mask×seed trajectory grid.

- diffusion_config.py
    Declarative RunConfig presets.

Role of this file
-----------------
`diffusion_core.py` defines the concrete diffusion model wrapper:
- implements the denoiser contract required by diffusion dynamics,
- defines training targets and losses (eps / eta),
- orchestrates inference (multi-seed, multi-masking) and applies reducers.

Design invariant
----------------
A single canonical trajectory runner executes the diffusion process.
All higher-level features (UQ, multi-sampling, diagnostics) are implemented
via orchestration + configurable readouts and reducers, never via parallel
diffusion pipelines.

-------------------------------------------------------------------------------
Core mechanics (high-level view)
-------------------------------------------------------------------------------

Diffusion schedule & transitions
    Implemented in `diffusion_schedule.py` via DiffusionScheduleMixin:
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
        p(x_{t-1} | x_t, cond) via DDPM update (optionally sub-sampled).

Structured forward noise (optional)
    eta ~ N(0, I), eps = K(eta)
    - train_target = "eps": predict eps (classic DDPM)
    - train_target = "eta": predict eta, eps_hat = K(eta_hat)
    Default: IdentityKernel. Optional: TemporalConvKernel.

Denoiser contract
    Subclass must implement:
        eps_model(x_t, t, cond, training)
    Internally normalized to:
        (pred_mean_target, pred_log_var_target | None)
    Supported heads:
        - Point head: (B,T,C)
        - Gaussian head: diagonal (mean, log_var)

-------------------------------------------------------------------------------
Canonical trajectory primitive
-------------------------------------------------------------------------------

_infer_one_trajectory(...) -> TrajectoryResult

This is the atomic executable unit:
    - executes reverse sampling,
    - applies optional projection (hard/soft),
    - emits optional instrumentation payload,
    - returns structured outputs for reducers.

All public inference APIs are thin wrappers over this primitive:
    - sample_unconditional(...)
    - sample_conditional_inpaint(...)
    - predict(...)
    - run(cfg) for sweeps (seeds × masks) + reducers.

-------------------------------------------------------------------------------
Instrumentation payload (single-source-of-truth hook)
-------------------------------------------------------------------------------

Reverse sampling exposes exactly one instrumentation interface:
a standardized payload dictionary emitted at each timestep.

Contract:
    diffusion dynamics  -> produce payload
    orchestration layer -> selects what to collect
    reducers            -> operate only on collected data

The payload dictionary may include:
   - "t": current timestep (B,)
   - "x_t": current noisy state (B,T,C)
   - "eps_hat": eps-space prediction used in the reverse update
   - "eps_log_var": log-variance in eps-space (if Gaussian head active)
   - "pred_mean_target": prediction in denoiser target space
   - "pred_log_var_target": target-space log-variance (if available)
   - "x0_pred": reconstructed x0 at this step
   - "cond": conditioning tensor (if any)

Reducers must never access internal diffusion tensors directly.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple, Callable, Union,  List,Literal, Sequence

import tensorflow as tf
from tensorflow.keras import metrics
from uqmodels.modelization.TF_estimator.base.model import BaseKModel
from uqmodels.modelization.TF_estimator.ts_diffusion.diffusion_masking import ConditionalMaskingMixin
from uqmodels.modelization.TF_estimator.ts_diffusion.diffusion_schedule import DiffusionScheduleMixin, _extract_t
from uqmodels.modelization.TF_estimator.ts_diffusion.diffusion_dataclass import TrajectoryResult, RunConfig, RunResult, CollectSpec
from uqmodels.modelization.TF_estimator.ts_diffusion.diffusion_reducer import REDUCER_REGISTRY,ReducerInputs

# -----------------------------------------------------------------------------
# Utils utilities
# -----------------------------------------------------------------------------

def normal_like(shape,
                *,
                rng: Optional[tf.random.Generator] = None,
                dtype: tf.dtypes.DType = tf.float32) -> tf.Tensor:
    """
    Generate a tensor from a standard normal distribution.

    Parameters
    ----------
    shape : TensorShape or tuple
        Output tensor shape.
    rng : tf.random.Generator, optional
        Per-trajectory generator.
    dtype : tf.DType
        Output dtype.

    Returns
    -------
    tf.Tensor
        Random normal tensor.
    """
    if rng is not None:
        return rng.normal(shape, dtype=dtype)
    return tf.random.normal(shape, dtype=dtype)

class BaseDiffusionModel(DiffusionScheduleMixin, ConditionalMaskingMixin, BaseKModel):
    """
    Base diffusion model orchestrating:
    - schedule + forward/reverse dynamics
    - optional conditional masking (inpainting)
    - modular denoiser output decoding (point vs Gaussian stats)
    - modular target/loss (MSE vs NLL)
    - optional uncertainty quantification (UQ) utilities
 

    Subclasses MUST define:
    - self.eps_model: callable with signature eps_model(x_t, t, cond=None, training=...)
      returning eps_hat same shape as x_t.

    UQ methods supported
    --------------------
    1) Multi-sampling UQ (same condition): empirical mean/var across trajectories.
    2) Multi-masking UQ: empirical mean/var across different masks.
    3) 5B UQ (eps distribution): predicted log-variance on eps converted to x0 variance
       via a step-wise analytic factor (approx).


    Data format
    ----------
    - data can be X or (X, y); y is ignored by default.
    - X is expected shape (B,T,C) (time-series windows), but works for any rank>=2
      if your eps_model supports it (here, broadcasting assumes (B,T,C)).
    """

    def __init__(self, *, 
                 name: str = "Diffusion",
                 default_run_cfg: Optional[RunConfig] = None, 
                 **kwargs):
        super().__init__(name=name, **kwargs)

        # Trackers (aligned with BaseAutoencoder style)
        self.noise_loss_tracker = metrics.Mean(name="noise_loss")
        self.total_loss_tracker = metrics.Mean(name="loss")
        self.default_run_cfg = default_run_cfg

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.noise_loss_tracker]

    # ----- loss -----

    def _noise_loss_mse(self, noise: tf.Tensor, eps_hat: tf.Tensor) -> tf.Tensor:
        """Default DDPM loss: MSE(noise, eps_hat) reduced over non-batch dims."""
        noise = tf.cast(noise, tf.float32)
        eps_hat = tf.cast(eps_hat, tf.float32)
        sq = tf.square(noise - eps_hat)
        axes = tf.range(1, tf.rank(sq))
        per_ex = tf.reduce_mean(sq, axis=axes)
        return tf.reduce_mean(per_ex)

    # ------------------------------------------------------------------
    # 0) Configuration knobs (minimal, backward-compatible defaults)
    # ------------------------------------------------------------------
    denoiser_output_mode: str = "auto"   # "auto" | "point" | "gaussian"
    loss_mode: str = "auto"             # "auto" | "mse" | "nll"
    log_var_clip: Tuple[float, float] = (-10.0, 10.0)

    # ==================================================================
    # 1) Canonical hooks: decode stats / compute target / compute loss
    # ==================================================================

    def _decode_denoiser_output(
        self,
        out: Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor], List[tf.Tensor]],
        *,
        channels: Optional[int] = None,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        """
        Decode denoiser output into (mean, log_var).

        Supported outputs:
            - point: Tensor (B,T,C) -> (out, None)
            - gaussian tuple: (mean, log_var) each (B,T,C)
            - gaussian concat: Tensor (B,T,2C) -> split using `channels`
        """
        if isinstance(out, (tuple, list)):
            if len(out) != 2:
                raise ValueError("Gaussian denoiser output as tuple/list must have length 2: (mean, log_var).")
            mean, log_var = out
            return tf.cast(mean, tf.float32), tf.cast(log_var, tf.float32)

        if not isinstance(out, tf.Tensor):
            raise TypeError(f"Unsupported denoiser output type: {type(out)}")

        out = tf.cast(out, tf.float32)
        if out.shape.rank != 3:
            raise ValueError(f"Denoiser output must be rank-3 (B,T,C). Got rank={out.shape.rank}.")

        last = out.shape[-1]
        if channels is None:
            # Without channels, we can only safely interpret (B,T,C) as point output.
            return out, None

        if last is None:
            # Dynamic last dim: still safe to split only if `channels` provided.
            # We enforce runtime check.
            out_last = tf.shape(out)[-1]
            tf.debugging.assert_equal(out_last, 2 * channels, message="Expected concatenated Gaussian head with last dim = 2*channels.")
            return out[:, :, :channels], out[:, :, channels:]

        # Static last dim checks
        if last == channels:
            return out, None
        if last == 2 * channels:
            return out[:, :, :channels], out[:, :, channels:]
        raise ValueError(f"Unexpected denoiser output last dim={last}. Expected C={channels} (point) or 2C={2*channels} (gaussian).")

    def denoiser_stats(
        self, x_t: tf.Tensor, 
        t: tf.Tensor, 
        cond: Optional[tf.Tensor], 
        *, training: bool) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        """
        Canonical denoiser stats prediction.

        Returns
        -------
        pred_mean : tf.Tensor (B,T,C)
        pred_log_var : tf.Tensor (B,T,C) or None
        """
        # Your eps_model signature should support (x_t, t, cond, training=...)
        out = self.eps_model(x_t, t, cond=cond, training=training)
        C = tf.shape(x_t)[-1]
        # For static graphs, prefer python int channels if you store it
        channels = int(x_t.shape[-1]) if x_t.shape[-1] is not None else None
        if channels is None:
            # Graph-safe behavior:
            # - If denoiser returns a tuple/list (mean, log_var): OK (no need for channels)
            # - If denoiser returns concatenated (B,T,2C): we cannot split safely without knowing C
            #   -> force the user to either (1) make the denoiser return a tuple, or (2) ensure static x_t.shape[-1]
            if isinstance(out, (tuple, list)) and len(out) == 2:
                mean, log_var = out
            else:
                raise ValueError(
                    "Cannot decode denoiser output when x_t.shape[-1] is dynamic. "
                    "Make the denoiser return a tuple (mean, log_var) for Gaussian mode, "
                    "or ensure x_t has a static channel dimension (so C is known at build time)."
                )
        else:
            mean, log_var = self._decode_denoiser_output(out, channels=channels)

        # Enforce mode if requested
        mode = str(getattr(self, "denoiser_output_mode", "auto")).lower()
        if mode == "point":
            log_var = None
        elif mode == "gaussian" and log_var is None:
            raise ValueError("denoiser_output_mode='gaussian' but denoiser did not return log_var.")

        return mean, log_var

    def _compute_loss_from_stats(
        self,
        target: tf.Tensor,
        pred_mean: tf.Tensor,
        pred_log_var: Optional[tf.Tensor],
        *,
        loss_mode: Optional[str] = None,
        weight: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """
        Compute training loss from predicted Gaussian stats.

        - auto: MSE if pred_log_var is None, else NLL
        - mse:  always MSE (ignores pred_log_var)
        - nll:  requires pred_log_var, else raises

        Supports optional weighting (e.g., holdout-only).
        """
        mode = str(loss_mode or getattr(self, "loss_mode", "auto")).lower().strip()

        # ---- MSE path ----
        if mode == "mse" or (mode == "auto" and pred_log_var is None):
            sq = tf.square(tf.cast(target, tf.float32) - tf.cast(pred_mean, tf.float32))
            if weight is not None:
                w = tf.cast(weight, tf.float32)
                sq = sq * w
                return tf.reduce_sum(sq) / (tf.reduce_sum(w) + 1e-8)
            return tf.reduce_mean(sq)

        # ---- NLL path ----
        if pred_log_var is None:
            raise ValueError("loss_mode='nll' requires pred_log_var (Gaussian head).")

        lo, hi = getattr(self, "log_var_clip", (-10.0, 10.0))
        pred_log_var = tf.clip_by_value(tf.cast(pred_log_var, tf.float32), float(lo), float(hi))

        loss_map = 0.5 * (tf.exp(-pred_log_var) * tf.square(tf.cast(target, tf.float32) - tf.cast(pred_mean, tf.float32)) + pred_log_var)

        if weight is not None:
            w = tf.cast(weight, tf.float32)
            loss_map = loss_map * w
            return tf.reduce_sum(loss_map) / (tf.reduce_sum(w) + 1e-8)

        return tf.reduce_mean(loss_map)

# ==================================================================
# 2) Minimal training hook using the canonical hooks above
# ==================================================================

    def forward_and_losses(self, data: Any) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Hook driving train_step/test_step (BaseKModel style).

        Expected input:
        - data can be X or (X, y). y ignored.
        - X shape (B,T,C)
        """
        x0 = data[0] if isinstance(data, (tuple, list)) else data
        x0 = tf.cast(x0, tf.float32)

        B = tf.shape(x0)[0]
        t = tf.random.uniform([B], minval=0, maxval=self.schedule.num_steps, dtype=tf.int32)

        # conditioning
        cond, y_obs, mask = (None, None, None)
        if getattr(self, "masking_cfg", None) is not None:
            cond, y_obs, mask = self.make_condition(x0)

        # forward diffusion | Here we could 
        eps, target = self.sample_forward_noise_and_target(x0,rng = None)
        x_t = self.q_sample(x0, t, noise=eps)
        
        # predict stats in target space
        pred_mean, pred_log_var = self.denoiser_stats(x_t, t, cond, training=True)

        # optional loss weighting: focus on holdout region if mask exists
        weight = None
        if mask is not None:
            weight = 1.0 - tf.cast(mask, tf.float32)

        loss = self._compute_loss_from_stats(target, pred_mean, pred_log_var, weight=weight)

        logs = {
            "loss": loss,
        }
        return loss, logs


    # ----- keras steps -----

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            total_loss, logs = self.forward_and_losses(data)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        if "noise_loss" in logs:
            self.noise_loss_tracker.update_state(logs["noise_loss"])

        res = {"loss": self.total_loss_tracker.result()}
        for k, v in logs.items():
            tracker = getattr(self, f"{k}_tracker", None)
            res[k] = tracker.result() if tracker is not None else v
        return res

    def test_step(self, data):
        total_loss, logs = self.forward_and_losses(data)
        self.total_loss_tracker.update_state(total_loss)
        if "noise_loss" in logs:
            self.noise_loss_tracker.update_state(logs["noise_loss"])

        res = {"loss": self.total_loss_tracker.result()}
        for k, v in logs.items():
            tracker = getattr(self, f"{k}_tracker", None)
            res[k] = tracker.result() if tracker is not None else v
        return res
    

# ==================================================================
# Helper
# ==================================================================

    def _sigma_t(self, t: tf.Tensor, ref: tf.Tensor) -> tf.Tensor:
        """
        Compute sigma_t = sqrt((1 - alpha_bar_t) / alpha_bar_t) from DDPM schedule.

        Parameters
        ----------
        t : tf.Tensor
            Timesteps, shape (B,), int32.
        ref : tf.Tensor
            Reference tensor (B,T,C) for broadcasting.

        Returns
        -------
        tf.Tensor
            sigma_t, shape (B,1,1), float32.
        """
        ref = tf.cast(ref, tf.float32)
        sqrt_ab = _extract_t(self.schedule.sqrt_alpha_bars, t)
        sqrt_1mab = _extract_t(self.schedule.sqrt_one_minus_alpha_bars, t)
        sigma = sqrt_1mab / (sqrt_ab + 1e-12)
        B = tf.shape(ref)[0]
        return tf.reshape(tf.cast(sigma, tf.float32), (B, 1, 1))

    
    def _make_projector(
        self,
        *,
        y_obs: tf.Tensor,
        mask: tf.Tensor,
        mode: str = "hard",
    ):
        """
        Build a projection function proj_fn(x, t) -> x to enforce observations.

        Parameters
        ----------
        y_obs : tf.Tensor
            Observed values (B,T,C).
        mask : tf.Tensor
            Observation mask (B,T,C), 1=observed.
        mode : str
            "hard" or "soft".
        """
        mode = str(mode).lower().strip()
        y_obs = tf.cast(y_obs, tf.float32)
        mask = tf.cast(mask, tf.float32)

        if mode == "hard":
            def proj_fn(x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
                return self.project_hard(x, y_obs, mask)
            return proj_fn

        if mode == "soft":
            def proj_fn(x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
                sigma_t = self._sigma_t(t, x)  # expects broadcastable (B,1,1)
                return self.project_soft(x, y_obs, mask, sigma_t=sigma_t)
            return proj_fn

        raise ValueError("Projection mode must be 'hard' or 'soft'.")

    def _reverse_chain(
        self,
        x_init: tf.Tensor,
        *,
        cond: Optional[tf.Tensor],
        num_steps: Optional[int] = None,
        rng: Optional[tf.random.Generator] = None,
        proj_fn: Optional[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]] = None,
        collect_fn: Optional[Callable[[dict], None]] = None,
    ) -> tf.Tensor:
        """
        Canonical reverse diffusion loop (Single Source of Truth).

        Parameters
        ----------
        x_init : tf.Tensor
            Initial state (B,T,C), typically sampled from N(0,I).
        cond : tf.Tensor or None
            Conditioning passed to p_sample_ddpm.
        num_steps : int, optional
            Number of reverse steps. Defaults to schedule.num_steps.
        proj_fn : callable, optional
            proj_fn(x, t) -> x applied after each reverse update.
        collect_fn : callable, optional
            collect_fn(payload_dict) called inside p_sample_ddpm at each step
        """

        T = int(self.schedule.num_steps)

        # Allow skip sampling: num_steps is the number of reverse iterations (<= T).
        if num_steps is None:
            num_steps = T
        else:
            num_steps = int(num_steps)
            if num_steps <= 0:
                raise ValueError("num_steps must be >= 1.")
            if num_steps > T:
                raise ValueError(f"num_steps={num_steps} cannot be > schedule.num_steps={T}.")

        t_seq = self._make_timestep_sequence(num_steps)

        B = tf.shape(x_init)[0]
        x = x_init
        for i in tf.unstack(t_seq):
            t = tf.fill([B], i)
            x = self.p_sample_ddpm(x, t, cond=cond, rng=rng, collect_fn=collect_fn)
            if proj_fn is not None:
                x = proj_fn(x, t)
        return x
    
# ==================================================================
# Wrapper of _reverse_chain for unconditional sampling
# ==================================================================

    def sample_unconditional(
        self,
        shape: Tuple[int, int, int],
        *,
        num_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> tf.Tensor:
        """
        Unconditional sampling: start from x_T ~ N(0,I) and run reverse steps.

        Args:
            shape: (B,T,C)
            num_steps: override number of reverse steps (default: schedule length)
            seed: optional seed for x_init
        """
        x_init = normal_like(shape, seed=seed)
        return self._reverse_chain(
            x_init,
            cond=None,
            num_steps=int(num_steps) if num_steps is not None else int(self.schedule.num_steps),
            proj_fn=None,
            collect_fn=None,
        )

# ==================================================================
# Core Function related to noise-trajectory inference 
# ==================================================================
    def _init_x(self, shape, rng=None, dtype=tf.float32):
        """Initialize reverse trajectory from standard Gaussian noise."""
        return normal_like(shape, rng=rng, dtype=dtype)

    def _infer_one_trajectory(
        self,
        x_ref: tf.Tensor,
        *,
        num_steps: Optional[int] = None,
        projection: str = "hard",
        seed: Optional[int] = None,
        collect_spec: Optional[CollectSpec] = None) -> TrajectoryResult:
        """
        Rich internal wrapper: builds (cond, y_obs, mask), projector, runs reverse chain.

        Parameters
        ----------
        x_ref : tf.Tensor
            Reference batch (B,T,C) used to build conditioning and shape.
        num_steps : int, optional
            Number of reverse steps, default schedule.num_steps.
        projection : str
            "hard" or "soft".
        seed : int, optional
            Noise seed for x_T init.
        collect_spec : callable, optional
            active instrumentation payload collection
            “payload emitted by p_sample_ddpm”
            “reduced to TrajectoryResult.collect”

        Returns
        -------
        x_hat : tf.Tensor
            Final reconstruction (B,T,C).
        y_obs : tf.Tensor
            Observed tensor used for conditioning (B,T,C).
        mask : tf.Tensor
            Observation mask (B,T,C).
        """
        rng = tf.random.Generator.from_seed(seed)
        x_ref = tf.cast(x_ref, tf.float32)
        x_init = self._init_x(tf.shape(x_ref), rng=rng, dtype=tf.float32)

        if self.masking_cfg is None:
            cond, y_obs, mask = None, None, None

            proj_fn = None
        else:
            self.reset_mask_context()
            cond, y_obs, mask = self.make_condition(x_ref)
            if cond is None or y_obs is None or mask is None:
                raise RuntimeError("make_condition did not produce cond/y_obs/mask.")
            proj_fn = self._make_projector(y_obs=y_obs, mask=mask, mode=projection)




        collected: Optional[Dict[str, Any]] = None
        collector = None
        
        if collect_spec is not None and collect_spec.enabled:
            collected = {}

            keys_set = set(collect_spec.keys) if collect_spec.keys is not None else None

            def collector(payload: Dict[str, Any]) -> None:
                # V0: "last" reduction -> overwrite at each step; last call wins.
                if keys_set is None:
                    for k, v in payload.items():
                        collected[k] = v
                else:
                    for k in keys_set:
                        if k in payload:
                            collected[k] = payload[k]

        x_hat = self._reverse_chain(
            x_init,
            cond=cond,
            num_steps=num_steps,
            proj_fn=proj_fn,
            rng=rng,
            collect_fn=collector)
        
        return TrajectoryResult(x_hat=x_hat, y_obs=y_obs, mask=mask, collect=collected)

# ==================================================================
# Wrapper of _infer_one_trajectory
# ==================================================================

    def sample_conditional_inpaint(
        self,
        x0_ref: tf.Tensor,
        *,
        hard: bool = True,
        num_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> tf.Tensor:
        """
        Conditional inpainting sampling (wrapper).

        Args:
            x0_ref: reference signal used to build conditioning (B,T,C)
            hard: enforce observed values strictly if True, else soft projection
            num_steps: override number of reverse steps (default: schedule length)
            seed: optional seed forwarded to the trajectory runner

        Returns:
            x_hat: reconstructed sample (B,T,C)
        """
        res = self._infer_one_trajectory(
            tf.cast(x0_ref, tf.float32),
            num_steps=int(num_steps) if num_steps is not None else int(self.schedule.num_steps),
            projection="hard" if hard else "soft",
            seed=seed,
            collect_spec=None,
        )
        return res.x_hat
    
    
# =====================================================================
# Reducer Related to aggregation of multi trajectory mode
# =====================================================================
        
    def _extract_field(self, r: TrajectoryResult, path: str) -> tf.Tensor:
        return r.get(path)

# =============================================================================
# Reducer framework (post-trajectory readout layer)
# =============================================================================
#
# Philosophy
# ----------
# Reducers implement post-processing logic over collections of TrajectoryResult
# objects produced by run(cfg). They never execute diffusion steps themselves.
#
# The diffusion core (forward, reverse, conditioning) is executed exactly once
# per trajectory by the canonical runner. Reducers operate purely on its outputs.
#
# Levels of operation
# -------------------
# - results : flat list of TrajectoryResult (all runs)
# - groups  : hierarchical grouping (mask outer, seed inner)
#
# Reducers may:
#   * extract tensors (x_hat, collect.* fields, mask, etc.)
#   * aggregate across seeds and/or masks
#   * compute derived quantities (e.g., variance propagation, anomaly scores)
#
# =============================================================================
    def _apply_reducer(self, spec: Dict[str, Any], inputs: ReducerInputs) -> Dict[str, tf.Tensor]:
        rtype = str(spec.get("type", "")).strip()
        if not rtype:
            raise ValueError("Each reducer spec must define a non-empty 'type'.")

        fn = REDUCER_REGISTRY.get(rtype, None)
        if fn is None:
            raise ValueError(f"Unknown reducer type: {rtype!r}")

        return fn(spec, inputs)

# =====================================================================
# Core Function Run build on mutli-scheme _infer_one_trajectory call
# =====================================================================

    def run(self, x_ref: tf.Tensor, *, cfg: RunConfig) -> RunResult:
        """
        Run one or many trajectories according to cfg (V0 sweep runner).

        Returns
        -------
        results : list[TrajectoryResult]
            Flat list of results. Ordering: mask_sweep outer, seed_sweep inner.
        """
        # Seed list
        if cfg.seed_sweep.mode == "none":
            seeds = [cfg.seed]
        else:
            base = 0 if cfg.seed is None else int(cfg.seed)
            seeds = [base + i for i in range(int(cfg.seed_sweep.n))]

        # Mask sweep count
        if cfg.mask_sweep.mode == "none":
            mask_indices = [0]
        else:
            mask_indices = list(range(int(cfg.mask_sweep.n)))


        groups: List[List[TrajectoryResult]] = []
        for _m in mask_indices:
            group_m: List[TrajectoryResult] = []
            # V0: rely on masking_cfg internal randomness to resample masks across calls.

            for s in seeds:
                res = self._infer_one_trajectory(
                    x_ref,
                    num_steps=int(cfg.num_steps),
                    projection=cfg.projection,
                    seed=s,
                    collect_spec=cfg.collect_spec,
                )
                group_m.append(res)
            groups.append(group_m)

        results = [r for group in groups for r in group]

        inputs = ReducerInputs(
            groups=groups,
            ctx = {
                "schedule": {
                    "alpha_bars": self.schedule.alpha_bars,
                    # optionally add more:
                    # "betas": self.schedule.betas,
                    # "posterior_variance": self.schedule.posterior_variance,
                },
                "data": {
                    "x_ref": x_ref,
                },
                "sweep": {
                    "seed_ids": seeds,   # len N flat (optional)
                    "mask_ids": mask_indices,   # len N flat (optional)
                },
                # "views": {"groups": groups, "runs": runs}  # optional if you keep them here
            }
        )

        reduced: Dict[str, Any] = {}
        for spec in getattr(cfg, "reducers", []):
            name = spec.get("name", None)
            if not name:
                raise ValueError("Each reducer spec must define a non-empty 'name'.")
            reduced[name] = self._apply_reducer(spec, inputs)

        return RunResult(results=results, groups=inputs.groups, reduced=reduced)

    def _resolved_run_cfg(
        self,
        *,
        cfg: Optional[RunConfig] = None,
        num_steps: Optional[int] = None,
        projection: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> RunConfig:
        base = cfg if cfg is not None else self.default_run_cfg
        if base is None:
            raise ValueError("No RunConfig provided and self.default_run_cfg is None.")

        # Make a shallow copy (dataclasses.replace) to avoid mutating defaults.
        out = replace(base)
        if num_steps is not None:
            out.num_steps = int(num_steps)
        if projection is not None:
            out.projection = str(projection)
        if seed is not None:
            out.seed = int(seed)
        return out

    def predict(
        self,
        x_ref: tf.Tensor,
        *,
        cfg: Optional[RunConfig] = None,
        num_steps: Optional[int] = None,
        projection: Optional[str] = None,
        seed: Optional[int] = None,
        all_details=False,
    ) -> tf.Tensor:
        """
        Predict / inpaint using the default run configuration (or an override).

        Returns
        -------
        x_hat : tf.Tensor (B,T,C)
            Output of the first trajectory in the run.
        """
        run_cfg = self._resolved_run_cfg(cfg=cfg, num_steps=num_steps, projection=projection, seed=seed)
        out = self.run(tf.cast(x_ref, tf.float32), cfg=run_cfg)
        
        if(all_details):
            return out
        else:
            return out.results[0].x_hat