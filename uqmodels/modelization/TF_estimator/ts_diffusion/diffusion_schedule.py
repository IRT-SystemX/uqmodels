"""
diffusion_schedule.py
=====================

Core diffusion dynamics for time-series DDPM-style models.

This module implements:

- DiffusionSchedule:
    Precomputes forward-process coefficients (betas, alphas, alpha_bars,
    posterior variance) from a beta schedule.

- DiffusionScheduleMixin:
    Implements the forward diffusion (q_sample) and reverse sampling
    dynamics (p_sample_ddpm, reverse_chain) independently of the denoiser
    architecture.

Design Principles
-----------------
- The schedule encodes the mathematical definition of the forward and
  reverse processes.
- The denoiser is injected via a contractual API:
      _predict_target_stats(x_t, t, cond, training)
  which must be implemented by the concrete diffusion model.
- Skip sampling (num_steps < T) is supported via an explicit timestep
  mapping, but remains an approximation of full DDPM unless a dedicated
  method (e.g., DDIM) is used.

This module is model-agnostic and does not define losses, reducers,
or orchestration logic.
"""
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple, Callable, Union,  List,Literal, Sequence


import tensorflow as tf
from uqmodels.modelization.TF_estimator.ts_diffusion.diffusion_noise import make_noise_model,make_kernel_operator

def _extract_t(coeffs_1d: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
    """
    Gather coefficients at timesteps t and reshape to (B,1,1) for broadcasting.
    """
    b = tf.shape(t)[0]
    out = tf.gather(coeffs_1d, t)        # (B,)
    return tf.reshape(out, (b, 1, 1))    # broadcast to (B,T,C)

def linear_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> tf.Tensor:
    """Return a float32 linear beta schedule of length num_steps."""
    return tf.cast(tf.linspace(beta_start, beta_end, num_steps), tf.float32)

def _make_timestep_sequence(self, num_steps: int) -> tf.Tensor:
    """
    Build a strictly decreasing sequence of timesteps in [0, T-1].

    - If num_steps == T: returns [T-1, ..., 0]
    - If num_steps < T: returns a monotonic, unique sub-sequence (skip sampling).
    """
    T = int(self.schedule.num_steps)
    S = int(num_steps)

    if S <= 0:
        raise ValueError("num_steps must be >= 1.")
    if S > T:
        raise ValueError(f"num_steps={S} cannot be > schedule.num_steps={T}.")

    if S == T:
        return tf.range(T - 1, -1, -1, dtype=tf.int32)

    # Choose S indices approximately evenly spaced, then enforce uniqueness.
    idx = tf.cast(tf.round(tf.linspace(0.0, float(T - 1), S)), tf.int32)
    idx = tf.clip_by_value(idx, 0, T - 1)

    # Enforce uniqueness while preserving order:
    # tf.unique keeps first occurrences; since idx is non-decreasing, it preserves order.
    uniq, _ = tf.unique(idx)

    # If duplicates removed reduced the count, pad by adding missing indices from the end.
    # (Deterministic and keeps decreasing sequence stable.)
    missing = S - tf.shape(uniq)[0]
    def pad():
        # candidates: all timesteps, remove those already in uniq
        all_t = tf.range(0, T, dtype=tf.int32)
        # mask-out existing uniq
        mask = tf.reduce_all(all_t[:, None] != uniq[None, :], axis=1)
        remaining = tf.boolean_mask(all_t, mask)
        # take last 'missing' to preserve coverage of large timesteps
        extra = remaining[-missing:]
        return tf.concat([uniq, extra], axis=0)

    uniq = tf.cond(missing > 0, pad, lambda: uniq)

    # Sort increasing then reverse to get decreasing sampling order.
    uniq = tf.sort(uniq)
    return tf.reverse(uniq, axis=[0])



@dataclass
class DiffusionSchedule:
    """
    Precomputed DDPM schedule coefficients.
    """
    num_steps: int
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    def __post_init__(self):
        betas = linear_beta_schedule(self.num_steps, self.beta_start, self.beta_end)  # (N,)
        alphas = 1.0 - betas
        alpha_bars = tf.math.cumprod(alphas, axis=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

        self.sqrt_alpha_bars = tf.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = tf.sqrt(1.0 - alpha_bars)

        # Reverse step helpers
        self.sqrt_recip_alphas = tf.sqrt(1.0 / alphas)

        # posterior variance (DDPM)
        alpha_bars_prev = tf.concat([tf.ones((1,), tf.float32), alpha_bars[:-1]], axis=0)
        self.posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

        self.betas = tf.cast(self.betas, tf.float32)
        self.alphas = tf.cast(self.alphas, tf.float32)
        self.alpha_bars = tf.cast(self.alpha_bars, tf.float32)
        self.sqrt_recip_alphas = tf.cast(self.sqrt_recip_alphas, tf.float32)
        self.sqrt_alpha_bars = tf.cast(self.sqrt_alpha_bars, tf.float32)
        self.sqrt_one_minus_alpha_bars = tf.cast(self.sqrt_one_minus_alpha_bars, tf.float32)
        self.posterior_variance = tf.cast(self.posterior_variance, tf.float32)

class DiffusionScheduleMixin:
    """
    Provides schedule + q_sample + reverse sampling, with pluggable noise models.
    """

    def __init__(
        self,
        *,
        num_steps: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        kernel_cfg: Optional[Dict[str, Any]] = None,
        train_target: str = "eps",  # "eps" (default) or "eta"
        noise_forward_cfg: Optional[Dict[str, Any]] = None,
        noise_reverse_cfg: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.diffusion_cfg = {
            "num_steps": int(num_steps),
            "beta_start": float(beta_start),
            "beta_end": float(beta_end),
        }
        self.schedule = DiffusionSchedule(**self.diffusion_cfg)

        # Noise models
        self.noise_model_forward = make_noise_model(noise_forward_cfg)
        self.noise_model_reverse = make_noise_model(noise_reverse_cfg)

        self.kernel = make_kernel_operator(kernel_cfg)
        train_target = str(train_target).lower().strip()
        if train_target not in {"eps", "eta"}:
            raise ValueError("train_target must be 'eps' or 'eta'.")
        self.train_target = train_target

    def denoiser_stats(
        self,
        x_t: tf.Tensor,
        t: tf.Tensor,
        cond: Optional[tf.Tensor],
        *,
        training: bool,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        """
        Contract: predict denoiser statistics in *target space*.

        This mixin implements diffusion dynamics and calls this method from
        p_sample_ddpm(...) and training hooks.

        Implementations must return:
            pred_mean_target: tf.Tensor of shape (B,T,C)
            pred_log_var_target: tf.Tensor of shape (B,T,C) or None

        Notes
        -----
        - The returned space is the training target space: "eps" or "eta"
        depending on self.train_target.
        - If pred_log_var_target is provided (Gaussian head), it may be used for
        instrumentation and/or hybrid sampling variance, depending on configuration.
        """
        raise NotImplementedError(
            "DiffusionScheduleMixin requires denoiser_stats(...) to be implemented "
            "by the concrete model (e.g., BaseDiffusionModel)."
        )

    def to_eps_space(self, pred_target: tf.Tensor) -> tf.Tensor:
        """
        Convert a prediction from target space to eps space (noise space).

        If train_target == 'eps': pred_target is already eps_hat.
        If train_target == 'eta': pred_target is eta_hat and eps_hat = K(eta_hat).

        Args:
            pred_target: Tensor (B,T,C), prediction in target space.

        Returns:
            eps_hat: Tensor (B,T,C), prediction in eps space.
        """
        if self.train_target == "eta":
            return self.kernel.apply(pred_target)
        return pred_target
    
    def target_stats_to_eps_stats(
        self,
        pred_mean_target: tf.Tensor,
        pred_log_var_target: Optional[tf.Tensor],
        ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        """
        Convert target-space stats to eps-space stats.

        Args:
            pred_mean_target: (B,T,C) mean in target space (eps or eta).
            pred_log_var_target: (B,T,C) log-variance in target space or None.

        Returns:
            eps_mean: (B,T,C)
            eps_log_var: (B,T,C) or None
        """
        eps_mean = self.to_eps_space(pred_mean_target)

        if pred_log_var_target is None:
            return eps_mean, None

        # If target is already eps-space
        if self.train_target != "eta":
            return eps_mean, pred_log_var_target

        # train_target == "eta": map variance if kernel supports it
        if hasattr(self.kernel, "apply_variance"):
            var_eta = tf.exp(pred_log_var_target)
            var_eps = self.kernel.apply_variance(var_eta)
            eps_log_var = tf.math.log(var_eps + 1e-12)
            return eps_mean, eps_log_var

        # Fallback: keep target-space log_var (document limitation)
        return eps_mean, pred_log_var_target
    
    def sample_forward_noise_and_target(self, 
                                        x0: tf.Tensor,
                                        *,
                                        rng: Optional[tf.random.Generator] = None,) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Sample the forward noise in eps space and the supervision target for the denoiser.
        - train_target == 'eps':
            eps ~ NoiseModelForward (or N(0,I))
            target = eps
        - train_target == 'eta':
            eta ~ N(0,I)
            eps = K(eta)
            target = eta

        Args: 
            x0: Tensor (B,T,C), clean signal.

        Returns:
            eps: Tensor (B,T,C), noise injected into q_sample.
            target: Tensor (B,T,C), training target for the denoiser.
        """
        if self.train_target == "eta":
            eta = normal_like(tf.shape(x0), rng=rng,dtype=tf.float32)
            eps = self.kernel.apply(eta)
            return eps, eta

        if self.noise_model_forward is not None:
            eps = self.noise_model_forward.sample_like(x0)
        else:
            eps = normal_like(tf.shape(x0), rng=rng, dtype=tf.float32)

        return eps, eps

        
    def q_sample(self, x0: tf.Tensor, t: tf.Tensor, noise: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Forward diffusion: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * noise
        """
        x0 = tf.cast(x0, tf.float32)

        if noise is None:
            raise ValueError("noise must be provided explicitly to q_sample.")

        noise = tf.cast(noise, tf.float32)

        a = _extract_t(self.schedule.sqrt_alpha_bars, t)
        b = _extract_t(self.schedule.sqrt_one_minus_alpha_bars, t)
        return a * x0 + b * noise

    def p_sample_ddpm(self,
                      x_t: tf.Tensor,
                      t: tf.Tensor,
                      cond: Optional[tf.Tensor],
                      *,
                      rng: Optional[tf.random.Generator] = None,
                      collect_fn: Optional[Callable[[dict], None]] = None) -> tf.Tensor:
        """
        One reverse step. The stochastic term uses noise_model_reverse (default: white).
        """
        x_t = tf.cast(x_t, tf.float32)

        betas_t = _extract_t(self.schedule.betas, t)
        sqrt_one_minus_ab_t = _extract_t(self.schedule.sqrt_one_minus_alpha_bars, t)
        sqrt_recip_alpha_t = _extract_t(self.schedule.sqrt_recip_alphas, t)

        # The network output interpretation depends on train_target:
        # - "eps": eps_hat is predicted directly
        # - "eta": network predicts eta_hat, then eps_hat = K * eta_hat
        pred_mean, pred_log_var = self.denoiser_stats(x_t, t, cond=cond, training=False)
        eps_hat, eps_log_var = self.target_stats_to_eps_stats(pred_mean, pred_log_var)

        mu = sqrt_recip_alpha_t * (x_t - (betas_t / sqrt_one_minus_ab_t) * eps_hat)

        # Hybrid reverse variance: blend closed-form DDPM variance with
        # learned variance (stability + effective use of Gaussian head).    
        base_var = _extract_t(self.schedule.posterior_variance, t)
        if eps_log_var is not None:
            learned_var = tf.exp(eps_log_var)
            var_t = 0.5 * base_var + 0.5 * learned_var
        else:
            var_t = base_var
        var_t = tf.maximum(var_t, 1e-12)

        # stochasticity for t>0
        if self.noise_model_reverse is not None:
            z = self.noise_model_reverse.sample_like(x_t)
        else:
            z = normal_like(tf.shape(x_t), dtype=x_t.dtype)

        t_is_zero = tf.reshape(tf.equal(t, 0), (-1, 1, 1))

        # Optional collection hook for UQ/debug
        if collect_fn is not None:
            # x0 prediction in DDPM parameterization
            alpha_bar_t = _extract_t(self.schedule.alpha_bars, t)
            x0_pred = (x_t - sqrt_one_minus_ab_t * eps_hat) / (tf.sqrt(alpha_bar_t) + 1e-12)

            collect_fn(
                {
                    "t": t,
                    "x_t": x_t,
                    "eps_hat": eps_hat,
                    "eps_log_var": eps_log_var,
                    "pred_mean_target": pred_mean,
                    "pred_log_var_target": pred_log_var,
                    "x0_pred": x0_pred,
                    "cond": cond,
                }
            )

        return tf.where(t_is_zero, mu, mu + tf.sqrt(var_t) * z)
    
    
    def _make_timestep_sequence(self, num_steps: int) -> tf.Tensor:
        """
        Build a strictly decreasing sequence of timesteps in [0, T-1].

        - If num_steps == T: returns [T-1, ..., 0]
        - If num_steps < T: returns a monotonic, unique sub-sequence (skip sampling).
        """
        T = int(self.schedule.num_steps)
        S = int(num_steps)

        if S <= 0:
            raise ValueError("num_steps must be >= 1.")
        if S > T:
            raise ValueError(f"num_steps={S} cannot be > schedule.num_steps={T}.")

        if S == T:
            return tf.range(T - 1, -1, -1, dtype=tf.int32)

        # Choose S indices approximately evenly spaced, then enforce uniqueness.
        idx = tf.cast(tf.round(tf.linspace(0.0, float(T - 1), S)), tf.int32)
        idx = tf.clip_by_value(idx, 0, T - 1)

        # Enforce uniqueness while preserving order:
        # tf.unique keeps first occurrences; since idx is non-decreasing, it preserves order.
        uniq, _ = tf.unique(idx)

        # If duplicates removed reduced the count, pad by adding missing indices from the end.
        # (Deterministic and keeps decreasing sequence stable.)
        missing = S - tf.shape(uniq)[0]
        def pad():
            # candidates: all timesteps, remove those already in uniq
            all_t = tf.range(0, T, dtype=tf.int32)
            # mask-out existing uniq
            mask = tf.reduce_all(all_t[:, None] != uniq[None, :], axis=1)
            remaining = tf.boolean_mask(all_t, mask)
            # take last 'missing' to preserve coverage of large timesteps
            extra = remaining[-missing:]
            return tf.concat([uniq, extra], axis=0)

        uniq = tf.cond(missing > 0, pad, lambda: uniq)

        # Sort increasing then reverse to get decreasing sampling order.
        uniq = tf.sort(uniq)
        return tf.reverse(uniq, axis=[0])
