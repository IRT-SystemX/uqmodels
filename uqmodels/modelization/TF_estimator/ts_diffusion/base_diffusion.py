"""
base_diffusion.py
=================

Modular diffusion models on top of BaseKModel (Keras wrapper) using a single
hook-based training interface similar to BaseAutoencoder / BaseVAE.

This implementation supports:
- Classic DDPM (epsilon-parameterization, white noise)
- Structured temporally correlated noise via a fixed kernel operator
- Optional reparameterization of the training target (predicting eta instead of epsilon)
- Identity fallback (no kernel required)

Core layers (conceptual)
------------------------

1) Denoiser model (core network) [to be provided by subclass]:
   - eps_model(x_t, t, cond) -> prediction
   - The prediction is interpreted depending on `train_target`:
       * "eps" : predicts epsilon (classic DDPM)
       * "eta" : predicts eta (whitened noise), with epsilon = K(eta)
   - cond can be None or any tensor (e.g., concat(y_obs, mask))

2) Diffusion process (schedule mixin, closed-form DDPM):
   - beta schedule and precomputed coefficients (alphas, alpha_bars, etc.)
   - forward sampling:
         x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
   - reverse step p(x_{t-1} | x_t, cond) using the predicted epsilon
   - When `train_target="eta"`, epsilon_hat is reconstructed as:
         epsilon_hat = K(eta_hat)

3) Noise model abstraction:
   - Forward noise is generated as:
         eta ~ N(0, I)
         epsilon = K(eta)
   - K is a fixed linear operator applied along time (per channel)
   - Default operator is Identity (classic white noise case)
   - Optional TemporalConvKernel allows realistic temporally correlated noise
   - Reverse noise remains configurable (white or structured)

4) Conditioning / masking (mixin, optional):
   - provides y_obs, mask from a configurable masking pipeline
   - supports different mask lifecycles via MaskPolicy (persistent vs resampled)
   - compatible with both epsilon and eta training targets

5) Training hook:
   - forward_and_losses(data) -> total_loss, logs
   - Default objective:
       * train_target="eps" : MSE(epsilon, epsilon_hat)
       * train_target="eta" : MSE(eta, eta_hat)
   - Reparameterization via eta improves numerical conditioning when K is
     smoothing or non-invertible.

6) Inference utilities:
   - sample_unconditional(...)
   - sample_conditional_inpaint(...)
   - Reverse update remains DDPM closed-form; when using eta training,
     epsilon_hat is reconstructed via the kernel operator before applying
     the standard reverse formula.

Design philosophy
-----------------
- Keep DDPM closed-form mechanics for simplicity and stability.
- Introduce structured noise realism without requiring full SDE/ODE solvers.
- Preserve backward compatibility (IdentityKernel = classic DDPM).
- Maintain modular separation between:
    * schedule mechanics,
    * kernel operator,
    * noise model,
    * conditioning pipeline,
    * training hook.

References / alignment
----------------------
- Mirrors the hook-based training design of BaseAutoencoder / BaseVAE:
  forward_and_losses(data) driving train_step/test_step.
- Uses BaseKModel compile/fit convenience behavior.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import metrics
from uqmodels.modelization.TF_estimator.base.model import BaseKModel
from uqmodels.modelization.TF_estimator.base.masking import MaskGenerator, MaskPolicy, MaskContext  # type: ignore
from uqmodels.modelization.TF_estimator.ts_diffusion.noise import make_noise_model


# Your project import (adjust if needed)
from uqmodels.modelization.TF_estimator.base.model import BaseKModel

# -----------------------------------------------------------------------------
# Schedule utilities
# -----------------------------------------------------------------------------

def linear_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> tf.Tensor:
    """Return a float32 linear beta schedule of length num_steps."""
    return tf.cast(tf.linspace(beta_start, beta_end, num_steps), tf.float32)


def _extract_t(coeffs_1d: tf.Tensor, t: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
    """
    Gather coefficients at timesteps t and reshape to (B,1,1) for broadcasting.
    """
    b = tf.shape(t)[0]
    out = tf.gather(coeffs_1d, t)        # (B,)
    return tf.reshape(out, (b, 1, 1))    # broadcast to (B,T,C)


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


# -----------------------------------------------------------------------------
# Mixins
# -----------------------------------------------------------------------------

# DiffusionScheduleMixin encapsulates the mathematical dynamics of diffusion
# (forward noise process and reverse update rule) independently of training
# logic or conditioning. This allows swapping different diffusion
# parameterizations (e.g., DDPM, DDIM, EDM) without modifying the core model.
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

    def q_sample(self, x0: tf.Tensor, t: tf.Tensor, noise: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Forward diffusion: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * noise
        If noise is None, it is sampled using noise_model_forward.
        """
        x0 = tf.cast(x0, tf.float32)
        if noise is None:
            noise = self.noise_model_forward.sample_like(x0)
        noise = tf.cast(noise, tf.float32)

        a = _extract_t(self.schedule.sqrt_alpha_bars, t, x0)
        b = _extract_t(self.schedule.sqrt_one_minus_alpha_bars, t, x0)
        return a * x0 + b * noise

    def p_sample_ddpm(self, x_t: tf.Tensor, t: tf.Tensor, cond: Optional[tf.Tensor]) -> tf.Tensor:
        """
        One reverse step. The stochastic term uses noise_model_reverse (default: white).
        """
        x_t = tf.cast(x_t, tf.float32)

        betas_t = _extract_t(self.schedule.betas, t, x_t)
        sqrt_one_minus_ab_t = _extract_t(self.schedule.sqrt_one_minus_alpha_bars, t, x_t)
        sqrt_recip_alpha_t = _extract_t(self.schedule.sqrt_recip_alphas, t, x_t)

        # The network output interpretation depends on train_target:
        # - "eps": eps_hat is predicted directly
        # - "eta": network predicts eta_hat, then eps_hat = K * eta_hat
        pred = self.eps_model(x_t, t, cond=cond, training=False)  # must be provided
        eps_hat = pred if self.train_target == "eps" else self.kernel.apply(pred)
        mu = sqrt_recip_alpha_t * (x_t - (betas_t / sqrt_one_minus_ab_t) * eps_hat)

        var_t = _extract_t(self.schedule.posterior_variance, t, x_t)

        # stochasticity for t>0
        z = self.noise_model_reverse.sample_like(x_t)
        t_is_zero = tf.reshape(tf.equal(t, 0), (-1, 1, 1))
        return tf.where(t_is_zero, mu, mu + tf.sqrt(var_t) * z)
    
# Kernel operator

class KernelOperator:
    """
    Fixed linear operator K applied along time only (no inter-channel mixing).
    Used to build correlated noise eps = K eta, and to map eta_hat -> eps_hat.
    """
    def apply(self, x: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError


@dataclass
class IdentityKernel(KernelOperator):
    """K = I."""
    def apply(self, x: tf.Tensor) -> tf.Tensor:
        return tf.cast(x, tf.float32)


@dataclass
class TemporalConvKernel(KernelOperator):
    """
    Temporal convolution kernel applied independently per channel.

    Notes
    -----
    - Kernel is L2-normalized (sum k^2 = 1) to keep variance approximately stable:
      Var(K * eta) ≈ Var(eta) when eta ~ N(0, I).
    - Uses depthwise_conv2d trick on (B,T,1,C) to avoid channel mixing.
    """
    kernel: Any  # list/tuple/np array or tf.Tensor
    l2_normalize: bool = True

    def __post_init__(self) -> None:
        k = tf.convert_to_tensor(self.kernel, dtype=tf.float32)
        if k.shape.rank != 1:
            raise ValueError("TemporalConvKernel.kernel must be 1D.")
        if self.l2_normalize:
            norm = tf.sqrt(tf.reduce_sum(tf.square(k))) + 1e-12
            k = k / norm
        self._k = k  # (K,)

    def apply(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.cast(x, tf.float32)
        # Expect shape (B,T,C)
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        C = tf.shape(x)[2]
        K = tf.shape(self._k)[0]

        x4 = tf.reshape(x, (B, T, 1, C))  # (B,T,1,C)
        k = tf.reshape(self._k, (K, 1, 1, 1))        # (K,1,1,1)
        k = tf.tile(k, (1, 1, C, 1))                 # (K,1,C,1)

        y4 = tf.nn.depthwise_conv2d(x4, k, strides=[1, 1, 1, 1], padding="SAME")
        return tf.reshape(y4, (B, T, C))

def make_kernel_operator(cfg: Optional[Dict[str, Any]]) -> KernelOperator:
    """
    Factory for kernel operator.

    Examples
    --------
    None -> IdentityKernel()
    {"type": "identity"}
    {"type": "temporal_conv", "kernel": [0.25, 0.5, 0.25], "l2_normalize": True}
    """
    if cfg is None:
        return IdentityKernel()
    t = str(cfg.get("type", "identity")).lower().strip()
    if t in {"identity", "id", "none"}:
        return IdentityKernel()
    if t in {"temporal_conv", "conv", "kernel"}:
        if "kernel" not in cfg:
            raise ValueError("kernel_cfg for temporal_conv requires 'kernel'.")
        return TemporalConvKernel(kernel=cfg["kernel"], l2_normalize=bool(cfg.get("l2_normalize", True)))
    raise ValueError(f"Unknown kernel operator type: {t}")


# ConditionalMaskingMixin encapsulates measurement-based conditioning
# (mask generation, persistence policy, and projection operators).
# It keeps the conditioning mechanism orthogonal to both the diffusion
# dynamics and the training loop, enabling modular conditional variants
# (inpainting, forecasting, channel masking) without altering the base model.
class ConditionalMaskingMixin:
    """
    Optional mixin to create conditioning tensors (y_obs, mask) from a masking config.
    Requires masking.py providing MaskGenerator/MaskPolicy/MaskContext.
    """
    
    def __init__(
        self,
        *,
        masking_cfg: Optional[Dict[str, Any]] = None,
        mask_policy: str = "per_forward_pass",
        **kwargs):
        
        super().__init__(**kwargs)

        self.masking_cfg = masking_cfg
        self.mask_policy_mode = mask_policy

        self._mask_ctx = None
        if masking_cfg is not None:
            if MaskGenerator is None or MaskPolicy is None or MaskContext is None:
                raise ImportError("masking.py not available/importable but masking_cfg was provided.")
            gen = MaskGenerator(masking_cfg)
            pol = MaskPolicy(mode=mask_policy)
            self._mask_ctx = MaskContext(generator=gen, policy=pol)

    def reset_mask_context(self) -> None:
        """Reset persistent mask cache (useful for per_inference policy)."""
        if self._mask_ctx is not None:
            self._mask_ctx.reset()

    def make_condition(self, x0: tf.Tensor) -> Tuple[Optional[tf.Tensor], Optional[tf.Tensor], Optional[tf.Tensor]]:
        """
        Return (cond, y_obs, mask). If no masking_cfg: (None, None, None).

        Default cond is concat(y_obs, mask) along channels.
        """
        if self._mask_ctx is None:
            return None, None, None

        x0 = tf.cast(x0, tf.float32)
        m = self._mask_ctx.mask(x0)        # (B,T,C)
        y_obs = m * x0                     # (B,T,C)
        cond = tf.concat([y_obs, m], axis=-1)  # (B,T,2C)
        return cond, y_obs, m

    def project_hard(self, x: tf.Tensor, y_obs: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Hard projection: enforce observed entries."""
        x = tf.cast(x, tf.float32)
        y_obs = tf.cast(y_obs, tf.float32)
        mask = tf.cast(mask, tf.float32)
        return mask * y_obs + (1.0 - mask) * x

    def project_soft(self, x: tf.Tensor, y_obs: tf.Tensor, mask: tf.Tensor, sigma_t: tf.Tensor) -> tf.Tensor:
        """
        Soft projection using lambda_t = 1/(1+sigma_t^2).
        sigma_t should be broadcastable to (B,1,1) or (B,T,C).
        """
        x = tf.cast(x, tf.float32)
        y_obs = tf.cast(y_obs, tf.float32)
        mask = tf.cast(mask, tf.float32)
        sigma_t = tf.cast(sigma_t, tf.float32)
        lam = 1.0 / (1.0 + tf.square(sigma_t))
        return lam * mask * y_obs + (1.0 - lam * mask) * x


# -----------------------------------------------------------------------------
# Base Diffusion Model (hook-based training)
# -----------------------------------------------------------------------------

class BaseDiffusionModel(DiffusionScheduleMixin, ConditionalMaskingMixin, BaseKModel):
    """
    Base diffusion model using the same hook structure as BaseKModel:
    forward_and_losses(data) drives train_step/test_step. :contentReference[oaicite:5]{index=5}

    Subclasses MUST define:
    - self.eps_model: callable with signature eps_model(x_t, t, cond=None, training=...)
      returning eps_hat same shape as x_t.

    Data format
    ----------
    - data can be X or (X, y); y is ignored by default.
    - X is expected shape (B,T,C) (time-series windows), but works for any rank>=2
      if your eps_model supports it (here, broadcasting assumes (B,T,C)).
    """

    def __init__(self, *, name: str = "Diffusion", **kwargs):
        super().__init__(name=name, **kwargs)

        # Trackers (aligned with BaseAutoencoder style)
        self.noise_loss_tracker = metrics.Mean(name="noise_loss")
        self.total_loss_tracker = metrics.Mean(name="loss")
        set_ANOMALY_PRESETS(self)

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

    # ----- hook -----

    def forward_and_losses(self, data):
        X = data[0] if isinstance(data, (tuple, list)) else data
        X = tf.cast(X, tf.float32)

        B = tf.shape(X)[0]
        t = tf.random.uniform([B], minval=0, maxval=self.schedule.num_steps, dtype=tf.int32)

        # --- NEW: reparameterized noise path --------------------------------
        # Always sample eta ~ N(0,I). Then eps = K eta.
        # - train_target="eps": target is eps (classic epsilon-parameterization)
        # - train_target="eta": target is eta (whitened noise), while reverse uses eps_hat=K*eta_hat
        eta = tf.random.normal(tf.shape(X), dtype=tf.float32)
        eps = self.kernel.apply(eta)
        x_t = self.q_sample(X, t, noise=eps)
        
        cond, _, _ = self.make_condition(X)
        pred = self.eps_model(x_t, t, cond=cond, training=True)
        target = eps if self.train_target == "eps" else eta
        noise_loss = self._noise_loss_mse(target, pred)
        
        logs = {"noise_loss": noise_loss}
        return noise_loss, logs

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
        sqrt_ab = _extract_t(self.schedule.sqrt_alpha_bars, t, ref)
        sqrt_1mab = _extract_t(self.schedule.sqrt_one_minus_alpha_bars, t, ref)
        sigma = sqrt_1mab / (sqrt_ab + 1e-12)
        B = tf.shape(ref)[0]
        return tf.reshape(tf.cast(sigma, tf.float32), (B, 1, 1))

    def sample_unconditional(
        self,
        shape: Tuple[int, int, int],
        *,
        num_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> tf.Tensor:
        """
        Unconditional sampling: start from x_T ~ N(0,I) and run reverse steps.

        Parameters
        ----------
        shape : tuple (B,T,C)
            Output sample shape.
        num_steps : int, optional
            Number of reverse steps. Defaults to schedule.num_steps.
        seed : int, optional
            Seed for reproducible initialization.

        Returns
        -------
        tf.Tensor
            Sample of shape (B,T,C).
        """
        B, T, C = shape
        if num_steps is None:
            num_steps = int(self.schedule.num_steps)

        if seed is not None:
            g = tf.random.Generator.from_seed(int(seed))
            x = g.normal((B, T, C), dtype=tf.float32)
        else:
            x = tf.random.normal((B, T, C), dtype=tf.float32)

        for step in range(num_steps - 1, -1, -1):
            t = tf.fill([B], tf.cast(step, tf.int32))
            x = self.p_sample_ddpm(x, t, cond=None)
        return x

    def sample_conditional_inpaint(
        self,
        x0_ref: tf.Tensor,
        *,
        hard: bool = True,
        num_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> tf.Tensor:
        """
        Conditional inpainting sampling using the model masking_cfg.

        - mask policy 'per_inference' is recommended: one mask per trajectory.
        - x0_ref is only used to build (y_obs, mask) and to define sample shape.

        Parameters
        ----------
        x0_ref : tf.Tensor
            Reference window (B,T,C) used only to build conditioning.
        hard : bool, default=True
            If True, enforce hard projection. If False, use soft projection with sigma_t.
        num_steps : int, optional
            Number of reverse steps. Defaults to schedule.num_steps.
        seed : int, optional
            Seed for reproducible initialization.

        Returns
        -------
        tf.Tensor
            Reconstructed sample consistent with (y_obs, mask), shape (B,T,C).
        """
        if self.masking_cfg is None:
            raise ValueError("sample_conditional_inpaint requires masking_cfg to be set.")

        x0_ref = tf.cast(x0_ref, tf.float32)
        B = tf.shape(x0_ref)[0]

        if num_steps is None:
            num_steps = int(self.schedule.num_steps)

        # Ensure one mask for this inference trajectory if policy is persistent
        self.reset_mask_context()
        cond, y_obs, mask = self.make_condition(x0_ref)
        if cond is None or y_obs is None or mask is None:
            raise RuntimeError("Masking context did not produce cond/y_obs/mask.")

        # Init from noise
        if seed is not None:
            g = tf.random.Generator.from_seed(int(seed))
            x = g.normal(tf.shape(x0_ref), dtype=tf.float32)
        else:
            x = tf.random.normal(tf.shape(x0_ref), dtype=tf.float32)

        for step in range(num_steps - 1, -1, -1):
            t = tf.fill([B], tf.cast(step, tf.int32))
            x = self.p_sample_ddpm(x, t, cond=cond)

            # Enforce measurement constraint
            if hard:
                x = self.project_hard(x, y_obs, mask)
            else:
                sigma_t = self._sigma_t(t, x)
                x = self.project_soft(x, y_obs, mask, sigma_t=sigma_t)

        return x

    def _broadcast_mask(self, mask: tf.Tensor, B: tf.Tensor) -> tf.Tensor:
        """
        Ensure mask is (B,T,C) by tiling if provided as (1,T,C).
        """
        mask = tf.cast(mask, tf.float32)
        if tf.shape(mask)[0] == 1:
            mask = tf.tile(mask, [B, 1, 1])
        return mask

    def _residual_map(self, x_hat: tf.Tensor, x_ref: tf.Tensor, kind: str) -> tf.Tensor:
        """
        Pointwise residual map.

        kind: "abs" | "sq"
        """
        x_hat = tf.cast(x_hat, tf.float32)
        x_ref = tf.cast(x_ref, tf.float32)
        kind = str(kind).lower().strip()
        if kind == "abs":
            return tf.abs(x_hat - x_ref)
        if kind == "sq":
            return tf.square(x_hat - x_ref)
        raise ValueError("residual_kind must be 'abs' or 'sq'.")

    def _reverse_denoise_from_condition(
        self,
        *,
        x_shape: tf.Tensor,
        cond: tf.Tensor,
        y_obs: tf.Tensor,
        mask: tf.Tensor,
        num_steps: int,
        projection: str,
        seed: Optional[int] = None,
    ) -> tf.Tensor:
        """
        Reverse diffusion given explicit (cond, y_obs, mask). Returns final x_hat.

        Parameters
        ----------
        x_shape : tf.Tensor
            Shape for initialization, typically tf.shape(x_ref).
        cond : tf.Tensor
            Conditioning tensor passed to eps_model (e.g. concat(y_obs, mask)).
        y_obs : tf.Tensor
            Observed values, shape (B,T,C).
        mask : tf.Tensor
            Observation mask, shape (B,T,C).
        num_steps : int
            Number of reverse steps.
        projection : str
            "hard" | "soft".
        seed : int, optional
            Seed for reproducible initialization.

        Returns
        -------
        tf.Tensor
            Final reconstruction x_hat, shape (B,T,C).
        """
        projection = str(projection).lower().strip()
        if projection not in {"hard", "soft"}:
            raise ValueError("projection must be 'hard' or 'soft'.")

        if seed is not None:
            g = tf.random.Generator.from_seed(int(seed))
            x = g.normal(x_shape, dtype=tf.float32)
        else:
            x = tf.random.normal(x_shape, dtype=tf.float32)

        B = tf.shape(x)[0]
        y_obs = tf.cast(y_obs, tf.float32)
        mask = tf.cast(mask, tf.float32)

        for step in range(num_steps - 1, -1, -1):
            t = tf.fill([B], tf.cast(step, tf.int32))
            x = self.p_sample_ddpm(x, t, cond=cond)

            if projection == "hard":
                x = self.project_hard(x, y_obs, mask)
            else:
                sigma_t = self._sigma_t(t, x)
                x = self.project_soft(x, y_obs, mask, sigma_t=sigma_t)

        return x

    # =====================================================================
    # Meta method: anomaly residual map prediction (B,T,C)
    # =====================================================================

    def predict(
        self,
        x_ref: tf.Tensor,
        *,
        n_masks: int = 16,
        num_steps: int = 30,
        projection: str = "hard",      # "hard" | "soft"
        residual_kind: str = "abs",    # "abs" | "sq"
        agg: str = "mean",             # "mean" | "max"
        weight_cfg: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> tf.Tensor:
        """
        Monte-Carlo anomaly residual map aggregation (B,T,C).

        - Draw K masks via make_condition(x_ref) (one per trajectory).
        - Reconstruct x_hat via N reverse steps + projection.
        - Compute residual on held-out region (1 - mask).
        - Aggregate across K with mean or max, optionally weighted by temporal distance.

        Returns
        -------
        tf.Tensor
            Aggregated residual map (B,T,C).
        """
        if self.masking_cfg is None:
            raise ValueError("predict_anomaly_residual_map requires masking_cfg to be set.")

        x_ref = tf.cast(x_ref, tf.float32)
        B = tf.shape(x_ref)[0]

        state = self._init_residual_aggregator(x_ref, agg)

        for k in range(int(n_masks)):
            self.reset_mask_context()
            cond, y_obs, mask = self.make_condition(x_ref)
            if cond is None or y_obs is None or mask is None:
                raise RuntimeError("make_condition did not produce cond/y_obs/mask.")

            mask = self._broadcast_mask(mask, B)  # assume you already have this helper
            y_obs = tf.cast(y_obs, tf.float32)
            holdout = 1.0 - mask

            x_hat = self._reverse_denoise_from_condition(
                x_shape=tf.shape(x_ref),
                cond=cond,
                y_obs=y_obs,
                mask=mask,
                num_steps=int(num_steps),
                projection=str(projection).lower().strip(),
                seed=None if seed is None else int(seed) + k,
            )

            resid = self._residual_map(x_hat, x_ref, residual_kind)
            weight = self._aggregation_weight_map(mask, weight_cfg)

            state = self._update_residual_aggregator(
                state, resid=resid, holdout=holdout, weight=weight
            )

        return self._finalize_residual_aggregator(state)    
# =====================================================================
# Distance & poids (brique indépendante)
# =====================================================================


    def _temporal_nearest_observed_distance(self, mask: tf.Tensor) -> tf.Tensor:
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

    def _distance_weight(self, d: tf.Tensor, *, mode: str, tau: float) -> tf.Tensor:
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

    def _aggregation_weight_map(self, mask: tf.Tensor, cfg: Optional[Dict[str, Any]]) -> tf.Tensor:
        """
        Compute per-point aggregation weights for a given mask.

        If cfg is None or disabled, returns ones (no weighting).
        """
        if not cfg or not bool(cfg.get("enabled", False)):
            return tf.ones_like(tf.cast(mask, tf.float32))

        kind = str(cfg.get("kind", "temporal_nearest_observed")).lower().strip()
        if kind != "temporal_nearest_observed":
            raise ValueError("Unknown weight kind. Supported: 'temporal_nearest_observed'.")

        mode = str(cfg.get("mode", "exp"))
        tau = float(cfg.get("tau", 5.0))

        d = self._temporal_nearest_observed_distance(mask)
        return self._distance_weight(d, mode=mode, tau=tau)

# =====================================================================
# Agrégateur multi-masques (mean/max) avec pondération optionnelle
# =====================================================================

    def _init_residual_aggregator(self, x_ref: tf.Tensor, agg: str):
        agg = str(agg).lower().strip()
        if agg == "mean":
            return dict(
                agg="mean",
                accum=tf.zeros_like(x_ref, tf.float32),
                denom=tf.zeros_like(x_ref, tf.float32),
            )
        if agg == "max":
            return dict(
                agg="max",
                current=tf.fill(tf.shape(x_ref), tf.constant(-1e30, tf.float32)),
            )
        raise ValueError("agg must be 'mean' or 'max' (TFP-free).")

    def _update_residual_aggregator(
        self,
        state: Dict[str, tf.Tensor],
        *,
        resid: tf.Tensor,
        holdout: tf.Tensor,
        weight: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """
        Update aggregator state with one mask draw.
        """
        resid = tf.cast(resid, tf.float32)
        holdout = tf.cast(holdout, tf.float32)
        weight = tf.cast(weight, tf.float32)

        if state["agg"] == "mean":
            w = holdout * weight
            state["accum"] = state["accum"] + resid * w
            state["denom"] = state["denom"] + w
            return state

        # max: ignore observed points (holdout==0) by -inf
        neg_inf = tf.constant(-1e30, tf.float32)
        cand = tf.where(holdout > 0.0, resid * weight, neg_inf)
        state["current"] = tf.maximum(state["current"], cand)
        return state

    def _finalize_residual_aggregator(self, state: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Finalize aggregator state into an aggregated residual map (B,T,C).
        """
        if state["agg"] == "mean":
            return state["accum"] / (state["denom"] + 1e-12)

        # max: points never held out remain -inf -> set to 0
        cur = state["current"]
        return tf.where(cur < -1e20, tf.zeros_like(cur), cur)

def set_ANOMALY_PRESETS(self):
    self._ANOMALY_PRESETS = {
    "mc_holdout_mean": dict(
        n_masks=40,
        num_steps=50,
        projection="hard",     # "hard" | "soft"
        residual_kind="abs",   # "abs" | "sq"
        agg="mean",            # "mean" | "max"
        seed=None,
    ),
    "mc_holdout_max": dict(
        n_masks=40,
        num_steps=50,
        projection="hard",
        residual_kind="abs",
        agg="max",
        seed=None,
    )}