"""
diffusion_masking.py
====================

Conditioning and masking primitives for time-series diffusion models.

This module provides the core utilities used to build conditional diffusion
inputs for inpainting / forecasting / partial observation settings.

Main components
---------------
- ConditionalMaskingMixin:
    Utilities to build conditioning tensors (e.g., concat(y_obs, mask)),
    apply masking policies, and enforce observed-point consistency when needed.

Design principles
-----------------
- Masking/conditioning is independent from:
    * diffusion mathematics (see diffusion_schedule.py),
    * model orchestration (see base_diffusion.py),
    * post-run reductions (see diffusion_reducer.py).
- The goal is to keep conditioning logic reusable across tasks and denoiser
  architectures, while remaining compatible with the runner payload contract.

Notes
-----
- Mask sweep behavior may rely on internal randomness of the masking config,
  depending on the chosen masking policy.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable
import tensorflow as tf
from uqmodels.modelization.TF_estimator.base.masking import MaskContext,MaskGenerator,MaskPolicy

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
        Heuristic soft projection using lambda_t = 1/(1+sigma_t^2).
        sigma_t should be broadcastable to (B,1,1) or (B,T,C).
        """
        x = tf.cast(x, tf.float32)
        y_obs = tf.cast(y_obs, tf.float32)
        mask = tf.cast(mask, tf.float32)
        sigma_t = tf.cast(sigma_t, tf.float32)
        lam = 1.0 / (1.0 + tf.square(sigma_t))
        return lam * mask * y_obs + (1.0 - lam * mask) * x