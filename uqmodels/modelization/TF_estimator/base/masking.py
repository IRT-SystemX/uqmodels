"""
masking.py
==========

A small, reusable masking “brick” for time-series / tensor data, designed to
support multiple use-cases with different mask lifecycles (e.g., dropout-like
non-persistent masks vs diffusion-like persistent masks), while sharing the same
mask generation strategies.

Conceptual layers
-----------------
1) Mask generation (strategy):
   - How a mask is sampled (iid, time-blocks, channel-dropout, forecasting split, ...).
   - Produces a binary mask m in {0,1} with shape (B, T, C).

2) Mask combination (composition):
   - Combine several strategies into a single mask (AND / OR / SEQUENCE).

3) Mask lifecycle (policy):
   - When a mask is resampled, and how long it persists:
     - per_forward_pass: new mask each call (dropout-style)
     - per_inference: one mask for a whole inference trajectory (diffusion-style)
     - per_batch: one mask per batch context
     - per_step: new mask each step (rare; can be used for experiments)

4) Mask application semantics (operators):
   - Dropout application: x <- m * x
   - Measurement operator: y = M(x) (here: diagonal mask)
   - Projection / clamping: x <- m*y_obs + (1-m)*x (hard projection)
   - Soft projection (optional): weighted projection depending on sigma_t

Included mask strategies
------------------------
- "iid":              element-wise Bernoulli keep_prob
- "channel_dropout":  drop full channels per sample with keep_prob
- "time_blocks":      mask contiguous time segments (n_blocks, block_size)
- "forecasting":      keep first 'past' steps, mask the rest
- "per_channel_iid":  per-channel Bernoulli with keep_probs[C]

Main entry points
-----------------
- apply_masking(x, cfg) -> (y_obs, mask)
- MaskGenerator(strategy_cfg).sample(x) -> mask
- MaskPolicy(mode).get_mask(generator, x) -> mask (handles persistence)
- MaskContext(generator, policy): convenience for dropout/projection usage

Notes
-----
- All outputs are float32 tensors.
- Shapes assume (B, T, C). The code is agnostic to the meaning of T/C.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import tensorflow as tf


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _as_float32(x: tf.Tensor) -> tf.Tensor:
    """Cast tensor to float32."""
    return tf.cast(x, tf.float32)


def _check_3d(x: tf.Tensor, name: str = "x") -> None:
    """Best-effort shape check for (B,T,C)."""
    if x.shape.rank is not None and x.shape.rank != 3:
        raise ValueError(f"{name} must have rank 3 (B,T,C). Got rank={x.shape.rank} shape={x.shape}.")


def _extract_kwargs(d: Dict[str, Any], drop_keys: Tuple[str, ...] = ("type",)) -> Dict[str, Any]:
    """Return dict without specific keys."""
    return {k: v for k, v in d.items() if k not in drop_keys}


# ---------------------------------------------------------------------
# Mask strategies (each returns mask m in {0,1} with shape (B,T,C))
# ---------------------------------------------------------------------

def mask_iid(x: tf.Tensor, *, keep_prob: float) -> tf.Tensor:
    """IID masking: each entry kept with probability keep_prob."""
    _check_3d(x)
    x = _as_float32(x)
    kp = tf.cast(keep_prob, tf.float32)
    return tf.cast(tf.random.uniform(tf.shape(x)) < kp, tf.float32)


def mask_channel_dropout(x: tf.Tensor, *, keep_prob: float) -> tf.Tensor:
    """Channel dropout: keep entire channels with probability keep_prob (per sample)."""
    _check_3d(x)
    x = _as_float32(x)
    B = tf.shape(x)[0]
    T = tf.shape(x)[1]
    C = tf.shape(x)[2]
    kp = tf.cast(keep_prob, tf.float32)
    keep = tf.cast(tf.random.uniform((B, 1, C)) < kp, tf.float32)
    return tf.tile(keep, [1, T, 1])


def mask_time_blocks(
    x: tf.Tensor,
    *,
    block_size: int,
    n_blocks: int = 1,
    same_blocks_across_channels: bool = True,
) -> tf.Tensor:
    """
    Time-block masking: set contiguous time segments to 0.

    Parameters
    ----------
    block_size : int
        Length of each masked time block.
    n_blocks : int
        Number of blocks per sample.
    same_blocks_across_channels : bool
        If True, the same time blocks are applied to all channels for a given sample.
        If False, blocks are sampled independently per (sample, channel).
    """
    _check_3d(x)
    x = _as_float32(x)
    B = tf.shape(x)[0]
    T = tf.shape(x)[1]
    C = tf.shape(x)[2]

    block_size = tf.cast(block_size, tf.int32)
    n_blocks = tf.cast(n_blocks, tf.int32)

    def build_time_mask_1d(_: tf.Tensor) -> tf.Tensor:
        # start indices in [0, T-block_size]
        max_start = tf.maximum(1, T - block_size + 1)
        starts = tf.random.uniform((n_blocks,), minval=0, maxval=max_start, dtype=tf.int32)
        time = tf.range(T, dtype=tf.int32)  # (T,)

        in_any_block = tf.reduce_any(
            (time[None, :] >= starts[:, None]) & (time[None, :] < (starts[:, None] + block_size)),
            axis=0,
        )
        return tf.cast(~in_any_block, tf.float32)  # (T,) 1 kept, 0 masked

    if same_blocks_across_channels:
        time_masks = tf.map_fn(
            build_time_mask_1d,
            tf.range(B),
            fn_output_signature=tf.float32,
        )  # (B,T)
        return tf.tile(time_masks[:, :, None], [1, 1, C])

    # independent per (B,C)
    bc = tf.range(B * C)
    time_masks = tf.map_fn(build_time_mask_1d, bc, fn_output_signature=tf.float32)  # (B*C, T)
    time_masks = tf.reshape(time_masks, (B, C, T))  # (B,C,T)
    return tf.transpose(time_masks, (0, 2, 1))  # (B,T,C)


def mask_forecasting(x: tf.Tensor, *, past: int) -> tf.Tensor:
    """Forecasting mask: keep [0:past) and mask [past:T)."""
    _check_3d(x)
    x = _as_float32(x)
    B = tf.shape(x)[0]
    T = tf.shape(x)[1]
    C = tf.shape(x)[2]
    past = tf.cast(past, tf.int32)
    past = tf.clip_by_value(past, 0, T)

    m_past = tf.ones((B, past, C), dtype=tf.float32)
    m_fut = tf.zeros((B, T - past, C), dtype=tf.float32)
    return tf.concat([m_past, m_fut], axis=1)


def mask_per_channel_iid(x: tf.Tensor, *, keep_probs) -> tf.Tensor:
    """
    Per-channel IID masking: each channel has its own keep probability.

    keep_probs: list/tuple/np array length C, or tf.Tensor shape (C,)
    """
    _check_3d(x)
    x = _as_float32(x)
    keep_probs = tf.convert_to_tensor(keep_probs, dtype=tf.float32)  # (C,)
    p = keep_probs[None, None, :]  # (1,1,C)
    return tf.cast(tf.random.uniform(tf.shape(x)) < p, tf.float32)


MASK_REGISTRY: Dict[str, Callable[..., tf.Tensor]] = {
    "iid": mask_iid,
    "channel_dropout": mask_channel_dropout,
    "time_blocks": mask_time_blocks,
    "forecasting": mask_forecasting,
    "per_channel_iid": mask_per_channel_iid,
}


# ---------------------------------------------------------------------
# Mask composition
# ---------------------------------------------------------------------

def combine_masks_and(masks: List[tf.Tensor]) -> tf.Tensor:
    """Intersection: stricter masking (multiply binary masks)."""
    if not masks:
        raise ValueError("combine_masks_and: empty masks list.")
    m = tf.ones_like(masks[0], dtype=tf.float32)
    for mi in masks:
        m = m * tf.cast(mi, tf.float32)
    return m


def combine_masks_or(masks: List[tf.Tensor]) -> tf.Tensor:
    """Union: more permissive masking (1 - Π(1-mi))."""
    if not masks:
        raise ValueError("combine_masks_or: empty masks list.")
    inv = tf.ones_like(masks[0], dtype=tf.float32)
    for mi in masks:
        inv = inv * (1.0 - tf.cast(mi, tf.float32))
    return 1.0 - inv


def apply_masking(x: tf.Tensor, cfg: Dict[str, Any]) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Apply configurable masking strategies.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor, shape (B,T,C).
    cfg : dict
        Example:
          {
            "combine": "and" | "or" | "sequence",
            "transforms": [
              {"type": "iid", "keep_prob": 0.8},
              {"type": "time_blocks", "block_size": 12, "n_blocks": 2, "same_blocks_across_channels": True},
              ...
            ],
          }

    Returns
    -------
    y_obs : tf.Tensor
        Masked observations, shape (B,T,C), float32.
    mask : tf.Tensor
        Binary mask in {0,1}, shape (B,T,C), float32.
    """
    _check_3d(x)
    x = _as_float32(x)

    transforms = cfg.get("transforms", [])
    if not transforms:
        mask = tf.ones_like(x, dtype=tf.float32)
        return x, mask

    mode = str(cfg.get("combine", "and")).lower()
    if mode not in {"and", "or", "sequence"}:
        raise ValueError(f"Unknown combine mode '{mode}'. Use 'and', 'or', or 'sequence'.")

    if mode == "sequence":
        # sequentially refine a global mask; each transform sees y (current masked view) by default
        mask = tf.ones_like(x, dtype=tf.float32)
        y = x
        for tr in transforms:
            ttype = tr["type"]
            fn = MASK_REGISTRY.get(ttype)
            if fn is None:
                raise ValueError(f"Unknown masking type '{ttype}'. Available: {sorted(MASK_REGISTRY)}")
            mi = fn(y, **_extract_kwargs(tr))
            mask = mask * tf.cast(mi, tf.float32)
            y = mask * x
        return y, mask

    masks = []
    for tr in transforms:
        ttype = tr["type"]
        fn = MASK_REGISTRY.get(ttype)
        if fn is None:
            raise ValueError(f"Unknown masking type '{ttype}'. Available: {sorted(MASK_REGISTRY)}")
        masks.append(fn(x, **_extract_kwargs(tr)))

    mask = combine_masks_and(masks) if mode == "and" else combine_masks_or(masks)
    y_obs = mask * x
    return y_obs, mask


# ---------------------------------------------------------------------
# Lifecycle / persistence policy
# ---------------------------------------------------------------------

class MaskPolicy:
    """
    Controls when masks are resampled and how long they persist.

    Modes
    -----
    - per_forward_pass: resample at each get_mask call (dropout-style).
    - per_inference:    sample once, reuse until reset() (diffusion trajectory).
    - per_batch:        sample once, reuse until reset() (batch-scoped).
    - per_step:         resample every call (alias of per_forward_pass, semantically step-based).
    """

    def __init__(self, mode: str = "per_forward_pass"):
        self.mode = str(mode).lower()
        self._cached: Optional[tf.Tensor] = None

    def reset(self) -> None:
        """Clear cached mask (relevant for persistent modes)."""
        self._cached = None

    def get_mask(self, generator: "MaskGenerator", x: tf.Tensor) -> tf.Tensor:
        """Return a mask according to the policy."""
        if self.mode in {"per_forward_pass", "per_step"}:
            return generator.sample(x)

        if self.mode in {"per_inference", "per_batch"}:
            if self._cached is None:
                self._cached = generator.sample(x)
            return self._cached

        raise ValueError(
            f"Unknown MaskPolicy mode '{self.mode}'. "
            "Use: per_forward_pass, per_inference, per_batch, per_step."
        )


# ---------------------------------------------------------------------
# Generator + Context
# ---------------------------------------------------------------------

@dataclass
class MaskGenerator:
    """
    A thin wrapper around apply_masking() that produces only the mask.
    """
    strategy_cfg: Dict[str, Any]

    def sample(self, x: tf.Tensor) -> tf.Tensor:
        """Sample a mask (float32, shape (B,T,C))."""
        _check_3d(x)
        x = _as_float32(x)
        _, m = apply_masking(x, self.strategy_cfg)
        return tf.cast(m, tf.float32)


@dataclass
class MaskContext:
    """
    Convenience object combining generator + policy + common mask applications.
    """
    generator: MaskGenerator
    policy: MaskPolicy

    def reset(self) -> None:
        self.policy.reset()

    def mask(self, x: tf.Tensor) -> tf.Tensor:
        """Get a mask consistent with the policy."""
        return self.policy.get_mask(self.generator, x)

    def apply_dropout(self, x: tf.Tensor) -> tf.Tensor:
        """Dropout-style masking: x <- m * x."""
        x = _as_float32(x)
        m = self.mask(x)
        return m * x

    def project_hard(self, x: tf.Tensor, y_obs: tf.Tensor) -> tf.Tensor:
        """Hard projection / clamping: x <- m*y_obs + (1-m)*x."""
        x = _as_float32(x)
        y_obs = _as_float32(y_obs)
        m = self.mask(x)
        return m * y_obs + (1.0 - m) * x

    def project_soft(self, x: tf.Tensor, y_obs: tf.Tensor, sigma_t: tf.Tensor) -> tf.Tensor:
        """
        Soft projection: weight the clamping according to noise level sigma_t.

        lambda_t = 1 / (1 + sigma_t^2)
        x <- lambda_t*m*y_obs + (1 - lambda_t*m)*x

        sigma_t can be scalar or broadcastable to (B,1,1) / (B,T,C).
        """
        x = _as_float32(x)
        y_obs = _as_float32(y_obs)
        sigma_t = tf.cast(sigma_t, tf.float32)
        lam = 1.0 / (1.0 + tf.square(sigma_t))
        m = self.mask(x)
        return lam * m * y_obs + (1.0 - lam * m) * x


# ---------------------------------------------------------------------
# Measurement operator (optional but often handy for inverse problems)
# ---------------------------------------------------------------------

@dataclass
class DiagonalMaskOperator:
    """
    Measurement operator y = M(x) with M being a diagonal mask.

    - forward:  y = m * x
    - adjoint:  M^T = M for diagonal masks
    - project:  hard clamping onto observed entries
    """
    mask: tf.Tensor  # (B,T,C), float32 in {0,1}

    def __post_init__(self) -> None:
        _check_3d(self.mask, "mask")
        self.mask = tf.cast(self.mask, tf.float32)

    def forward(self, x: tf.Tensor) -> tf.Tensor:
        x = _as_float32(x)
        return self.mask * x

    def adjoint(self, x: tf.Tensor) -> tf.Tensor:
        x = _as_float32(x)
        return self.mask * x

    def project(self, x: tf.Tensor, y_obs: tf.Tensor) -> tf.Tensor:
        x = _as_float32(x)
        y_obs = _as_float32(y_obs)
        return self.mask * y_obs + (1.0 - self.mask) * x


# ---------------------------------------------------------------------
# Example configs (keep as reference)
# ---------------------------------------------------------------------

EXAMPLE_CFG_INPAINT_MIXED: Dict[str, Any] = {
    "combine": "and",
    "transforms": [
        {"type": "time_blocks", "block_size": 12, "n_blocks": 2, "same_blocks_across_channels": True},
        {"type": "channel_dropout", "keep_prob": 0.9},
        {"type": "iid", "keep_prob": 0.8},
    ],
}

EXAMPLE_CFG_FORECASTING: Dict[str, Any] = {
    "combine": "and",
    "transforms": [
        {"type": "forecasting", "past": 48},
    ],
}

EXAMPLE_CFG_PER_CHANNEL: Dict[str, Any] = {
    "combine": "and",
    "transforms": [
        {"type": "per_channel_iid", "keep_probs": [0.95, 0.7, 0.6]},
        {"type": "time_blocks", "block_size": 8, "n_blocks": 1, "same_blocks_across_channels": True},
	],
}