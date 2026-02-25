"""
diffusion_noise.py
==================

Forward noise models for time-series diffusion.

This module defines the primitives responsible for sampling the epsilon term
used in the forward diffusion process. A noise model generates a tensor
matching the shape of a reference input, typically (B, T, C).

Provided components
-------------------
- NoiseModel:
    Base interface for shape-driven noise sampling.

- WhiteNoise:
    i.i.d. Gaussian noise N(0, std^2 I).

- TemporalConvFilteredWhiteNoise:
    White noise filtered by a fixed 1D temporal kernel (no inter-channel mixing).
    The kernel is L2-normalized to approximately preserve per-channel variance.

- Optional kernel utilities:
    Structured eta -> eps transformations and diagonal variance propagation
    for approximate uncertainty handling.
    - KernelOperator:
        Base interface for kernelized noise transforms.
    - IdentityKernel:
        White noise (classic DDPM behavior).
    - TemporalConvKernel:
        Depthwise temporal convolution kernel to induce time correlation.


Design notes
------------
- Noise generation is stateless apart from configuration parameters.
- All outputs are float32 tensors matching the reference input shape.
- This module only concerns forward noise sampling.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence
import tensorflow as tf

class NoiseModel:
    """Interface for noise sampling. Output shape matches the provided reference tensor."""
    def sample_like(self, x: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError


@dataclass
class WhiteNoise(NoiseModel):
    """i.i.d. Gaussian white noise: N(0, I)."""
    std: float = 1.0

    def sample_like(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.cast(x, tf.float32)
        return tf.random.normal(tf.shape(x), mean=0.0, stddev=float(self.std), dtype=tf.float32)


@dataclass
class TemporalConvFilteredWhiteNoise(NoiseModel):
    """
    White noise filtered by a fixed 1D convolution kernel along time ONLY (no inter-channel mixing).

    Implementation details
    ----------------------
    - Generate eta ~ N(0, I) then apply a depthwise conv along the time axis per channel.
    - Kernel is normalized to unit L2 (sum k^2 = 1) so the output variance remains ~1 (per channel).
    """
    kernel: Sequence[float]
    std: float = 1.0

    def __post_init__(self) -> None:
        k = tf.convert_to_tensor(self.kernel, dtype=tf.float32)
        if k.shape.rank != 1:
            raise ValueError("kernel must be 1D.")
        # L2 normalize to keep Var approximately stable: Var(K * eta) ≈ ||K||_2^2 Var(eta)
        norm = tf.sqrt(tf.reduce_sum(tf.square(k))) + 1e-12
        self._k = k / norm  # (K,)

    def sample_like(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.cast(x, tf.float32)
        eta = tf.random.normal(tf.shape(x), mean=0.0, stddev=float(self.std), dtype=tf.float32)  # (B,T,C)

        B = tf.shape(eta)[0]
        T = tf.shape(eta)[1]
        C = tf.shape(eta)[2]
        K = tf.shape(self._k)[0]

        # depthwise conv2d trick to get per-channel independent temporal conv:
        # reshape (B,T,C) -> (B,T,1,C)
        eta_4d = tf.reshape(eta, (B, T, 1, C))

        # filter shape for depthwise_conv2d: (K, 1, in_channels, channel_multiplier)
        # We replicate the same temporal kernel for each channel.
        k = tf.reshape(self._k, (K, 1, 1, 1))                # (K,1,1,1)
        k = tf.tile(k, (1, 1, C, 1))                         # (K,1,C,1)

        out = tf.nn.depthwise_conv2d(
            eta_4d,
            k,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )  # (B,T,1,C)

        return tf.reshape(out, (B, T, C))


def make_noise_model(cfg: Optional[Dict[str, Any]]) -> NoiseModel:
    """
    Factory for noise models.

    Example cfg
    -----------
    {"type": "white", "std": 1.0}
    {"type": "temporal_conv", "kernel": [0.25, 0.5, 0.25], "std": 1.0}
    """
    if cfg is None:
        return WhiteNoise(std=1.0)

    t = str(cfg.get("type", "white")).lower()
    if t == "white":
        return WhiteNoise(std=float(cfg.get("std", 1.0)))
    if t in {"temporal_conv", "conv", "filtered_white"}:
        kernel = cfg.get("kernel", None)
        if kernel is None:
            raise ValueError("temporal_conv noise requires 'kernel'.")
        return TemporalConvFilteredWhiteNoise(kernel=kernel, std=float(cfg.get("std", 1.0)))

    raise ValueError(f"Unknown noise type: {t}")

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable


import tensorflow as tf
from uqmodels.modelization.TF_estimator.base.masking import MaskGenerator, MaskPolicy, MaskContext  # type: ignore

class KernelOperator:
    """
    Fixed linear operator K applied along time only (no inter-channel mixing).
    Used to build correlated noise eps = K eta, and to map eta_hat -> eps_hat.
    """
    def apply(self, x: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError
    
    def apply_variance(self, var: tf.Tensor) -> tf.Tensor:
        """
        Propagate a diagonal (per-time, per-channel) variance through the kernel operator.
        Args: var: Tensor of shape (B, T, C) representing Var(input).
        Returns: Tensor of shape (B, T, C) representing an approximation of Var(K(input)).
        """
        raise NotImplementedError

@dataclass
class IdentityKernel(KernelOperator):
    """K = I."""
    def apply(self, x: tf.Tensor) -> tf.Tensor:
        return tf.cast(x, tf.float32)
    
    def apply_variance(self, var: tf.Tensor) -> tf.Tensor:
        return var

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
    
    def apply_variance(self, var: tf.Tensor) -> tf.Tensor:
        """
        Approximate variance propagation for depthwise temporal convolution:
            eps = conv_w(eta)
            Var(eps_t) ≈ sum_k w_k^2 Var(eta_{t-k})
            Assumes input noise is temporally independent (diagonal covariance).

        Warning : depthwise_kernel and keras version may cause issues.
        """
        var = tf.cast(var, tf.float32)
        
        K = tf.shape(self._k)[0]
        B = tf.shape(var)[0]
        T = tf.shape(var)[1]
        C = tf.shape(var)[2]

        # Build a depthwise kernel with squared weights
        k = tf.reshape(self._k, (K, 1, 1, 1))
        k = tf.tile(k, (1, 1, C, 1))
        w2 = tf.square(k)

        # Depthwise conv in 1D expects (B, T, C) input for DepthwiseConv1D call
        # We apply the same op as forward but with squared weights.
        # Workaround: use tf.nn.depthwise_conv2d by expanding a dummy spatial dim.
        x = tf.expand_dims(var, axis=2)  # (B, T, 1, C)
        # tf.nn.depthwise_conv2d expects filter shape [fh, fw, inC, channel_multiplier]
        # We want to convolve along T => fh=K, fw=1, inC=C, mult=1
        filt = w2  # already (K, 1, C, 1)
        y = tf.nn.depthwise_conv2d(
            x,
            filt,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )  # (B, T, 1, C)
        return tf.squeeze(y, axis=2)  # (B, T, C)

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
