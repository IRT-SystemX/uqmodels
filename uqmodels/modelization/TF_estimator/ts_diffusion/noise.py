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