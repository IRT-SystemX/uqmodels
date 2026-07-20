"""
Convolutional layers and sub-networks.

Provides reusable 1D/2D convolutional blocks and configurable CNN
sub-networks for encoder and decoder architectures.
"""

from math import ceil
from typing import Literal
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from uqmodels.utils import add_random_state
from uqmodels.modelization.TF_estimator.layers.layers import normalize_type_output,MLPBlock,DenseHeadBlock,ProbProcessingLayers,EDLProcessingLayers

@tf.keras.utils.register_keras_serializable(package="UQModels_layers")
class ConvBlock1D(layers.Layer):
    def __init__(
        self,
        num_channels: int,
        filters: int = 32,
        kernel: int = 2,
        strides: int = 2,
        dp: float = 0.02,
        mc_dropout: bool = False,
        random_state: int | None = None,
        padding: str = "causal",
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_channels = int(num_channels)
        self.filters = int(filters)
        self.kernel = int(kernel)
        self.strides = int(strides)
        self.dp = float(dp)
        self.mc_dropout = bool(mc_dropout)
        self.random_state = random_state
        self.padding = padding
        self.activation = activation

        self.conv = layers.Conv1D(
            filters * num_channels,
            kernel,
            strides=strides,
            padding=padding,
            groups=num_channels,
            activation=activation,
        )
        self.bn = layers.BatchNormalization()
        self.drop = layers.Dropout(dp, seed=random_state) if dp and dp > 0 else None

    def call(self, x, training=False):
        y = self.conv(x)
        y = self.bn(y, training=training)
        if self.drop is not None:
            y = self.drop(y, training=(training or self.mc_dropout))
        return y

    def get_config(self):
        return {
            **super().get_config(),
            "num_channels": self.num_channels,
            "filters": self.filters,
            "kernel": self.kernel,
            "strides": self.strides,
            "dp": self.dp,
            "mc_dropout": self.mc_dropout,
            "random_state": self.random_state,
            "padding": self.padding,
            "activation": self.activation,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="UQModels_layers")
class ConvBlock2D(layers.Layer):
    def __init__(
        self,
        num_channels: int,
        filters: int = 32,
        kernel: int | tuple[int, int] = 5,
        strides: int | tuple[int, int] = (2, 1),
        dp: float = 0.02,
        mc_dropout: bool = False,
        random_state: int | None = None,
        padding: str = "valid",
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_channels = int(num_channels)
        self.filters = int(filters)
        self.kernel = kernel
        self.strides = strides
        self.dp = float(dp)
        self.mc_dropout = bool(mc_dropout)
        self.random_state = random_state
        self.padding = padding
        self.activation = activation

        self.conv = layers.Conv2D(
            filters, kernel, strides=strides, padding=padding, activation=activation
        )
        self.bn = layers.BatchNormalization()
        self.drop = layers.Dropout(dp, seed=random_state) if dp and dp > 0 else None

    def call(self, x, training=False):
        y = self.conv(x)
        y = self.bn(y, training=training)
        if self.drop is not None:
            y = self.drop(y, training=(training or self.mc_dropout))
        return y

    def get_config(self):
        return {
            **super().get_config(),
            "num_channels": self.num_channels,
            "filters": self.filters,
            "kernel": self.kernel,
            "strides": self.strides,
            "dp": self.dp,
            "mc_dropout": self.mc_dropout,
            "random_state": self.random_state,
            "padding": self.padding,
            "activation": self.activation,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package="UQModels_layers")
class TConvBlock1D(layers.Layer):
    def __init__(
        self,
        dim_out: int,
        filters: int = 32,
        kernel: int = 2,
        strides: int = 2,
        dp: float = 0.02,
        mc_dropout: bool = False,
        random_state: int | None = None,
        padding: str = "same",
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim_out = int(dim_out)
        self.filters = int(filters)
        self.kernel = int(kernel)
        self.strides = int(strides)
        self.dp = float(dp)
        self.mc_dropout = bool(mc_dropout)
        self.random_state = random_state
        self.padding = padding
        self.activation = activation

        self.tconv = layers.Conv1DTranspose(
            filters * dim_out,
            kernel,
            strides=strides,
            padding=padding,
            activation=activation,
        )
        self.bn = layers.BatchNormalization()
        self.drop = layers.Dropout(dp, seed=random_state) if dp and dp > 0 else None

    def call(self, x, training=False):
        y = self.tconv(x)
        y = self.bn(y, training=training)
        if self.drop is not None:
            y = self.drop(y, training=(training or self.mc_dropout))
        return y

    def get_config(self):
        return {
            **super().get_config(),
            "dim_out": self.dim_out,
            "filters": self.filters,
            "kernel": self.kernel,
            "strides": self.strides,
            "dp": self.dp,
            "mc_dropout": self.mc_dropout,
            "random_state": self.random_state,
            "padding": self.padding,
            "activation": self.activation,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package="UQModels_layers")
class TConvBlock2D(layers.Layer):
    def __init__(
        self,
        dim_out: int,
        filters: int = 32,
        kernel: int = 5,
        strides: int | tuple[int, int] = (2, 1),
        dp: float = 0.02,
        mc_dropout: bool = False,
        random_state: int | None = None,
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim_out = int(dim_out)
        self.filters = int(filters)
        self.kernel = int(kernel)
        self.strides = strides
        self.dp = float(dp)
        self.mc_dropout = bool(mc_dropout)
        self.random_state = random_state
        self.activation = activation

        # NB: respecte exactement ton implémentation (kernel, dim_out)
        self.tconv = layers.Conv2DTranspose(
            filters, (kernel, dim_out), strides=strides, activation=activation
        )
        self.bn = layers.BatchNormalization()
        self.drop = layers.Dropout(dp, seed=random_state) if dp and dp > 0 else None

    def call(self, x, training=False):
        y = self.tconv(x)
        y = self.bn(y, training=training)
        if self.drop is not None:
            y = self.drop(y, training=(training or self.mc_dropout))
        return y

    def get_config(self):
        return {
            **super().get_config(),
            "dim_out": self.dim_out,
            "filters": self.filters,
            "kernel": self.kernel,
            "strides": self.strides,
            "dp": self.dp,
            "mc_dropout": self.mc_dropout,
            "random_state": self.random_state,
            "activation": self.activation,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# ---------------------------------
#   CNNSubNet
# ---------------------------------

@tf.keras.utils.register_keras_serializable(package="UQModels_layers")
class CNNSubNet(layers.Layer):
    """
    Configurable convolutional sub-network.

    Encoder
    -------
    Input
        -> convolution blocks
        -> flatten
        -> optional MLP block
        -> DenseHeadBlock

    Decoder
    -------
    Latent input
        -> optional MLP block
        -> dense projection
        -> reshape
        -> transposed convolution blocks
        -> DenseHeadBlock
    """

    VALID_MODES = {"encoder", "decoder"}
    VALID_BLOCKS = {"1D", "2D"}

    def __init__(
        self,
        mode: str,
        dim_seq: int,
        dim_in: int | None = None,
        dim_out: int | None = None,
        num_channels: int = 1,
        list_filters: list[int] | tuple[int, ...] = (64, 64, 32),
        list_kernels: list | tuple = (4, 4, 4),
        list_strides: list | tuple = (2, 2, 2),
        block: str = "1D",
        dim_z: int = 200,
        dp: float = 0.02,
        type_output: str | None = None,
        logvar_min: float = -6.0,
        cfg_mlp: dict | None = None,
        cfg_head: dict | None = None,
        random_state: int | None = None,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.mode = mode
        self.dim_seq = int(dim_seq)

        self.dim_in = (
            None if dim_in is None else int(dim_in)
        )

        self.dim_out = (
            None if dim_out is None else int(dim_out)
        )

        self.num_channels = int(num_channels)

        self.list_filters = list(list_filters)
        self.list_kernels = list(list_kernels)
        self.list_strides = list(list_strides)

        self.block = block
        self.dim_z = int(dim_z)

        self.dp = float(dp)
        self.type_output = normalize_type_output(type_output)
        self.logvar_min = float(logvar_min)

        self.cfg_mlp = (
            None
            if cfg_mlp is None
            else dict(cfg_mlp)
        )

        self.cfg_head = (
            {}
            if cfg_head is None
            else dict(cfg_head)
        )

        self.random_state = random_state

        self.mc_dropout = (
            self.type_output == "mc_dropout"
        )

        self._validate_config()
        self._prepare_block_configs()

        if self.mode == "encoder":
            self._build_encoder()
        else:
            self._build_decoder()

    def _prepare_block_configs(self) -> None:
        """Prepare internal block configurations."""

        if self.cfg_mlp is not None:
            self.cfg_mlp.setdefault(
                "mc_dropout",
                self.mc_dropout,
            )

            self.cfg_mlp.setdefault(
                "random_state",
                self.random_state,
            )

            self.cfg_mlp.setdefault(
                "name",
                f"{self.mode}_mlp_block",
            )

        head_dim_out = (
            self.dim_z
            if self.mode == "encoder"
            else self.dim_out
        )

        self.cfg_head.setdefault(
            "dim_out",
            head_dim_out,
        )

        self.cfg_head.setdefault(
            "type_output",
            self.type_output,
        )

        self.cfg_head.setdefault(
            "logvar_min",
            self.logvar_min,
        )

        self.cfg_head.setdefault(
            "name",
            "dense_head_block",
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------



    def _validate_config(self) -> None:
        """Validate sub-network configuration."""

        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"Unsupported mode={self.mode!r}. "
                f"Expected one of {self.VALID_MODES}."
            )

        if self.block not in self.VALID_BLOCKS:
            raise ValueError(
                f"Unsupported block={self.block!r}. "
                f"Expected one of {self.VALID_BLOCKS}."
            )

        if self.mode == "encoder" and self.dim_in is None:
            raise ValueError(
                "dim_in must be provided in encoder mode."
            )

        if self.mode == "decoder" and self.dim_out is None:
            raise ValueError(
                "dim_out must be provided in decoder mode."
            )

        if self.num_channels <= 0:
            raise ValueError(
                "num_channels must be strictly positive."
            )

        lengths = (
            len(self.list_filters),
            len(self.list_kernels),
            len(self.list_strides),
        )

        if len(set(lengths)) != 1:
            raise ValueError(
                "list_filters, list_kernels and list_strides "
                "must have identical lengths."
            )

        if len(self.list_filters) == 0:
            raise ValueError(
                "At least one convolution block must be defined."
            )

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    def _make_mlp_block(
        self,
        name: str,
    ) -> MLPBlock | None:
        """Build the optional MLP block."""

        if self.cfg_mlp is None:
            return None

        config = dict(self.cfg_mlp)

        config.setdefault(
            "mc_dropout",
            self.mc_dropout,
        )

        config.setdefault(
            "random_state",
            self.random_state,
        )

        config.setdefault(
            "name",
            name,
        )

        return MLPBlock(**config)

    @staticmethod
    def _stride_value(stride) -> int:
        """
        Extract temporal stride from scalar or tuple specification.
        """

        if isinstance(stride, (tuple, list)):
            return int(stride[0])

        return int(stride)

    def _compute_encoded_temporal_size(self) -> int:
        """
        Compute temporal feature-map size after encoder downsampling.

        Assumes SAME-like convolutional padding.
        """

        size = self.dim_seq

        for stride in self.list_strides:
            temporal_stride = self._stride_value(stride)

            size = ceil(
                size / temporal_stride
            )

        return size

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def _build_encoder(self) -> None:
        """Build encoder-specific layers."""
        if self.dim_in % self.num_channels != 0:
            raise ValueError(
                "dim_in must be divisible by num_channels."
            )

        self.dim_per_channel = (
            self.dim_in // self.num_channels
        )

        self.pre = self._make_encoder_preprocessing()

        self.blocks = []

        for index, (filters, kernel, strides) in enumerate(
            zip(
                self.list_filters,
                self.list_kernels,
                self.list_strides,
            )
        ):
            seed = add_random_state(
                self.random_state,
                index + 1,
            )

            if self.block == "2D":
                conv_block = ConvBlock2D(
                    dim_in=self.dim_per_channel,
                    filters=filters,
                    kernel=kernel,
                    strides=strides,
                    dp=self.dp,
                    mc_dropout=self.mc_dropout,
                    random_state=seed,
                    name=f"conv_block_{index}",
                )

            else:
                conv_block = ConvBlock1D(
                    num_channels=self.dim_in,
                    filters=filters,
                    kernel=kernel,
                    strides=strides,
                    dp=self.dp,
                    mc_dropout=self.mc_dropout,
                    random_state=seed,
                    name=f"conv_block_{index}",
                )

            self.blocks.append(conv_block)

        self.flatten = layers.Flatten(
            name="flatten",
        )

        self.mlp_block = self._make_mlp_block(
            name="encoder_mlp_block",
        )

        self.head_block = DenseHeadBlock(
            **self.cfg_head
        )

    def _make_encoder_preprocessing(self):
        """Create encoder input preprocessing."""

        if self.block == "1D":
            return None

        if self.num_channels == 1:
            return layers.Lambda(
                lambda x: K.expand_dims(
                    x,
                    axis=-1,
                ),
                name="encoder_expand_channel",
            )

        return layers.Lambda(
            lambda x: K.reshape(
                x,
                (
                    -1,
                    self.dim_seq,
                    self.dim_per_channel,
                    self.num_channels,
                ),
            ),
            name="encoder_reshape",
        )

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------

    def _build_decoder(self) -> None:
        """Build standard CNN autoencoder decoder."""

        self.mlp_block = self._make_mlp_block(
            name="decoder_mlp_block",
        )

        self.encoded_temporal_size = (
            self._compute_encoded_temporal_size()
        )

        self.initial_filters = int(
            self.list_filters[-1]
        )

        if self.block == "1D":
            self.encoded_shape = (
                self.encoded_temporal_size,
                self.initial_filters,
            )

        else:
            if self.dim_out % self.num_channels != 0:
                raise ValueError(
                    "dim_out must be divisible by "
                    "num_channels in 2D decoder mode."
                )

            self.dim_per_channel = (
                self.dim_out // self.num_channels
            )

            self.encoded_spatial_size = (
                self._compute_encoded_spatial_size()
            )

            self.encoded_shape = (
                self.encoded_temporal_size,
                self.encoded_spatial_size,
                self.initial_filters,
            )

        self.encoded_flat_dim = 1

        for dim in self.encoded_shape:
            self.encoded_flat_dim *= dim

        self.initial_projection = layers.Dense(
            units=self.encoded_flat_dim,
            activation="relu",
            name="decoder_initial_projection",
        )

        self.initial_reshape = layers.Reshape(
            target_shape=self.encoded_shape,
            name="decoder_initial_reshape",
        )

        self.blocks = []

        decoder_filters = list(
            reversed(self.list_filters[:-1])
        )

        decoder_kernels = list(
            reversed(self.list_kernels)
        )

        decoder_strides = list(
            reversed(self.list_strides)
        )

        for index, (
            filters,
            kernel,
            strides,
        ) in enumerate(
            zip(
                decoder_filters,
                decoder_kernels[:-1],
                decoder_strides[:-1],
            )
        ):
            seed = add_random_state(
                self.random_state,
                index,
            )

            if self.block == "2D":
                tconv_block = TConvBlock2D(
                    dim_out=self.dim_out,
                    filters=filters,
                    kernel=kernel,
                    strides=strides,
                    dp=self.dp,
                    mc_dropout=self.mc_dropout,
                    random_state=seed,
                    name=f"tconv_block_{index}",
                )

            else:
                tconv_block = TConvBlock1D(
                    dim_out=self.dim_out,
                    filters=filters,
                    kernel=kernel,
                    strides=strides,
                    dp=self.dp,
                    mc_dropout=self.mc_dropout,
                    random_state=seed,
                    name=f"tconv_block_{index}",
                )

            self.blocks.append(tconv_block)

        self.final_upsampling = self._make_final_upsampling(
            kernel=decoder_kernels[-1],
            stride=decoder_strides[-1],
        )

        self.crop_output = layers.Lambda(
            lambda x: x[:, :self.dim_seq, ...],
            name="output_crop",
        )

        self.head_block = DenseHeadBlock(
            **self.cfg_head
        )

    def _make_final_upsampling(
        self,
        kernel,
        stride,
    ):
        """Create final decoder upsampling layer."""

        if self.block == "2D":
            return layers.Conv2DTranspose(
                filters=self.list_filters[0],
                kernel_size=kernel,
                strides=stride,
                padding="same",
                activation="relu",
                name="final_upsampling",
            )

        return layers.Conv1DTranspose(
            filters=self.list_filters[0],
            kernel_size=kernel,
            strides=stride,
            padding="same",
            activation="relu",
            name="final_upsampling",
        )

    def _compute_encoded_spatial_size(self) -> int:
        """
        Compute encoded spatial feature size for 2D convolutions.

        Assumes SAME-like padding.
        """

        size = self.dim_per_channel

        for stride in self.list_strides:

            if isinstance(stride, (tuple, list)):
                spatial_stride = int(stride[1])
            else:
                spatial_stride = int(stride)

            size = ceil(
                size / spatial_stride
            )

        return size

    def _make_decoder_output_projection(
        self,
        kernel,
        stride,
    ):
        """
        Create final decoder projection.

        The final transposed convolution reconstructs the original
        temporal resolution and output feature dimension.
        """

        num_output_channels = (
            self._get_num_channels_out()
        )

        if self.block == "2D":

            return layers.Conv2DTranspose(
                filters=num_output_channels,
                kernel_size=kernel,
                strides=stride,
                padding="same",
                activation="linear",
                name="output_projection",
            )

        return layers.Conv1DTranspose(
            filters=(
                self.dim_out
                * num_output_channels
            ),
            kernel_size=kernel,
            strides=stride,
            padding="same",
            activation="linear",
            name="output_projection",
        )

    def _get_num_channels_out(self) -> int:
        """
        Return output parameter multiplicity.
        """

        if self.type_output in {
            "Variational",
            "MC_Dropout",
            "Deep_ensemble",
        }:
            return 2

        if self.type_output == "EDL":
            return 4

        return 1

    def _make_probabilistic_processing(self):
        """Create optional probabilistic output processing."""

        if self.type_output in {
            "Variational",
            "MC_Dropout",
            "Deep_ensemble",
        }:
            return ProbProcessingLayers(
                self.logvar_min
            )

        if self.type_output == "EDL":
            return EDLProcessingLayers(
                self.logvar_min
            )

        return None

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def call(
        self,
        inputs,
        training=False,
    ):
        """Run encoder or decoder forward pass."""

        if self.mode == "encoder":
            return self._call_encoder(
                inputs,
                training=training,
            )

        return self._call_decoder(
            inputs,
            training=training,
        )

    def _call_encoder(
        self,
        inputs,
        training=False,
    ):
        """Run encoder forward pass."""

        x = inputs

        if self.pre is not None:
            x = self.pre(x)

        for block in self.blocks:
            x = block(
                x,
                training=training,
            )

        x = self.flatten(x)

        if self.mlp_block is not None:
            x = self.mlp_block(
                x,
                training=training,
            )

        x = self.head_block(x)

        return x

    def _call_decoder(
        self,
        inputs,
        training=False,
    ):
        """Run decoder forward pass."""

        x = inputs

        if self.mlp_block is not None:
            x = self.mlp_block(
                x,
                training=training,
            )

        x = self.initial_projection(x)
        x = self.initial_reshape(x)

        for block in self.blocks:
            x = block(
                x,
                training=training,
            )

        x = self.final_upsampling(x)

        x = self.crop_output(x)

        if self.block == "2D":
            x = tf.reshape(
                x,
                (
                    tf.shape(x)[0],
                    tf.shape(x)[1],
                    -1,
                ),
            )

        x = self.head_block(x)

        return x

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mode": self.mode,
                "dim_seq": self.dim_seq,
                "dim_in": self.dim_in,
                "dim_out": self.dim_out,
                "num_channels": self.num_channels,
                "list_filters": self.list_filters,
                "list_kernels": self.list_kernels,
                "list_strides": self.list_strides,
                "block": self.block,
                "dim_z": self.dim_z,
                "dp": self.dp,
                "type_output": self.type_output,
                "logvar_min": self.logvar_min,
                "cfg_mlp": self.cfg_mlp,
                "cfg_head": self.cfg_head,
                "random_state": self.random_state,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Rebuild layer from serialized configuration."""
        return cls(**config)

    # ------------------------------------------------------------------
    # Configuration helper
    # ------------------------------------------------------------------

    @staticmethod
    def make_config(
        mode: str,
        dim_seq: int,
        dim_in: int | None = None,
        dim_out: int | None = None,
        num_channels: int = 1,
        list_filters: list[int] | tuple[int, ...] = (
            64,
            64,
            32,
        ),
        list_kernels: list | tuple = (
            4,
            4,
            4,
        ),
        list_strides: list | tuple = (
            2,
            2,
            2,
        ),
        block: str = "1D",
        dim_z: int = 200,
        dp: float = 0.02,
        type_output: str | None = None,
        logvar_min: float = -6.0,
        cfg_mlp: dict | None = None,
        cfg_head: dict | None = None,
        random_state: int | None = None,
        name: str | None = "cnn_subnet",
    ) -> dict:
        """Build a CNNSubNet configuration dictionary."""

        return {
            "mode": mode,
            "dim_seq": dim_seq,
            "dim_in": dim_in,
            "dim_out": dim_out,
            "num_channels": num_channels,
            "list_filters": list(list_filters),
            "list_kernels": list(list_kernels),
            "list_strides": list(list_strides),
            "block": block,
            "dim_z": dim_z,
            "dp": dp,
            "type_output": type_output,
            "logvar_min": logvar_min,
            "cfg_mlp": (
                None
                if cfg_mlp is None
                else dict(cfg_mlp)
            ),
            "cfg_head": (
                {}
                if cfg_head is None
                else dict(cfg_head)
            ),
            "random_state": random_state,
            "name": name,
        }