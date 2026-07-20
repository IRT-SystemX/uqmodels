"""
Attention layers and sub-networks.

Provides positional embeddings, Transformer blocks, and configurable
attention-based sub-networks for sequence encoder and decoder architectures.
"""

import tensorflow as tf
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow.keras import layers
from uqmodels.modelization.TF_estimator.layers.layers import MLPSubNet,MLPBlock,DenseHeadBlock

@tf.keras.utils.register_keras_serializable(package="UQModels")
class PositionalEmbedding(layers.Layer):
    """
    Learnable positional embedding layer.

    Inputs
    ------
    Shape: (B, T, dim_hidden)

    Outputs
    -------
    Shape: (B, T, dim_hidden)

    Parameters
    ----------
    dim_seq:
        Maximum supported sequence length.
    dim_hidden:
        Internal representation dimension.
    """

    def __init__(
        self,
        dim_seq: int,
        dim_hidden: int,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.dim_seq = int(dim_seq)
        self.dim_hidden = int(dim_hidden)

        self._validate_config()

        self.pos_emb = self.add_weight(
            name="pos_embedding",
            shape=(self.dim_seq, self.dim_hidden),
            initializer="random_normal",
            trainable=True,
        )

    def _validate_config(self) -> None:
        """Validate positional embedding configuration."""

        if self.dim_seq <= 0:
            raise ValueError(
                "dim_seq must be strictly positive."
            )

        if self.dim_hidden <= 0:
            raise ValueError(
                "dim_hidden must be strictly positive."
            )

    def call(self, inputs):
        """Add learnable positional embeddings."""

        seq_len = tf.shape(inputs)[1]

        return inputs + self.pos_emb[:seq_len]

    def get_config(self):
        """Return serializable configuration."""

        config = super().get_config()

        config.update(
            {
                "dim_seq": self.dim_seq,
                "dim_hidden": self.dim_hidden,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        """Rebuild layer from serialized configuration."""

        return cls(**config)

    @staticmethod
    def make_config(
        dim_seq: int,
        dim_hidden: int,
        name: str | None = "positional_embedding",
    ) -> dict:
        """Build a configuration dictionary for PositionalEmbedding."""

        return {
            "dim_seq": dim_seq,
            "dim_hidden": dim_hidden,
            "name": name,
        }


@tf.keras.utils.register_keras_serializable(package="UQModels")
class TransformerEncoderBlock(layers.Layer):
    """
    Transformer encoder block using PreNorm architecture.

    Architecture
    ------------
    Input
        -> LayerNorm
        -> Multi-Head Self-Attention
        -> Dropout
        -> Residual connection
        -> LayerNorm
        -> Feed-Forward Network
        -> Dropout
        -> Residual connection

    Parameters
    ----------
    dim_hidden:
        Internal Transformer representation dimension.
    n_heads:
        Number of attention heads.
    dim_ff:
        Hidden dimension of the feed-forward network.
    dp:
        Dropout rate.
    activation:
        Activation function used in the feed-forward network.
    mc_dropout:
        Whether dropout remains active during inference.
    """

    def __init__(
        self,
        dim_hidden: int = 128,
        n_heads: int = 4,
        dim_ff: int = 256,
        dp: float = 0.1,
        activation: str | None = "relu",
        mc_dropout: bool = False,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.dim_hidden = int(dim_hidden)
        self.n_heads = int(n_heads)
        self.dim_ff = int(dim_ff)
        self.dp = float(dp)
        self.activation = activation
        self.mc_dropout = bool(mc_dropout)

        self._validate_config()

        self.dim_head = (
            self.dim_hidden // self.n_heads
        )

        self.norm_attention = layers.LayerNormalization(
            epsilon=1e-6,
            name="norm_attention",
        )

        self.self_attention = layers.MultiHeadAttention(
            num_heads=self.n_heads,
            key_dim=self.dim_head,
            name="self_attention",
        )

        self.dropout_attention = layers.Dropout(
            rate=self.dp,
            name="dropout_attention",
        )

        self.norm_ff = layers.LayerNormalization(
            epsilon=1e-6,
            name="norm_ff",
        )

        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(
                    units=self.dim_ff,
                    activation=self.activation,
                    name="ff_dense_hidden",
                ),
                layers.Dense(
                    units=self.dim_hidden,
                    activation=None,
                    name="ff_dense_output",
                ),
            ],
            name="feed_forward",
        )

        self.dropout_ff = layers.Dropout(
            rate=self.dp,
            name="dropout_ff",
        )

    def _validate_config(self) -> None:
        """Validate Transformer encoder block configuration."""

        if self.dim_hidden <= 0:
            raise ValueError(
                "dim_hidden must be strictly positive."
            )

        if self.n_heads <= 0:
            raise ValueError(
                "n_heads must be strictly positive."
            )

        if self.dim_hidden % self.n_heads != 0:
            raise ValueError(
                "dim_hidden must be divisible by n_heads."
            )

        if self.dim_ff <= 0:
            raise ValueError(
                "dim_ff must be strictly positive."
            )

        if not 0.0 <= self.dp < 1.0:
            raise ValueError(
                "dp must satisfy 0 <= dp < 1."
            )

    def call(
        self,
        inputs,
        training=False,
        mask=None,
    ):
        """Run Transformer encoder block."""

        dropout_training = (
            training or self.mc_dropout
        )

        # Self-attention branch.
        h = self.norm_attention(inputs)

        h = self.self_attention(
            query=h,
            value=h,
            key=h,
            attention_mask=mask,
            training=dropout_training,
        )

        h = self.dropout_attention(
            h,
            training=dropout_training,
        )

        x = inputs + h

        # Feed-forward branch.
        h = self.norm_ff(x)

        h = self.ffn(
            h,
            training=dropout_training,
        )

        h = self.dropout_ff(
            h,
            training=dropout_training,
        )

        return x + h

    def get_config(self):
        """Return serializable configuration."""

        config = super().get_config()

        config.update(
            {
                "dim_hidden": self.dim_hidden,
                "n_heads": self.n_heads,
                "dim_ff": self.dim_ff,
                "dp": self.dp,
                "activation": self.activation,
                "mc_dropout": self.mc_dropout,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        """Rebuild layer from serialized configuration."""

        return cls(**config)

    @staticmethod
    def make_config(
        dim_hidden: int = 128,
        n_heads: int = 4,
        dim_ff: int = 256,
        dp: float = 0.1,
        activation: str | None = "relu",
        mc_dropout: bool = False,
        name: str | None = "transformer_encoder_block",
    ) -> dict:
        """Build a configuration dictionary for TransformerEncoderBlock."""

        return {
            "dim_hidden": dim_hidden,
            "n_heads": n_heads,
            "dim_ff": dim_ff,
            "dp": dp,
            "activation": activation,
            "mc_dropout": mc_dropout,
            "name": name,
        }


@tf.keras.utils.register_keras_serializable(package="UQModels")
class TransformerDecoderBlock(layers.Layer):
    """
    Transformer decoder block using self-attention only.

    This block is designed for autoencoder architectures where the latent
    representation is first projected into a sequence.

    Architecture
    ------------
    Input
        -> LayerNorm
        -> Multi-Head Self-Attention
        -> Dropout
        -> Residual connection
        -> LayerNorm
        -> Feed-Forward Network
        -> Dropout
        -> Residual connection

    Parameters
    ----------
    dim_hidden:
        Internal Transformer representation dimension.
    n_heads:
        Number of attention heads.
    dim_ff:
        Hidden dimension of the feed-forward network.
    dp:
        Dropout rate.
    activation:
        Activation function used in the feed-forward network.
    mc_dropout:
        Whether dropout remains active during inference.
    """

    def __init__(
        self,
        dim_hidden: int = 128,
        n_heads: int = 4,
        dim_ff: int = 256,
        dp: float = 0.1,
        activation: str | None = "relu",
        mc_dropout: bool = False,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.dim_hidden = int(dim_hidden)
        self.n_heads = int(n_heads)
        self.dim_ff = int(dim_ff)
        self.dp = float(dp)
        self.activation = activation
        self.mc_dropout = bool(mc_dropout)

        self._validate_config()

        self.dim_head = (
            self.dim_hidden // self.n_heads
        )

        self.norm_attention = layers.LayerNormalization(
            epsilon=1e-6,
            name="norm_attention",
        )

        self.self_attention = layers.MultiHeadAttention(
            num_heads=self.n_heads,
            key_dim=self.dim_head,
            name="self_attention",
        )

        self.dropout_attention = layers.Dropout(
            rate=self.dp,
            name="dropout_attention",
        )

        self.norm_ff = layers.LayerNormalization(
            epsilon=1e-6,
            name="norm_ff",
        )

        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(
                    units=self.dim_ff,
                    activation=self.activation,
                    name="ff_dense_hidden",
                ),
                layers.Dense(
                    units=self.dim_hidden,
                    activation=None,
                    name="ff_dense_output",
                ),
            ],
            name="feed_forward",
        )

        self.dropout_ff = layers.Dropout(
            rate=self.dp,
            name="dropout_ff",
        )

    def _validate_config(self) -> None:
        """Validate Transformer decoder block configuration."""

        if self.dim_hidden <= 0:
            raise ValueError(
                "dim_hidden must be strictly positive."
            )

        if self.n_heads <= 0:
            raise ValueError(
                "n_heads must be strictly positive."
            )

        if self.dim_hidden % self.n_heads != 0:
            raise ValueError(
                "dim_hidden must be divisible by n_heads."
            )

        if self.dim_ff <= 0:
            raise ValueError(
                "dim_ff must be strictly positive."
            )

        if not 0.0 <= self.dp < 1.0:
            raise ValueError(
                "dp must satisfy 0 <= dp < 1."
            )

    def call(
        self,
        inputs,
        training=False,
        mask=None,
    ):
        """Run Transformer decoder block."""

        dropout_training = (
            training or self.mc_dropout
        )

        # Self-attention branch.
        h = self.norm_attention(inputs)

        h = self.self_attention(
            query=h,
            value=h,
            key=h,
            attention_mask=mask,
            training=dropout_training,
        )

        h = self.dropout_attention(
            h,
            training=dropout_training,
        )

        x = inputs + h

        # Feed-forward branch.
        h = self.norm_ff(x)

        h = self.ffn(
            h,
            training=dropout_training,
        )

        h = self.dropout_ff(
            h,
            training=dropout_training,
        )

        return x + h

    def get_config(self):
        """Return serializable configuration."""

        config = super().get_config()

        config.update(
            {
                "dim_hidden": self.dim_hidden,
                "n_heads": self.n_heads,
                "dim_ff": self.dim_ff,
                "dp": self.dp,
                "activation": self.activation,
                "mc_dropout": self.mc_dropout,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        """Rebuild layer from serialized configuration."""

        return cls(**config)

    @staticmethod
    def make_config(
        dim_hidden: int = 128,
        n_heads: int = 4,
        dim_ff: int = 256,
        dp: float = 0.1,
        activation: str | None = "relu",
        mc_dropout: bool = False,
        name: str | None = "transformer_decoder_block",
    ) -> dict:
        """Build a configuration dictionary for TransformerDecoderBlock."""

        return {
            "dim_hidden": dim_hidden,
            "n_heads": n_heads,
            "dim_ff": dim_ff,
            "dp": dp,
            "activation": activation,
            "mc_dropout": mc_dropout,
            "name": name,
        }
    
from typing import Literal

import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable(package="UQModels")
class TransformerSubNet(layers.Layer):
    """
    Configurable Transformer sub-network.

    Supports two operating modes.

    Encoder
    -------
    Input sequence
        -> input projection
        -> positional embedding
        -> Transformer encoder blocks
        -> flatten
        -> optional MLP block
        -> dense output head

    Decoder
    -------
    Latent input
        -> optional MLP block
        -> dense projection to sequence
        -> reshape
        -> positional embedding
        -> Transformer decoder blocks
        -> dense output head

    The sub-network does not manage variational latent sampling.
    This responsibility belongs to the model builder and
    BaseVariationalAutoencoder.
    """

    VALID_MODES = {"encoder", "decoder"}

    def __init__(
        self,
        mode: Literal["encoder", "decoder"],
        dim_seq: int,
        dim_in: int | None = None,
        dim_out: int | None = None,
        dim_z: int = 100,
        dim_hidden: int = 128,
        n_blocks: int = 2,
        type_output: str | None = None,
        random_state: int | None = None,
        cfg_backbone: dict | None = None,
        cfg_mlp: dict | None = None,
        cfg_head: dict | None = None,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.mode = mode
        self.dim_seq = int(dim_seq)

        self.dim_in = (
            None if dim_in is None
            else int(dim_in)
        )

        self.dim_out = (
            None if dim_out is None
            else int(dim_out)
        )

        self.dim_z = int(dim_z)
        self.dim_hidden = int(dim_hidden)
        self.n_blocks = int(n_blocks)

        self.type_output = normalize_type_output(
            type_output
        )

        self.random_state = random_state

        self.cfg_backbone = (
            {}
            if cfg_backbone is None
            else dict(cfg_backbone)
        )

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

        validate_type_output(self.type_output)

        self._validate_config()
        self._prepare_block_configs()

        self.input_projection = None
        self.sequence_projection = None
        self.sequence_reshape = None

        self.positional_embedding = PositionalEmbedding(
            dim_seq=self.dim_seq,
            dim_hidden=self.dim_hidden,
            name="positional_embedding",
        )

        self.transformer_blocks = (
            self._make_transformer_blocks()
        )

        self.mlp_block = (
            None
            if self.cfg_mlp is None
            else MLPBlock(**self.cfg_mlp)
        )

        if self.mode == "encoder":
            self.input_projection = layers.Dense(
                units=self.dim_hidden,
                name="input_projection",
            )

            self.flatten = layers.Flatten(
                name="flatten",
            )

        else:
            self.sequence_projection = layers.Dense(
                units=self.dim_seq * self.dim_hidden,
                name="sequence_projection",
            )

            self.sequence_reshape = layers.Reshape(
                target_shape=(
                    self.dim_seq,
                    self.dim_hidden,
                ),
                name="sequence_reshape",
            )

        self.head_block = DenseHeadBlock(
            **self.cfg_head
        )

        self.input_spec = self._make_input_spec()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        """Validate Transformer sub-network configuration."""

        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"Unsupported mode={self.mode!r}. "
                f"Expected one of {self.VALID_MODES}."
            )

        if self.mode == "encoder" and self.dim_in is None:
            raise ValueError(
                "dim_in must be provided in encoder mode."
            )

        if self.mode == "decoder" and self.dim_out is None:
            raise ValueError(
                "dim_out must be provided in decoder mode."
            )

        if self.dim_seq <= 0:
            raise ValueError(
                "dim_seq must be strictly positive."
            )

        if self.dim_hidden <= 0:
            raise ValueError(
                "dim_hidden must be strictly positive."
            )

        if self.n_blocks <= 0:
            raise ValueError(
                "n_blocks must be strictly positive."
            )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _prepare_block_configs(self) -> None:
        """
        Complete internal block configurations with sub-network defaults.
        """

        mc_dropout = (
            self.type_output == "mc_dropout"
        )

        # Transformer backbone.
        self.cfg_backbone.setdefault(
            "dim_hidden",
            self.dim_hidden,
        )

        self.cfg_backbone.setdefault(
            "mc_dropout",
            mc_dropout,
        )

        # Optional MLP block.
        if self.cfg_mlp is not None:
            self.cfg_mlp.setdefault(
                "mc_dropout",
                mc_dropout,
            )

            self.cfg_mlp.setdefault(
                "random_state",
                self.random_state,
            )

            self.cfg_mlp.setdefault(
                "name",
                f"{self.mode}_mlp_block",
            )

        # Output head.
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
            "name",
            "dense_head_block",
        )

    # ------------------------------------------------------------------
    # Block construction
    # ------------------------------------------------------------------

    def _make_transformer_blocks(self) -> list[layers.Layer]:
        """Build the Transformer block stack."""

        blocks = []

        for idx in range(self.n_blocks):
            config = dict(
                self.cfg_backbone
            )

            config["name"] = (
                f"transformer_{self.mode}_block_{idx}"
            )

            if self.mode == "encoder":
                block = TransformerEncoderBlock(
                    **config
                )
            else:
                block = TransformerDecoderBlock(
                    **config
                )

            blocks.append(block)

        return blocks

    def _make_input_spec(self):
        """Create mode-dependent input specification."""

        if self.mode == "encoder":
            return layers.InputSpec(
                shape=(
                    None,
                    self.dim_seq,
                    self.dim_in,
                )
            )

        return layers.InputSpec(
            shape=(
                None,
                self.dim_z,
            )
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def call(
        self,
        inputs,
        training=False,
        mask=None,
    ):
        """Run encoder or decoder forward pass."""

        if self.mode == "encoder":
            return self._call_encoder(
                inputs,
                training=training,
                mask=mask,
            )

        return self._call_decoder(
            inputs,
            training=training,
            mask=mask,
        )

    def _call_encoder(
        self,
        inputs,
        training=False,
        mask=None,
    ):
        """Run Transformer encoder forward pass."""

        x = self.input_projection(
            inputs
        )

        x = self.positional_embedding(
            x
        )

        for block in self.transformer_blocks:
            x = block(
                x,
                training=training,
                mask=mask,
            )

        x = self.flatten(
            x
        )

        if self.mlp_block is not None:
            x = self.mlp_block(
                x,
                training=training,
            )

        x = self.head_block(
            x
        )

        return x

    def _call_decoder(
        self,
        inputs,
        training=False,
        mask=None,
    ):
        """Run Transformer decoder forward pass."""

        x = inputs

        if self.mlp_block is not None:
            x = self.mlp_block(
                x,
                training=training,
            )

        x = self.sequence_projection(
            x
        )

        x = self.sequence_reshape(
            x
        )

        x = self.positional_embedding(
            x
        )

        for block in self.transformer_blocks:
            x = block(
                x,
                training=training,
                mask=mask,
            )

        x = self.head_block(
            x
        )

        return x

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_param(self) -> int:
        """
        Return the number of output parameters per target dimension.
        """
        return self.head_block.n_param

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_config(self):
        """Return serializable configuration."""

        config = super().get_config()

        config.update(
            {
                "mode": self.mode,
                "dim_seq": self.dim_seq,
                "dim_in": self.dim_in,
                "dim_out": self.dim_out,
                "dim_z": self.dim_z,
                "dim_hidden": self.dim_hidden,
                "n_blocks": self.n_blocks,
                "type_output": self.type_output,
                "random_state": self.random_state,
                "cfg_backbone": (
                    self.cfg_backbone
                ),
                "cfg_mlp": (
                    self.cfg_mlp
                ),
                "cfg_head": (
                    self.cfg_head
                ),
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
        mode: Literal["encoder", "decoder"],
        dim_seq: int,
        dim_in: int | None = None,
        dim_out: int | None = None,
        dim_z: int = 100,
        dim_hidden: int = 128,
        n_blocks: int = 2,
        type_output: str | None = None,
        random_state: int | None = None,
        cfg_backbone: dict | None = None,
        cfg_mlp: dict | None = None,
        cfg_head: dict | None = None,
        name: str | None = "transformer_subnet",
    ) -> dict:
        """
        Build a configuration dictionary for TransformerSubNet.
        """

        return {
            "mode": mode,
            "dim_seq": dim_seq,
            "dim_in": dim_in,
            "dim_out": dim_out,
            "dim_z": dim_z,
            "dim_hidden": dim_hidden,
            "n_blocks": n_blocks,
            "type_output": type_output,
            "random_state": random_state,
            "cfg_backbone": (
                {}
                if cfg_backbone is None
                else dict(cfg_backbone)
            ),
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
            "name": name,
        }
