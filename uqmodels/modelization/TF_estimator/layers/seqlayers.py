"""
Sequential layers and sub-networks.

Provides reusable LSTM blocks and configurable recurrent sub-networks
for sequence encoder and decoder architectures.
"""

import tensorflow as tf
from typing import Literal
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras import backend as K
from uqmodels.utils import add_random_state
from uqmodels.modelization.DL_estimator.utils import set_global_determinism
from uqmodels.modelization.TF_estimator.layers.layers import MLPBlock,DenseHeadBlock

# Lstm Layers

@tf.keras.utils.register_keras_serializable(package="UQModels")

@tf.keras.utils.register_keras_serializable(package="UQModels")
class LstmBlock(layers.Layer):
    """
    Local LSTM computational block.

    Applies a repeated LSTM -> optional Dropout stack.

    The block does not manage latent reshaping, RepeatVector,
    output heads, or probabilistic post-processing.

    Dropout mechanisms
    ------------------
    dp:
        External dropout rate applied after each LSTM layer.

    dp_rec:
        Internal dropout rate used by each LSTM layer for both
        input dropout and recurrent dropout.

    mc_dropout:
        Whether external Dropout layers remain active during inference.

    mc_dropout_rec:
        Whether LSTM layers are forced into training mode during inference,
        keeping their internal dropout mechanisms active.

    Typical usage
    -------------
    Encoder:
        return_sequences=False for the last LSTM layer.

    Decoder:
        return_sequences=True for the last LSTM layer.
    """

    def __init__(
        self,
        layers_size: list[int] | tuple[int, ...] = (100, 50),
        dp: float = 0.01,
        dp_rec: float = 0.0,
        activation: str | list[str] | tuple[str, ...] | None = "tanh",
        recurrent_activation: str = "sigmoid",
        reg_W: tuple[float, float] = (1e-5, 1e-5),
        return_sequences: bool = True,
        return_state: bool = False,
        mc_dropout: bool = True,
        mc_dropout_rec: bool = False,
        random_state: int | None = None,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.layers_size = list(layers_size)

        self.dp = float(dp)
        self.dp_rec = float(dp_rec)

        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.reg_W = tuple(reg_W)

        self.return_sequences = bool(return_sequences)
        self.return_state = bool(return_state)

        self.mc_dropout = bool(mc_dropout)
        self.mc_dropout_rec = bool(mc_dropout_rec)

        self.random_state = random_state

        try:
            set_global_determinism(random_state)
        except Exception:
            pass

        self._validate_config()

        reg = regularizers.l1_l2(
            l1=self.reg_W[0],
            l2=self.reg_W[1],
        )

        self.lstm_layers = []
        self.dropout_layers = []

        for idx, dim_layer in enumerate(self.layers_size):
            is_last_layer = (
                idx == len(self.layers_size) - 1
            )

            self.lstm_layers.append(
                layers.LSTM(
                    units=dim_layer,
                    activation=self._get_activation(idx),
                    recurrent_activation=self.recurrent_activation,

                    # Internal stochastic mechanisms.
                    dropout=self.dp_rec,
                    recurrent_dropout=self.dp_rec,

                    return_sequences=(
                        True
                        if not is_last_layer
                        else self.return_sequences
                    ),
                    return_state=(
                        self.return_state
                        if is_last_layer
                        else False
                    ),

                    kernel_regularizer=reg,

                    name=f"lstm_{idx}",
                )
            )

            if self.dp > 0.0:
                dropout_layer = layers.Dropout(
                    rate=self.dp,
                    seed=add_random_state(
                        self.random_state,
                        idx,
                    ),
                    name=f"dropout_{idx}",
                )
            else:
                dropout_layer = None

            self.dropout_layers.append(
                dropout_layer
            )

    def _validate_config(self) -> None:
        """Validate LSTM block configuration."""

        if len(self.layers_size) == 0:
            raise ValueError(
                "layers_size must contain at least one LSTM layer."
            )

        if any(dim <= 0 for dim in self.layers_size):
            raise ValueError(
                "All values in layers_size must be strictly positive."
            )

        if not 0.0 <= self.dp < 1.0:
            raise ValueError(
                "dp must satisfy 0 <= dp < 1."
            )

        if not 0.0 <= self.dp_rec < 1.0:
            raise ValueError(
                "dp_rec must satisfy 0 <= dp_rec < 1."
            )

        if len(self.reg_W) != 2:
            raise ValueError(
                "reg_W must contain exactly two values: (l1, l2)."
            )

        if isinstance(self.activation, (list, tuple)):
            if len(self.activation) != len(self.layers_size):
                raise ValueError(
                    "When activation is a list or tuple, its length "
                    "must match the number of LSTM layers."
                )

        if self.mc_dropout_rec and self.dp_rec == 0.0:
            raise ValueError(
                "mc_dropout_rec=True requires dp_rec > 0."
            )

    def _get_activation(self, idx: int):
        """Return the activation associated with an LSTM layer."""

        if isinstance(self.activation, (list, tuple)):
            return self.activation[idx]

        return self.activation

    def call(
        self,
        inputs,
        training=False,
    ):
        """
        Run the LSTM stack.

        External and recurrent Monte Carlo dropout mechanisms are controlled
        independently through ``mc_dropout`` and ``mc_dropout_rec``.
        """

        x = inputs
        states = None

        for idx, lstm_layer in enumerate(self.lstm_layers):

            output = lstm_layer(
                x,
                training=(
                    training or self.mc_dropout_rec
                ),
            )

            if isinstance(output, (list, tuple)):
                x = output[0]
                states = output[1:]
            else:
                x = output

            dropout_layer = self.dropout_layers[idx]

            if dropout_layer is not None:
                x = dropout_layer(
                    x,
                    training=(
                        training or self.mc_dropout
                    ),
                )

        if self.return_state:
            if states is None:
                raise RuntimeError(
                    "return_state=True but no recurrent states were produced."
                )

            return [x, *states]

        return x

    @property
    def dim_out(self) -> int:
        """
        Return the output feature dimension of the last LSTM layer.
        """
        return int(self.layers_size[-1])

    def get_config(self):
        """Return serializable layer configuration."""

        config = super().get_config()

        config.update(
            {
                "layers_size": self.layers_size,
                "dp": self.dp,
                "dp_rec": self.dp_rec,
                "activation": self.activation,
                "recurrent_activation": self.recurrent_activation,
                "reg_W": self.reg_W,
                "return_sequences": self.return_sequences,
                "return_state": self.return_state,
                "mc_dropout": self.mc_dropout,
                "mc_dropout_rec": self.mc_dropout_rec,
                "random_state": self.random_state,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        """Rebuild layer from serialized configuration."""
        return cls(**config)

    @staticmethod
    def make_config(
        layers_size: list[int] | tuple[int, ...] = (100, 50),
        dp: float = 0.01,
        dp_rec: float = 0.0,
        activation: str | list[str] | tuple[str, ...] | None = "tanh",
        recurrent_activation: str = "sigmoid",
        reg_W: tuple[float, float] = (1e-5, 1e-5),
        return_sequences: bool = True,
        return_state: bool = False,
        mc_dropout: bool = True,
        mc_dropout_rec: bool = False,
        random_state: int | None = None,
        name: str | None = "lstm_block",
    ) -> dict:
        """
        Build a configuration dictionary for LstmBlock.

        Returns
        -------
        dict
            Configuration dictionary passed to LstmBlock.
        """

        return {
            "layers_size": layers_size,
            "dp": dp,
            "dp_rec": dp_rec,
            "activation": activation,
            "recurrent_activation": recurrent_activation,
            "reg_W": reg_W,
            "return_sequences": return_sequences,
            "return_state": return_state,
            "mc_dropout": mc_dropout,
            "mc_dropout_rec": mc_dropout_rec,
            "random_state": random_state,
            "name": name,
        }
    
@tf.keras.utils.register_keras_serializable(package="UQModels")
class LSTMSubNet(layers.Layer):
    """
    Configurable recurrent sub-network.

    Supports two operating modes:

    Encoder
    -------
    Input sequence
        -> LstmBlock
        -> optional MLPBlock
        -> DenseHeadBlock

    Decoder
    -------
    Latent input
        -> optional MLPBlock
        -> RepeatVector
        -> LstmBlock
        -> DenseHeadBlock

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

        self.type_output  = type_output

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

        self._validate_config()
        self._prepare_block_configs()

        self.lstm_block = LstmBlock(
            **self.cfg_backbone
        )

        self.mlp_block = (
            None
            if self.cfg_mlp is None
            else MLPBlock(**self.cfg_mlp)
        )

        if self.mode == "decoder":
            self.repeat_vector = layers.RepeatVector(
                self.dim_seq,
                name="repeat_vector",
            )
        else:
            self.repeat_vector = None

        self.head_block = DenseHeadBlock(
            **self.cfg_head
        )

        self.input_spec = self._make_input_spec()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        """Validate sub-network configuration."""

        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"Unsupported mode={self.mode!r}. "
                f"Expected one of {self.VALID_MODES}."
            )

        if self.mode == "encoder":
            if self.dim_in is None:
                raise ValueError(
                    "dim_in must be provided in encoder mode."
                )

        if self.mode == "decoder":
            if self.dim_out is None:
                raise ValueError(
                    "dim_out must be provided in decoder mode."
                )

    def _prepare_block_configs(self) -> None:
        """
        Complete internal block configurations with mode-dependent defaults.
        """

        mc_dropout = (
            self.type_output == "mc_dropout"
        )

        # LSTM backbone
        self.cfg_backbone.setdefault(
            "mc_dropout",
            mc_dropout,
        )

        self.cfg_backbone.setdefault(
            "random_state",
            self.random_state,
        )

        self.cfg_backbone.setdefault(
            "name",
            "lstm_block",
        )

        if self.mode == "encoder":
            self.cfg_backbone.setdefault(
                "return_sequences",
                False,
            )

        else:
            self.cfg_backbone.setdefault(
                "return_sequences",
                True,
            )

        # Optional MLP
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

        # Output head
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

        x = self.lstm_block(
            inputs,
            training=training,
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
    ):
        """Run decoder forward pass."""

        x = inputs

        if self.mlp_block is not None:
            x = self.mlp_block(
                x,
                training=training,
            )

        x = self.repeat_vector(
            x
        )

        x = self.lstm_block(
            x,
            training=training,
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
                "type_output": self.type_output,
                "random_state": self.random_state,
                "cfg_backbone": self.cfg_backbone,
                "cfg_mlp": self.cfg_mlp,
                "cfg_head": self.cfg_head,
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
        type_output: str | None = None,
        random_state: int | None = None,
        cfg_backbone: dict | None = None,
        cfg_mlp: dict | None = None,
        cfg_head: dict | None = None,
        name: str | None = "lstm_subnet",
    ) -> dict:
        """
        Build a configuration dictionary for LSTMSubNet.
        """

        return {
            "mode": mode,
            "dim_seq": dim_seq,
            "dim_in": dim_in,
            "dim_out": dim_out,
            "dim_z": dim_z,
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