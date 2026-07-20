"""
Transformer autoencoder models.

Provides deterministic and variational Transformer autoencoders built from
configurable attention-based sub-networks.
"""

import inspect
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from uqmodels.modelization.TF_estimator.vae.base_vae import BaseAutoencoder,VariationalBlock,BaseVariationalAutoencoder
from uqmodels.modelization.TF_estimator.layers.attlayers import TransformerSubNet,TransformerDecoderBlock,TransformerEncoderBlock
# ---------------------------------------------------------------------
# Builders: Transformer Encoder / Decoder
# ---------------------------------------------------------------------

def build_transformervae_encoder(
    dim_seq: int,
    dim_in: int,
    dim_z: int,
    variational: bool = False,
    cfg_subnet: dict | None = None,
    random_state: int | None = None,
    name: str = "transformer_encoder",
) -> tf.keras.Model:
    """
    Build a Transformer encoder.

    Architecture
    ------------
    Input (B, dim_seq, dim_in)
        -> TransformerSubNet(mode="encoder")
        -> optional VariationalBlock

    Parameters
    ----------
    dim_seq:
        Input sequence length.
    dim_in:
        Number of input features per time step.
    dim_z:
        Latent dimension.
    variational:
        If True, applies VariationalBlock and returns
        ``(z_mean, z_log_var)``.
        Otherwise, returns a deterministic latent representation.
    cfg_subnet:
        Optional configuration dictionary passed to TransformerSubNet.
    random_state:
        Random seed propagated to the sub-network when not already defined.
    name:
        Keras model name.

    Returns
    -------
    tf.keras.Model
        Configured Transformer encoder.
    """

    encoder_input = layers.Input(
        shape=(dim_seq, dim_in),
        name="encoder_input",
    )

    if cfg_subnet is None:
        cfg_subnet = TransformerSubNet.make_config(
            mode="encoder",
            dim_seq=dim_seq,
            dim_in=dim_in,
            dim_z=dim_z,
            dim_hidden=128,
            n_blocks=3,
            type_output=None,
            random_state=random_state,
            cfg_backbone=TransformerEncoderBlock.make_config(
                dim_hidden=128,
                n_heads=4,
                dim_ff=256,
                dp=0.1,
                mc_dropout=False,
            ),
            name="transformer_encoder_subnet",
        )

    else:
        cfg_subnet = dict(cfg_subnet)

        cfg_subnet.setdefault(
            "mode",
            "encoder",
        )

        cfg_subnet.setdefault(
            "dim_seq",
            dim_seq,
        )

        cfg_subnet.setdefault(
            "dim_in",
            dim_in,
        )

        cfg_subnet.setdefault(
            "dim_z",
            dim_z,
        )

        cfg_subnet.setdefault(
            "random_state",
            random_state,
        )

        cfg_subnet.setdefault(
            "name",
            "transformer_encoder_subnet",
        )

    subnet = TransformerSubNet(
        **cfg_subnet
    )

    x = subnet(
        encoder_input
    )

    if variational:
        outputs = VariationalBlock(
            dim_z=dim_z,
            name="variational_block",
        )(x)

    else:
        outputs = x

    return tf.keras.Model(
        inputs=encoder_input,
        outputs=outputs,
        name=name,
    )

def build_transformervae_decoder(
    dim_seq: int,
    dim_out: int,
    dim_z: int,
    cfg_subnet: dict | None = None,
    random_state: int | None = None,
    name: str = "transformer_decoder",
) -> tf.keras.Model:
    """
    Build a Transformer decoder.

    Architecture
    ------------
    Input latent representation
        -> TransformerSubNet(mode="decoder")
        -> reconstructed sequence

    Parameters
    ----------
    dim_seq:
        Output sequence length.

    dim_out:
        Number of reconstructed features per time step.

    dim_z:
        Latent input dimension.

    cfg_subnet:
        Optional configuration dictionary passed to TransformerSubNet.

    random_state:
        Random seed propagated to the sub-network when not already defined.

    name:
        Keras model name.

    Returns
    -------
    tf.keras.Model
        Configured Transformer decoder.
    """

    decoder_input = layers.Input(
        shape=(dim_z,),
        name="decoder_input",
    )

    if cfg_subnet is None:
        cfg_subnet = TransformerSubNet.make_config(
            mode="decoder",
            dim_seq=dim_seq,
            dim_out=dim_out,
            dim_z=dim_z,
            dim_hidden=128,
            n_blocks=3,
            type_output=None,
            random_state=random_state,
            cfg_backbone=TransformerDecoderBlock.make_config(
                dim_hidden=128,
                n_heads=4,
                dim_ff=256,
                dp=0.1,
                mc_dropout=False,
            ),
            name="transformer_decoder_subnet",
        )

    else:
        cfg_subnet = dict(cfg_subnet)

        cfg_subnet.setdefault(
            "mode",
            "decoder",
        )

        cfg_subnet.setdefault(
            "dim_seq",
            dim_seq,
        )

        cfg_subnet.setdefault(
            "dim_out",
            dim_out,
        )

        cfg_subnet.setdefault(
            "dim_z",
            dim_z,
        )

        cfg_subnet.setdefault(
            "random_state",
            random_state,
        )

        cfg_subnet.setdefault(
            "name",
            "transformer_decoder_subnet",
        )

    subnet = TransformerSubNet(
        **cfg_subnet
    )

    outputs = subnet(
        decoder_input
    )

    return tf.keras.Model(
        inputs=decoder_input,
        outputs=outputs,
        name=name,
    )

# ---------------------------------------------------------------------
# AE / VAE classes (compose with your framework)
# ---------------------------------------------------------------------

class TransformerAE(BaseAutoencoder):
    """
    Deterministic Transformer autoencoder.

    The encoder and decoder are fully configured through dedicated
    builder configuration dictionaries.

    The encoder configuration must explicitly define:
        variational=False
    """

    def __init__(
        self,
        cfg_encoder: dict,
        cfg_decoder: dict,
        name: str = "transformer_ae",
        **kwargs,
    ):
        frame_locals = inspect.currentframe().f_locals

        explicit_args = {
            key: value
            for key, value in frame_locals.items()
            if key not in ("self", "kwargs", "frame_locals")
        }

        all_init_params = {
            **explicit_args,
            **kwargs,
        }

        super().__init__(**all_init_params)

        self.cfg_encoder = dict(cfg_encoder)
        self.cfg_decoder = dict(cfg_decoder)

        self._validate_encoder_config(
            self.cfg_encoder
        )

        self.encoder = build_transformervae_encoder(
            **self.cfg_encoder
        )

        self.decoder = build_transformervae_decoder(
            **self.cfg_decoder
        )


class TransformerVAE(BaseVariationalAutoencoder):
    """
    Variational Transformer autoencoder.

    The encoder and decoder are fully configured through dedicated
    builder configuration dictionaries.

    The encoder configuration must explicitly define:
        variational=True
    """

    def __init__(
        self,
        cfg_encoder: dict,
        cfg_decoder: dict,
        kl_weight: float = 1.0,
        name: str = "transformer_vae",
        **kwargs,
    ):
        frame_locals = inspect.currentframe().f_locals

        explicit_args = {
            key: value
            for key, value in frame_locals.items()
            if key not in ("self", "kwargs", "frame_locals")
        }

        all_init_params = {
            **explicit_args,
            **kwargs,
        }

        super().__init__(**all_init_params)

        self.cfg_encoder = dict(cfg_encoder)
        self.cfg_decoder = dict(cfg_decoder)

        self._validate_encoder_config(
            self.cfg_encoder
        )

        self.encoder = build_transformervae_encoder(
            **self.cfg_encoder
        )

        self.decoder = build_transformervae_decoder(
            **self.cfg_decoder
        )