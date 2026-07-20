"""
Dense autoencoder models.

Provides deterministic and variational dense autoencoders built from
MLP sub-networks and reusable model builders.
"""

import tensorflow as tf
import inspect
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from uqmodels.modelization.TF_estimator.vae.base_vae import VariationalBlock,BaseAutoencoder,BaseVariationalAutoencoder,VariationalMixin
from uqmodels.modelization.TF_estimator.layers.layers import MLPSubNet,MLPBlock

def build_densevae_encoder(
    dim_seq: int,
    dim_in: int,
    dim_z: int,
    variational: bool = True,
    cfg_subnet: dict | None = None,
    random_state: int | None = None,
    name: str = "encoder_dense",
) -> tf.keras.Model:
    """
    Build a dense encoder.

    Architecture
    ------------
    Input (B, T, F)
        -> MLPSubNet
        -> optional VariationalBlock

    Parameters
    ----------
    dim_seq:
        Sequence length.
    dim_in:
        Input feature dimension per time step.
    dim_z:
        Latent dimension.
    variational:
        If True, applies VariationalBlock and returns
        ``(z_mean, z_log_var)``.
        Otherwise, returns a deterministic latent representation.
    cfg_subnet:
        Optional configuration dictionary passed to MLPSubNet.
    random_state:
        Random seed propagated to MLPSubNet when not already configured.
    name:
        Keras model name.

    Returns
    -------
    tf.keras.Model
        Configured dense encoder.
    """

    enc_in = Input(
        shape=(dim_seq, dim_in),
        name="encoder_input",
    )

    if cfg_subnet is None:
        cfg_subnet = MLPSubNet.make_config(
            dim_in=dim_seq * dim_in,
            dim_out=dim_z,
            shape_in=(dim_seq, dim_in),
            shape_out=None,
            type_output=None,
            random_state=random_state,
            cfg_backbone=MLPBlock.make_config(
                layers_size=(100, 50),
                random_state=random_state,
                name="enc_mlp_block",
            ),
            name="encoder_mlp_subnet",
        )
    else:
        cfg_subnet = dict(cfg_subnet)
        cfg_subnet.setdefault("random_state", random_state)
        cfg_subnet.setdefault("name", "encoder_mlp_subnet")

    subnet = MLPSubNet(**cfg_subnet)

    x = subnet(enc_in)

    if variational:
        outputs = VariationalBlock(
            dim_z=dim_z,
            name="variational_block",
        )(x)
    else:
        outputs = x

    return Model(
        inputs=enc_in,
        outputs=outputs,
        name=name,
    )


def build_densevae_decoder(
    dim_seq: int,
    dim_in: int,
    dim_z: int,
    cfg_subnet: dict | None = None,
    random_state: int | None = None,
    name: str = "decoder_dense",
) -> tf.keras.Model:
    """
    Build a dense decoder.

    Architecture
    ------------
    Input (B, dim_z)
        -> MLPSubNet
        -> Output (B, dim_seq, dim_in)

    Parameters
    ----------
    dim_seq:
        Output sequence length.
    dim_in:
        Output feature dimension per time step.
    dim_z:
        Latent input dimension.
    cfg_subnet:
        Optional configuration dictionary passed to MLPSubNet.
    random_state:
        Random seed propagated to MLPSubNet when not already configured.
    name:
        Keras model name.

    Returns
    -------
    tf.keras.Model
        Configured dense decoder model.
    """

    dec_in = Input(
        shape=(dim_z,),
        name="decoder_input",
    )

    if cfg_subnet is None:
        cfg_subnet = MLPSubNet.make_config(
            dim_in=dim_z,
            dim_out=dim_seq * dim_in,
            shape_in=None,
            shape_out=(dim_seq, dim_in),
            type_output=None,
            random_state=random_state,
            cfg_backbone=MLPBlock.make_config(
                layers_size=(50, 100),
                random_state=random_state,
                name="dec_mlp_block",
            ),
            name="decoder_mlp_subnet",
        )
    else:
        cfg_subnet = dict(cfg_subnet)
        cfg_subnet.setdefault("random_state", random_state)
        cfg_subnet.setdefault("name", "decoder_mlp_subnet")

    decoder = MLPSubNet(**cfg_subnet)

    out = decoder(dec_in)

    return Model(
        inputs=dec_in,
        outputs=out,
        name=name,
    )
class DenseAE(BaseAutoencoder):
    """
    Dense deterministic autoencoder.

    The encoder and decoder are fully configured through dedicated
    builder configuration dictionaries.

    The encoder configuration must explicitly define:
        variational=False
    """

    def __init__(
        self,
        cfg_encoder: dict,
        cfg_decoder: dict,
        name: str = "dense_ae",
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

        self.encoder = build_densevae_encoder(
            **self.cfg_encoder
        )

        self.decoder = build_densevae_decoder(
            **self.cfg_decoder
        )


class DenseVAE(BaseVariationalAutoencoder):
    """
    Dense variational autoencoder.

    The encoder and decoder are fully configured through dedicated
    builder configuration dictionaries.

    The encoder configuration must explicitly define:
        variational=True
    """

    def __init__(
        self,
        cfg_encoder: dict,
        cfg_decoder: dict,
        kl_weight: float = 0.1,
        name: str = "dense_vae",
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

        self.encoder = build_densevae_encoder(
            **self.cfg_encoder
        )

        self.decoder = build_densevae_decoder(
            **self.cfg_decoder
        )
