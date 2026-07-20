"""
Convolutional autoencoder models.

Provides deterministic and variational CNN autoencoders built from
configurable convolutional sub-networks.
"""

import tensorflow as tf
import math
import inspect
from tensorflow.keras import layers,Model
from tensorflow.keras.layers import Input
from uqmodels.modelization.TF_estimator.vae.base_vae import VariationalBlock,BaseAutoencoder,BaseVariationalAutoencoder,VariationalMixin
from uqmodels.modelization.TF_estimator.layers.convlayers import CNNSubNet

def _enc_time_after_downsampling(dim_seq: int, n_layers: int) -> int:
    """
    Avec Conv1D(strides=2, padding='same'), la longueur est ~ ceil(dim_seq / 2^n_layers)
    """
    return int(math.ceil(dim_seq / (2 ** n_layers)))


def build_cnnvae_encoder(
    dim_seq: int,
    dim_in: int,
    dim_z: int,
    num_channels:int=1,
    variational: bool = True,
    cfg_subnet: dict | None = None,
    random_state: int | None = None,
    name: str = "cnn_encoder",
    basic=False,
    ) -> tf.keras.Model:
    """
    Encoder (B, T, F) -> (z_mean, z_log_var, z), latent (B, dim_z).
    conv_filters: ex [50, 100, 200] (comme ton code)
    """
    enc_in = Input(shape=(dim_seq, dim_in), name="encoder_input")

    if(cfg_subnet is None):
        cfg_subnet = CNNSubNet.make_config(
            mode="encoder",
            dim_seq=dim_seq,
            dim_in=dim_in,
            num_channels=num_channels,
            list_filters=(64, 64, 32),
            list_kernels=(10, 10, 10),
            list_strides=(2, 2, 2),
            random_state=random_state,
            name="cnn_subnet",
            block="1D",
            dim_z=200)

    x = enc_in
    if (basic):
        for i, nf in enumerate(cfg_subnet['list_filters']):
            x = layers.Conv1D(
                filters=nf, kernel_size=3, strides=2, padding="same",
                activation="relu", name=f"enc_conv_{i}"
            )(x)
        x = layers.Flatten(name="enc_flatten")(x)

        cfg_mlp = cfg_subnet.get("cfg_mlp")
        
        for i, units in enumerate(cfg_mlp['layers_size']):
            x = layers.Dense(units, activation="relu", name=f"enc_dense_{i}")(x)

        x = layers.Dense(
            units=dim_z,
            name="latent_projection")(x)

    else:
        encoder = CNNSubNet(**cfg_subnet)
        x = encoder(x)    

    if variational:
        output = VariationalBlock(
            dim_z=dim_z,
            name="variational_block",
        )(x)
    else:
        output = x

    return Model(inputs=enc_in,outputs=output,name=name)

def build_cnnvae_decoder(
    dim_seq: int,
    dim_out: int,
    dim_z: int,
    num_channels: int = 1,
    cfg_subnet: dict | None = None,
    random_state: int | None = None,
    name: str = "cnn_decoder",
    basic: bool = False,
) -> tf.keras.Model:
    """
    Build a convolutional decoder.

    Input
    -----
    Shape: (B, dim_z)

    Output
    ------
    Shape: (B, dim_seq, dim_out)

    Parameters
    ----------
    dim_seq:
        Output sequence length.
    dim_out:
        Number of reconstructed features per time step.
    dim_z:
        Latent input dimension.
    num_channels:
        Number of output channels used by CNNSubNet.
    cfg_subnet:
        Optional configuration dictionary passed to CNNSubNet.
    random_state:
        Random seed propagated to internal layers.
    name:
        Keras model name.
    basic:
        Whether to use the basic reference decoder instead of CNNSubNet.

    Returns
    -------
    tf.keras.Model
        Configured decoder model.
    """

    if cfg_subnet is None:
        cfg_subnet = CNNSubNet.make_config(
            mode="decoder",
            dim_seq=dim_seq,
            dim_out=dim_out,
            num_channels=num_channels,
            list_filters=(64, 64, 32),
            list_kernels=(10, 10, 10),
            list_strides=(2, 2, 2),
            block="1D",
            dim_z=dim_z,
            random_state=random_state,
            name="cnn_subnet",
        )

    dec_in = Input(
        shape=(dim_z,),
        name="decoder_input",
    )

    if basic:
        conv_filters = cfg_subnet["list_filters"]

        n_layers = len(conv_filters)
        last_nf = conv_filters[-1]

        t_encoded = _enc_time_after_downsampling(
            dim_seq,
            n_layers,
        )

        encoded_flat_dim = t_encoded * last_nf

        x = layers.Dense(
            units=encoded_flat_dim,
            activation="relu",
            name="dec_dense",
        )(dec_in)

        x = layers.Reshape(
            target_shape=(t_encoded, last_nf),
            name="dec_reshape",
        )(x)

        for i, nf in enumerate(
            reversed(conv_filters[:-1])
        ):
            x = layers.Conv1DTranspose(
                filters=nf,
                kernel_size=3,
                strides=2,
                padding="same",
                activation="relu",
                name=f"dec_deconv_{i}",
            )(x)

        x = layers.Conv1DTranspose(
            filters=dim_out,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="linear",
            name=f"dec_deconv_{len(conv_filters) - 1}",
        )(x)

        x = layers.Flatten(
            name="dec_flatten",
        )(x)

        x = layers.Dense(
            units=dim_seq * dim_out,
            name="decoder_dense_final",
        )(x)

        out = layers.Reshape(
            target_shape=(dim_seq, dim_out),
            name="decoder_reshape_final",
        )(x)

    else:
        decoder = CNNSubNet(
            **cfg_subnet
        )

        out = decoder(
            dec_in
        )

    return Model(
        inputs=dec_in,
        outputs=out,
        name=name,
    )

class CnnAE(BaseAutoencoder):
    """
    Convolutional autoencoder.

    The encoder and decoder are fully configured through dedicated
    configuration dictionaries.
    """

    def __init__(
        self,
        cfg_encoder: dict,
        cfg_decoder: dict,
        name: str = "conv_ae",
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

        self._validate_encoder_config(cfg_encoder)

        self.cfg_encoder = dict(cfg_encoder)
        self.cfg_decoder = dict(cfg_decoder)

        self.encoder = build_cnnvae_encoder(
            **self.cfg_encoder
        )

        self.decoder = build_cnnvae_decoder(
            **self.cfg_decoder
        )

class CnnVAE(BaseVariationalAutoencoder):
    """
    Convolutional variational autoencoder.

    The encoder and decoder are fully configured through dedicated
    configuration dictionaries.

    The encoder configuration must explicitly define:
        variational=True
    """

    def __init__(
        self,
        cfg_encoder: dict,
        cfg_decoder: dict,
        name: str = "conv_vae",
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

        self._validate_encoder_config(cfg_encoder)

        self.cfg_encoder = dict(cfg_encoder)
        self.cfg_decoder = dict(cfg_decoder)

        self.encoder = build_cnnvae_encoder(
            **self.cfg_encoder
        )

        self.decoder = build_cnnvae_decoder(
            **self.cfg_decoder
        )