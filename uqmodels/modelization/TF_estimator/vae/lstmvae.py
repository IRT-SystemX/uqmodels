"""
LSTM autoencoder models.

Provides deterministic and variational recurrent autoencoders built from
configurable LSTM sub-networks.

Minimal Sequence Variational Autoencoder built by composing:
    SequenceMixin + VariationalMixin + BaseAutoencoder

Encoder (B,T,F) -> (z_mean,z_log_var,z) of shape (B,T,D)
Decoder (B,T,D) -> (B,T,F)
Training is handled by BaseAutoencoder via the forward_and_losses hook.
"""
import inspect
import tensorflow as tf
from tensorflow.keras import layers, Model as KModel
from uqmodels.modelization.TF_estimator.layers.seqlayers import LstmBlock,LSTMSubNet
from uqmodels.modelization.TF_estimator.vae.base_vae import BaseAutoencoder,VariationalBlock,BaseVariationalAutoencoder,HybridMixin


def build_lstmvae_encoder(
    dim_seq: int,
    dim_in: int,
    dim_z: int,
    variational: bool = True,
    cfg_subnet: dict | None = None,
    random_state: int | None = None,
    name: str = "encoder_lstm",
) -> tf.keras.Model:
    """
    Build an LSTM encoder.

    Architecture
    ------------
    Input (B, T, F)
        -> LSTMSubNet(mode="encoder")
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
        Optional configuration dictionary passed to LSTMSubNet.
    random_state:
        Random seed propagated to LSTMSubNet when not already configured.
    name:
        Keras model name.

    Returns
    -------
    tf.keras.Model
        Configured LSTM encoder.
    """

    enc_in = layers.Input(
        shape=(dim_seq, dim_in),
        name="encoder_input",
    )

    if cfg_subnet is None:
        cfg_subnet = LSTMSubNet.make_config(
            mode="encoder",
            dim_seq=dim_seq,
            dim_in=dim_in,
            dim_z=dim_z,
            type_output=None,
            random_state=random_state,
            cfg_backbone=LstmBlock.make_config(
                layers_size=(100, 50),
                return_sequences=False,
                return_state=False,
                random_state=random_state,
                name="encoder_lstm_block",
            ),
            name="encoder_lstm_subnet",
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
            "encoder_lstm_subnet",
        )

    subnet = LSTMSubNet(
        **cfg_subnet
    )

    x = subnet(
        enc_in
    )

    if variational:
        outputs = VariationalBlock(
            dim_z=dim_z,
            name="variational_block",
        )(x)

    else:
        outputs = x

    return tf.keras.Model(
        inputs=enc_in,
        outputs=outputs,
        name=name,
    )

def build_lstmvae_decoder(
    dim_seq: int,
    dim_out: int,
    dim_z: int,
    cfg_subnet: dict | None = None,
    random_state: int | None = None,
    name: str = "decoder_lstm",
) -> tf.keras.Model:
    """
    Build an LSTM decoder.

    Architecture
    ------------
    Input (B, dim_z)
        -> LSTMSubNet(mode="decoder")
        -> Output (B, dim_seq, dim_out)

    Parameters
    ----------
    dim_seq:
        Output sequence length.

    dim_out:
        Number of reconstructed features per time step.

    dim_z:
        Latent input dimension.

    cfg_subnet:
        Optional configuration dictionary passed to LSTMSubNet.

    random_state:
        Random seed propagated to LSTMSubNet when not already configured.

    name:
        Keras model name.

    Returns
    -------
    tf.keras.Model
        Configured LSTM decoder.
    """

    dec_in = layers.Input(
        shape=(dim_z,),
        name="decoder_input",
    )

    if cfg_subnet is None:
        cfg_subnet = LSTMSubNet.make_config(
            mode="decoder",
            dim_seq=dim_seq,
            dim_out=dim_out,
            dim_z=dim_z,
            type_output=None,
            random_state=random_state,
            cfg_backbone=LstmBlock.make_config(
                layers_size=(50, 100),
                return_sequences=True,
                return_state=False,
                random_state=random_state,
                name="decoder_lstm_block",
            ),
            name="decoder_lstm_subnet",
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
            "decoder_lstm_subnet",
        )

    subnet = LSTMSubNet(
        **cfg_subnet
    )

    outputs = subnet(
        dec_in
    )

    return tf.keras.Model(
        inputs=dec_in,
        outputs=outputs,
        name=name,
    )

class LstmAE(BaseAutoencoder):
    """
    Deterministic LSTM autoencoder.

    The encoder and decoder are fully configured through dedicated
    builder configuration dictionaries.

    The encoder configuration must explicitly define:
        variational=False
    """

    def __init__(
        self,
        cfg_encoder: dict,
        cfg_decoder: dict,
        name: str = "lstm_ae",
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

        self.encoder = build_lstmvae_encoder(
            **self.cfg_encoder
        )

        self.decoder = build_lstmvae_decoder(
            **self.cfg_decoder
        )

class LstmVAE(BaseVariationalAutoencoder):
    """
    Variational LSTM autoencoder.

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
        name: str = "lstm_vae",
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

        self.encoder = build_lstmvae_encoder(
            **self.cfg_encoder
        )

        self.decoder = build_lstmvae_decoder(
            **self.cfg_decoder
        )