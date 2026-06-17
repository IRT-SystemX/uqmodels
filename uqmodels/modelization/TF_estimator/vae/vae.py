import tensorflow as tf
import math
import inspect
from tensorflow.keras import layers,Model
from tensorflow.keras.layers import Input
from uqmodels.modelization.TF_estimator.vae.base_vae import BaseAutoencoder,BaseVariationalAutoencoder,VariationalMixin

from uqmodels.modelization.TF_estimator.base.layers import MLPBlock
from uqmodels.modelization.TF_estimator.base.convlayers import CNNEncoder,CNNDecoder

def build_dense_vae_encoder(
    dim_seq: int,
    dim_in: int,
    dim_z: int,
    variational: bool = True,
    cfg_MLPBlock: dict | None = None,
    random_state: int | None = None,
    name: str = "encoder_dense",
) -> tf.keras.Model:
    """
    Build a dense VAE encoder.

    Architecture:
        Input(B, T, F) -> Flatten -> MLPBlock -> latent head

    Parameters
    ----------
    dim_seq:
        Sequence length.
    dim_in:
        Input feature dimension.
    dim_z:
        Latent dimension.
    variational:
        If True, returns [z_mean, z_log_var].
        If False, returns z_mean only.
    cfg_MLPBlock:
        Configuration dictionary passed to MLPBlock.
    random_state:
        Random seed propagated to MLPBlock if not already provided.
    name:
        Keras model name.

    Returns
    -------
    tf.keras.Model
        Dense encoder model.
    """
    cfg_MLPBlock = {} if cfg_MLPBlock is None else dict(cfg_MLPBlock)
    cfg_MLPBlock.setdefault("random_state", random_state)
    cfg_MLPBlock.setdefault("name", "enc_mlp_block")

    enc_in = Input(shape=(dim_seq, dim_in), name="encoder_input")

    x = layers.Flatten(name="enc_flatten")(enc_in)

    x = MLPBlock(**cfg_MLPBlock)(x)

    if variational:
        z_mean = layers.Dense(dim_z, activation=None, name="z_mean")(x)
        z_log_var = layers.Dense(dim_z, activation=None, name="z_log_var")(x)
        outputs = [z_mean, z_log_var]
    else:
        outputs = layers.Dense(dim_z, activation=None, name="z_mean")(x)

    return Model(enc_in, outputs, name=name)


def build_dense_vae_decoder(
    dim_seq: int,
    dim_in: int,
    dim_z: int,
    cfg_MLPBlock: dict | None = None,
    random_state: int | None = None,
    name: str = "decoder_dense") -> tf.keras.Model:
    """
    Build a dense VAE decoder.

    Architecture:
        Input(B, dim_z) -> MLPBlock -> Dense(T * F) -> Reshape(B, T, F)

    Parameters
    ----------
    dim_seq:
        Sequence length.
    dim_in:
        Output feature dimension.
    dim_z:
        Latent dimension.
    cfg_MLPBlock:
        Configuration dictionary passed to MLPBlock.
        For decoder symmetry, provide reversed dimensions if needed.
    random_state:
        Random seed propagated to MLPBlock if not already provided.
    name:
        Keras model name.

    Returns
    -------
    tf.keras.Model
        Dense decoder model.
    """
    cfg_MLPBlock = {} if cfg_MLPBlock is None else dict(cfg_MLPBlock)
    cfg_MLPBlock.setdefault("random_state", random_state)
    cfg_MLPBlock.setdefault("name", "dec_mlp_block")

    dec_in = Input(shape=(dim_z,), name="decoder_input")

    x = MLPBlock(**cfg_MLPBlock)(dec_in)

    x = layers.Dense(
        units=dim_seq * dim_in,
        activation=None,
        name="decoder_final_dense",
    )(x)

    out = layers.Reshape(target_shape=(dim_seq, dim_in),name="decoder_reshape")(x)
    return Model(dec_in, out, name=name)


def _enc_time_after_downsampling(dim_seq: int, n_layers: int) -> int:
    """
    Avec Conv1D(strides=2, padding='same'), la longueur est ~ ceil(dim_seq / 2^n_layers)
    """
    return int(math.ceil(dim_seq / (2 ** n_layers)))


def build_conv_vae_encoder(dim_seq: int, dim_in: int, dim_z: int, conv_filters: list[int],layers_size: list[int],variational=True,basic=True) -> tf.keras.Model:
    """
    Encoder (B, T, F) -> (z_mean, z_log_var, z), latent (B, dim_z).
    conv_filters: ex [50, 100, 200] (comme ton code)
    """
    enc_in = Input(shape=(dim_seq, dim_in), name="encoder_input")
    x = enc_in
    if (basic):
        for i, nf in enumerate(conv_filters):
            x = layers.Conv1D(
                filters=nf, kernel_size=3, strides=2, padding="same",
                activation="relu", name=f"enc_conv_{i}"
            )(x)
        x = layers.Flatten(name="enc_flatten")(x)
        
        for i, units in enumerate(layers_size):
            x = layers.Dense(units, activation="relu", name=f"dec_dense_{i}")(x)

    else:
        x = CNNEncoder(size_subseq_enc= dim_seq,
                       dim_target=dim_in,
                       dim_chan= 1,
        list_filters= conv_filters,
        list_kernels= [3 for i in conv_filters],
        list_strides= [2 for i in conv_filters],
        block= "1D",                 
        dim_z= dim_z,
        dp= 0.02,
        type_output=None,    
        random_state=None)(x)

    if(variational==True):
        z_mean = layers.Dense(dim_z, name="z_mean")(x)
        z_log_var = layers.Dense(dim_z, name="z_log_var")(x)
        output = [z_mean, z_log_var]
    else:
        output = layers.Dense(dim_z, name="z_mean")(x)

    return Model(enc_in, output, name="encoder_conv")


def build_conv_vae_decoder(dim_seq: int, dim_in: int, dim_z: int, conv_filters: list[int],basic=True) -> tf.keras.Model:
    """
    Decoder (B, dim_z) -> (B, T, F), miroir de l'encoder.
    On reconstruit la taille temps "encodée" T_enc, puis on remonte.
    """
    n_layers = len(conv_filters)
    last_nf = conv_filters[-1]
    T_enc = _enc_time_after_downsampling(dim_seq, n_layers)
    enc_flat_dim = T_enc * last_nf  # taille dense avant reshape (comme ton self.encoder_last_dense_dim)

    dec_in = Input(shape=(dim_z,), name="decoder_input")
    x = layers.Dense(enc_flat_dim, activation="relu", name="dec_dense")(dec_in)
    x = layers.Reshape((T_enc, last_nf), name="dec_reshape")(x)

    if(basic):
        # deconvs (miroir des convs sauf la dernière, cf. ton code)
        for i, nf in enumerate(reversed(conv_filters[:-1])):
            x = layers.Conv1DTranspose(
                filters=nf, kernel_size=3, strides=2, padding="same",
                activation="relu", name=f"dec_deconv_{i}"
            )(x)

        # dernière deconv pour revenir au nb de features, strides=2
        x = layers.Conv1DTranspose(
            filters=dim_in, kernel_size=3, strides=2, padding="same",
            activation="relu", name=f"dec_deconv_{len(conv_filters)-1}"
        )(x)
    else:
        x = CNNDecoder(size_subseq_dec=dim_seq,
                       dim_out=dim_seq,
                       dim_chan=dim_in,
                       list_filters= conv_filters,
                       list_kernels= [3 for i in conv_filters],
                       list_strides= [2 for i in conv_filters],
                       block= "1D",
                       dim_z= dim_z,
                       dp=0.02,
                       type_output=None,
                       min_logvar=-6.0,
                       random_state=None)(x)

    # finition identique à ton implé: flatten -> dense(T*F) -> reshape
    x = layers.Flatten(name="dec_flatten")(x)
    x = layers.Dense(dim_seq * dim_in, name="decoder_dense_final")(x)
    out = layers.Reshape((dim_seq, dim_in), name="decoder_reshape_final")(x)
    return Model(dec_in, out, name="decoder_conv")

# Model instanciation : 

class DenseAE(BaseAutoencoder):
    def __init__(
        self,
        dim_seq: int,
        dim_in: int,
        dim_z: int,
        cfg_MLPBlock: dict | None = None,
        name: str = "dense_ae",
        random_state: int | None = None,
        **kwargs):

        frame_locals = inspect.currentframe().f_locals
        explicit_args = {
            k: v for k, v in frame_locals.items()
            if k not in ("self", "kwargs")
        }
        all_init_params = {**explicit_args, **kwargs}
        super().__init__(**all_init_params)

        cfg_encoder_MLPBlock = {} if cfg_MLPBlock is None else dict(cfg_MLPBlock)
        cfg_encoder_MLPBlock.setdefault("random_state", random_state)
        cfg_encoder_MLPBlock.setdefault("name", "encoder_mlp_block")

        cfg_decoder_MLPBlock = {} if cfg_MLPBlock is None else dict(cfg_MLPBlock)
        cfg_decoder_MLPBlock.setdefault("random_state", random_state)
        cfg_decoder_MLPBlock.setdefault("name", "decoder_mlp_block")

        if "dim_layers" in cfg_decoder_MLPBlock:
            cfg_decoder_MLPBlock["dim_layers"] = tuple(reversed(cfg_decoder_MLPBlock["dim_layers"]))

        self.encoder = build_dense_vae_encoder(
            dim_seq=dim_seq,
            dim_in=dim_in,
            dim_z=dim_z,
            variational=False,
            cfg_MLPBlock=cfg_encoder_MLPBlock,
            random_state=random_state,
            name="encoder_dense")

        self.decoder = build_dense_vae_decoder(
            dim_seq=dim_seq,
            dim_in=dim_in,
            dim_z=dim_z,
            cfg_MLPBlock=cfg_decoder_MLPBlock,
            random_state=random_state,
            name="decoder_dense")

class DenseVAE(BaseVariationalAutoencoder):
    def __init__(
        self,
        dim_seq: int,
        dim_in: int,
        dim_z: int,
        cfg_MLPBlock: dict | None = None,
        kl_weight: float = 0.1,
        name: str = "dense_vae",
        random_state: int | None = None,
        **kwargs,
    ):
        frame_locals = inspect.currentframe().f_locals
        explicit_args = {
            k: v for k, v in frame_locals.items()
            if k not in ("self", "kwargs")
        }
        all_init_params = {**explicit_args, **kwargs}
        super().__init__(**all_init_params)

        cfg_encoder_MLPBlock = {} if cfg_MLPBlock is None else dict(cfg_MLPBlock)
        cfg_encoder_MLPBlock.setdefault("random_state", random_state)
        cfg_encoder_MLPBlock.setdefault("name", "encoder_mlp_block")

        cfg_decoder_MLPBlock = {} if cfg_MLPBlock is None else dict(cfg_MLPBlock)
        cfg_decoder_MLPBlock.setdefault("random_state", random_state)
        cfg_decoder_MLPBlock.setdefault("name", "decoder_mlp_block")

        if "dim_layers" in cfg_decoder_MLPBlock:
            cfg_decoder_MLPBlock["dim_layers"] = tuple(
                reversed(cfg_decoder_MLPBlock["dim_layers"])
            )

        self.encoder = build_dense_vae_encoder(
            dim_seq=dim_seq,
            dim_in=dim_in,
            dim_z=dim_z,
            variational=True,
            cfg_MLPBlock=cfg_encoder_MLPBlock,
            random_state=random_state,
            name="encoder_dense",
        )

        self.decoder = build_dense_vae_decoder(
            dim_seq=dim_seq,
            dim_in=dim_in,
            dim_z=dim_z,
            cfg_MLPBlock=cfg_decoder_MLPBlock,
            random_state=random_state,
            name="decoder_dense",
        )

# ConvAE
class ConvAE(BaseAutoencoder):
    def __init__(self, dim_seq, dim_in, dim_z, conv_filters=(32,64,128), name="conv_ae", basic=True,**kwargs):
        frame_locals = inspect.currentframe().f_locals
        explicit_args = {k: v for k, v in frame_locals.items() if k not in ("self", "kwargs")}
        all_init_params = {**explicit_args, **kwargs}
        super().__init__(**all_init_params)
        self.encoder = build_conv_vae_encoder(dim_seq, dim_in, dim_z, list(conv_filters),variational=False,basic=basic)
        self.decoder = build_conv_vae_decoder(dim_seq, dim_in, dim_z, list(conv_filters),basic=basic)


class ConvVAE(BaseVariationalAutoencoder):
    def __init__(self, dim_seq, dim_in, dim_z, conv_filters=(32,64,128), kl_weight=1.0, name="conv_vae", basic=True,**kwargs):
        frame_locals = inspect.currentframe().f_locals
        explicit_args = {k: v for k, v in frame_locals.items() if k not in ("self", "kwargs")}
        all_init_params = {**explicit_args, **kwargs}
        super().__init__(**all_init_params)
        self.encoder = build_conv_vae_encoder(dim_seq, dim_in, dim_z, list(conv_filters),variational=True,basic=basic)
        self.decoder = build_conv_vae_decoder(dim_seq, dim_in, dim_z, list(conv_filters),basic=basic)