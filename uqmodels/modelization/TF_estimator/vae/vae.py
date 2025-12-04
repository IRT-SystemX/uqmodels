import tensorflow as tf
import math
import inspect
from tensorflow.keras import layers,Model
from tensorflow.keras.layers import Input
from uqmodels.modelization.TF_estimator.vae.base_vae import BaseAutoencoder,BaseVariationalAutoencoder,VariationalMixin

from uqmodels.modelization.TF_estimator.base.layers import MLPLayer
from uqmodels.modelization.TF_estimator.base.convlayers import CNNEncoder,CNNDecoder

def build_dense_vae_encoder(seq_len: int, input_dim: int, latent_dim: int, hidden_layer_sizes: list[int], variational=True,basic=True) -> tf.keras.Model:
    """
    Encoder (B, T, F) -> (z_mean, z_log_var, z) with z of shape (B, latent_dim)
    """
    enc_in = Input(shape=(seq_len, input_dim), name="encoder_input")

    if(basic):
        x = layers.Flatten(name="enc_flatten")(enc_in)
        for i, units in enumerate(hidden_layer_sizes):
            x = layers.Dense(units, activation="relu", name=f"enc_dense_{i}")(x)

    else:
        mlp = MLPLayer(dim_in=seq_len*input_dim,
                            dim_out=latent_dim,
                            layers_size=hidden_layer_sizes,
                            dp=0.01,
                            mc_dropout=False,
                            type_output=None,
                            logvar_min= -10.0,
                            regularizer_W = (1e-9, 1e-9),
                            shape_2D= (seq_len, input_dim),
                            shape_2D_out = None,
                            random_state = None,
                            activation_hidden= None)    
        x = mlp(enc_in[:,:,:,None])

    if(variational==True):
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        output = [z_mean, z_log_var]
    else:
           output = layers.Dense(latent_dim, name="z_mean")(x)

    return Model(enc_in, output, name="encoder_dense")


def build_dense_vae_decoder(seq_len: int, input_dim: int, latent_dim: int, hidden_layer_sizes: list[int],basic=True) -> tf.keras.Model:
    """
    Decoder (B, latent_dim) -> (B, T, F)
    """
    dec_in = Input(shape=(latent_dim,), name="decoder_input")
    x = dec_in
    if(basic):
        for i, units in enumerate(reversed(hidden_layer_sizes)):
            x = layers.Dense(units, activation="relu", name=f"dec_dense_{i}")(x)
        x = layers.Dense(seq_len * input_dim, name="decoder_final_dense")(x)
        out = layers.Reshape((seq_len, input_dim), name="decoder_reshape")(x)
    else:
        mlp = MLPLayer(dim_in=latent_dim,
                       dim_out=seq_len * input_dim,
                       layers_size=hidden_layer_sizes,
                       dp=0.01,
                       mc_dropout=False,
                       type_output=None,
                       logvar_min= -10.0,
                       regularizer_W = (1e-9, 1e-9),
                       shape_2D = None,
                       shape_2D_out =(seq_len,input_dim),
                       random_state = None,
                       activation_hidden= None)    
        out = mlp(dec_in)
    return Model(dec_in, out, name="decoder_dense")


def _enc_time_after_downsampling(seq_len: int, n_layers: int) -> int:
    """
    Avec Conv1D(strides=2, padding='same'), la longueur est ~ ceil(seq_len / 2^n_layers)
    """
    return int(math.ceil(seq_len / (2 ** n_layers)))


def build_conv_vae_encoder(seq_len: int, input_dim: int, latent_dim: int, conv_filters: list[int],variational=True,basic=True) -> tf.keras.Model:
    """
    Encoder (B, T, F) -> (z_mean, z_log_var, z), latent (B, latent_dim).
    conv_filters: ex [50, 100, 200] (comme ton code)
    """
    enc_in = Input(shape=(seq_len, input_dim), name="encoder_input")
    x = enc_in
    if (basic):
        for i, nf in enumerate(conv_filters):
            x = layers.Conv1D(
                filters=nf, kernel_size=3, strides=2, padding="same",
                activation="relu", name=f"enc_conv_{i}"
            )(x)
        x = layers.Flatten(name="enc_flatten")(x)

    else:
        x = CNNEncoder(size_subseq_enc= seq_len,
                       dim_target=input_dim,
                       dim_chan= 1,
        list_filters= conv_filters,
        list_kernels= [3 for i in conv_filters],
        list_strides= [2 for i in conv_filters],
        block= "1D",                 
        dim_z= latent_dim,
        dp= 0.02,
        type_output=None,    
        random_state=None)(x)


    if(variational==True):
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        output = [z_mean, z_log_var]
    else:
        output = layers.Dense(latent_dim, name="z_mean")(x)

    return Model(enc_in, output, name="encoder_conv")


def build_conv_vae_decoder(seq_len: int, input_dim: int, latent_dim: int, conv_filters: list[int],basic=True) -> tf.keras.Model:
    """
    Decoder (B, latent_dim) -> (B, T, F), miroir de l'encoder.
    On reconstruit la taille temps "encodée" T_enc, puis on remonte.
    """
    n_layers = len(conv_filters)
    last_nf = conv_filters[-1]
    T_enc = _enc_time_after_downsampling(seq_len, n_layers)
    enc_flat_dim = T_enc * last_nf  # taille dense avant reshape (comme ton self.encoder_last_dense_dim)

    dec_in = Input(shape=(latent_dim,), name="decoder_input")
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
            filters=input_dim, kernel_size=3, strides=2, padding="same",
            activation="relu", name=f"dec_deconv_{len(conv_filters)-1}"
        )(x)
    else:
        x = CNNDecoder(size_subseq_dec=seq_len,
                       dim_out=seq_len,
                       dim_chan=input_dim,
                       list_filters= conv_filters,
                       list_kernels= [3 for i in conv_filters],
                       list_strides= [2 for i in conv_filters],
                       block= "1D",
                       dim_z= latent_dim,
                       dp=0.02,
                       type_output=None,
                       min_logvar=-6.0,
                       random_state=None)(x)

    # finition identique à ton implé: flatten -> dense(T*F) -> reshape
    x = layers.Flatten(name="dec_flatten")(x)
    x = layers.Dense(seq_len * input_dim, name="decoder_dense_final")(x)
    out = layers.Reshape((seq_len, input_dim), name="decoder_reshape_final")(x)
    return Model(dec_in, out, name="decoder_conv")

# Model instanciation : 

class DenseAE(BaseAutoencoder):
    def __init__(self, seq_len, input_dim, latent_dim, hidden=(128,128), name="dense_ae", basic=True,**kwargs):
        frame_locals = inspect.currentframe().f_locals
        explicit_args = {k: v for k, v in frame_locals.items() if k not in ("self", "kwargs")}
        all_init_params = {**explicit_args, **kwargs}
        super().__init__(**all_init_params)
        self.encoder = build_dense_vae_encoder(seq_len, input_dim, latent_dim, list(hidden),variational=False,basic=basic)
        self.decoder = build_dense_vae_decoder(seq_len, input_dim, latent_dim, list(hidden),basic=basic)

class DenseVAE(BaseVariationalAutoencoder):
    def __init__(self, seq_len, input_dim, latent_dim, hidden=(128,128), kl_weight=1.0, name="dense_vae", basic=True,**kwargs):
        frame_locals = inspect.currentframe().f_locals
        explicit_args = {k: v for k, v in frame_locals.items() if k not in ("self", "kwargs")}
        all_init_params = {**explicit_args, **kwargs}
        super().__init__(**all_init_params)
        self.encoder = build_dense_vae_encoder(seq_len, input_dim, latent_dim, list(hidden),variational=True,basic=basic)
        self.decoder = build_dense_vae_decoder(seq_len, input_dim, latent_dim, list(hidden),basic=basic)

# ConvAE
class ConvAE(BaseAutoencoder):
    def __init__(self, seq_len, input_dim, latent_dim, conv_filters=(32,64,128), name="conv_ae", basic=True,**kwargs):
        frame_locals = inspect.currentframe().f_locals
        explicit_args = {k: v for k, v in frame_locals.items() if k not in ("self", "kwargs")}
        all_init_params = {**explicit_args, **kwargs}
        super().__init__(**all_init_params)
        self.encoder = build_conv_vae_encoder(seq_len, input_dim, latent_dim, list(conv_filters),variational=False,basic=basic)
        self.decoder = build_conv_vae_decoder(seq_len, input_dim, latent_dim, list(conv_filters),basic=basic)


class ConvVAE(BaseVariationalAutoencoder):
    def __init__(self, seq_len, input_dim, latent_dim, conv_filters=(32,64,128), kl_weight=1.0, name="conv_vae", basic=True,**kwargs):
        frame_locals = inspect.currentframe().f_locals
        explicit_args = {k: v for k, v in frame_locals.items() if k not in ("self", "kwargs")}
        all_init_params = {**explicit_args, **kwargs}
        super().__init__(**all_init_params)
        self.encoder = build_conv_vae_encoder(seq_len, input_dim, latent_dim, list(conv_filters),variational=True,basic=basic)
        self.decoder = build_conv_vae_decoder(seq_len, input_dim, latent_dim, list(conv_filters),basic=basic)