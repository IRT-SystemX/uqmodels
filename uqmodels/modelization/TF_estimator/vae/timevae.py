import tensorflow as tf
import math
import inspect
from tensorflow.keras import layers,Model
from tensorflow.keras.layers import Input
from uqmodels.modelization.TF_estimator.vae.base_vae import VariationalMixin,BaseAutoencoder,BaseVariationalAutoencoder,BaseVariationalAutoencoder

# ---------- Composantes decoder ----------
class TrendLayer(layers.Layer):
    """Construit une tendance polynomiale (jusqu'à ordre P) pour chaque feature, à partir de z."""
    def __init__(self, input_dim: int, trend_poly: int, seq_len: int, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.trend_poly = int(trend_poly)
        self.seq_len = int(seq_len)
        self.trend_dense1 = layers.Dense(self.input_dim * self.trend_poly, activation="relu", name="trend_params")
        self.trend_dense2 = layers.Dense(self.input_dim * self.trend_poly, name="trend_params2")
        self.reshape_layer = layers.Reshape(target_shape=(self.input_dim, self.trend_poly))

    def call(self, z):
        # (N, D_lat) -> (N, D_feat, P) -> appliquer base polynomiale (P,T) -> (N,T,D_feat)
        trend_params = self.reshape_layer(self.trend_dense2(self.trend_dense1(z)))
        lin_space = tf.range(0.0, float(self.seq_len), 1.0) / float(self.seq_len)       # (T,)
        poly_space = tf.stack([lin_space ** float(p + 1) for p in range(self.trend_poly)], axis=0)  # (P, T)
        trend_vals = tf.matmul(trend_params, poly_space)   # (N, D_feat, T)
        trend_vals = tf.transpose(trend_vals, perm=[0, 2, 1])  # (N, T, D_feat)
        return tf.cast(trend_vals, tf.float32)

class SeasonalLayer(layers.Layer):
    """Saisonnalités custom: liste de (num_seasons, len_per_season)."""
    def __init__(self, input_dim: int, seq_len: int, custom_seas: list[tuple[int, int]], **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.custom_seas = custom_seas
        self.dense_layers = [layers.Dense(input_dim * s, name=f"season_params_{i}") for i, (s, L) in enumerate(custom_seas)]
        self.reshape_layers = [layers.Reshape(target_shape=(input_dim, s)) for s, L in custom_seas]

    def _get_season_indexes_over_seq(self, num_seasons, len_per_season):
        season_idx = tf.range(num_seasons)[:, None] + tf.zeros((num_seasons, len_per_season), dtype=tf.int32)
        season_idx = tf.reshape(season_idx, [-1])  # (num_seasons*len_per_season,)
        season_idx = tf.tile(season_idx, [self.seq_len // len_per_season + 1])[: self.seq_len]  # (T,)
        return season_idx  # (T,)

    def call(self, z):
        N = tf.shape(z)[0]
        ones = tf.ones(shape=[N, self.input_dim, self.seq_len], dtype=tf.int32)  # (N, D, T)
        all_vals = []
        for i, (num_seasons, L) in enumerate(self.custom_seas):
            params = self.reshape_layers[i](self.dense_layers[i](z))  # (N, D, S)
            idx_t = self._get_season_indexes_over_seq(num_seasons, L)  # (T,)
            dim2_idx = ones * tf.reshape(idx_t, (1, 1, -1))            # (N, D, T)
            vals = tf.gather(params, dim2_idx, batch_dims=-1)          # (N, D, T)
            all_vals.append(vals)
        stacked = tf.stack(all_vals, axis=-1)        # (N, D, T, S_groups)
        summed = tf.reduce_sum(stacked, axis=-1)     # (N, D, T)
        return tf.transpose(summed, perm=[0, 2, 1])  # (N, T, D)
    
def build_time_vae_encoder(seq_len: int, input_dim: int, latent_dim: int, conv_filters: list[int],variational=True) -> tf.keras.Model:
    """
    Encoder (B, T, F) -> (z_mean, z_log_var, z) avec z de shape (B, latent_dim).
    Conv1D strides=2 "same" comme dans le script TimeVAE.
    """
    enc_in = Input(shape=(seq_len, input_dim), name="encoder_input")
    x = enc_in
    for i, nf in enumerate(conv_filters):
        x = layers.Conv1D(filters=nf, kernel_size=3, strides=2, padding="same",
                          activation="relu", name=f"enc_conv_{i}")(x)
    x = layers.Flatten(name="enc_flatten")(x)
    
    if(variational==True):
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        output = [z_mean, z_log_var]
    else:
        output = layers.Dense(latent_dim, name="z_mean")(x)
    return Model(enc_in, output, name="encoder_time")

def _enc_time_after_downsampling(seq_len: int, n_layers: int) -> int:
    # Approx de TimeVAE: Conv1D stride=2 "same" => ceil(T / 2^n)
    return int(math.ceil(seq_len / (2 ** n_layers)))

def build_time_vae_decoder(
    seq_len: int,
    input_dim: int,
    latent_dim: int,
    conv_filters: list[int],
    *,
    trend_poly: int = 0,
    custom_seas: list[tuple[int, int]] | None = None,
    use_residual_conn: bool = True,
) -> tf.keras.Model:
    """
    Decoder (B, latent_dim) -> (B, T, F).
    Reprend le schéma: level + (trend?) + (seasonalities?) + (residual conv-deconv?) additionnés.
    Retourne un TENSEUR (pas [outputs]) pour compatibilité Keras.
    """
    dec_in = Input(shape=(latent_dim,), name="decoder_input")

    # --- composante "level" (constante dans le temps) ---
    level = layers.Dense(input_dim, activation="relu", name="level_params")(dec_in)
    level = layers.Dense(input_dim, name="level_params2")(level)
    level = layers.Reshape((1, input_dim))(level)                # (N,1,F)
    ones_T = tf.ones((1, seq_len, 1), dtype=tf.float32)
    level_vals = level * ones_T                                 # (N,T,F)

    outputs = level_vals

    # --- tendance polynomiale (optionnelle) ---
    if trend_poly and trend_poly > 0:
        trend_vals = TrendLayer(input_dim, trend_poly, seq_len)(dec_in)  # (N,T,F)
        outputs = outputs + trend_vals

    # --- saisonnalités custom (optionnelles) ---
    if custom_seas and len(custom_seas) > 0:
        seas_vals = SeasonalLayer(input_dim, seq_len, custom_seas)(dec_in)  # (N,T,F)
        outputs = outputs + seas_vals

    # --- résiduel conv-transpose miroir (optionnel) ---
    if use_residual_conn and conv_filters:
        n_layers = len(conv_filters)
        last_nf = conv_filters[-1]
        T_enc = _enc_time_after_downsampling(seq_len, n_layers)
        enc_flat_dim = T_enc * last_nf

        x = layers.Dense(enc_flat_dim, activation="relu", name="dec_dense")(dec_in)
        x = layers.Reshape((T_enc, last_nf), name="dec_reshape")(x)

        for i, nf in enumerate(reversed(conv_filters[:-1])):
            x = layers.Conv1DTranspose(filters=nf, kernel_size=3, strides=2, padding="same",
                                       activation="relu", name=f"dec_deconv_{i}")(x)
        x = layers.Conv1DTranspose(filters=input_dim, kernel_size=3, strides=2, padding="same",
                                   activation="relu", name=f"dec_deconv_{len(conv_filters)-1}")(x)

        x = layers.Flatten(name="dec_flatten")(x)
        x = layers.Dense(seq_len * input_dim, name="decoder_dense_final")(x)
        residuals = layers.Reshape((seq_len, input_dim), name="decoder_reshape_final")(x)
        outputs = outputs + residuals

    return Model(dec_in, outputs, name="decoder_time")


class TimeAE(BaseAutoencoder):
    def __init__(self, seq_len, input_dim, latent_dim, conv_filters=(50,100,200),
                 trend_poly=0, custom_seas=None, use_residual_conn=True, name="time_vae", **kwargs):
        frame_locals = inspect.currentframe().f_locals
        explicit_args = {k: v for k, v in frame_locals.items() if k not in ("self", "kwargs")}
        all_init_params = {**explicit_args, **kwargs}
        super().__init__(**all_init_params)
        self.encoder = build_time_vae_encoder(seq_len, input_dim, latent_dim, list(conv_filters),variational=False)
        self.decoder = build_time_vae_decoder(
            seq_len, input_dim, latent_dim, list(conv_filters),
            trend_poly=trend_poly, custom_seas=custom_seas, use_residual_conn=use_residual_conn
        )

class TimeVAE(BaseVariationalAutoencoder):
    def __init__(self, seq_len, input_dim, latent_dim, conv_filters=(50,100,200),
                 trend_poly=0, custom_seas=None, use_residual_conn=True,
                 kl_weight=1.0, name="time_vae", **kwargs):

        frame_locals = inspect.currentframe().f_locals
        explicit_args = {k: v for k, v in frame_locals.items() if k not in ("self", "kwargs")}
        all_init_params = {**explicit_args, **kwargs}
        super().__init__(**all_init_params)
        self.encoder = build_time_vae_encoder(seq_len, input_dim, latent_dim, list(conv_filters),variational=True)
        self.decoder = build_time_vae_decoder(
            seq_len, input_dim, latent_dim, list(conv_filters),
            trend_poly=trend_poly, custom_seas=custom_seas, use_residual_conn=use_residual_conn
        )