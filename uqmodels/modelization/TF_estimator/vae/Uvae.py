#################################################
# In progress : doesn't currently work

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras import optimizers
from uqmodels.modelization.TF_estimator.vae.base_vae import BaseAutoencoder,BaseVariationalAutoencoder

def _double_conv(x, nf, name):
    x = layers.Conv2D(nf, 3, padding="same", activation="relu", name=f"{name}_conv1")(x)
    x = layers.Conv2D(nf, 3, padding="same", activation="relu", name=f"{name}_conv2")(x)
    return x


def build_unet_vae_encoder(
    in_shape: tuple[int, int, int], latent_dim: int, base_filters: int = 32, depth: int = 4,mode_vae=True
) -> tf.keras.Model:
    """
    Encoder U-Net (2D):
      - retourne (z_mean, z_log_var, z, skips)
      - 'skips' = liste de tensors des features descendantes (pour la remontée)
    """
    inp = Input(shape=in_shape, name="unet_enc_input")
    x = inp
    skips = []
    nf = base_filters

    # Down path
    for d in range(depth):
        x = _double_conv(x, nf, name=f"down{d}")
        skips.append(x)
        x = layers.MaxPooling2D(2, name=f"down{d}_pool")(x)
        nf *= 2

    # Bottleneck (spatialement réduit)
    x = _double_conv(x, nf, name="bottleneck")

    # Vec latent
    x_flat = layers.Flatten(name="bottleneck_flat")(x)
    
    if(mode_vae==True):
        z_mean = layers.Dense(latent_dim, name="z_mean")(x_flat)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x_flat)
        output = [z_mean, z_log_var]
    else:
        z_mean = layers.Dense(latent_dim, name="z_mean")(x_flat)
        output = [z_mean]

    # On renvoie aussi la shape spatiale du bottleneck pour reshape côté decoder
    bottleneck_shape = tf.shape(x)  # dynamique; on passera plutôt la taille attendue dans le decoder

    # Keras Model ne peut pas renvoyer des objets Python; on renvoie la liste 'skips' comme sorties séparées
    # Convention: [z_mean, z_log_var, z] + skips (du plus profond au plus superficiel)
    return Model(inp, output + skips, name="unet_encoder")


def build_unet_vae_decoder(
    out_shape: tuple[int, int, int], latent_dim: int, base_filters: int = 32, depth: int = 4
) -> tf.keras.Model:
    """
    Decoder U-Net (2D):
      - inputs = [z] + skips (dans le même ordre que la sortie encoder)
      - reconstruit l'image (B, H, W, C)
    """
    # Entrées: z + une entrée par skip
    z_in = Input(shape=(latent_dim,), name="z_in")
    skip_ins = [Input(shape=None, name=f"skip_in_{i}") for i in range(depth)]  # shapes dynamiques → None

    # On a besoin de connaître la taille spatiale du bottleneck avant flatten.
    # Astuce: la re-projeter depuis z via un Dense, en supposant H_bott = H / 2**depth (idem pour W)
    H, W, C = out_shape
    H_b = H // (2 ** depth)
    W_b = W // (2 ** depth)
    nf = (base_filters * (2 ** (depth - 1))) * 2  # même nf que dans le bottleneck encoder

    x = layers.Dense(H_b * W_b * nf, activation="relu", name="z_proj")(z_in)
    x = layers.Reshape((H_b, W_b, nf), name="z_reshape")(x)

    # Up path (concat skips en ordre inverse)
    nf //= 2
    for d in reversed(range(depth)):
        x = layers.Conv2DTranspose(nf, 2, strides=2, padding="same", name=f"up{d}_up")(x)
        x = layers.Concatenate(name=f"up{d}_concat")([x, skip_ins[d]])
        x = _double_conv(x, nf, name=f"up{d}_conv")
        nf //= 2

    out = layers.Conv2D(C, 1, activation=None, name="recon_out")(x)
    return Model([z_in] + skip_ins, out, name="unet_decoder")

class UNetAE(BaseAutoencoder):
    def __init__(self, seq_len,input_dim, latent_dim, base_filters=32, depth=4, name="unet_ae"):
        in_shape = (seq_len,input_dim)
        super().__init__(name=name)
        self.encoder = build_unet_vae_encoder(in_shape, latent_dim, base_filters, depth,mode_vae=False)
        self.decoder = build_unet_vae_decoder(in_shape, latent_dim, base_filters, depth)

class UNetVAE(BaseVariationalAutoencoder):
    def __init__(self, seq_len,input_dim, latent_dim, base_filters=32, depth=4, kl_weight=1.0, name="unet_vae"):
        in_shape = (seq_len,input_dim)
        super().__init__(kl_weight=kl_weight, name=name)
        self.encoder = build_unet_vae_encoder(in_shape, latent_dim, base_filters, depth,mode_vae=False)
        self.decoder = build_unet_vae_decoder(in_shape, latent_dim, base_filters, depth)