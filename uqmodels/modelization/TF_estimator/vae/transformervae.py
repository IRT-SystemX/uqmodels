import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from uqmodels.modelization.TF_estimator.vae.base_vae import BaseAutoencoder,BaseVariationalAutoencoder
# ---------------------------------------------------------------------
# Positional embedding & Transformer blocks
# ---------------------------------------------------------------------

class PositionalEmbedding(layers.Layer):
    """
    Learnable positional embeddings added to token representations.
    Inputs:  (B, T, D)
    Outputs: (B, T, D)
    """
    def __init__(self, seq_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.pos_emb = self.add_weight(
            name="pos_embedding",
            shape=(self.seq_len, self.d_model),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, x):
        # x: (B, T, D)
        return x + self.pos_emb

class TransformerEncoderBlock(layers.Layer):
    """
    Standard Transformer encoder block: MHSA + FFN (PreNorm).
    """
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.drop1 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation="relu"),
            layers.Dense(d_model),
        ])
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False, mask=None):
        # Self-attention
        h = self.norm1(x)
        h = self.mha(h, h, attention_mask=mask, training=training)
        h = self.drop1(h, training=training)
        x = x + h
        # FFN
        h2 = self.norm2(x)
        h2 = self.ffn(h2, training=training)
        h2 = self.drop2(h2, training=training)
        return x + h2

class TransformerDecoderBlock(layers.Layer):
    """
    Simple Transformer decoder block with self-attention only (no cross-attn).
    For autoencoding, this works well when we pre-project latent into a sequence.
    """
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.drop1 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation="relu"),
            layers.Dense(d_model),
        ])
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False, mask=None):
        # Self-attention (optionally causal mask can be provided via mask)
        h = self.norm1(x)
        h = self.mha(h, h, attention_mask=mask, training=training)
        h = self.drop1(h, training=training)
        x = x + h
        # FFN
        h2 = self.norm2(x)
        h2 = self.ffn(h2, training=training)
        h2 = self.drop2(h2, training=training)
        return x + h2

# ---------------------------------------------------------------------
# Builders: Transformer Encoder / Decoder
# ---------------------------------------------------------------------

def build_transformer_encoder(
    seq_len: int,
    input_dim: int,
    latent_dim: int,
    *,
    num_layers: int = 3,
    d_model: int = 128,
    num_heads: int = 4,
    dff: int = 256,
    dropout: float = 0.1,
    variational: bool = False,
    latent_as_sequence: bool = False,
    pooling: str = "mean",  # mean | last | cls
    name: str = "t_encoder",
) -> tf.keras.Model:
    """
    Build a Transformer encoder for sequences (B, T, F).
    If variational=False: returns z
      - z shape: (B, latent_dim) if latent_as_sequence=False; (B, T, latent_dim) otherwise.
    If variational=True: returns [z_mean, z_log_var, z] with same shape logic.

    Notes:
      - When latent_as_sequence=False, we project to (B, T, d_model), add pos-emb,
        stack encoder blocks, then pool over time to a vector, then Dense(latent_dim).
      - When latent_as_sequence=True, we keep sequence length T and emit (B, T, latent_dim).
    """
    inp = Input(shape=(seq_len, input_dim), name=f"{name}_input")
    x = layers.Dense(d_model, name=f"{name}_proj")(inp)
    x = PositionalEmbedding(seq_len, d_model, name=f"{name}_pos")(x)

    for i in range(num_layers):
        x = TransformerEncoderBlock(d_model, num_heads, dff, dropout, name=f"{name}_blk{i}")(x)

    if latent_as_sequence:
        # keep sequence, project per token
        z_core = layers.Dense(latent_dim, name=f"{name}_to_latent_td")(x)  # (B, T, D_lat)
    else:
        # pool to a vector
        if pooling == "mean":
            z_pool = tf.reduce_mean(x, axis=1)  # (B, D)
        elif pooling == "last":
            z_pool = x[:, -1, :]               # (B, D)
        elif pooling == "cls":
            # prepend a learnable [CLS] token equivalent
            # (simple option: just take first token)
            z_pool = x[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        z_core = layers.Dense(latent_dim, name=f"{name}_to_latent")(z_pool)  # (B, D_lat)

    if not variational:
        return Model(inp, z_core, name=name)

    # Variational heads + sampling
    if latent_as_sequence:
        z_mean = layers.Dense(latent_dim, name=f"{name}_z_mean_td")(x)
        z_log_var = layers.Dense(latent_dim, name=f"{name}_z_log_var_td")(x)
    else:
        z_mean = layers.Dense(latent_dim, name=f"{name}_z_mean")(z_core)
        z_log_var = layers.Dense(latent_dim, name=f"{name}_z_log_var")(z_core)

    # Sampling that supports (B, D) or (B, T, D)
    def sample(inputs):
        mu, logv = inputs
        eps = tf.random.normal(tf.shape(mu))
        return mu + tf.exp(0.5 * logv) * eps

    z = layers.Lambda(sample, name=f"{name}_z_sample")([z_mean, z_log_var])

    return Model(inp, [z_mean, z_log_var, z], name=name)


def build_transformer_decoder(
    seq_len: int,
    input_dim: int,
    latent_dim: int,
    *,
    num_layers: int = 3,
    d_model: int = 128,
    num_heads: int = 4,
    dff: int = 256,
    dropout: float = 0.1,
    latent_as_sequence: bool = False,
    name: str = "t_decoder",
) -> tf.keras.Model:
    """
    Build a Transformer decoder that maps latent -> reconstructed sequence (B, T, F).

    - If latent_as_sequence=False:
        input: (B, latent_dim) -> Dense(T * d_model) -> reshape (B,T,D)
        + positional embedding -> N x decoder blocks -> Dense(F)
    - If latent_as_sequence=True:
        input: (B, T, latent_dim) -> Dense(d_model) -> +pos -> decoder blocks -> Dense(F)
    """
    if latent_as_sequence:
        inp = Input(shape=(seq_len, latent_dim), name=f"{name}_input_seq")
        x = layers.Dense(d_model, name=f"{name}_proj_td")(inp)         # (B, T, D)
    else:
        inp = Input(shape=(latent_dim,), name=f"{name}_input_vec")
        x = layers.Dense(seq_len * d_model, name=f"{name}_proj_vec")(inp)
        x = layers.Reshape((seq_len, d_model), name=f"{name}_reshape")(x)

    x = PositionalEmbedding(seq_len, d_model, name=f"{name}_pos")(x)

    # (Optional) causal mask if you want autoregressive style:
    # causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

    for i in range(num_layers):
        x = TransformerDecoderBlock(d_model, num_heads, dff, dropout, name=f"{name}_blk{i}")(x)

    out = layers.Dense(input_dim, name=f"{name}_to_feat")(x)  # (B, T, F)
    return Model(inp, out, name=name)

# ---------------------------------------------------------------------
# AE / VAE classes (compose with your framework)
# ---------------------------------------------------------------------

class AETransformer(BaseAutoencoder):
    """
    Transformer Autoencoder for sequences.
    - Encoder outputs z (vector or sequence).
    """
    def __init__(
        self,
        seq_len: int,
        input_dim: int,
        latent_dim: int,
        *,
        num_layers: int = 3,
        d_model: int = 128,
        num_heads: int = 4,
        dff: int = 256,
        dropout: float = 0.1,
        latent_as_sequence: bool = False,
        pooling: str = "mean",
        name: str = "ae_transformer",
        **kwargs
    ):
        super().__init__(name=name,**kwargs)
        self.seq_len = int(seq_len)
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)

        self.encoder = build_transformer_encoder(
            seq_len, input_dim, latent_dim,
            num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, dropout=dropout,
            variational=False, latent_as_sequence=latent_as_sequence, pooling=pooling,
            name="ae_tenc",
        )
        self.decoder = build_transformer_decoder(
            seq_len, input_dim, latent_dim,
            num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, dropout=dropout,
            latent_as_sequence=latent_as_sequence,
            name="ae_tdec",
        )

class VAETransformer(BaseVariationalAutoencoder):
    """
    Transformer VAE for sequences.
    - Relies on BaseVariationalAutoencoder to compute recon + KL automatically.
    - Encoder returns (z_mean, z_log_var, z).
    """
    def __init__(
        self,
        seq_len: int,
        input_dim: int,
        latent_dim: int,
        *,
        num_layers: int = 2,
        d_model: int = 64,
        num_heads: int = 2,
        dff: int = 128,
        dropout: float = 0.1,
        latent_as_sequence: bool = False,
        pooling: str = "mean",
        kl_weight: float = 1.0,
        name: str = "vae_transformer",
        **kwargs
    ):
        # Compose bases
        super().__init__(name=name,kl_weight=kl_weight, **kwargs)
        # SequenceMixin has no state; no explicit __init__ needed.

        self.seq_len = int(seq_len)
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)

        self.encoder = build_transformer_encoder(
            seq_len, input_dim, latent_dim,
            num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, dropout=dropout,
            variational=True, latent_as_sequence=latent_as_sequence, pooling=pooling,
            name="vae_tenc",
        )
        self.decoder = build_transformer_decoder(
            seq_len, input_dim, latent_dim,
            num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, dropout=dropout,
            latent_as_sequence=latent_as_sequence,
            name="vae_tdec",
        )