####### Depreciated ###############

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers

class TransformerEncoderBlock(layers.Layer):
    """
    PreNorm Transformer encoder block: LN -> MHSA -> Dropout -> Residual
                                     -> LN -> MLP  -> Dropout -> Residual
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: int = 4,
        attn_dropout: float = 0.0,
        dropout: float = 0.1,
        activation: str = "gelu",
        mc_dropout: bool = False,
        seed: int | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn_dropout = attn_dropout
        self.dropout = dropout
        self.activation = activation
        self.mc_dropout = mc_dropout
        self.seed = seed

        # Layers
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=attn_dropout,
        )
        self.drop1 = layers.Dropout(dropout, seed=seed)

        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            layers.Dense(mlp_ratio * d_model, activation=activation),
            layers.Dense(d_model),
        ])
        self.drop2 = layers.Dropout(dropout, seed=None if seed is None else seed + 1)

    def call(self, x, training=False, mask=None):
        # Self-attention (PreNorm)
        h = self.norm1(x)
        # Keras MHA expects attention_mask shaped (batch, q, k) or broadcastable; pass None if not provided
        h = self.attn(h, h, attention_mask=mask, training=training)
        h = self.drop1(h, training=(training or self.mc_dropout))
        x = x + h

        # FFN (PreNorm)
        h2 = self.norm2(x)
        h2 = self.ffn(h2, training=training)
        h2 = self.drop2(h2, training=(training or self.mc_dropout))
        return x + h2

    def get_config(self):
        return {
            **super().get_config(),
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "attn_dropout": self.attn_dropout,
            "dropout": self.dropout,
            "activation": self.activation,
            "mc_dropout": self.mc_dropout,
            "seed": self.seed,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class TransformerEncoder(layers.Layer):
    def __init__(self, depth: int, **block_kwargs):
        super().__init__()
        self.blocks = [TransformerEncoderBlock(**block_kwargs) for _ in range(depth)]

    def call(self, x, training=False, mask=None):
        for b in self.blocks:
            x = b(x, training=training, mask=mask)
        return x

    def get_config(self):
        # On ne sérialise que la config du premier bloc (supposée identique) + depth
        if not self.blocks:
            base_cfg = {}
        else:
            base_cfg = self.blocks[0].get_config()
            base_cfg.pop("name", None)
            base_cfg.pop("trainable", None)
            base_cfg.pop("dtype", None)
        return {**super().get_config(), "depth": len(self.blocks), "block_config": base_cfg}

    @classmethod
    def from_config(cls, config):
        depth = config.pop("depth")
        block_config = config.pop("block_config")
        return cls(depth=depth, **block_config)

class TransformerDecoderBlock(layers.Layer):
    """
    PreNorm Transformer decoder block.
    - Self-attention (optionnellement causal)
    - (Optionnel) Cross-attention avec l'encodeur
    - FFN
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: int = 4,
        attn_dropout: float = 0.0,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_cross_attention: bool = True,
        use_causal_self_attn: bool = True,
        mc_dropout: bool = False,
        seed: int | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        key_dim = d_model // num_heads

        self.d_model = d_model
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn_dropout = attn_dropout
        self.dropout = dropout
        self.activation = activation
        self.use_cross_attention = use_cross_attention
        self.use_causal_self_attn = use_causal_self_attn
        self.mc_dropout = mc_dropout
        self.seed = seed

        # Self-attn
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.self_attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=attn_dropout
        )
        self.drop1 = layers.Dropout(dropout, seed=seed)

        # Cross-attn (optionnel)
        if use_cross_attention:
            self.norm2 = layers.LayerNormalization(epsilon=1e-6)
            self.cross_attn = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=key_dim, dropout=attn_dropout
            )
            self.drop2 = layers.Dropout(dropout, seed=None if seed is None else seed + 1)
        else:
            self.norm2 = None
            self.cross_attn = None
            self.drop2 = None

        # FFN
        self.norm3 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            layers.Dense(mlp_ratio * d_model, activation=activation),
            layers.Dense(d_model),
        ])
        self.drop3 = layers.Dropout(dropout, seed=None if seed is None else seed + 2)

    def call(
        self,
        target,                  # (B, T, D)
        training: bool = False,
        self_mask=None,          # (B, T, T) bool/binary, optionnel (en plus du causal)
        enc_out=None,            # (B, S, D), requis si use_cross_attention
        enc_mask=None,           # (B, T, S) ou (B, 1, S), optionnel
    ):
        # Self-attention (causal + mask externe)
        x = target
        h = self.norm1(x)
        h = self.self_attn(
            h, h,
            attention_mask=self_mask,
            use_causal_mask=self.use_causal_self_attn,
            training=training
        )
        h = self.drop1(h, training=(training or self.mc_dropout))
        x = x + h

        # Cross-attention (optionnel)
        if self.use_cross_attention:
            h2 = self.norm2(x)
            if enc_out is None:
                raise ValueError("enc_out is required when use_cross_attention=True.")
            h2 = self.cross_attn(
                h2, enc_out, enc_out,
                attention_mask=enc_mask,
                training=training
            )
            h2 = self.drop2(h2, training=(training or self.mc_dropout))
            x = x + h2

        # FFN
        h3 = self.norm3(x)
        h3 = self.ffn(h3, training=training)
        h3 = self.drop3(h3, training=(training or self.mc_dropout))
        return x + h3

    def get_config(self):
        return {
            **super().get_config(),
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "attn_dropout": self.attn_dropout,
            "dropout": self.dropout,
            "activation": self.activation,
            "use_cross_attention": self.use_cross_attention,
            "use_causal_self_attn": self.use_causal_self_attn,
            "mc_dropout": self.mc_dropout,
            "seed": self.seed,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class TransformerDecoder(layers.Layer):
    """
    Empile `depth` blocs de decoder homogènes.
    - Si `use_cross_attention=True`, fournir `enc_out`/`enc_mask` au call.
    """
    def __init__(self, depth: int, **block_kwargs):
        super().__init__()
        self.depth = depth
        self.block_config = dict(block_kwargs)  # pour sérialiser
        self.blocks = [TransformerDecoderBlock(**block_kwargs) for _ in range(depth)]

    def call(self, target, training=False, self_mask=None, enc_out=None, enc_mask=None):
        x = target
        for b in self.blocks:
            x = b(x, training=training, self_mask=self_mask, enc_out=enc_out, enc_mask=enc_mask)
        return x

    def get_config(self):
        return {
            **super().get_config(),
            "depth": self.depth,
            "block_config": self.block_config,
        }

    @classmethod
    def from_config(cls, config):
        depth = config.pop("depth")
        block_config = config.pop("block_config")
        return cls(depth=depth, **block_config)