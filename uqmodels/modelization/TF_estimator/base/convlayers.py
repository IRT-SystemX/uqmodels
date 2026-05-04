import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from uqmodels.modelization.DL_estimator.utils import set_global_determinism
from uqmodels.utils import add_random_state, generate_random_state, get_fold_nstep
from uqmodels.modelization.TF_estimator.base.layers import ProbabilisticProcessing, EDLProcessing


@tf.keras.utils.register_keras_serializable(package="UQModels_layers")
class ConvBlock1D(layers.Layer):
    def __init__(
        self,
        dim_chan: int,
        filters: int = 32,
        kernel: int = 2,
        strides: int = 2,
        dp: float = 0.02,
        mc_dropout: bool = False,
        random_state: int | None = None,
        padding: str = "causal",
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim_chan = int(dim_chan)
        self.filters = int(filters)
        self.kernel = int(kernel)
        self.strides = int(strides)
        self.dp = float(dp)
        self.mc_dropout = bool(mc_dropout)
        self.random_state = random_state
        self.padding = padding
        self.activation = activation

        self.conv = layers.Conv1D(
            filters * dim_chan,
            kernel,
            strides=strides,
            padding=padding,
            groups=dim_chan,
            activation=activation,
        )
        self.bn = layers.BatchNormalization()
        self.drop = layers.Dropout(dp, seed=random_state) if dp and dp > 0 else None

    def call(self, x, training=False):
        y = self.conv(x)
        y = self.bn(y, training=training)
        if self.drop is not None:
            y = self.drop(y, training=(training or self.mc_dropout))
        return y

    def get_config(self):
        return {
            **super().get_config(),
            "dim_chan": self.dim_chan,
            "filters": self.filters,
            "kernel": self.kernel,
            "strides": self.strides,
            "dp": self.dp,
            "mc_dropout": self.mc_dropout,
            "random_state": self.random_state,
            "padding": self.padding,
            "activation": self.activation,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="UQModels_layers")
class ConvBlock2D(layers.Layer):
    def __init__(
        self,
        dim_chan: int,                 # non utilisé fonctionnellement, gardé pour compat API
        filters: int = 32,
        kernel: int | tuple[int, int] = 5,
        strides: int | tuple[int, int] = (2, 1),
        dp: float = 0.02,
        mc_dropout: bool = False,
        random_state: int | None = None,
        padding: str = "valid",
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim_chan = int(dim_chan)
        self.filters = int(filters)
        self.kernel = kernel
        self.strides = strides
        self.dp = float(dp)
        self.mc_dropout = bool(mc_dropout)
        self.random_state = random_state
        self.padding = padding
        self.activation = activation

        self.conv = layers.Conv2D(
            filters, kernel, strides=strides, padding=padding, activation=activation
        )
        self.bn = layers.BatchNormalization()
        self.drop = layers.Dropout(dp, seed=random_state) if dp and dp > 0 else None

    def call(self, x, training=False):
        y = self.conv(x)
        y = self.bn(y, training=training)
        if self.drop is not None:
            y = self.drop(y, training=(training or self.mc_dropout))
        return y

    def get_config(self):
        return {
            **super().get_config(),
            "dim_chan": self.dim_chan,
            "filters": self.filters,
            "kernel": self.kernel,
            "strides": self.strides,
            "dp": self.dp,
            "mc_dropout": self.mc_dropout,
            "random_state": self.random_state,
            "padding": self.padding,
            "activation": self.activation,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="UQModels_layers")
class TConvBlock1D(layers.Layer):
    def __init__(
        self,
        dim_out: int,
        filters: int = 32,
        kernel: int = 2,
        strides: int = 2,
        dp: float = 0.02,
        mc_dropout: bool = False,
        random_state: int | None = None,
        padding: str = "same",
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim_out = int(dim_out)
        self.filters = int(filters)
        self.kernel = int(kernel)
        self.strides = int(strides)
        self.dp = float(dp)
        self.mc_dropout = bool(mc_dropout)
        self.random_state = random_state
        self.padding = padding
        self.activation = activation

        self.tconv = layers.Conv1DTranspose(
            filters * dim_out,
            kernel,
            strides=strides,
            padding=padding,
            groups=dim_out,
            activation=activation,
        )
        self.bn = layers.BatchNormalization()
        self.drop = layers.Dropout(dp, seed=random_state) if dp and dp > 0 else None

    def call(self, x, training=False):
        y = self.tconv(x)
        y = self.bn(y, training=training)
        if self.drop is not None:
            y = self.drop(y, training=(training or self.mc_dropout))
        return y

    def get_config(self):
        return {
            **super().get_config(),
            "dim_out": self.dim_out,
            "filters": self.filters,
            "kernel": self.kernel,
            "strides": self.strides,
            "dp": self.dp,
            "mc_dropout": self.mc_dropout,
            "random_state": self.random_state,
            "padding": self.padding,
            "activation": self.activation,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package="UQModels_layers")
class TConvBlock2D(layers.Layer):
    def __init__(
        self,
        dim_out: int,
        filters: int = 32,
        kernel: int = 5,
        strides: int | tuple[int, int] = (2, 1),
        dp: float = 0.02,
        mc_dropout: bool = False,
        random_state: int | None = None,
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim_out = int(dim_out)
        self.filters = int(filters)
        self.kernel = int(kernel)
        self.strides = strides
        self.dp = float(dp)
        self.mc_dropout = bool(mc_dropout)
        self.random_state = random_state
        self.activation = activation

        # NB: respecte exactement ton implémentation (kernel, dim_out)
        self.tconv = layers.Conv2DTranspose(
            filters, (kernel, dim_out), strides=strides, activation=activation
        )
        self.bn = layers.BatchNormalization()
        self.drop = layers.Dropout(dp, seed=random_state) if dp and dp > 0 else None

    def call(self, x, training=False):
        y = self.tconv(x)
        y = self.bn(y, training=training)
        if self.drop is not None:
            y = self.drop(y, training=(training or self.mc_dropout))
        return y

    def get_config(self):
        return {
            **super().get_config(),
            "dim_out": self.dim_out,
            "filters": self.filters,
            "kernel": self.kernel,
            "strides": self.strides,
            "dp": self.dp,
            "mc_dropout": self.mc_dropout,
            "random_state": self.random_state,
            "activation": self.activation,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# ---------------------------------
#   ENCODER / DECODER (Layer)
# ---------------------------------

@tf.keras.utils.register_keras_serializable(package="UQModels_layers")
class CNNEncoder(layers.Layer):
    """
    Reproduit la logique de cnn_enc_bis en tant que Layer.
    Entrée attendue: (B, size_subseq_enc, dim_target)
    Sortie: (B, dim_z)
    """
    def __init__(
        self,
        size_subseq_enc: int = 60,
        dim_target: int = 52,
        dim_chan: int = 4,
        list_filters: list[int] = (64, 64, 32),
        list_kernels: list = ((10, 3), 10, 10),
        list_strides: list = ((2, 1), (2, 1), (2, 1)),
        block: str = "2D",                 # "2D" ou "1D"
        dim_z: int = 200,
        dp: float = 0.02,
        type_output: str | None = None,    # pour MC_Dropout flag
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size_subseq_enc = int(size_subseq_enc)
        self.dim_target = int(dim_target)
        self.dim_chan = int(dim_chan)
        self.list_filters = list(list_filters)
        self.list_kernels = list(list_kernels)
        self.list_strides = list(list_strides)
        self.block = block
        self.dim_z = int(dim_z)
        self.dp = float(dp)
        self.type_output = type_output
        self.random_state = random_state

        self.mc_dropout = bool(type_output in ["MC_Dropout"])
        self.dim_space = int(self.dim_target / self.dim_chan)

        # Pré-traitement (reshape/expand)
        if self.dim_chan == 1:
            self.pre = layers.Lambda(lambda x: K.expand_dims(x, axis=-1))
        else:
            self.pre = layers.Lambda(
                lambda x: K.reshape(x, (-1, self.size_subseq_enc, self.dim_space, self.dim_chan))
            )

        # Empilement de blocs conv
        self.blocks = []
        for n, (f, k, s) in enumerate(zip(self.list_filters, self.list_kernels, self.list_strides)):
            seed = add_random_state(self.random_state, 1 + n)
            if self.block == "2D":
                self.blocks.append(
                    ConvBlock2D(self.dim_space, filters=f, kernel=k, strides=s,
                                dp=self.dp, mc_dropout=self.mc_dropout, random_state=seed)
                )
            else:
                # 1D
                seed = add_random_state(self.random_state, 1 + len(self.list_filters) + n)
                self.blocks.append(
                    ConvBlock1D(self.dim_space, filters=f, kernel=k, strides=s,
                                dp=self.dp, mc_dropout=self.mc_dropout, random_state=seed)
                )

        self.flatten = layers.Flatten()
        self.proj = layers.Dense(self.dim_z)
        self.drop = layers.Dropout(self.dp, seed=self.random_state) if self.dp and self.dp > 0 else None

    def call(self, x, training=False):
        y = self.pre(x)
        for b in self.blocks:
            y = b(y, training=training)
        y = self.flatten(y)
        y = self.proj(y)
        if self.drop is not None:
            y = self.drop(y, training=(training or self.mc_dropout))
        return y

    def get_config(self):
        return {
            **super().get_config(),
            "size_subseq_enc": self.size_subseq_enc,
            "dim_target": self.dim_target,
            "dim_chan": self.dim_chan,
            "list_filters": self.list_filters,
            "list_kernels": self.list_kernels,
            "list_strides": self.list_strides,
            "block": self.block,
            "dim_z": self.dim_z,
            "dp": self.dp,
            "type_output": self.type_output,
            "random_state": self.random_state,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="UQModels_layers")
class CNNDecoder(layers.Layer):
    """
    Reproduit la logique de cnn_dec_bis en tant que Layer.
    Entrée attendue: (B, dim_z)
    Sortie: (B, size_subseq_dec, dim_space * dim_chan_out)
    """
    def __init__(
        self,
        size_subseq_dec: int,
        dim_out: int,
        dim_chan: int = 1,
        list_filters: list[int] = [64, 64],
        list_strides: list[int] = [(1, 1),(1, 1)],
        list_kernels: list[int] = [4, 4],
        dim_z: int = 200,
        type_output: str | None = None,    # None | "Variational" | "MC_Dropout" | "Deep_ensemble" | "EDL"
        min_logvar: float = -6.0,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size_subseq_dec = int(size_subseq_dec)
        self.dim_out = int(dim_out)
        self.dim_chan = int(dim_chan)
        self.type_output = type_output
        self.min_logvar = float(min_logvar)
        self.list_filters = list(list_filters)
        self.list_strides = list(list_strides)
        self.list_kernels = list(list_kernels)
        self.dim_z = int(dim_z)
        self.random_state = random_state

        self.mc_dropout = bool(type_output in ["MC_Dropout"])
        self.dim_space = int(self.dim_out / self.size_subseq_dec)

        # Facteur de canaux en sortie
        self.dim_chan_out = self.dim_chan
        if type_output in ["Variational","MC_Dropout", "Deep_ensemble"]:
            self.dim_chan_out = 2 * self.dim_chan
        elif type_output == "EDL":
            self.dim_chan_out = 4 * self.dim_chan

        # (B, dim_z) -> (B, 1, 1, dim_z)
        self.expand = layers.Lambda(lambda x: x[:, None, None, :])

        # Empilement de TConv2D blocks
        self.blocks = []
        for n, k in enumerate(self.list_kernels):
            d_tar = self.dim_space if n == 0 else 1
            seed = add_random_state(self.random_state, n)
            self.blocks.append(
                TConvBlock2D(d_tar, filters=self.list_filters[n], kernel=k,
                             strides=self.list_strides[n], dp=0.02, mc_dropout=self.mc_dropout,
                             random_state=seed)
            )

        # Dernière proj. Conv2DTranspose vers dim_chan_out
        self.final_tconv = layers.Conv2DTranspose(self.dim_chan_out, (1, 1), activation="linear")

        # Post-traitements probabilistes
        self.post_prob = None
        if self.type_output in ["Variational","MC_Dropout", "Deep_ensemble"]:
            self.post_prob = ProbabilisticProcessing(self.min_logvar)
        elif self.type_output == "EDL":
            self.post_prob = EDLProcessing(self.min_logvar)

        # Reshape final vers (B, size_subseq_dec, dim_space * dim_chan_out)
        self.final_reshape = layers.Lambda(
            lambda t: K.reshape(t, (-1, self.size_subseq_dec, self.dim_space * self.dim_chan_out))
        )

    def call(self, x, training=False):
        y = self.expand(x)
        for b in self.blocks:
            y = b(y, training=training)
        y = self.final_tconv(y)
        if self.post_prob is not None:
            y = self.post_prob(y)
        y = self.final_reshape(y)
        return y

    def get_config(self):
        return {
            **super().get_config(),
            "size_subseq_dec": self.size_subseq_dec,
            "dim_out": self.dim_out,
            "dim_chan": self.dim_chan,
            "type_output": self.type_output,
            "min_logvar": self.min_logvar,
            "list_filters": self.list_filters,
            "list_strides": self.list_strides,
            "list_kernels": self.list_kernels,
            "dim_z": self.dim_z,
            "random_state": self.random_state,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# ---------------------------------
#   FACTORIES (tf.keras.Model)
# ---------------------------------

def make_cnn_encoder_model(
    size_subseq_enc=60,
    dim_target=52,
    dim_chan=4,
    list_filters=[64, 64, 32],
    list_kernels=[(10, 3), 10, 10],
    list_strides=[(2, 1), (2, 1), (2, 1)],
    type_output=None,
    block="2D",
    dim_z=200,
    dp=0.02,
    random_state=None,
    name: str = "conv_lag_enc",
):
    """
    Équivalent fonctionnel à cnn_enc_bis() mais basé sur CNNEncoderBis.
    Entrée: (B, size_subseq_enc, dim_target)  ->  Sortie: (B, dim_z)
    """
    inp = layers.Input(shape=(size_subseq_enc, dim_target), name="st")
    enc_layer = CNNEncoder(
        size_subseq_enc=size_subseq_enc,
        dim_target=dim_target,
        dim_chan=dim_chan,
        list_filters=list_filters,
        list_kernels=list_kernels,
        list_strides=list_strides,
        block=block,
        dim_z=dim_z,
        dp=dp,
        type_output=type_output,
        random_state=random_state,
        name=f"{name}_layer",
    )
    out = enc_layer(inp)
    return tf.keras.Model(inp, out, name=name)


def make_cnn_decoder_model(
    size_subseq_dec,
    dim_out,
    dim_chan=1,
    type_output=None,
    min_logvar=-6,
    list_filters=[32,64, 64],
    list_strides=[(1,1),(1,1),(1,1)],
    list_kernels=[10, 10, 10],
    dim_z=200,
    random_state=None,
    name: str = "conv_lag_dec",
):
    """
    Équivalent fonctionnel à cnn_dec_bis() mais basé sur CNNDecoderBis.
    Entrée: (B, dim_z)  ->  Sortie: (B, size_subseq_dec, dim_space * dim_chan_out)
    """
    inp = layers.Input(shape=(dim_z,), name="st")
    dec_layer = CNNDecoder(
        size_subseq_dec=size_subseq_dec,
        dim_out=dim_out,
        dim_chan=dim_chan,
        type_output=type_output,
        min_logvar=min_logvar,
        list_filters=list_filters,
        list_strides=list_strides,
        list_kernels=list_kernels,
        dim_z=dim_z,
        random_state=random_state,
        name=f"{name}_layer")
    
    out = dec_layer(inp)
    model = tf.keras.Model(inp, out, name=name)

    # Alerte de compat (même check que ton code)
    # (si les tailles ne concordent pas, K.reshape ci-dessous plantera plus tard)
    if model.layers[-2].output_shape[1] != model.layers[-1].output_shape[1]:
        print("Warning : inadequate deconvolution window size : model will crash")

    return model