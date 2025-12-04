import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras import backend as K

# Hypothèse: ces utilitaires existent dans ton codebase
# set_global_determinism, add_random_state, EDLProcessing, ProbabilisticProcessing

from uqmodels.modelization.DL_estimator.utils import set_global_determinism

from uqmodels.utils import add_random_state, generate_random_state, get_fold_nstep

# EDL head
# https://github.com/aamini/evidential-deep-learning/blob/main/evidential_deep_learning/layers/dense.py

# tf.keras.utils.get_custom_objects().clear()


@tf.keras.utils.register_keras_serializable(package="UQModels_layers")
class EDLProcessing(layers.Layer):
    def __init__(self, min_logvar=-6, **kwargs):
        self.min_logvar = min_logvar
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        """Apply EDLProcessing

        Args:
            x (_type_): input

        Returns:
            _type_: _description_
        """
        mu, logv, logalpha, logbeta = tf.split(x, 4, axis=-1)
        v = tf.nn.softplus(logv) + 10e-6
        alpha = tf.nn.softplus(logalpha) + 1
        beta = tf.nn.softplus(logbeta) + 10e-6
        return tf.concat([mu, v, alpha, beta], axis=-1)

    def get_config(self):
        return {"min_logvar": self.min_logvar}


@tf.keras.utils.register_keras_serializable(package="UQModels_layers")
class ProbabilisticProcessing(layers.Layer):
    """_summary_

    Args:
        Layer (_type_): _description_
    """

    def __init__(self, min_logvar=-10, max_logvar=10, **kwargs):
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        """Apply ProbabilisticProcessing to x

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        mu, logsigma = tf.split(x, 2, axis=-1)
        logsigma = tf.where(logsigma > self.min_logvar, logsigma, self.min_logvar)

        logsigma = tf.where(logsigma < self.max_logvar, logsigma, self.max_logvar)
        # logsigma = tf.nn.softplus(logsigma)
        return tf.concat([mu, logsigma], axis=-1)

    def get_config(self):
        return {"min_logvar": self.min_logvar, "max_logvar": self.max_logvar}

# Dense Layers

@tf.keras.utils.register_keras_serializable(package="UQModels_layers")
class MLPLayer(layers.Layer):
    """
    Couche MLP configurable:
      - MC Dropout en inference (mc_dropout=True)
      - Têtes: None | "MC_Dropout"/"Deep_ensemble" (mu, logvar) | "EDL" | "classif"
      - Entrée 1D ou 2D (H,W) aplatie vers dim_in; sortie réarrangeable en 2D
    """
    def __init__(
        self,
        dim_in: int = 10,
        dim_out: int | None = 1,
        layers_size: list[int] = (100, 50),
        dp: float = 0.01,
        mc_dropout: bool = True,
        type_output: str | None = None,        # None | "Variational" | "MC_Dropout" | "Deep_ensemble" | "EDL" | "variational" | "classif"
        logvar_min: float = -10.0,
        regularizer_W: tuple[float, float] = (1e-5, 1e-5),
        shape_2D: tuple[int, int] | None = None,      # (H, W) d'entrée
        shape_2D_out: tuple[int, int] | None = None,  # (H_out, W_out) de sortie
        random_state: int | None = None,
        activation_hidden: str | None = None,  # None -> LeakyReLU(0.01)
        **kwargs,
    ):
        super().__init__(**kwargs)
        try:
            set_global_determinism(random_state)
        except Exception:
            pass

        # Params (pour get_config)
        self.dim_in = int(dim_in)
        self.dim_out = None if dim_out is None else int(dim_out)
        self.layers_size = list(layers_size)
        self.dp = float(dp)
        self.mc_dropout = bool(mc_dropout)
        self.type_output = type_output
        self.logvar_min = float(logvar_min)
        self.regularizer_W = tuple(regularizer_W)
        self.shape_2D = None if shape_2D is None else tuple(shape_2D)
        self.shape_2D_out = None if shape_2D_out is None else tuple(shape_2D_out)
        self.random_state = random_state
        self.activation_hidden = activation_hidden

        reg = regularizers.l1_l2(l1=self.regularizer_W[0], l2=self.regularizer_W[1])

        # Hidden stack
        self.hidden_dense = []
        self.hidden_act = []
        self.hidden_drop = []
        for n, width in enumerate(self.layers_size):
            self.hidden_dense.append(
                layers.Dense(
                    width, activation=None, name=f"MLP_{n}", kernel_regularizer=reg
                )
            )
            if self.activation_hidden is None:
                self.hidden_act.append(layers.LeakyReLU(alpha=0.01, name=f"act_{n}"))
            else:
                self.hidden_act.append(layers.Activation(self.activation_hidden, name=f"act_{n}"))

            if self.dp and self.dp > 0.0:
                self.hidden_drop.append(
                    layers.Dropout(self.dp, seed=add_random_state(self.random_state, n), name=f"drop_{n}")
                )
            else:
                self.hidden_drop.append(None)

        # Heads
        self.head_dense = None
        self.edl_proc = None
        self.prob_proc = None
        if self.type_output == "EDL":
            self.head_dense = layers.Dense(4 * (self.dim_out or 1), name="EDL", activation=None)
            self.edl_proc = EDLProcessing(self.logvar_min)
        elif self.type_output in ("Variational","MC_Dropout", "Deep_ensemble"):
            self.head_dense = layers.Dense(2 * (self.dim_out or 1), name="Mu_logvar", activation=None)
            self.prob_proc = ProbabilisticProcessing(self.logvar_min)
        elif self.type_output in ("Variational","MC_Dropout", "Deep_ensemble"):
            self.head_dense = layers.Dense(2 * (self.dim_out or 1), name="Mu_logvar", activation=None)
            self.prob_proc = ProbabilisticProcessing(self.logvar_min)
        elif self.type_output == "classif":
            self.head_dense = layers.Dense((self.dim_out or 1), name="Prob", activation="softmax")
        elif self.dim_out is not None:
            self.head_dense = layers.Dense(self.dim_out, name="Output")

    @property
    def n_param(self) -> int:
        if self.type_output == "EDL":
            return 4
        if self.type_output in ("Variational","MC_Dropout", "Deep_ensemble"):
            return 2
        return 1

    def call(self, inputs, training=False):
        x = inputs
        if self.shape_2D is not None:
            # (B, H, W) -> (B, dim_in)
            x = tf.reshape(x, (-1, self.dim_in))

        # Hidden stack
        for dense, act, drop in zip(self.hidden_dense, self.hidden_act, self.hidden_drop):
            x = dense(x)
            x = act(x)
            if drop is not None:
                x = drop(x, training=(training or self.mc_dropout))

        # Head
        y = x
        if self.head_dense is not None:
            y = self.head_dense(y)
            if self.edl_proc is not None:
                y = self.edl_proc(y)
            elif self.prob_proc is not None:
                y = self.prob_proc(y)

        # Sortie 2D optionnelle
        if self.shape_2D_out is not None:
            H_out, W_out = self.shape_2D_out
            y = tf.reshape(y, (-1, W_out * self.n_param, H_out))
            y = tf.transpose(y, perm=[0, 2, 1])  # (B, H_out, W_out * n_param)
        return y

    # ---- Serialization ----
    def get_config(self):
        base = super().get_config()
        base.update({
            "dim_in": self.dim_in,
            "dim_out": self.dim_out,
            "layers_size": self.layers_size,
            "dp": self.dp,
            "mc_dropout": self.mc_dropout,
            "type_output": self.type_output,
            "logvar_min": self.logvar_min,
            "regularizer_W": self.regularizer_W,
            "shape_2D": self.shape_2D,
            "shape_2D_out": self.shape_2D_out,
            "random_state": self.random_state,
            "activation_hidden": self.activation_hidden,
        })
        return base

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
def make_mlp_model(
    dim_in: int = 10,
    dim_out: int | None = 1,
    layers_size: list[int] = (100, 50),
    name: str = "MLP",
    dp: float = 0.01,
    mc_dropout: bool = True,
    type_output: str | None = None,
    logvar_min: float = -10.0,
    regularizer_W: tuple[float, float] = (1e-5, 1e-5),
    shape_2D: tuple[int, int] | None = None,
    shape_2D_out: tuple[int, int] | None = None,
    random_state: int | None = None,
    activation_hidden: str | None = None,
):
    """
    Construit un modèle fonctionnel Keras (Input -> MLPLayer -> Output)
    """
    if shape_2D is None:
        inp = layers.Input(shape=(dim_in,), name=f"input_{name}")
    else:
        H, W = shape_2D
        assert H * W == dim_in, "dim_in doit être égal à H*W lorsque shape_2D est fourni."
        inp = layers.Input(shape=(H, W), name=f"input_{name}")

    mlp_layer = MLPLayer(
        dim_in=dim_in,
        dim_out=dim_out,
        layers_size=layers_size,
        dp=dp,
        mc_dropout=mc_dropout,
        type_output=type_output,
        logvar_min=logvar_min,
        regularizer_W=regularizer_W,
        shape_2D=shape_2D,
        shape_2D_out=shape_2D_out,
        random_state=random_state,
        activation_hidden=activation_hidden,
        name=f"{name}_layer",
    )
    out = mlp_layer(inp)
    return Model(inp, out, name=name)

# Dense Layers