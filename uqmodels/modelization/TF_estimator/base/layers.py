import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras import backend as K

# Hypothèse: ces utilitaires existent dans ton codebase

from uqmodels.modelization.DL_estimator.utils import set_global_determinism

from uqmodels.utils import add_random_state, generate_random_state, get_fold_nstep

# EDL head
# https://github.com/aamini/evidential-deep-learning/blob/main/evidential_deep_learning/layers/dense.py

# tf.keras.utils.get_custom_objects().clear()

# Fonction d'harmonisation
VALID_TYPE_OUTPUT = {
    None,
    "variational",
    "mc_dropout",
    "deep_ensemble",
    "edl",
    "classif",
}

def normalize_type_output(type_output: str | None) -> str | None:
    """Normalize output type to canonical lowercase values."""
    if type_output is None:
        return None
    return str(type_output).lower()

def validate_type_output(type_output: str | None) -> None:
    """Validate canonical output type."""
    if type_output not in VALID_TYPE_OUTPUT:
        valid_values = sorted(v for v in VALID_TYPE_OUTPUT if v is not None)
        raise ValueError(
            f"Unknown type_output={type_output!r}. "
            f"Expected one of {valid_values} or None."
        )
# Test

@tf.keras.utils.register_keras_serializable(package="UQModels_layers")
class EDLProcessingLayers(layers.Layer):
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
class ProbProcessingLayers(layers.Layer):
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
        """Apply ProbProcessingLayers to x

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

@tf.keras.utils.register_keras_serializable(package="UQModels")
@tf.keras.utils.register_keras_serializable(package="UQModels")
class DenseHeadBlock(layers.Layer):
    """
    Dense output head block.

    Projects an input representation into deterministic, probabilistic,
    evidential, or classification outputs.

    Responsibilities:
    - build the final Dense projection;
    - adapt the output dimension according to type_output;
    - apply probabilistic or EDL post-processing when required.
    """

    def __init__(
        self,
        dim_out: int | None = 1,
        type_output: str | None = None,
        logvar_min: float = -10.0,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.dim_out = None if dim_out is None else int(dim_out)
        self.type_output = normalize_type_output(type_output)
        self.logvar_min = float(logvar_min)

        validate_type_output(self.type_output)

        self.head_dense = self._build_dense_head()
        self.output_processing = self._build_output_processing()

    @property
    def n_param(self) -> int:
        """Return the number of output parameters per target dimension."""
        if self.type_output == "edl":
            return 4

        if self.type_output in {"variational", "mc_dropout", "deep_ensemble"}:
            return 2

        return 1

    def _build_dense_head(self):
        """Build the dense projection associated with the output semantics."""
        if self.dim_out is None:
            return None

        if self.type_output == "edl":
            return layers.Dense(
                units=4 * self.dim_out,
                activation=None,
                name="edl_projection",
            )

        if self.type_output in {"variational", "mc_dropout", "deep_ensemble"}:
            return layers.Dense(
                units=2 * self.dim_out,
                activation=None,
                name="mu_logvar_projection",
            )

        if self.type_output == "classif":
            return layers.Dense(
                units=self.dim_out,
                activation="softmax",
                name="classif_projection",
            )

        return layers.Dense(
            units=self.dim_out,
            activation=None,
            name="output_projection",
        )

    def _build_output_processing(self):
        """Build the final output processing layer when needed."""
        if self.type_output == "edl":
            return EDLProcessingLayers(self.logvar_min)

        if self.type_output in {"variational", "mc_dropout", "deep_ensemble"}:
            return ProbProcessingLayers(self.logvar_min)

        return None

    def call(self, inputs):
        x = inputs

        if self.head_dense is not None:
            x = self.head_dense(x)

        if self.output_processing is not None:
            x = self.output_processing(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim_out": self.dim_out,
                "type_output": self.type_output,
                "logvar_min": self.logvar_min,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def make_cfg_MLPBlock(
    layers_size: list[int] | tuple[int, ...] = (100, 50),
    dp: float = 0.01,
    activation: str | list[str] | tuple[str, ...] | None = "relu",
    reg_W: tuple[float, float] = (1e-5, 1e-5),
    mc_dropout: bool = True,
    random_state: int | None = None,
    name: str | None = "mlp_block",
) -> dict:
    """
    Build a configuration dictionary for MLPBlock.

    Parameters
    ----------
    layers_size:
        Hidden layer dimensions.
    dp:
        Dropout rate.
    activation:
        Dense activation. Can be a string, None, or one activation per layer.
    reg_W:
        L1/L2 regularization factors for Dense kernels.
    mc_dropout:
        Whether dropout remains active during inference.
    random_state:
        Random seed propagated to dropout layers.
    name:
        Keras layer name.

    Returns
    -------
    dict
        Configuration dictionary passed to MLPBlock.
    """
    return {
        "layers_size": layers_size,
        "dp": dp,
        "activation": activation,
        "reg_W": reg_W,
        "mc_dropout": mc_dropout,
        "random_state": random_state,
        "name": name,
    }

@tf.keras.utils.register_keras_serializable(package="UQModels")
class MLPBlock(layers.Layer):
    """
    Local MLP computational block.

    Applies a repeated Dense -> optional Dropout stack.
    The block does not manage input/output reshaping, output heads,
    or probabilistic post-processing.
    """

    def __init__(
        self,
        layers_size: list[int] | tuple[int, ...] = (100, 50),
        dp: float = 0.01,
        activation: str | list[str] | tuple[str, ...] | None = "relu",
        reg_W: tuple[float, float] = (1e-5, 1e-5),
        mc_dropout: bool = True,
        random_state: int | None = None,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.layers_size = list(layers_size)
        self.dp = float(dp)
        self.activation = activation
        self.reg_W = tuple(reg_W)
        self.mc_dropout = bool(mc_dropout)
        self.random_state = random_state

        try:
            set_global_determinism(random_state)
        except Exception:
            pass

        self._validate_activation()

        reg = regularizers.l1_l2(l1=self.reg_W[0], l2=self.reg_W[1])

        self.layers_stack = []

        for idx, dim_layer in enumerate(self.layers_size):
            self.layers_stack.append(
                layers.Dense(
                    units=dim_layer,
                    activation=self._get_activation(idx),
                    kernel_regularizer=reg,
                    name=f"dense_{idx}",
                )
            )

            if self.dp > 0.0:
                self.layers_stack.append(
                    layers.Dropout(
                        rate=self.dp,
                        seed=add_random_state(self.random_state, idx),
                        name=f"dropout_{idx}",
                    )
                )

    def _validate_activation(self):
        """Validate activation specification."""
        if isinstance(self.activation, (list, tuple)):
            if len(self.activation) != len(self.layers_size):
                raise ValueError(
                    "When activation is a list or tuple, its length must match "
                    "the number of MLP layers."
                )

    def _get_activation(self, idx: int):
        """Return the activation associated with a layer."""
        if isinstance(self.activation, (list, tuple)):
            return self.activation[idx]
        return self.activation

    def call(self, inputs, training=False):
        x = inputs

        for layer in self.layers_stack:
            if isinstance(layer, layers.Dropout):
                x = layer(x, training=(training or self.mc_dropout))
            else:
                x = layer(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "layers_size": self.layers_size,
                "dp": self.dp,
                "activation": self.activation,
                "reg_W": self.reg_W,
                "mc_dropout": self.mc_dropout,
                "random_state": self.random_state,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def make_cfg_DenseHeadBlock(
    dim_out: int | None = 1,
    type_output: str | None = None,
    logvar_min: float = -10.0,
    name: str | None = "dense_head_block",
) -> dict:
    """
    Build a configuration dictionary for DenseHeadBlock.

    Parameters
    ----------
    dim_out:
        Output dimension.
    type_output:
        Output semantics: None, "variational", "mc_dropout",
        "deep_ensemble", "edl", or "classif".
    logvar_min:
        Lower numerical bound for probabilistic output processing.
    name:
        Keras layer name.

    Returns
    -------
    dict
        Configuration dictionary passed to DenseHeadBlock.
    """
    return {
        "dim_out": dim_out,
        "type_output": type_output,
        "logvar_min": logvar_min,
        "name": name,
    }

@tf.keras.utils.register_keras_serializable(package="UQModels")
class MLPSubNet(layers.Layer):
    """
    Structured MLP sub-network.

    Applies:
    optional input reshape -> MLPBlock -> DenseHeadBlock
    -> optional output reshape.

    MLPBlock handles the hidden dense stack.
    DenseHeadBlock handles the output projection and final output formalization.
    """

    def __init__(
        self,
        dim_in: int = 10,
        dim_out: int | None = 1,
        shape_in: tuple[int, int] | None = None,
        shape_out: tuple[int, int] | None = None,
        type_output: str | None = None,
        random_state: int | None = None,
        cfg_MLPBlock: dict | None = None,
        cfg_DenseHeadBlock: dict | None = None,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        try:
            set_global_determinism(random_state)
        except Exception:
            pass

        self.dim_in = int(dim_in)
        self.dim_out = None if dim_out is None else int(dim_out)
        self.shape_in = None if shape_in is None else tuple(shape_in)
        self.shape_out = None if shape_out is None else tuple(shape_out)
        self.type_output = normalize_type_output(type_output)
        self.random_state = random_state

        validate_type_output(self.type_output)
        self._validate_shapes()

        self.cfg_MLPBlock = {} if cfg_MLPBlock is None else dict(cfg_MLPBlock)
        mc_dropout = False
        if(self.type_output == "mc_dropout"):
            mc_dropout = True
        self.cfg_MLPBlock.setdefault("mc_dropout", mc_dropout)
        self.cfg_MLPBlock.setdefault("random_state", self.random_state)
        self.cfg_MLPBlock.setdefault("name", "mlp_block")

        self.cfg_DenseHeadBlock = (
            {} if cfg_DenseHeadBlock is None else dict(cfg_DenseHeadBlock))
        self.cfg_DenseHeadBlock.setdefault("dim_out", self.dim_out)
        self.cfg_DenseHeadBlock.setdefault("type_output", self.type_output)
        self.cfg_DenseHeadBlock.setdefault("name", "dense_head_block")

        self.mlp_block = MLPBlock(**self.cfg_MLPBlock)
        self.head_block = DenseHeadBlock(**self.cfg_DenseHeadBlock)

    @property
    def n_param(self) -> int:
        """Return the number of output parameters per target dimension."""
        return self.head_block.n_param

    def _validate_shapes(self) -> None:
        """Validate consistency between flat dimensions and structured shapes."""
        if self.shape_in is not None:
            if len(self.shape_in) != 2:
                raise ValueError("shape_in must be a tuple of length 2: (H, W).")

            height, width = self.shape_in
            if self.dim_in != height * width:
                raise ValueError(
                    "dim_in must be equal to H * W when shape_in is provided."
                )

        if self.shape_out is not None:
            if len(self.shape_out) != 2:
                raise ValueError(
                    "shape_out must be a tuple of length 2: (H_out, W_out)."
                )

            if self.dim_out is None:
                raise ValueError("dim_out must be provided when shape_out is provided.")

            height_out, width_out = self.shape_out
            if self.dim_out != height_out * width_out:
                raise ValueError(
                    "dim_out must be equal to H_out * W_out when shape_out is provided."
                )

    def call(self, inputs, training=False):
        x = inputs

        if self.shape_in is not None:
            x = tf.reshape(x, (-1, self.dim_in))

        x = self.mlp_block(x, training=training)
        x = self.head_block(x)

        if self.shape_out is not None:
            height_out, width_out = self.shape_out
            x = tf.reshape(x, (-1, width_out * self.n_param, height_out))
            x = tf.transpose(x, perm=[0, 2, 1])

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim_in": self.dim_in,
                "dim_out": self.dim_out,
                "shape_in": self.shape_in,
                "shape_out": self.shape_out,
                "type_output": self.type_output,
                "random_state": self.random_state,
                "cfg_MLPBlock": self.cfg_MLPBlock,
                "cfg_DenseHeadBlock": self.cfg_DenseHeadBlock,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    
def make_mlp_model(
    dim_in: int = 10,
    dim_out: int | None = 1,
    shape_in: tuple[int, int] | None = None,
    shape_out: tuple[int, int] | None = None,
    type_output: str | None = None,
    logvar_min: float = -10.0,
    random_state: int | None = None,
    cfg_MLPBlock: dict | None = None,
    cfg_DenseHeadBlock: dict | None = None,
    name: str = "MLP") -> Model:
    """
    Build a functional Keras MLP model.

    Architecture:
        Input -> MLPSubNet -> Output

    Parameters
    ----------
    dim_in:
        Flattened input dimension.
    dim_out:
        Flattened output dimension.
    shape_in:
        Optional structured input shape, excluding batch dimension.
    shape_out:
        Optional structured output shape, excluding batch dimension.
    type_output:
        Output semantics handled by DenseHeadBlock.
    logvar_min:
        Lower numerical bound for probabilistic output processing.
    random_state:
        Random seed propagated to stochastic subcomponents.
    cfg_MLPBlock:
        Configuration dictionary passed to MLPBlock.
    cfg_DenseHeadBlock:
        Configuration dictionary passed to DenseHeadBlock.
    name:
        Keras model name.

    Returns
    -------
    tf.keras.Model
        Functional Keras model.
    """
    if shape_in is None:
        inputs = layers.Input(shape=(dim_in,), name=f"input_{name}")
    else:
        if len(shape_in) != 2:
            raise ValueError("shape_in must be a tuple of length 2: (H, W).")

        height, width = shape_in
        if dim_in != height * width:
            raise ValueError(
                "dim_in must be equal to H * W when shape_in is provided."
            )

        inputs = layers.Input(shape=shape_in, name=f"input_{name}")

    mlp_subnet = MLPSubNet(
        dim_in=dim_in,
        dim_out=dim_out,
        shape_in=shape_in,
        shape_out=shape_out,
        type_output=type_output,
        logvar_min=logvar_min,
        random_state=random_state,
        cfg_MLPBlock=cfg_MLPBlock,
        cfg_DenseHeadBlock=cfg_DenseHeadBlock,
        name=f"{name}_subnet",
    )

    outputs = mlp_subnet(inputs)

    return Model(inputs, outputs, name=name)

# Dense Layers