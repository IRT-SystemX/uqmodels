import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from keras import Model, layers
from keras.layers import RNN, Dense, Dropout, Lambda, Layer, LSTMCell, TimeDistributed

from uqmodels.modelization.DL_estimator.utils import set_global_determinism

from ...utils import add_random_state, generate_random_state, get_fold_nstep

# EDL head
# https://github.com/aamini/evidential-deep-learning/blob/main/evidential_deep_learning/layers/dense.py

# tf.keras.utils.get_custom_objects().clear()


@tf.keras.utils.register_keras_serializable(package="UQModels_layers")
class EDLProcessing(Layer):
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
class ProbabilisticProcessing(Layer):
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


def mlp(
    dim_in=10,
    dim_out=1,
    layers_size=[100, 50],
    name="",
    dp=0.01,
    with_mc_dp=True,
    type_output=None,
    logvar_min=-10,
    regularizer_W=(0.00001, 0.00001),
    shape_2D=None,
    shape_2D_out=None,
    random_state=None,
    **kwargs,
):
    """Generate a keras MLP model to make preprocessing or head subpart".

    Args:
        dim_in (int): Input dimension, erase by shape_2D if 2D input_size
        dim_out (int or None):  Input dimension, if None take the last layers_size values
        layers_size (list of  in, optional): List of size of layers. Defaults to [100, 50].
        name (str, optional): Name of model. Defaults to "".
        dp (float, optional): Percentage of dropout. Defaults to 0.01.
        type_output (_type_, optional): Specify Head last layers among
        ['None':pred,MC_Dropout:(Pred,var),"EDL":(Pred,mu,alpha,beta) ]
        logvar_min (int, optional): Cut off for small variance estimations
        regularizer_W (tuple, optional):Regularisation on Dense layers. Defaults to (0.00001, 0.00001).
        shape_2D (tupple or None, optional): if intput shape is 2D. Defaults to None.
        shape_2D_out:
        random_state (bool): handle experimental random using seed.

    Returns:
        _type_: _description_
    """

    set_global_determinism(random_state)

    reg_l1l2 = tf.keras.regularizers.l1_l2(l1=regularizer_W[0], l2=regularizer_W[1])

    flag_mc = None
    if with_mc_dp:
        flag_mc = 1

    if shape_2D is None:
        inputs = tf.keras.layers.Input(shape=(dim_in), name="input_" + name)
        output = inputs

    else:
        inputs = tf.keras.layers.Input(
            shape=(shape_2D[0], shape_2D[1]), name="input_" + name
        )
        output = tf.keras.layers.Lambda(lambda x: K.reshape(x, shape=(-1, dim_in)))(
            inputs
        )

    for n, i in enumerate(layers_size):
        layer = tf.keras.layers.Dense(
            i,
            activation=tf.keras.layers.LeakyReLU(alpha=0.01),
            name="MLP_" + str(n) + "_" + name,
            kernel_regularizer=reg_l1l2,
        )
        if dp > 0:
            output = tf.keras.layers.Dropout(
                dp, seed=add_random_state(random_state, n)
            )(layer(output), training=flag_mc)

    # Probablistic NN
    if type_output == "EDL":
        n_param = 4
        EDL_ouput = tf.keras.layers.Dense(
            n_param * dim_out, name="EDL", activation=None
        )(output)
        output = EDLProcessing(logvar_min)(EDL_ouput)

    elif (type_output == "MC_Dropout") or (type_output == "Deep_ensemble"):
        n_param = 2
        Prob_output = tf.keras.layers.Dense(
            n_param * dim_out, name="Mu_logvar", activation=None
        )(output)
        output = ProbabilisticProcessing(logvar_min)(Prob_output)

    elif type_output == "classif":
        n_param = 1
        tf.keras.layers.Dense(n_param * dim_out, name="Prob", activation="softmax")(
            output
        )

    elif dim_out is not None:
        n_param = 1
        output = tf.keras.layers.Dense(n_param * dim_out, name="Output_" + name)(output)

    else:
        pass

    if shape_2D_out is not None:
        output = tf.keras.layers.Lambda(
            lambda x: K.reshape(x, (-1, shape_2D_out[1] * n_param, shape_2D_out[0]))
        )(output)

        output = tf.keras.layers.Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(
            output
        )

    mlp = tf.keras.Model(inputs, output, name="MLP_" + name)
    return mlp


# Improvemement to do : Transform moving_slice in Keras layers
# Dev : Preprocessing et reconstruction basé cnn 1D for independant multivariate TS : depreciated

# Old


def stack_and_roll_layer(
    inputs, size_window, size_subseq, padding, name="", format="tf_slice"
):
    """Layers that produce a stack rolled layers to produce a batch of subsequence

    Args:
        inputs (_type_): layers
        size_window (_type_): size of subseqeuence
        size_subseq (_type_): _description_
        padding (_type_): _description_
        name (str, optional): _description_. Defaults to "".
        format (str, optional): _description_. Defaults to "tf_slice".

    Returns:
        _type_: _description_
    """
    # Slice and stack tensor to produce folded representation
    slide_tensor = []
    n_step = get_fold_nstep(size_window, size_subseq, padding)
    # Implementation numpy

    if format == "tf_slice":  # tf slice based
        n_step = get_fold_nstep(size_window, size_subseq, padding)
        z_slice = []
        steps = range(0, n_step * padding, padding)
        for i, step in enumerate(steps):
            z_slice.append(
                tf.slice(inputs, [0, step, 0], [-1, size_subseq, -1])[:, None]
            )
        x = tf.concat(z_slice, axis=1)
        return x
    elif format == "np_slice":  # np slice based
        for i in range(n_step):
            slide_tensor.append(
                inputs[:, (i * padding) : (i * padding) + size_subseq, :][:, None]
            )
        return Lambda(lambda x: K.concatenate(x, axis=1), name=name + "_rollstack")(
            slide_tensor
        )

    elif format == "tf_map":  # tf map based
        x = tf.map_fn(
            lambda i: inputs[:, (i * padding) : (i * padding) + size_subseq, :],
            tf.range(n_step),
            dtype=tf.float32,
        )
        x = tf.transpose(x, [1, 0, 2, 3])
        return x

    else:
        exit(
            f"Unknown format {format}. Supported formats are: tf_slice, np_slice and tf_map"
        )


def cnn_enc_1D(
    size_subseq_enc,
    dim_out,
    dim_z,
    dim_lt,
    type_output=None,
    k=32,
    dp=0.05,
    random_state=None,
    **kwarg,
):
    """cnn_enc_1D layers

    Args:
        size_subseq_enc (int, optional): _description_. Defaults to 100.
        dim_out (int, optional): _description_. Defaults to 10.
        dim_z (int, optional): _description_. Defaults to 100.
        dim_lt (int, optional): _description_. Defaults to 100.
        type_output (_type_, optional): _description_. Defaults to None.
        k (int, optional): _description_. Defaults to 32.
        dp (float, optional): _description_. Defaults to 0.05.
        random_state (bool): handle experimental random using seed.

    Returns:
        _type_: _description_
    """

    flag_mc = False
    if type_output in ["MC_Dropout", "Deep_ensemble"]:
        flag_mc = True

    inputs = tf.keras.layers.Input(shape=(size_subseq_enc, dim_out), name="st")

    output = layers.SeparableConv1D(
        k,
        int(dim_lt / 4),
        strides=1,
        depth_multiplier=10,
        activation="relu",
        padding="same",
    )(inputs)

    output = layers.AveragePooling1D((5), padding="same")(output)
    if dp > 0:
        output = tf.keras.layers.Dropout(dp, seed=random_state)(
            output, training=flag_mc
        )
    output = layers.SeparableConv1D(
        k * 2,
        int(dim_lt / 8),
        strides=1,
        depth_multiplier=10,
        activation="relu",
        padding="same",
    )(output)

    output = layers.AveragePooling1D((2), padding="same")(output)
    if dp > 0:
        output = tf.keras.layers.Dropout(dp, seed=add_random_state(random_state, 1))(
            output, training=flag_mc
        )
    output = layers.SeparableConv1D(
        k * 2,
        int(dim_lt / 16),
        strides=1,
        depth_multiplier=10,
        activation="relu",
        padding="same",
    )(output)

    output = layers.AveragePooling1D((2), padding="same")(output)
    output = tf.keras.layers.Flatten()(output)
    output = tf.keras.layers.Dense(dim_z)(output)
    enc = Model(inputs, output)
    return enc


def cnn_dec_1D(
    size_subseq_dec=10,
    dim_out=10,
    dim_z=100,
    type_output=None,
    k=32,
    dp=0.05,
    random_state=None,
    **kwarg,
):
    """_summary_

    Args:
        size_subseq_dec (int, optional): _description_. Defaults to 10.
        dim_out (int, optional): _description_. Defaults to 10.
        dim_z (int, optional): _description_. Defaults to 100.
        type_output (_type_, optional): _description_. Defaults to None.
        k (int, optional): _description_. Defaults to 32.
        dp (float, optional): _description_. Defaults to 0.05.
        random_state (bool): handle experimental random using seed.

    Returns:
        _type_: _description_
    """

    dim_chan_out = 1
    flag_mc = False
    if type_output in ["MC_Dropout", "Deep_ensemble"]:
        dim_chan_out = 2
        flag_mc = True

    elif type_output == "EDL":
        dim_chan_out = 4

    inputs = layers.Input(shape=(dim_z), name="st")
    output = tf.keras.layers.Lambda(lambda x: x[:, None, :])(inputs)
    output = layers.Conv1DTranspose(
        k, (10), strides=5, activation="relu", padding="same"
    )(output)
    if dp > 0:
        output = tf.keras.layers.Dropout(dp, seed=random_state)(
            output, training=flag_mc
        )
    output = layers.Conv1DTranspose(
        k, (50), strides=2, activation="relu", padding="same"
    )(output)
    if dp > 0:
        output = tf.keras.layers.Dropout(dp, seed=add_random_state(random_state, 1))(
            output, training=flag_mc
        )
    output = layers.SeparableConv1D(
        dim_out * dim_chan_out,
        10,
        strides=1,
        depth_multiplier=10,
        activation="sigmoid",
        padding="same",
    )(output)
    # aamini/evidential-deep-learning
    if type_output in ["MC_Dropout", "Deep_ensemble"]:
        output = ProbabilisticProcessing()(output)

    if type_output == "EDL":
        output = EDLProcessing()(output)

    output = tf.keras.layers.Lambda(
        lambda x: K.reshape(x, (-1, size_subseq_dec * dim_out * dim_chan_out))
    )(output)
    dec = tf.keras.Model(inputs, output, name="conv_lag_dec")
    return dec


# Dev : Preprocessing CNN based representation layers for 2D TS without spatial structure


def cnn_enc(
    size_subseq_enc,
    dim_out,
    dim_chan=1,
    k1=10,
    reduction_x1=8,
    reduction_x2=1,
    dim_z=100,
    dp=0.02,
    random_state=None,
    **kwarg,
):
    """Warning depreciated CNN_enc implementation

    Args:
        size_subseq_enc (_type_): _description_
        dim_out (_type_): _description_
        dim_chan (int, optional): _description_. Defaults to 1.
        k1 (int, optional): _description_. Defaults to 10.
        reduction_x1 (int, optional): _description_. Defaults to 8.
        reduction_x2 (int, optional): _description_. Defaults to 1.
        dim_z (int, optional): _description_. Defaults to 100.
        dp (float, optional): _description_. Defaults to 0.02.
        random_state (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    dim_out_reshape = int(dim_out / dim_chan)
    inputs = tf.keras.layers.Input(shape=(size_subseq_enc, dim_out), name="st")

    output = tf.keras.layers.Lambda(
        lambda x: K.reshape(x, (-1, size_subseq_enc, dim_out_reshape, dim_chan))
    )(inputs)

    output = tf.keras.layers.Conv2D(
        k1 * 3,
        (reduction_x1, reduction_x2),
        strides=1,
        activation="relu",
    )(output)

    output = tf.keras.layers.AveragePooling2D((2, reduction_x2), padding="same")(output)

    # output = tf.keras.layers.Dropout(dp)(output, training=True)
    output = tf.keras.layers.BatchNormalization()(output)

    output = tf.keras.layers.Conv2D(
        k1,
        (reduction_x1, dim_out_reshape),
        strides=1,
        activation="relu",
    )(output)

    output = tf.keras.layers.AveragePooling2D((2, 1), padding="same")(output)

    # output = tf.keras.layers.Dropout(dp)(output, training=True)
    output = tf.keras.layers.BatchNormalization()(output)

    output = tf.keras.layers.Conv2D(
        k1 * 2, (reduction_x1, 1), strides=1, activation="relu"
    )(output)

    output = tf.keras.layers.Flatten()(output)
    output = tf.keras.layers.Dense(dim_z)(output)
    if dp > 0:
        output = tf.keras.layers.Dropout(dp, seed=random_state)(output, training=True)
    enc = tf.keras.Model(inputs, output, name="conv_lag_enc")
    return enc


def cnn_dec(
    size_subseq_dec,
    dim_out,
    dim_chan=1,
    type_output=None,
    k1=10,
    min_logvar=-6,
    dim_z=100,
    dp=0.01,
    random_state=None,
    **kwarg,
):
    """Warning depreciated CNN_dec implementation

    Args:
        size_subseq_dec (_type_): _description_
        dim_out (_type_): _description_
        dim_chan (int, optional): _description_. Defaults to 1.
        type_output (_type_, optional): _description_. Defaults to None.
        k1 (int, optional): _description_. Defaults to 10.
        min_logvar (int, optional): _description_. Defaults to -6.
        dim_z (int, optional): _description_. Defaults to 100.
        dp (float, optional): _description_. Defaults to 0.01.
        random_state (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    dim_chan_out = 1
    if type_output in ["MC_Dropout", "Deep_ensemble"]:
        dim_chan_out = 2

    elif type_output == "EDL":
        dim_chan_out = 4

    dim_out_reshape = int(dim_out / dim_chan)
    inputs = tf.keras.layers.Input(shape=(dim_z), name="st")
    output = tf.keras.layers.Lambda(lambda x: x[:, None, None, :])(inputs)

    output = tf.keras.layers.Conv2DTranspose(
        k1 * 2,
        (int(np.floor((size_subseq_dec + 1) / 2)), dim_out_reshape * dim_chan),
        strides=(1, 1),
        activation="relu",
    )(output)

    output = tf.keras.layers.BatchNormalization()(output)
    if dp > 0:
        output = tf.keras.layers.Dropout(dp, seed=random_state)(output, training=True)

    output = tf.keras.layers.Conv2DTranspose(
        k1,
        (int(np.ceil((size_subseq_dec + 1) / 2)), 1),
        strides=(1, 1),
        activation="relu",
    )(output)

    output = tf.keras.layers.BatchNormalization()(output)
    if dp > 0:
        output = tf.keras.layers.Dropout(dp, seed=add_random_state(random_state, 1))(
            output, training=True
        )

    output = tf.keras.layers.Conv2DTranspose(dim_chan_out, (1, 1), activation="linear")(
        output
    )

    # Probablistic NN
    if type_output in ["MC_Dropout", "Deep_ensemble"]:
        output = ProbabilisticProcessing(min_logvar)(output)

    # aamini/evidential-deep-learning
    elif type_output == "EDL":
        output = EDLProcessing(min_logvar)(output)

    else:
        pass

    output = tf.keras.layers.Lambda(
        lambda x: K.reshape(x, (-1, size_subseq_dec * dim_out * dim_chan_out))
    )(output)

    dec = tf.keras.Model(inputs, output, name="conv_lag_dec")
    return dec


def conv_block_1D(
    inputs,
    dim_chan,
    filters=32,
    kernel=2,
    strides=2,
    dp=0.02,
    flag_mc=False,
    random_state=None,
):

    output = tf.keras.layers.Conv1D(
        filters * dim_chan,
        kernel,
        strides=strides,
        padding="causal",
        groups=dim_chan,
        activation="relu",
    )(inputs)
    output = tf.keras.layers.BatchNormalization()(output)
    if dp > 0:
        output = tf.keras.layers.Dropout(dp, seed=random_state)(
            output, training=flag_mc
        )
    return output


def conv_block_2D(
    inputs,
    dim_chan,
    filters=32,
    kernel=5,
    strides=(2, 1),
    dp=0.02,
    flag_mc=False,
    random_state=None,
):

    output = tf.keras.layers.Conv2D(
        filters, kernel, strides=strides, padding="valid", activation="relu"
    )(inputs)
    output = tf.keras.layers.BatchNormalization()(output)
    if dp > 0:
        output = tf.keras.layers.Dropout(dp, seed=random_state)(
            output, training=flag_mc
        )
    return output


# Version actuelle : génération d'une structure CNN_ENC parametrable


def cnn_enc_bis(
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
    **kwarg,
):
    """Produce a cnn_enn subpart of a deep learning predictor

    Returns:
        _type_: _description_
    """
    flag_mc = False
    if type_output in ["MC_Dropout"]:
        flag_mc = True

    dim_space = int(dim_target / dim_chan)
    if dim_chan == 1:
        inputs = tf.keras.layers.Input(shape=(size_subseq_enc, dim_target), name="st")

        output = tf.keras.layers.Lambda(lambda x: K.expand_dims(x, axis=-1))(inputs)
    else:
        inputs = tf.keras.layers.Input(shape=(size_subseq_enc, dim_target), name="st")

        output = tf.keras.layers.Lambda(
            lambda x: K.reshape(x, (-1, size_subseq_enc, dim_space, dim_chan))
        )(inputs)

    for n, (filters, kernel, strides) in enumerate(
        zip(list_filters, list_kernels, list_strides)
    ):
        if block == "2D":
            output = conv_block_2D(
                output,
                dim_space,
                filters=filters,
                kernel=kernel,
                strides=strides,
                flag_mc=flag_mc,
                random_state=add_random_state(random_state, 1 + n),
            )

        if block == "1D":
            output = conv_block_1D(
                output,
                dim_space,
                filters=filters,
                kernel=kernel,
                strides=strides,
                flag_mc=flag_mc,
                random_state=add_random_state(random_state, 1 + len(list_filters) + n),
            )

    output = tf.keras.layers.Flatten()(output)
    output = tf.keras.layers.Dense(dim_z)(output)
    if dp > 0:
        output = tf.keras.layers.Dropout(dp, seed=random_state)(
            output, training=flag_mc
        )
    enc = tf.keras.Model(inputs, output, name="conv_lag_enc")
    return enc


def Tconv_block_1D(
    inputs,
    dim_out,
    filters=32,
    kernel=2,
    strides=2,
    dp=0.02,
    flag_mc=False,
    random_state=None,
):

    output = tf.keras.layers.Conv1DTranspose(
        filters * dim_out,
        kernel,
        strides=strides,
        padding="same",
        groups=dim_out,
        activation="relu",
    )(inputs)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dropout(dp, seed=random_state)(output, training=flag_mc)
    return output


def Tconv_block_2D(
    inputs,
    dim_out,
    filters=32,
    kernel=5,
    strides=(2, 1),
    dp=0.02,
    flag_mc=False,
    random_state=None,
):

    output = tf.keras.layers.Conv2DTranspose(
        filters, (kernel, dim_out), strides=strides, activation="relu"
    )(inputs)
    output = tf.keras.layers.BatchNormalization()(output)
    if dp > 0:
        output = tf.keras.layers.Dropout(dp, seed=random_state)(
            output, training=flag_mc
        )
    return output


def cnn_dec_bis(
    size_subseq_dec,
    dim_out,
    dim_chan=1,
    type_output=None,
    min_logvar=-6,
    list_filters=[64, 64],
    strides=(1, 1),
    list_kernels=[4, 4],
    dim_z=200,
    random_state=None,
    **kwarg,
):

    flag_mc = False
    if type_output in ["MC_Dropout"]:
        flag_mc = True

    dim_chan_out = 1 * dim_chan
    if type_output in ["MC_Dropout", "Deep_ensemble"]:
        dim_chan_out = 2 * dim_chan

    elif type_output == "EDL":
        dim_chan_out = 4 * dim_chan

    dim_space = int(dim_out / size_subseq_dec)
    inputs = tf.keras.layers.Input(shape=(dim_z), name="st")
    output = tf.keras.layers.Lambda(lambda x: x[:, None, None, :])(inputs)

    for n, (filters, kernel) in enumerate(zip(list_filters, list_kernels)):
        d_tar = 1
        if n == 0:
            d_tar = dim_space
        output = Tconv_block_2D(
            output,
            d_tar,
            filters=filters,
            kernel=kernel,
            strides=strides,
            flag_mc=flag_mc,
            random_state=add_random_state(random_state, n),
        )

    output = tf.keras.layers.Conv2DTranspose(dim_chan_out, (1, 1), activation="linear")(
        output
    )

    # Probablistic NN
    if type_output in ["MC_Dropout", "Deep_ensemble"]:
        output = ProbabilisticProcessing(min_logvar)(output)

    # aamini/evidential-deep-learning
    elif type_output == "EDL":
        output = EDLProcessing(min_logvar)(output)

    output = tf.keras.layers.Lambda(
        lambda x: K.reshape(x, (-1, size_subseq_dec, dim_space * dim_chan_out))
    )(output)

    dec = tf.keras.Model(inputs, output, name="conv_lag_dec")
    if dec.layers[-2].output.shape[1] != dec.layers[-1].output.shape[1]:
        print("Warning : inadequate deconvolution window size : model will crash")

    return dec


# Dev : Preprocessing CNN based representation layers for 2D TS without spatial structure


def dense2D_enc_dec(
    size_subseq_enc,
    size_subseq_dec,
    dim_in,
    dim_out,
    layers_size=[100, 50],
    dim_z=100,
    dp=0.05,
    enc_only=False,
    type_output=None,
    random_state=None,
):

    inputs = tf.keras.layers.Input(shape=(size_subseq_enc, dim_in), name="st")
    inputs_flatten = Lambda(lambda x: K.reshape(x, (-1, size_subseq_enc * dim_in)))(
        inputs
    )

    layers_size_enc = layers_size
    layers_size_enc.append(dim_z)

    mlp_enc = mlp(
        size_subseq_enc * dim_in,
        dim_out=None,
        layers_size=layers_size_enc,
        name="",
        dp=dp,
        type_output=None,
        random_state=random_state,
    )(inputs_flatten)
    enc = tf.keras.Model(inputs, mlp_enc, name="MLP_enc")

    if enc_only:
        return (enc, None)

    inputs = tf.keras.layers.Input(shape=(dim_z), name="embedding")
    z_flatten = Lambda(lambda x: K.reshape(x, (-1, dim_z)))(inputs)
    mlp_dec = mlp(
        dim_z,
        dim_out=size_subseq_dec * dim_out,
        layers_size=layers_size[::-1],
        name="",
        dp=dp,
        type_output=type_output,
        random_state=add_random_state(random_state, 100),
    )(z_flatten)

    if type_output == "EDL":
        dim_out = dim_out * 4

    elif type_output == "MC_Dropout":
        dim_out = dim_out * 2

    reshape_dec = Lambda(lambda x: K.reshape(x, (-1, size_subseq_dec, dim_out)))(
        mlp_dec
    )
    dec = tf.keras.Model(inputs, reshape_dec, name="MLP_dec")
    return (enc, dec)


def moving_slice_map(inputs, n_step, padding, kick_off=0, depth_slice=1):
    """Apply Layers on n_step slices of input with padding
    Args:
        input (TF.tensor): Input tensors
        Layers (Keras.model): Submodels
        n_step (int): N_slide
        padding (int): padding"""

    steps = range(kick_off, kick_off + n_step * padding, padding)
    z_slice = []
    for i, step in enumerate(steps):
        if len(inputs.shape) == 3:
            slice = tf.slice(inputs, [0, step, 0], [-1, depth_slice, -1])
        else:
            slice = tf.slice(inputs, [0, step, 0, 0], [-1, depth_slice, -1, -1])

        if depth_slice > 1:
            slice = slice[:, None]

        z_slice.append(slice)

    z_slice = Lambda(lambda x: K.concatenate(x, axis=1))(z_slice)
    return z_slice


class Moving_slice_layer(Layer):
    """Layer that apply moving_slice_map"""

    def __init__(
        self, n_step, padding, kick_off=0, depth_slice=1, ignore_futur=None, **kwargs
    ):
        self.n_step = n_step
        self.padding = padding
        self.kick_off = kick_off
        self.depth_slice = depth_slice
        self.ignore_futur = ignore_futur
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, input_data):
        """Apply moving_slice_map to generate window sequnce

        Args:
            input_data (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.ignore_futur:
            input_data = input_data[:, : self.ignore_futur]
        z_slice = moving_slice_map(
            input_data, self.n_step, self.padding, self.kick_off, self.depth_slice
        )
        return z_slice

    def compute_output_shape(self, input_shape):
        step = range(
            self.kick_off, self.kick_off + self.n_step * self.padding, self.padding
        )
        if self.depth_slice > 1:
            output_shape = (input_shape[0], len(step), input_shape[-1])
        else:
            output_shape = (
                input_shape[0],
                len(step),
                self.depth_slice,
                input_shape[-1],
            )
        return output_shape


class Double_Moving_slice_layer(Layer):
    """Layer that apply double moving_slice_map

    Args:
        Layer (_type_): _description_
    """

    def __init__(
        self, n_step_out, n_step_in, padding_out, padding_in, depth_slice=1, **kwargs
    ):
        self.n_step_out = n_step_out
        self.n_step_in = n_step_in
        self.padding_out = padding_out
        self.padding_in = padding_in
        self.depth_slice = depth_slice
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, input_data):
        """_summary_

        Args:
            input_data (_type_): _description_

        Returns:
            _type_: _description_
        """
        z_slice_out = []
        for i in range(self.n_step_out):
            kick_off = i * self.padding_out
            z_slice_in = moving_slice_map(
                input_data,
                self.n_step_in,
                self.padding_in,
                kick_off=kick_off,
                depth_slice=self.depth_slice,
            )
            z_slice_out.append(z_slice_in[:, None])
        z_slice = Lambda(lambda x: K.concatenate(x, axis=1))(z_slice_out)
        return z_slice

    def compute_output_shape(self, input_shape):
        step_out = range(0, self.n_step_out * self.padding_out, self.padding_out)
        step_in = range(0, self.n_step_in * self.padding_in, self.padding_in)

        if self.depth_slice > 1:
            output_shape = (
                input_shape[0],
                len(step_out),
                len(step_in),
                input_shape[-1],
            )
        else:
            output_shape = (
                input_shape[0],
                len(step_out),
                len(step_in),
                self.depth_slice,
                input_shape[-1],
            )
        return output_shape


# Modelisation ensemble de code lié à la construction d'un modèle LSTM ED à prédiction multihorizon
# Processing layers : LSTM cell modification to handle state passing.


class LSTMCellReturnCellState(LSTMCell):
    """Layer LSTM returning output and state jointly

    Args:
        LSTMCell (_type_): _description_
    """

    def call(self, inputs, states, training=None):
        """_summary_

        Args:
            inputs (_type_): _description_
            states (_type_): _description_
            training (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        outputs, [h, c] = super().call(inputs, states, training=training)
        return tf.concat([h, c], axis=1), [h, c]


class LSTMCellMidsize(LSTMCell):
    """Hack to take into accout state in cell : size in dim_z*2 | size out dim_z

    Args:
        LSTMCell (_type_): _description_
    """

    def build(self, input_shape):
        input_shape = (None, int(input_shape[-1] / 2))
        super().build(input_shape)


# Processing layers : RNN cell


class RNN_states_in_inputs(RNN):
    """RNN class dispatching jointly [H,C]

    Args:
        RNN (_type_): _description_
    """

    def call(
        self, inputs, mask=None, training=None, initial_state=None, constants=None
    ):
        """_summary_

        Args:
            inputs (_type_): _description_
            mask (_type_, optional): _description_. Defaults to None.
            training (_type_, optional): _description_. Defaults to None.
            initial_state (_type_, optional): _description_. Defaults to None.
            constants (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        h, c = tf.split(inputs, 2, axis=-1)
        states = (h[:, -1], c[:, -1])
        return super().call(
            h, initial_state=states, mask=mask, training=training, constants=constants
        )


def LSTM_EProcessing(
    n_step,
    dim_in,
    dim_z,
    flag_mc,
    dp=0.05,
    dp_r=0.02,
    l1_l2_reg=(0.0000001, 0.0000001),
    random_state=None,
):
    """Encoder Processing as block aim to capture dynamic
    Input (batch,n_step,dim_in) output (batch,n_step,dim_z*2) with Z_h and Z_state concatenate)


    Args:
        n_step (_type_): _description_
        dim_in (_type_): _description_
        dim_z (_type_): _description_
        flag_mc (_type_): _description_
        dp (float, optional): _description_. Defaults to 0.05.
        dp_r (float, optional): _description_. Defaults to 0.02.
        l1_l2_reg (tuple, optional): _description_. Defaults to (0.0000001, 0.0000001).
        random_state (bool): handle experimental random using seed.
    Returns:
        Model: Keras model as LSTM Encoder block
    """

    lstm_cell_enc = LSTMCellReturnCellState(
        int(dim_z),
        name="LSTM_enc_0",
        activation="sigmoid",
        recurrent_dropout=dp_r,
        seed=random_state,
        kernel_regularizer=tf.keras.regularizers.l1_l2(
            l1=l1_l2_reg[0], l2=l1_l2_reg[1]
        ),
    )
    lstm_enc = tf.keras.layers.RNN(lstm_cell_enc, return_sequences=True)

    EProcessing_in = tf.keras.layers.Input(
        shape=(n_step, dim_in), name="input_EProcessing"
    )
    Z_input_lstm = Dropout(dp, seed=add_random_state(random_state, 100))(
        TimeDistributed(Dense(dim_z))(EProcessing_in)
    )
    Z_lstm_output = Dropout(dp, seed=add_random_state(random_state, 101))(
        lstm_enc(Z_input_lstm), training=flag_mc
    )

    # Extended with 0 to not affect state part in skip connexion.
    Z_input_lstm_extended = tf.keras.layers.Lambda(lambda x: K.concatenate([x, x * 0]))(
        Z_input_lstm
    )

    EProcessing_out = tf.keras.layers.Lambda(
        lambda x: tf.keras.layers.Add()([x[0], x[1]]), name="skip_EProcessing"
    )([Z_lstm_output, Z_input_lstm_extended])

    LSTM_EProcessing_ = tf.keras.Model(
        EProcessing_in, EProcessing_out, name="EProcessing"
    )

    return LSTM_EProcessing_


def LSTM_DProcessing(
    n_step,
    dim_z,
    flag_mc,
    dp=0.05,
    dp_r=0.02,
    l1_l2_reg=(0.0000001, 0.0000001),
    random_state=None,
):
    """Decoder Processing as block : aim to make temporal projection (Usefull to hold query information)
    Input (batch,n_step,dim_z*2) output (batch,n_step,dim_z) with Zdecoding latent space)

    Args:
        n_step (_type_): _description_
        dim_z (_type_): _description_
        flag_mc (_type_): _description_
        dp (float, optional): _description_. Defaults to 0.05.
        dp_r (float, optional): _description_. Defaults to 0.02.
        l1_l2_reg (tuple, optional): _description_. Defaults to (0.0000001, 0.0000001).
        random_state (bool): handle experimental random using seed.

    Returns:
        Model: Keras model as LSTM Decoder block
    """

    lstm_cell_dec = LSTMCellMidsize(
        int(dim_z),
        name="LSTM_dec_0",
        activation="sigmoid",
        recurrent_dropout=dp_r,
        seed=random_state,
        kernel_regularizer=tf.keras.regularizers.l1_l2(
            l1=l1_l2_reg[0], l2=l1_l2_reg[1]
        ),
    )

    lstm_dec = RNN_states_in_inputs(lstm_cell_dec, return_sequences=True)

    DProcessing_in = tf.keras.layers.Input(
        shape=(n_step, dim_z * 2), name="input_DProcessing"
    )
    Z_lstm_output = Dropout(dp, seed=add_random_state(random_state + 100))(
        lstm_dec(DProcessing_in), training=flag_mc
    )
    DProcessing_out = tf.keras.layers.Lambda(
        lambda x: tf.keras.layers.Add()([x[0], x[1][:, :, :dim_z]]),
        name="skip_DProcessing",
    )([Z_lstm_output, DProcessing_in])

    LSTM_DProcessing_ = tf.keras.Model(
        DProcessing_in, DProcessing_out, name="DProcessing"
    )
    return LSTM_DProcessing_


class Add_query_to_Z_Processing_with_state(Layer):
    def __init__(self, dim_z):
        self.dim_z = dim_z
        self.layer = Dense(dim_z)
        super().__init__()

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.dim_z * 2)

    def call(self, ZProcessing, Query):
        """_summary_

        Args:
            ZProcessing (_type_): _description_
            Query (_type_): _description_

        Returns:
            _type_: _description_
        """
        Z = tf.concat([ZProcessing[:, :, :, : self.dim_z], Query], axis=-1)
        new_Z = TimeDistributed(TimeDistributed(self.layer), name="wtf")(Z)
        Z = tf.concat([new_Z, ZProcessing[:, :, :, self.dim_z :]], axis=-1)
        return Z


def get_cnn_enc_params(dim_target, size_subseq_enc=1, dim_z=50, random_state=None):
    """Produce dict params that can instanciate cnn_enn_bloc
    Args:
        dim_target (_type_): dimension of motifs to convolute
        size_subseq_enc (int, optional): length of motifs to convulute
        dim_z (int, optional): latent dimension
        random_state (bool): handle experimental random using seed.

    """

    dict_params = {
        "builder": cnn_enc_bis,
        "size_subseq_enc": size_subseq_enc,
        "dim_target": dim_target,
        "dim_chan": 1,
        "list_filters": [64, 64, 32],
        "list_kernels": [10, 10, 10],
        "block": "2D",
        "dim_z": dim_z,
        "dp": 0.05,
        "random_state": random_state,
    }

    return dict_params


def get_cnn_dec_params(dim_target, size_subseq_dec=1, dim_z=50, random_state=None):
    """Produce dict params that can instanciate cnn_dec_bloc
    Args:
        dim_target (_type_): dimension of motifs to convolute
        size_subseq_dec (int, optional): length of motifs to convulute
        dim_z (int, optional): latent dimension
        random_state (bool): handle experimental random using seed.
    """
    dict_params = {
        "builder": cnn_dec_bis,
        "dim_z": dim_z,
        "size_subseq_dec": size_subseq_dec,
        "dim_target": dim_target,
        "dim_chan": 1,
        "list_filters": [64, 64],
        "strides": (2, 1),
        "list_kernels": [4, 4],
        "min_logvar": -10,
        "random_state": random_state,
    }

    return dict_params
