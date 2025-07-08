import tensorflow as tf
from keras.layers import Input, TimeDistributed

from uqmodels.modelization.DL_estimator.data_embedding import Mouving_conv_Embedding
from uqmodels.modelization.DL_estimator.metalayers import mlp


def cnn_mlp(
    dim_dyn,
    dim_target,
    size_window=40,
    n_windows=10,
    step=1,
    dim_chan=1,
    dim_z=50,
    type_output="MC_Dropout",
    dp=0.08,
    name="",
):
    """CNN processing with timed distributed MLP
        [ |    |    |  ] *10
         mlp  mlp  mlp
         val  val  val

    Args:
        dim_dyn (_type_): _description_
        dim_target (_type_): _description_
        size_window (int, optional): _description_. Defaults to 40.
        n_windows (int, optional): _description_. Defaults to 10.
        step (int, optional): _description_. Defaults to 1.
        dim_chan (int, optional): _description_. Defaults to 1.
        dim_z (int, optional): _description_. Defaults to 50.
        name (str, optional): _description_. Defaults to "".

    Returns:
        _type_: _description_
    """

    inputs = Input(shape=(size_window + n_windows * step - 1, dim_dyn), name="ST")
    MWE = Mouving_conv_Embedding(
        size_window=size_window,
        n_windows=n_windows,
        step=step,
        dim_d=dim_dyn,
        dim_chan=dim_chan,
        conv2D=True,
        list_strides=[2, 2, 1],
        list_filters=[32, 32, 32],
        list_kernels=None,
        dp=dp,
        flag_mc=True,
    )
    output = MWE(inputs)
    Interpretor = mlp(
        dim_in=32,
        dim_out=dim_target,
        layers_size=[200, 150, 75],
        dp=dp,
        type_output=type_output,
        name="Interpretor",
    )
    output = TimeDistributed(Interpretor)(output)
    cnn_mlp = tf.keras.Model(inputs, output, name="CNN_MLP_" + name)
    return cnn_mlp
