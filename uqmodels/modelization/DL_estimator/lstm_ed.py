import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.layers import LSTM, Dropout, Input, Lambda, TimeDistributed

from uqmodels.modelization.DL_estimator.data_embedding import (
    Factice_Time_Extension,
    Mouving_conv_Embedding,
    Mouving_Windows_Embedding,
)
from uqmodels.modelization.DL_estimator.neural_network_UQ import (
    NN_UQ,
    get_training_parameters,
    get_UQEstimator_parameters,
    mlp,
)
from uqmodels.modelization.DL_estimator.utils import (
    Folder_Generator,
    set_global_determinism,
)
from uqmodels.utils import add_random_state, apply_mask, stack_and_roll

# Basic memory module


def build_lstm_stacked(
    size_window=20,
    n_windows=5,
    step=1,
    dim_target=3,
    dim_chan=1,
    dim_horizon=5,
    dim_ctx=18,
    dim_z=200,
    dp=0.05,
    dp_rec=0.03,
    type_output=None,
    num_lstm_enc=1,
    num_lstm_dec=1,
    k_reg=(10e-6, 10e-6),
    layers_enc=[75, 150, 75],
    layers_dec=[150, 75],
    list_strides=[2, 1],
    list_kernels=None,
    list_filters=None,
    with_ctx_input=True,
    with_convolution=True,
    dim_dyn=None,
    random_state=None,
    **kwarg
):
    """Builder for LSTM ED UQ with convolutif preprocessing for lag values

    Args:
        size_window (int, optional): Size of window for lag values. Defaults to 10.
        n_windows (int, optional): Number of window in past. Defaults to 5.
        step (int, optional): step between windows. Defaults to 1.
        dim_target (int, optional): dimension of TS. Defaults to 1.
        dim_chan (int, optional): Number of channel of TS. Defaults to 1.
        dim_horizon (int, optional): futur_horizon to predict. Defaults to 3.
        dim_ctx (int, optional): Number of ctx_features. Defaults to 20.
        dim_z (int, optional): Size of latent sapce. Defaults to 100.
        layers_enc (list, optional):size of MLP preprocessing
        (after concatenation of past values embeding + ctx) Defaults to [150].
        layers_dec (list, optional): size of MLP interpretor. Defaults to 2.
        dp (float, optional): dropout. Defaults to 0.05.
        dp_rec (float, optional): transformer dropout. Defaults to 0.1.
        k_reg (tuple, optional): _description_. Defaults to (0.00001, 0.00001).
        with_positional_embedding (bool, optional): _description_. Defaults to False.
        with_ctx_input (bool, optional): Expect ctx features in addition to lag. Defaults to True.
        with_convolution (bool, optional): use convolution rather than
        whole lag values in the windows. Defaults to True.
        type_output (_type_, optional): mode of UQ (see NN_UQ). Defaults to None.
        random_state (bool): handle experimental random using seed.


    Returns:
        transformer : multi-step forecaster with UQ
    """
    set_global_determinism(random_state)

    if dim_dyn is None:
        dim_dyn = dim_target

    residuals_link = False
    flag_mc = 0
    if type_output in ["BNN", "MC_Dropout"]:
        flag_mc = 1

    # Embedding_interpretor
    Interpretor = mlp(
        dim_in=dim_z,
        dim_out=dim_target,
        layers_size=layers_dec,
        dp=dp,
        type_output=type_output,
        name="Interpretor",
        random_state=random_state,
    )

    # Encoder LSTM
    Encoders = []
    for i in range(num_lstm_enc):
        Encoders.append(
            LSTM(
                dim_z,
                name="LSTM_enc_" + str(i),
                return_sequences=True,
                return_state=True,
                activation="sigmoid",
                recurrent_dropout=dp_rec,
                dropout=dp,
                kernel_regularizer=tf.keras.regularizers.l1_l2(
                    l1=k_reg[0],
                    l2=k_reg[1],
                ),
                seed=add_random_state(random_state, 100 + i),
            )
        )

    # Encoder decoder LSTM
    Decoders = []
    for i in range(num_lstm_dec):
        Decoders.append(
            LSTM(
                dim_z,
                name="LSTM_dec_" + str(i),
                return_sequences=True,
                stateful=False,
                return_state=True,
                activation="sigmoid",
                recurrent_dropout=dp_rec,
                dropout=dp,
                kernel_regularizer=tf.keras.regularizers.l1_l2(
                    l1=k_reg[0], l2=k_reg[1]
                ),
                seed=add_random_state(random_state, 200 + i),
            )
        )

    outputs = []

    # Input definition.

    list_inputs = []

    if with_ctx_input:
        CTX_inputs = Input(shape=(n_windows, dim_ctx), name="LT")
        list_inputs.append(CTX_inputs)

    Y_past = Input(shape=(size_window, dim_dyn), name="ST")
    list_inputs.append(Y_past)

    # Preprocessing layers :
    if with_convolution:
        MWE = Mouving_conv_Embedding(
            size_window,
            n_windows,
            step=step,
            dim_d=dim_dyn,
            dim_chan=dim_chan,
            use_conv2D=True,
            list_strides=list_strides,
            list_filters=list_filters,
            list_kernels=list_kernels,
            dp=0.05,
            flag_mc=flag_mc,
            seed=add_random_state(random_state, 300),
        )

    else:
        MWE = Mouving_Windows_Embedding(
            size_window, n_windows, step=step, dim_d=dim_dyn, dim_chan=dim_chan
        )

    FTE = Factice_Time_Extension(dim_horizon)

    dim_embedding = MWE.last_shape
    if with_ctx_input:
        dim_embedding += dim_ctx

    Embeddor_ctx = mlp(
        dim_in=dim_embedding,
        dim_out=None,
        layers_size=layers_enc,
        dp=dp,
        name="Embeddor",
        regularizer_W=k_reg,
        flag_mc=flag_mc,
        random_state=add_random_state(random_state, 500),
    )

    # Preprocessing computation
    Data = MWE(Y_past)
    # Concat with cat features
    if with_ctx_input:
        Data = K.concatenate([CTX_inputs, Data], axis=-1)

    # Factice time augmentation (actually useless but can be usefull for extended predict horizon)

    Embedding = TimeDistributed(Embeddor_ctx)(Data)

    # Encoder part
    Z_enc = Embedding
    state = None
    for Encoder in Encoders:
        Z_enc, H_enc, C_enc = Encoder(Embedding, initial_state=state)
        state = H_enc, C_enc

    if residuals_link:
        Z_enc = Z_enc + Embedding

    Z_enc = FTE(Z_enc)

    # Lattent embedding of each state (Z_t) and last current memory state (H et C)
    Z_enc = Dropout(dp, seed=add_random_state(random_state, 501))(
        Z_enc, training=flag_mc
    )
    H_enc = Dropout(dp, seed=add_random_state(random_state, 502))(
        H_enc, training=flag_mc
    )
    C_enc = Dropout(dp, seed=add_random_state(random_state, 503))(
        C_enc, training=flag_mc
    )
    state = H_enc, C_enc

    # Decoder part : Training inference without loop :
    for Decoder in Decoders:
        Z_dec, H_dec, C_dec = Decoder(Z_enc, initial_state=state)
        state = H_dec, C_dec

    if residuals_link:
        Z_dec = Z_dec + Z_enc

    outputs_training = TimeDistributed(Interpretor)(
        Dropout(dp)(Z_dec[:, -dim_horizon:, :], training=flag_mc)
    )

    # Inference loop
    # For i = 0 Z_enc tensor [Batch,size_window,dim_z] -> Stacked LSTM
    # Else Z_enc tensor [Batch,1,dim_z]
    if False:
        Z_enc_inference = Z_enc[:, :-dim_horizon]
        for i in range(dim_horizon):
            Z_dec, H_dec, C_dec = Decoder(Z_enc_inference)
            if residuals_link:
                Z_dec = Z_dec + Z_enc_inference

            output = Interpretor(Dropout(dp)(Z_dec[:, -1, :], training=flag_mc))
            outputs.append(output)

        #        if i != (dim_horizon) - 1:
        #            Data = Data_embedding(
        #                inputs_lt, Y_past, outputs, "encoder")
        #            Embedding = TimeDistributed(Embeddor_ctx)(Data)
        #            Z_enc_inference, H_enc, C_enc = Encoder(Embedding)
        #            if residuals_link:
        #                Z_enc_inference = Z_enc_inference + Embedding

        outputs = Lambda(lambda x: K.stack(x, axis=1))(outputs)

    if False:  # Autoreg
        pass
        # list_input = [inputs_lt, Y_past, Y_futur]
        # if not (loop_learning):
        #     tf.keras.Model(list_input, outputs_training)
        # model = tf.keras.Model(list_input, outputs)
    else:
        model = tf.keras.Model(list_inputs, outputs_training)
    return model


class Lstm_ED_UQ(NN_UQ):
    def __init__(
        self,
        model_parameters,
        factory_parameters=dict(),
        training_parameters=dict(),
        type_output=None,
        rescale=False,
        n_ech=5,
        train_ratio=0.9,
        name="",
        random_state=None,
    ):
        """LSTM_ED : Neural network with UQ using NN_UQ wrapper

        Args:
            model_parameters (_type_): _description_
            factory_parameters (_type_, optional): _description_. Defaults to dict().
            training_parameters (_type_, optional): _description_. Defaults to dict().
            type_output (_type_, optional): _description_. Defaults to None.
            rescale (bool, optional): _description_. Defaults to False.
            n_ech (int, optional): _description_. Defaults to 8.
            train_ratio (float, optional): _description_. Defaults to 0.9.
            name (str, optional): _description_. Defaults to "Lstm_stacked".
            random_state (bool): handle experimental random using seed.
        """
        if (random_state) is not None:
            print("Warning : issues non-deterministic behaviour even with random state")

        super().__init__(
            model_initializer=build_lstm_stacked,
            model_parameters=model_parameters,
            factory_parameters=factory_parameters,
            training_parameters=training_parameters,
            type_output=type_output,
            rescale=rescale,
            n_ech=n_ech,
            train_ratio=train_ratio,
            name=name,
            random_state=random_state,
        )

    def factory(self, X, y, mask=None, only_fit_scaler=False, **kwarg):
        model_params = self.model_parameters
        factory_params = self.factory_parameters

        with_ctx_input = model_params["with_ctx_input"]

        step = 1
        if "step" in model_params.keys():
            step = model_params["step"]

        X_none = False
        if X is None:
            X_none = True

        if X_none:
            inputs = None
        else:
            if with_ctx_input:
                X, X_lag = X
                X, X_lag, mask = super().factory(X, X_lag, mask)
                if only_fit_scaler:
                    return None
                X_lt = stack_and_roll(
                    X,
                    model_params["n_windows"],
                    lag=factory_params["factory_lag_lt"],
                    step=step,
                )

                X_st = stack_and_roll(
                    X_lag,
                    model_params["size_window"],
                    lag=factory_params["factory_lag_st"] - 1,
                    step=step,
                )

                inputs = [X_lt, X_st]
            else:
                X, _, mask = super().factory(X, None, mask)
                if only_fit_scaler:
                    return None
                X_lag = X
                X_st = stack_and_roll(
                    X,
                    model_params["size_window"],
                    lag=factory_params["factory_lag_st"] - 1,
                    step=step,
                )
                inputs = [X_st]

        new_y = None
        if y is not None:
            _, new_y, mask = super().factory(None, y, mask)
            new_y = stack_and_roll(
                y,
                model_params["dim_horizon"],
                lag=model_params["dim_horizon"] - 1,
                step=step,
            )

        return inputs, new_y, mask

    def Build_generator(self, X, y, batch=32, shuffle=True, train=True):
        return Folder_Generator(
            X,
            y,
            self,
            batch=batch,
            shuffle=shuffle,
            train=train,
            random_state=self.random_state,
        )


def get_params_dict(
    dim_ctx,
    dim_dyn,
    dim_target,
    size_window=20,
    n_windows=5,
    dim_horizon=5,
    step=1,
    dim_chan=1,
    dp=0.05,
    dp_rec=0.05,
    dim_z=50,
    k_reg=(10e-7, 10e-7),
    num_lstm_enc=1,
    num_lstm_dec=1,
    layers_enc=[150, 75],
    layers_dec=[200, 125, 75],
    list_strides=[2, 1, 1, 1],
    list_filters=[128, 128, 128],
    list_kernels=None,
    with_convolution=True,
    with_ctx_input=True,
    n_ech=3,
    type_output="MC_Dropout",
):
    dict_params = {
        "dim_ctx": dim_ctx,
        "dim_dyn": dim_dyn,
        "dim_target": dim_target,
        "dim_chan": dim_chan,
        "size_window": size_window,
        "n_windows": n_windows,
        "dim_horizon": dim_horizon,
        "type_output": type_output,
        "num_lstm_enc": num_lstm_enc,
        "num_lstm_dec": num_lstm_dec,
        "step": step,
        "dim_z": dim_z,
        "dp": dp,
        "dp_rec": dp_rec,
        "k_reg": k_reg,
        "layers_enc": layers_enc,
        "layers_dec": layers_dec,
        "list_strides": list_strides,
        "list_kernels": list_kernels,
        "list_filters": list_filters,
        "with_convolution": with_convolution,
        "n_ech": n_ech,
        "with_ctx_input": with_ctx_input,
    }
    return dict_params
