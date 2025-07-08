import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import TimeDistributed
from tensorflow import keras
from tensorflow.keras import Input, layers

from uqmodels.modelization.DL_estimator.data_embedding import (
    Factice_Time_Extension,
    Mouving_conv_Embedding,
    Mouving_Windows_Embedding,
    PositionalEmbedding,
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
from uqmodels.utils import add_random_state, stack_and_roll


@tf.keras.utils.register_keras_serializable(package="UQModels_layers")
class MultiHeadAttention(tf.keras.layers.MultiHeadAttention):
    pass


@tf.keras.utils.register_keras_serializable(package="UQModels_layers")
class LayerNormalization(tf.keras.layers.LayerNormalization):
    pass


@tf.keras.utils.register_keras_serializable(package="UQModels_layers")
class Dropout(tf.keras.layers.Dropout):
    pass


# Transformer Encoder Layer
@tf.keras.utils.register_keras_serializable(package="UQModels_layers")
class Dense(tf.keras.layers.Dense):
    pass


# Transformer Encoder Layer
@tf.keras.utils.register_keras_serializable(package="UQModels_layers")
class TransformerEncoder(layers.Layer):
    """Transformer Encoder Layer from https://keras.io/examples/audio/transformer_asr/"""

    def __init__(
        self,
        dim_z,
        num_heads,
        feed_forward_dim,
        dp_rec=0.1,
        flag_mc=False,
        random_state=None,
        **kwargs
    ):

        super().__init__()
        self.dim_z = dim_z
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.dp_rec = dp_rec
        self.flag_mc = flag_mc
        self.random_state = random_state
        set_global_determinism(self.random_state)

        # Layers instanciation
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=dim_z)
        self.dense1 = Dense(feed_forward_dim, activation="relu")
        self.dense2 = Dense(dim_z)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dp_rec, seed=self.random_state)
        self.dropout2 = Dropout(dp_rec, seed=add_random_state(self.random_state, 1))

    def call(self, inputs, training=None):
        """_summary_

        Args:
            inputs (_type_): _description_
            training (_type_): _description_

        Returns:
            _type_: _description_
        """
        if training is None:
            training = False

        attn_output = self.att(inputs, inputs)
        if self.dp_rec > 0:
            attn_output = self.dropout1(attn_output, training=training | self.flag_mc)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.dense2(self.dense1(out1))
        if self.dp_rec > 0:
            ffn_output = self.dropout2(ffn_output, training=training | self.flag_mc)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = {
            "dim_z": self.dim_z,
            "num_heads": self.num_heads,
            "feed_forward_dim": self.feed_forward_dim,
            "dp_rec": self.dp_rec,
            "flag_mc": self.flag_mc,
            "random_state": self.random_state,
            "att": tf.keras.utils.serialize_keras_object(self.att),
            "layernorm1": tf.keras.utils.serialize_keras_object(self.layernorm1),
            "layernorm2": tf.keras.utils.serialize_keras_object(self.layernorm2),
            "dense1": tf.keras.utils.serialize_keras_object(self.dense1),
            "dense2": tf.keras.utils.serialize_keras_object(self.dense2),
            "dropout1": tf.keras.utils.serialize_keras_object(self.dropout1),
            "dropout2": tf.keras.utils.serialize_keras_object(self.dropout2),
        }
        config = config
        return config

    @classmethod
    def from_config(cls, config):
        att = config.pop("att")
        layernorm1 = config.pop("layernorm1")
        layernorm2 = config.pop("layernorm2")
        dropout1 = config.pop("dropout1")
        dropout2 = config.pop("dropout2")
        dense1 = config.pop("dense1")
        dense2 = config.pop("dense2")

        obj = cls(**config)
        print(dense1)
        print(att)
        obj.att = tf.keras.utils.deserialize_keras_object(att)
        obj.layernorm1 = tf.keras.utils.deserialize_keras_object(layernorm1)
        obj.layernorm2 = tf.keras.utils.deserialize_keras_object(layernorm2)
        obj.dropout1 = tf.keras.utils.deserialize_keras_object(dropout1)
        obj.dropout2 = tf.keras.utils.deserialize_keras_object(dropout2)
        obj.dense1 = tf.keras.utils.deserialize_keras_object(dense1)
        obj.dense2 = tf.keras.utils.deserialize_keras_object(dense2)

        return obj


# Transformer Decoder Layer
@tf.keras.utils.register_keras_serializable(package="UQModels_layers")
class TransformerDecoder(layers.Layer):
    """Transformer Encoder Layer from https://keras.io/examples/audio/transformer_asr/"""

    def __init__(
        self,
        dim_z,
        dim_horizon,
        num_heads,
        feed_forward_dim,
        dp_rec=0.1,
        flag_mc=False,
        random_state=None,
        **kwargs
    ):

        super().__init__()
        self.dim_z = dim_z
        self.dim_horizon = dim_horizon
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.dp_rec = dp_rec
        self.flag_mc = flag_mc
        self.random_state = random_state
        set_global_determinism(self.random_state)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.self_att = MultiHeadAttention(num_heads=num_heads, key_dim=dim_z)
        self.enc_att = MultiHeadAttention(num_heads=num_heads, key_dim=dim_z)
        self.self_dropout = Dropout(dp_rec, seed=random_state)
        self.enc_dropout = Dropout(dp_rec, seed=add_random_state(random_state, 1))
        self.ffn_dropout = Dropout(dp_rec, seed=add_random_state(random_state, 2))
        self.dense1 = Dense(feed_forward_dim, activation="relu")
        self.dense2 = Dense(dim_z)

    def causal_attention_mask(self, batch_size, n_dest, n_src, dim_horizon, dtype):
        """Masks the upper half of the dot product matrix in self attention.

        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        len_past = n_dest - dim_horizon
        i = tf.concat(
            [
                tf.zeros(len_past, dtype=tf.int32) + len_past - 1,
                tf.range(dim_horizon) + len_past,
            ],
            0,
        )[:, None]
        j = tf.range(n_src)
        m = (i) >= (j - n_src + n_dest)
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    def call(self, enc_out, target, training=None):
        """_summary_

        Args:
            enc_out (_type_): _description_
            target (_type_): _description_

        Returns:
            _type_: _description_
        """
        if training is None:
            training = False

        input_shape = tf.shape(target)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(
            batch_size, seq_len, seq_len, self.dim_horizon, tf.bool
        )
        target_att = self.self_att(target, target, attention_mask=causal_mask)
        target_norm = self.layernorm1(
            target + self.self_dropout(target_att, training=training | self.flag_mc)
        )
        enc_out = self.enc_att(target_norm, enc_out)
        enc_out_norm = self.layernorm2(
            self.enc_dropout(enc_out, training=training | self.flag_mc) + target_norm
        )
        ffn_out = self.dense2(self.dense1(enc_out_norm))
        ffn_out_norm = self.layernorm3(
            enc_out_norm + self.ffn_dropout(ffn_out, training=training | self.flag_mc)
        )
        return ffn_out_norm

    def get_config(self):
        config = {
            "dim_z": self.dim_z,
            "dim_horizon": self.dim_horizon,
            "num_heads": self.num_heads,
            "feed_forward_dim": self.feed_forward_dim,
            "dp_rec": self.dp_rec,
            "flag_mc": self.flag_mc,
            "random_state": self.random_state,
            "layernorm1": tf.keras.utils.serialize_keras_object(self.layernorm1),
            "layernorm2": tf.keras.utils.serialize_keras_object(self.layernorm2),
            "layernorm3": tf.keras.utils.serialize_keras_object(self.layernorm3),
            "self_att": tf.keras.utils.serialize_keras_object(self.self_att),
            "enc_att": tf.keras.utils.serialize_keras_object(self.enc_att),
            "self_dropout": tf.keras.utils.serialize_keras_object(self.self_dropout),
            "enc_dropout": tf.keras.utils.serialize_keras_object(self.enc_dropout),
            "ffn_dropout": tf.keras.utils.serialize_keras_object(self.ffn_dropout),
            "dense1": tf.keras.utils.serialize_keras_object(self.dense1),
            "dense2": tf.keras.utils.serialize_keras_object(self.dense2),
        }
        return config

    @classmethod
    def from_config(cls, config):
        layernorm1 = config.pop("layernorm1")
        layernorm2 = config.pop("layernorm2")
        layernorm3 = config.pop("layernorm3")
        self_att = config.pop("self_att")
        enc_att = config.pop("enc_att")
        self_dropout = config.pop("self_dropout")
        enc_dropout = config.pop("enc_dropout")
        ffn_dropout = config.pop("ffn_dropout")
        dense1 = config.pop("dense1")
        dense2 = config.pop("dense2")
        print(config)
        obj = cls(**config)

        obj.layernorm1 = tf.keras.utils.deserialize_keras_object(layernorm1)
        obj.layernorm2 = tf.keras.utils.deserialize_keras_object(layernorm2)
        obj.layernorm3 = tf.keras.utils.deserialize_keras_object(layernorm3)
        obj.self_att = tf.keras.utils.deserialize_keras_object(self_att)
        obj.enc_att = tf.keras.utils.deserialize_keras_object(enc_att)
        obj.self_dropout = tf.keras.utils.deserialize_keras_object(self_dropout)
        obj.enc_dropout = tf.keras.utils.deserialize_keras_object(enc_dropout)
        obj.ffn_dropout = tf.keras.utils.deserialize_keras_object(ffn_dropout)
        obj.dense1 = tf.keras.utils.deserialize_keras_object(dense1)
        obj.dense2 = tf.keras.utils.deserialize_keras_object(dense2)
        return obj


# encoder
def build_transformer(
    size_window=10,
    n_windows=5,
    step=1,
    dim_target=1,
    dim_chan=1,
    dim_horizon=3,
    dim_ctx=20,
    dim_z=100,
    num_heads=2,
    num_feed_forward=128,
    num_layers_enc=3,
    num_layers_dec=2,
    layers_enc=[150],
    layers_dec=[150, 75],
    dp=0.05,
    dp_rec=0.03,
    k_reg=(0.00001, 0.00001),
    list_strides=[2, 1],
    list_filters=None,
    list_kernels=None,
    dim_dyn=None,
    with_positional_embedding=False,
    with_ctx_input=True,
    with_convolution=True,
    type_output=None,
    random_state=None,
    **kwargs
):
    """Builder for Transformer ED with convolutive preprocessing

    Args:
        size_window (int, optional): Size of window for lag values. Defaults to 10.
        n_windows (int, optional): Number of window in past. Defaults to 5.
        step (int, optional): step between windows. Defaults to 1.
        dim_target (int, optional): dimension of TS. Defaults to 1.
        dim_chan (int, optional): Number of channel of TS. Defaults to 1.
        dim_horizon (int, optional): futur_horizon to predict. Defaults to 3.
        dim_ctx (int, optional): Number of ctx_features. Defaults to 20.
        dim_z (int, optional): Size of latent sapce. Defaults to 100.
        num_heads (int, optional): num of heads transformer. Defaults to 2.
        num_feed_forward (int, optional): feed_forward transfomer dimension. Defaults to 128.
        num_layers_enc (int, optional): num of transformer enc block
        (after concatenation of past values embeding + ctx) . Defaults to 3.
        num_layers_dec (int, optional): num of transformer dec block Defaults to 2.
        layers_enc (list, optional):size of MLP preprocessing
        (after concatenation of past values embeding + ctx) Defaults to [150].
        layers_dec (list, optional): size of MLP interpretor. Defaults to 2.
        dp (float, optional): dropout. Defaults to 0.05.
        dp_t (float, optional): transformer dropout. Defaults to 0.1.
        k_reg (tuple, optional): _description_. Defaults to (0.00001, 0.00001).
        dim_dyn (int, None): size of dyn inputs, if None consider dim_dyn have same size than dim target
        with_positional_embedding (bool, optional): _description_. Defaults to False.
        with_ctx_input (bool, optional): Expect ctx features in addition to lag. Defaults to True.
        with_convolution (bool, optional): use convolution rather than
        whole lag values in the windows. Defaults to True.
        type_output (_type_, optional): mode of UQ (see NN_UQ). Defaults to None.
        random_state (bool): handle experimental random using seed.
    Returns:
        transformer : multi-step forecaster with UQ
    """
    if dim_dyn is None:
        dim_dyn = dim_target

    flag_mc = 0
    if type_output in ["BNN", "MC_Dropout"]:
        flag_mc = 1

    set_global_determinism(random_state)

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

    # dim_output_size = Interpretor.output.shape[-1]

    Pos_Embeddor = None
    if with_positional_embedding:
        Pos_Embeddor = PositionalEmbedding(dim_z, max_len=size_window + dim_horizon - 1)

    # Input definition

    list_input = []
    if with_ctx_input:
        CTX_inputs = Input(shape=(n_windows, dim_ctx), name="encoder_inputs")
        list_input.append(CTX_inputs)

    Y_past_in = Input(shape=(size_window, dim_dyn), name="past_inputs")
    list_input.append(Y_past_in)

    Y_past = Y_past_in

    # Preprocessing layers definition
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
            seed=add_random_state(random_state, 100),
        )
    else:
        MWE = Mouving_Windows_Embedding(
            size_window,
            n_windows,
            step=step,
            dim_d=dim_dyn,
            dim_chan=dim_chan,
            seed=add_random_state(random_state, 100),
        )

    FTE = Factice_Time_Extension(dim_horizon)
    layers_enc.append(dim_z)

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
        random_state=add_random_state(random_state, 200),
    )

    # Preprocessing computation
    Data = MWE(Y_past)
    # Concat with cat features
    if with_ctx_input:
        Data = K.concatenate([CTX_inputs, Data], axis=-1)
    # Factice time augmentation (actually useless but can be usefull for extended predict horizon)
    Data = FTE(Data)

    Embedding = TimeDistributed(Embeddor_ctx)(Data)

    # Static Pe that encode window position

    if Pos_Embeddor:
        Pe_Embedding = Pos_Embeddor(Embedding)
        Embedding = Embedding + Pe_Embedding

    # Encoder l'information passé
    enc_out = Embedding[:, :(-dim_horizon), :]
    encoder = []
    for i in range(num_layers_enc):
        encoder.append(
            TransformerEncoder(
                dim_z,
                num_heads,
                feed_forward_dim=50,
                num_feed_forward=num_feed_forward,
                dp_rec=dp_rec,
                flag_mc=flag_mc,
                random_state=add_random_state(random_state, 300 + i),
            )
        )
        enc_out = encoder[-1](enc_out)

    # For learning :
    decoder = []
    dec_out = enc_out
    for i in range(num_layers_dec):
        decoder.append(
            TransformerDecoder(
                dim_z=dim_z,
                dim_horizon=dim_horizon,
                feed_forward_dim=50,
                num_heads=num_heads,
                num_feed_forward=num_feed_forward,
                dp_rec=dp_rec,
                flag_mc=flag_mc,
                random_state=add_random_state(random_state, 400 + i),
            )
        )
        dec_out = decoder[-1](dec_out, Embedding)

    outputs = TimeDistributed(Interpretor)(dec_out[:, -(dim_horizon):])

    model = tf.keras.Model(list_input, outputs, name="model")
    return model


class Transformer_ED_UQ(NN_UQ):
    """Transformer_ED for forecasting with UQ : see build_transformer to check model parameters"""

    def __init__(
        self,
        model_parameters,
        factory_parameters={"factory_lag_lt": 0, "factory_lag_st": 0},
        training_parameters=dict(),
        type_output=None,
        rescale=False,
        n_ech=5,
        train_ratio=0.9,
        name="Lstm_stacked",
        random_state=None,
    ):
        """Initialization

        Args:
            model_parameters (_type_): _description_
            factory_parameters (dict, optional): _description_. Defaults to {'factory_lag_lt': 0, 'factory_lag_st': 0}.
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
            model_initializer=build_transformer,
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
                X, _, _ = super().factory(X, None, mask)
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
            _, y, _ = super().factory(None, y, mask)
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
    dim_chan=1,
    size_window=20,
    n_windows=5,
    dim_horizon=5,
    dim_z=50,
    dp=0.05,
    dp_rec=0.02,
    num_heads=2,
    num_feed_forward=128,
    num_layers_enc=3,
    num_layers_dec=2,
    layers_enc=[75, 150, 75],
    layers_dec=[200, 125, 75],
    list_strides=[2, 1, 1, 1],
    list_filters=[128, 128, 128],
    list_kernels=None,
    with_convolution=True,
    with_ctx_input=True,
    n_ech=3,
    type_output="MC_Dropout",
    random_state=None,
):
    dict_params = {
        "dim_ctx": dim_ctx,
        "size_window": size_window,
        "n_windows": n_windows,
        "dim_horizon": dim_horizon,
        "dim_target": dim_target,
        "dim_chan": dim_chan,
        "step": 1,
        "dim_z": dim_z,
        "dp": dp,
        "dp_rec": dp_rec,
        "dim_dyn": dim_dyn,
        "type_output": type_output,
        "num_heads": num_heads,
        "num_feed_forward": num_feed_forward,
        "num_layers_enc": num_layers_enc,
        "num_layers_dec": num_layers_dec,
        "k_reg": (10e-6, 10e-6),
        "layers_enc": layers_enc,
        "layers_dec": layers_dec,
        "list_strides": list_strides,
        "list_filters": list_filters,
        "list_kernels": list_kernels,
        "with_convolution": with_convolution,
        "with_ctx_input": with_ctx_input,
        "n_ech": n_ech,
        "random_state": random_state,
    }
    return dict_params
