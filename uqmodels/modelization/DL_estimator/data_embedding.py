import math

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.layers import LeakyReLU
from tensorflow.keras import layers

from uqmodels.modelization.DL_estimator.utils import (
    find_conv_kernel,
    set_global_determinism,
)

from ...utils import add_random_state

# tf.keras.utils.get_custom_objects().clear()


# Restructuration de code : Functionnalité de layers-preprocessing pour NN (Transformer ?)


class Mouving_Windows_Embedding(layers.Layer):
    def __init__(
        self, size_window, n_windows=1, step=1, dim_d=1, dim_chan=1, seed=None, **kwargs
    ):
        """_summary_

        Args:
            sub_seq_size (_type_): _description_
            size_window (_type_): _description_
            dim_out (int, optional): _description_. Defaults to 1.
            padding (int, optional): _description_. Defaults to 1.
            seed (bool): handle experimental random using seed.
        """
        super(Mouving_Windows_Embedding, self).__init__()
        self.size_window = size_window
        self.n_windows = n_windows
        self.step = step
        self.dim_d = dim_d
        self.dim_chan = dim_chan
        self.seed = seed
        self.last_shape = self.size_window * self.dim_d * self.dim_chan

    def call(self, inputs, mode="encoder"):
        """_summary_

        Args:
            inputs (_type_): _description_
            mode (str, optional): _description_. Defaults to "encoder".

        Returns:
            _type_: _description_
        """
        output = inputs
        if len(output.shape) == 3:
            output = output[:, :, :, None]

        slide_tensor = []
        for i in range(self.n_windows):
            slide_tensor.append(
                inputs[:, (i * self.step) : (i * self.step) + self.size_window]
            )
        MWE_raw = K.stack(slide_tensor, axis=1)
        MWE = K.reshape(
            MWE_raw, (-1, self.n_windows, self.size_window * self.dim_d * self.dim_chan)
        )

        return MWE


class Mouving_Window_Embedding(layers.Layer):
    def __init__(self, sub_seq_size, size_window, dim_out=1, padding=1, seed=None):
        super(Mouving_Window_Embedding, self).__init__()
        self.sub_seq_size = sub_seq_size
        self.dim_out = dim_out
        self.size_window = size_window
        self.padding = padding
        self.seed = seed

    def call(self, inputs, mode="encoder"):
        """_summary_

        Args:
            inputs (_type_): _description_
            mode (str, optional): _description_. Defaults to "encoder".

        Returns:
            _type_: _description_
        """
        if mode == "encoder":
            size_output = self.size_window
        elif mode == "decoder":
            size_output = self.size_window + self.dim_out - 1
        else:
            print("Mode error")
        if self.sub_seq_size == 1:
            return inputs
        else:
            slide_tensor = []
            for i in range(size_output):
                slide_tensor.append(
                    inputs[
                        :, (i * self.padding) : (i * self.padding) + self.sub_seq_size
                    ]
                )
            MWE_raw = K.stack(slide_tensor, axis=1)
            MWE = K.reshape(
                MWE_raw, (-1, size_output, self.sub_seq_size * inputs.shape[-1])
            )
        return MWE


@tf.keras.utils.register_keras_serializable(package="UQModels_data_embedding")
class Conv2D(tf.keras.layers.Conv2D):
    pass


@tf.keras.utils.register_keras_serializable(package="UQModels_data_embedding")
class Dropout(tf.keras.layers.Dropout):
    pass


@tf.keras.utils.register_keras_serializable(package="UQModels_data_embedding")
class Conv1D(tf.keras.layers.Conv1D):
    pass


@tf.keras.utils.register_keras_serializable(package="UQModels_data_embedding")
class Mouving_conv_Embedding(layers.Layer):
    def __init__(
        self,
        size_window,
        n_windows,
        step=1,
        dim_d=1,
        dim_chan=1,
        use_conv2D=True,
        list_filters=None,
        list_strides=[2, 1, 1],
        list_kernels=None,
        dp=0.01,
        flag_mc=False,
        seed=None,
        **kwargs
    ):
        """_summary_

        Args:
            sub_seq_size (_type_): _description_
            size_window (_type_): _description_
            dim_out (int, optional): _description_. Defaults to 1.
            padding (int, optional): _description_. Defaults to 1.
            seed (bool): handle experimental random using seed.
        """

        self.size_window = size_window
        self.n_windows = n_windows
        self.step = step
        self.dim_d = dim_d
        self.dim_chan = dim_chan
        self.use_conv2D = use_conv2D
        self.flag_mc = flag_mc
        self.seed = seed
        self.mutliscale = False
        set_global_determinism(self.seed)

        if list_filters is None:
            list_filters = [64 for i in list_strides]

        if list_kernels is None:
            list_kernels, list_strides = find_conv_kernel(
                self.size_window, n_windows, list_strides
            )
            list_filters.append(list_filters[-1])

        super().__init__(**kwargs)

        self.list_strides = list_strides
        self.list_kernels = list_kernels
        self.list_filters = list_filters

        self.layers = []
        for n, (filters, kernel, strides) in enumerate(
            zip(list_filters, list_kernels, list_strides)
        ):
            if use_conv2D:
                if n == 0:
                    kernel = (kernel, dim_d)
                else:
                    kernel = (kernel, 1)
                self.layers.append(
                    Conv2D(
                        filters,
                        kernel,
                        strides=strides,
                        padding="valid",
                        activation="relu",
                    )
                )
                if dp > 0:
                    self.layers.append(Dropout(dp, seed=add_random_state(seed, n)))

            else:
                self.layers.append(
                    Conv1D(
                        filters * dim_chan * dim_d,
                        kernel,
                        strides=strides,
                        padding="valid",
                        groups=dim_chan * dim_d,
                        activation="relu",
                    )
                )

            # self.layers.append(tf.keras.layers.BatchNormalization())
            # self.layers.append(tf.keras.layers.Dropout(dp))

            if use_conv2D:
                self.last_shape = list_filters[-1]
            else:
                self.last_shape = list_filters[-1] * self.dim_d * self.dim_chan

    def call(self, inputs):
        """_summary_

        Args:
            inputs (_type_): _description_
            mode (str, optional): _description_. Defaults to "encoder".

        Returns:
            _type_: _description_
        """
        output = inputs
        if len(output.shape) == 3:
            output = output[:, :, :, None]

        if not (self.use_conv2D):
            output = K.reshape(
                output, (-1, 1, self.size_window, self.dim_d * self.dim_chan)
            )

        # if (self.mutliscale):
        #    list_output = []

        for n, layer in enumerate(self.layers):
            if n == 2:  # dropout layers & end of block
                output = layer(output, training=self.flag_mc)
                # if (self.mutliscale):
                #    TO do : find how affect multiscale window to the good final step
                #    list_output.append(output)

            else:
                output = layer(output)
        if self.use_conv2D:
            output = output[:, :, 0, :]
        else:
            output = output[:, 0, :, :]

        return output

    def get_config(self):
        dict_config = {}
        dict_config["size_window"] = self.size_window
        dict_config["n_windows"] = self.n_windows
        dict_config["step"] = self.step
        dict_config["dim_d"] = self.dim_d
        dict_config["dim_chan"] = self.dim_chan
        dict_config["list_strides"] = self.list_strides
        dict_config["list_kernels"] = self.list_kernels
        dict_config["list_filters"] = self.list_filters
        dict_config["flag_mc"] = self.flag_mc
        dict_config["use_conv2D"] = self.use_conv2D
        dict_config["seed"] = self.seed

        dict_config["layers"] = []
        for layer in self.layers:
            dict_config["layers"].append(tf.keras.utils.serialize_keras_object(layer))
        return dict_config

    @classmethod
    def from_config(cls, config):
        layers_config = config.pop("layers")
        layers = []
        for layer_config in layers_config:
            layer = tf.keras.utils.deserialize_keras_object(layer_config)
            layers.append(layer)

        obj = cls(**config)
        obj.layers = layers
        return obj


class Data_embedding_TS(layers.Layer):
    def __init__(
        self,
        size_window,
        n_windows,
        step=1,
        dim_d=1,
        dim_chan=1,
        dim_out=1,
        flag_mc=False,
        seed=None,
    ):
        super(Data_embedding_TS, self).__init__()
        self.n_windows = n_windows
        self.size_window = size_window
        self.dim_d = dim_d
        self.seed = seed
        set_global_determinism(self.seed)

        if True:

            self.MWE = Mouving_Windows_Embedding(
                size_window,
                n_windows,
                step=step,
                dim_d=dim_d,
                seed=seed,
            )
        else:

            self.MWE = Mouving_conv_Embedding(
                size_window,
                n_windows,
                step=step,
                dim_d=dim_d,
                dim_chan=dim_chan,
                conv2D=True,
                list_filters=None,
                list_strides=[2, 1],
                list_kernels=None,
                dp=0.05,
                flag_mc=flag_mc,
                seed=seed,
            )

        self.FTE = Factice_Time_Extension(dim_out)

    def call(self, Z_enc, Y_past, extension):
        """_summary_

        Args:
            Z_enc (_type_): _description_
            Y_past (_type_): _description_
            Y_futur (_type_): _description_
            mode (str, optional): _description_. Defaults to "".

        Returns:
            _type_: _description_
        """

        MWE_past = self.MWE(Y_past)
        Z_enc = K.concatenate([Z_enc, MWE_past], axis=-1)
        Z_enc = self.FTE(Z_enc)
        return Z_enc


@tf.keras.utils.register_keras_serializable(package="UQModels_data_embedding")
class Factice_Time_Extension(layers.Layer):
    def __init__(self, dim_out, **kwargs):
        super().__init__(**kwargs)
        self.dim_out = dim_out

    def call(self, inputs, **kwargs):
        """_summary_

        Args:
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        last_input = inputs[:, -1, :]
        last_input_duplicated = K.repeat_elements(
            last_input[:, None, :], self.dim_out, 1
        )
        inputs_augmented = K.concatenate([inputs, last_input_duplicated], axis=1)
        return inputs_augmented

    def get_config(self):
        dict_config = {}
        dict_config["dim_out"] = self.dim_out
        return dict_config


# Define the way for positional encoding


class PositionalEmbedding(layers.Layer):
    def __init__(self, d_model, max_len=40, seed=None):
        super(PositionalEmbedding, self).__init__()
        self.seed = seed
        # Compute the positional encodings once in log space.
        pe = np.zeros((max_len, d_model), dtype=np.float32)

        position = np.expand_dims(np.arange(0, max_len, dtype=np.float32), 1)
        div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = tf.expand_dims(tf.convert_to_tensor(pe), 0)

    def call(self, inputs, **kwargs):
        """_summary_

        Args:
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.pe


class ValuesEmbedding(layers.Layer):
    def __init__(self, d_model, seed=None):
        super(ValuesEmbedding, self).__init__()
        self.seed = seed
        self.tokenConv = tf.keras.layers.Conv1D(
            filters=d_model, kernel_size=3, padding="causal", activation="linear"
        )
        self.activation = LeakyReLU()

    def call(self, inputs, **kwargs):
        """_summary_

        Args:
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.tokenConv(inputs[:, :, 1:])
        x = self.activation(x)
        return x


class FixedEmbedding(layers.Layer):
    def __init__(self, c_in, d_model, seed=None):
        super(FixedEmbedding, self).__init__()
        self.seed = seed
        w = np.zeros((c_in, d_model), dtype=np.float32)

        position = np.expand_dims(np.arange(0, c_in, dtype=np.float32), 1)
        div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model)
        )

        w[:, 0::2] = np.sin(position * div_term)
        w[:, 1::2] = np.cos(position * div_term)

        w = tf.convert_to_tensor(w)
        tf.stop_gradient(w)
        w = tf.keras.initializers.Constant(w)
        self.emb = tf.keras.layers.Embedding(c_in, d_model, embeddings_initializer=w)

    def call(self, inputs, **kargs):
        """_summary_

        Args:
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        embedding = self.emb(inputs)
        return embedding


class TemporalEmbedding(layers.Layer):
    def __init__(self, d_model, seq_len, seed=None):
        super(TemporalEmbedding, self).__init__()
        self.time_embed = FixedEmbedding(seq_len, d_model)
        self.seed = seed
        # self.minute_embed = FixedEmbedding(60, d_model)
        # self.hour_embed = FixedEmbedding(24, d_model)
        # self.weekday_embed = FixedEmbedding(7, d_model)
        # self.day_embed = FixedEmbedding(32, d_model)
        # self.month_embed = FixedEmbedding(13, d_model)

    def call(self, inputs, **kargs):
        """_summary_

        Args:
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        # x = x.long()
        return self.time_embed(inputs[:, :, 0])


class DataEmbedding_ITS(layers.Layer):
    def __init__(self, d_model, dropout=0.1, seq_len=96, seed=None):
        super(DataEmbedding_ITS, self).__init__()
        self.seq_len = seq_len
        self.value_embedding = ValuesEmbedding(d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, seq_len=seq_len)
        self.ctx_embedding = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout, seed=seed)
        self.seed = None

    def call(self, inputs, x_mark=None, **kwargs):
        """_summary_

        Args:
            inputs (_type_): _description_
            x_mark (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        x = (
            self.value_embedding(inputs)
            + self.position_embedding(inputs)
            + self.temporal_embedding(inputs)
        )

        return self.dropout(x)
