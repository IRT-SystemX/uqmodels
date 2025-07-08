import os
import random

import numpy as np
import tensorflow as tf

from uqmodels.utils import apply_mask


def identity(*args):
    return args


def sum_part_prod(array):
    """compute sum_part_prod
    array = [k1,...,kn]
    return (k1+k1k2+k1k2k3+..+k1..Kn)
    """
    s = 0
    for n in range(len(array)):
        s += np.prod(array[:n])
    return s


def size_post_conv(w, l_k, l_st):
    """provide size post conv (with padding=valid)
    w : size of window
    l_k : list kernel
    l_s : list_stride
    """
    curent_s = w
    for k, st in zip(l_k, l_st):
        curent_s = np.ceil((curent_s - k + 1) / st)
    return curent_s


def find_conv_kernel(window_initial, size_final, list_strides):
    """Return size of kernel according to :
    window_initial : size of window
    size_final : size final
    list_strides : list of strides

    return(list_kernel,list_strides)
    """

    val = sum_part_prod(list_strides[:-1])
    float_kernel = (size_final * np.prod(list_strides[:-1]) - window_initial) / val - 1
    kernel = int(max(np.floor(-float_kernel) - 1, 1))
    before_last_size = size_post_conv(
        window_initial, [kernel for i in list_strides[:-1]], list_strides[:-1]
    )
    last_kernel = (before_last_size - size_final + 1) / list_strides[-1]

    if last_kernel < 1:
        raise (ValueError("Incompatible list_strides values"))

    list_kernel = [kernel for i in list_strides]
    list_kernel[-1] = int(last_kernel)
    return (list_kernel, list_strides)


class Generator:
    def __init__(self, X, y, batch_min=64, shuffle=True, random_state=None):
        self.X = X
        self.y = y
        self.factory = identity
        self.shuffle = shuffle
        self.batch_min = batch_min
        self.random_state = random_state

    def load(self, idx):
        idx = idx * self.batch_min
        seuil_min = idx * self.batch_min
        seuil_max = (idx + 1) * self.batch_min
        return (self.X[seuil_min:seuil_max], self.y[seuil_min:seuil_max])

    def __len__(self):
        return self.X.shape[0] // self.batch_min

    def __getitem__(self, idx):
        x, y = self.load(idx)
        Inputs, Ouputs, _ = self.factory(x, y, fit_rescale=False)
        return Inputs, Ouputs

    def __call__(self):
        step = np.arange(0, self.__len__())
        if self.shuffle:
            random.seed(self.random_state)
            random.Random().shuffle(step)
        for i in step:
            yield self.__getitem__(i)
            if i > self.__len__() - 1:
                self.on_epoch_end()

    def on_epoch_end(self):
        pass


def set_seeds(seed=None):
    if seed is not None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)


def set_global_determinism(seed=None):
    if seed is not None:
        set_seeds(seed=seed)

        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

        # tf.config.threading.set_inter_op_parallelism_threads(1)
        # tf.config.threading.set_intra_op_parallelism_threads(1)


class Folder_Generator(tf.keras.utils.Sequence):
    def __init__(
        self, X, y, metamodel, batch=64, shuffle=True, train=True, random_state=None
    ):
        self.X = X
        self.y = y
        self.random_state = random_state
        if X is not None:
            self.len_ = X[0].shape[0]
        elif y is not None:
            self.len_ = y.shape[0]

        self.train = train
        self.seed = 0
        self.shuffle = shuffle
        self.batch = batch

        # self.scaler = metamodel.scaler
        self.factory = metamodel.factory
        self._format = metamodel._format
        self.rescale = metamodel.rescale

        self.causality_remove = None
        self.model_parameters = metamodel.model_parameters
        self.past_horizon = metamodel.model_parameters["size_window"]
        self.futur_horizon = (
            metamodel.model_parameters["dim_horizon"]
            * metamodel.model_parameters["step"]
        )
        self.size_seq = self.past_horizon + self.futur_horizon + self.batch
        self.size_window_futur = 1

        self.n_batch = int(np.ceil(self.len_ / self.batch))
        self.indices = np.arange(self.n_batch)

    def load(self, idx):
        """load seq of data locate at [idx*self.batch-past_horizon, idx*self.batch+self.futur_horizon]"""
        idx = idx * self.batch

        idx_min = max(0, idx - self.past_horizon)
        idx_max = max(self.size_seq + idx_min, idx + self.futur_horizon)
        # Hold case of last batch : load also end of previous batch to complete last batch
        if idx > 0:
            idx_min = max(idx_min - max(0, idx_max - self.len_), 0)
        y_batch = None

        if self.y is not None:
            y_batch = self.y[idx_min:idx_max]

        if self.X is None:
            return ([None, None], y_batch)

        else:
            return ([self.X[0][idx_min:idx_max], self.X[1][idx_min:idx_max]], y_batch)

    def __len__(self):
        return self.n_batch

    def __getitem__(self, idx):
        """Get batch by loading seq, apply factory on it, and select the relevant part

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indices)
            idx = self.indices[idx]

        x, y = self.load(idx)

        if self.train:
            pass
        Inputs, Ouputs, _ = self.factory(x, y, fit_rescale=False)
        selection = np.zeros(len(Inputs[0])) == 1

        idx_min = max(0, idx * self.batch - self.past_horizon)
        idx_max = max(
            self.size_seq + idx_min, idx * self.batch + self.batch + self.futur_horizon
        )

        if self.train:
            padding_test = 0
            selection[self.past_horizon : -self.futur_horizon] = True
            padding_test + self.past_horizon - self.futur_horizon
        else:  # hold case of predict for last batch

            idx_min = max(0, idx * self.batch - self.past_horizon)
            idx_max = max(
                self.size_seq + idx_min,
                idx * self.batch + self.batch + self.futur_horizon,
            )

            if idx == 0:
                if self.batch >= self.len_:
                    selection[0:] = True
                else:
                    selection[: -self.past_horizon - self.futur_horizon] = True

            else:
                # hold case of last batch
                padding_test = max(self.futur_horizon, idx_max - self.len_)

                selection[padding_test + self.past_horizon :] = True

        Inputs = apply_mask(Inputs, selection)
        Ouputs = apply_mask(Ouputs, selection)
        return Inputs, Ouputs

    # shuffles the dataset at the end of each epoch
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
