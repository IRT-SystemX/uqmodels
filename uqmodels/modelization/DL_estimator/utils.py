import os
import numpy as np
import tensorflow as tf
import random


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
