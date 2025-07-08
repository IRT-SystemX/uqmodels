import numpy as np
import pytest
import tensorflow as tf

RANDOM_SEED = 0


@pytest.fixture
def random_seed():
    return RANDOM_SEED


np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
