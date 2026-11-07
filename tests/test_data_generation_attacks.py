import numpy as np
from uqmodels.data_generation.Gen_basic_times_series import attack_mean


def test_attack_mean():
    y = np.zeros((5, 1))
    loc = np.array([1, 2, 3])
    out = attack_mean(y, loc, dim=(0,), f=1.0)
    assert out.shape == y.shape
    assert not np.allclose(out, y)
