import numpy as np
from uqmodels.preprocessing.structure import get_step_mask


def test_get_step_mask():
    step = np.arange(10)
    mask = get_step_mask(step, 3, 6, out_of_mask=True)
    assert mask.dtype == bool
    assert mask.sum() >= (6 - 3 + 1)
