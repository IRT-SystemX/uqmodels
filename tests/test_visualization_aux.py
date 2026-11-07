import numpy as np
from uqmodels.visualization.visualization import aux_get_var_color_sets


def test_aux_get_var_color_sets():
    out = aux_get_var_color_sets()
    assert out is not None
