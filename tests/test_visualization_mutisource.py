import numpy as np
from uqmodels.visualization.visualization_mutisource import apply_cmap, compute_dev_score


def test_apply_cmap():
    cmap = __import__("matplotlib").cm.get_cmap("viridis")
    val = np.array([0.0, 0.5, 1.0])
    colors = apply_cmap(val, 0.0, 1.0, cmap)
    assert len(colors) == 3


def test_compute_dev_score():
    val = np.array([1.0, 2.0])
    y = np.array([1.5, 1.5])
    vmin = np.array([0.0, 0.0])
    vmax = np.array([3.0, 3.0])
    r = compute_dev_score(val, y, vmin, vmax)
    assert r.shape == (2,)
