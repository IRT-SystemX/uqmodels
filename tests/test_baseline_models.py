import numpy as np
import pytest


def test_cnn_mlp_builds():
    from uqmodels.modelization.DL_estimator.baseline_models import cnn_mlp

    out = cnn_mlp(dim_dyn=3, dim_target=2, size_window=5, n_windows=2, step=1)
    assert "model" in out and "encoder" in out
    model = out["model"]
    assert model is not None
    x = np.zeros((2, 5 + 2 * 1 - 1, 3), dtype="float32")
    y = model.predict(x, verbose=0)
    assert y.shape[0] == 2
