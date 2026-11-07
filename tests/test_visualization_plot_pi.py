import numpy as np
from uqmodels.visualization.visualization import plot_pi


def test_plot_pi_runs():
    y = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    lower = y_pred - 0.5
    upper = y_pred + 0.5
    fig = plot_pi(y, y_pred, lower, upper, show_plot=False)
    assert fig is None or hasattr(fig, "savefig")
