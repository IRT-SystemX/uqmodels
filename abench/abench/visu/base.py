import numpy as np
import abench.store.api as api
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple, Union, Optional
def meta_plot(y, output, context, metadata, **kwarg):
    """Abstract visualisation function

    Args:
        y (array): Targets
        output (array): Model results
        train (boolean array): Identify training sample
        test (boolean array): Identify testing sample
        size (int,int): size of plot
    """
    pass

def plot_curve(
    storing, list_component_name, trainset_name,set_name, meta_plot, plot_param, size=(18, 2), names=None):
    """[summary]

    Args:
        dict_res ([type]): [description]
        list_name ([type]): [description]
        meta_plot ([type]): [description]
        cv_name ([type]): [description]
        plot_param ([type]): [description]
        size (tuple, optional): [description]. Defaults to (18, 2).
    """

    ABloader = api.get_ABloader(storing,set_name)
    for (_,y),context, metadata in ABloader:
        pass

    for n,component_name in enumerate(list_component_name):
        output = api.get_output(storing,component_name,trainset_name,set_name)
        meta_plot(
            y,
            output,
            context,
            metadata,
            size=size,
            name=component_name,
            show_legend=(n == 0),
            **plot_param
        )
    return


import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Iterable, Tuple, Any

def plot_grid(
    plot_fn: Callable[..., None],
    y: np.ndarray,
    output: np.ndarray,
    context: Optional[np.ndarray] = None,
    metadata: Optional[np.ndarray] = None,
    n: int = 3,
    idx: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    figsize_per_cell: float = 4.0,
    title_fn: Optional[Callable[..., str]] = None,
    **sample_kwargs: Any,
) -> Tuple[plt.Figure, np.ndarray, np.ndarray]:
    """
    Display an n x n grid of randomly selected samples (without replacement),
    calling a user-defined visualization function for each sample.

    Parameters
    ----------
    plot_fn : callable
        Visualization function applied to each sample.
        Expected signature:
            plot_fn(ax, y, output, context=None, metadata=None, **kwargs)
    y : np.ndarray
        Ground-truth data (shape: [N, ...]).
    output : np.ndarray
        Model outputs or reconstructions (shape: [N, ...]).
    context : np.ndarray | None
        Optional conditioning information or model inputs (shape: [N, ...]).
    metadata : np.ndarray | None
        Optional metadata for each sample (IDs, labels, etc.).
    n : int
        Number of rows and columns in the grid (n x n subplots).
    idx : np.ndarray | None
        Explicit list of indices to display. If None, samples are drawn randomly (without replacement).
    random_state : int | None
        Random seed for reproducible sampling.
    figsize_per_cell : float
        Size (in inches) of each subplot cell.
        The total figure size will be approximately (n * figsize_per_cell, n * figsize_per_cell).
    title_fn : callable | None
        Optional title generator function:
            title_fn(i, y_i, output_i, context_i, metadata_i) -> str
    **sample_kwargs :
        Additional keyword arguments passed directly to `plot_fn`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created matplotlib figure.
    axes : np.ndarray
        2D array of Axes with shape (n, n).
    idx_used : np.ndarray
        Array of indices actually used for plotting.
    """
    y = np.asarray(y)
    output = np.asarray(output)
    N = len(y)
    assert len(output) == N, "y and output must have the same length."

    if context is not None:
        assert len(context) == N, "context must have the same length as y."
    if metadata is not None:
        assert len(metadata) == N, "metadata must have the same length as y."

    nplots = n * n

    # Select indices (without replacement)
    if idx is None:
        rng = np.random.default_rng(random_state)
        k = min(nplots, N)
        idx_used = rng.choice(N, size=k, replace=False)
    else:
        idx_used = np.asarray(idx)
        assert idx_used.ndim == 1 and len(idx_used) <= nplots, "idx must be 1D and <= n*n."
        assert np.all((0 <= idx_used) & (idx_used < N)), "idx contains out-of-range values."

    # Create figure and axes grid
    fig, axes = plt.subplots(n, n, figsize=(figsize_per_cell * n, figsize_per_cell * n))
    if n == 1:
        axes = np.array([axes])
    axes_flat = axes.ravel()

    # Main plotting loop
    for ax, i in zip(axes_flat, idx_used):
        y_i = y[i]
        out_i = output[i]
        ctx_i = context[i] if context is not None else None

        # Call user-defined sample visualization
        plot_fn(ax, y_i, out_i, ctx_i, metadata, **sample_kwargs)

        # Optional title generator
        if title_fn is not None:
            ax.set_title(title_fn(i, y_i, out_i, ctx_i, metadata))

    # Hide empty axes if grid is not completely filled
    for ax in axes_flat[len(idx_used):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    return fig, axes, idx_used

