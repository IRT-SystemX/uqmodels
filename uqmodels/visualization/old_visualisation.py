import numpy as np
import matplotlib.pyplot as plt
import uqmodels.postprocessing.UQ_processing as UQ_proc


def plot_prediction_interval(
    y: np.array,
    y_pred_lower: np.array,
    y_pred_upper: np.array,
    X: np.array = None,
    y_pred: np.array = None,
    save_path: str = None,
    sort_X: bool = False,
    **kwargs,
) -> None:
    """Plot prediction intervals whose bounds are given by y_pred_lower and y_pred_upper.
    True values and point estimates are also plotted if given as argument.

    Args:
        y: label true values.
        y_pred_lower: lower bounds of the prediction interval.
        y_pred_upper: upper bounds of the prediction interval.
        X <optionnal>: abscisse vector.
        y_pred <optionnal>: predicted values.
        kwargs: plot parameters.
    """

    # Figure configuration
    if "figsize" in kwargs.keys():
        figsize = kwargs["figsize"]
    else:
        figsize = (15, 6)
    if "loc" not in kwargs.keys():
        loc = kwargs["loc"]
    else:
        loc = "upper left"
    plt.figure(figsize=figsize)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["ytick.labelsize"] = 15
    plt.rcParams["xtick.labelsize"] = 15
    plt.rcParams["axes.labelsize"] = 15
    plt.rcParams["legend.fontsize"] = 16

    if X is None:
        X = np.arange(len(y))
    elif sort_X:
        sorted_idx = np.argsort(X)
        X = X[sorted_idx]
        y = y[sorted_idx]
        y_pred = y_pred[sorted_idx]
        y_pred_lower = y_pred_lower[sorted_idx]
        y_pred_upper = y_pred_upper[sorted_idx]

    if y_pred_upper is None or y_pred_lower is None:
        miscoverage = np.array([False for _ in range(len(y))])
    else:
        miscoverage = (y > y_pred_upper) | (y < y_pred_lower)

    label = "Observation" if y_pred_upper is None else "Observation (inside PI)"
    plt.plot(
        X[~miscoverage],
        y[~miscoverage],
        "darkgreen",
        marker="X",
        markersize=2,
        linewidth=0,
        label=label,
        zorder=20,
    )

    label = "Observation" if y_pred_upper is None else "Observation (outside PI)"
    plt.plot(
        X[miscoverage],
        y[miscoverage],
        σ="red",
        marker="o",
        markersize=2,
        linewidth=0,
        label=label,
        zorder=20,
    )
    if y_pred_upper is not None and y_pred_lower is not None:
        plt.plot(X, y_pred_upper, "--", color="blue", linewidth=1, alpha=0.7)
        plt.plot(X, y_pred_lower, "--", color="blue", linewidth=1, alpha=0.7)
        plt.fill_between(
            x=X,
            y1=y_pred_upper,
            y2=y_pred_lower,
            alpha=0.2,
            fc="b",
            ec="None",
            label="Prediction Interval",
        )

    if y_pred is not None:
        plt.plot(X, y_pred, color="k", label="Prediction")

    plt.xlabel("X")
    plt.ylabel("Y")

    if "loc" not in kwargs.keys():
        loc = "upper left"
    else:
        loc = kwargs["loc"]

    plt.legend(loc=loc)
    if save_path:
        plt.savefig(f"{save_path}", format="pdf")
    else:
        plt.show()


def plot_sorted_pi(
    y: np.array,
    y_pred_lower: np.array,
    y_pred_upper: np.array,
    X: np.array = None,
    y_pred: np.array = None,
    **kwargs,
) -> None:
    """Plot prediction intervals in an ordered fashion (lowest to largest width),
    showing the upper and lower bounds for each prediction.
    Args:
        y: label true values.
        y_pred_lower: lower bounds of the prediction interval.
        y_pred_upper: upper bounds of the prediction interval.
        X <optionnal>: abscisse vector.
        y_pred <optionnal>: predicted values.
        kwargs: plot parameters.
    """

    if y_pred is None:
        y_pred = (y_pred_upper + y_pred_lower) / 2

    width = np.abs(y_pred_upper - y_pred_lower)
    sorted_order = np.argsort(width)

    # Figure configuration
    if "figsize" in kwargs.keys():
        figsize = kwargs["figsize"]
    else:
        figsize = (15, 6)
    if "loc" not in kwargs.keys():
        kwargs["loc"]
    else:
        pass
    plt.figure(figsize=figsize)

    if X is None:
        X = np.arange(len(y_pred_lower))

    # True values
    plt.plot(
        X,
        y_pred[sorted_order] - y_pred[sorted_order],
        color="black",
        markersize=2,
        zorder=20,
        label="Prediction",
    )

    misscoverage = (y > y_pred_upper) | (y < y_pred_lower)
    misscoverage = misscoverage[sorted_order]

    # True values
    plt.plot(
        X[~misscoverage],
        y[sorted_order][~misscoverage] - y_pred[sorted_order][~misscoverage],
        color="darkgreen",
        marker="o",
        markersize=2,
        linewidth=0,
        zorder=20,
        label="Observation (inside PI)",
    )

    plt.plot(
        X[misscoverage],
        y[sorted_order][misscoverage] - y_pred[sorted_order][misscoverage],
        color="red",
        marker="o",
        markersize=2,
        linewidth=0,
        zorder=20,
        label="Observation (outside PI)",
    )

    # PI Lower bound
    plt.plot(
        X,
        y_pred_lower[sorted_order] - y_pred[sorted_order],
        "--",
        label="Prediction Interval Bounds",
        color="blue",
        linewidth=1,
        alpha=0.7,
    )

    # PI upper bound
    plt.plot(
        X,
        y_pred_upper[sorted_order] - y_pred[sorted_order],
        "--",
        color="blue",
        linewidth=1,
        alpha=0.7,
    )

    plt.legend()

    plt.show()


def visu_latent_space(grid_dim, embedding, f_obs, context_grid, context_grid_name=None):
    fig = plt.figure(figsize=(15, 7))
    for i in range(grid_dim[0]):
        for j in range(grid_dim[1]):
            ax = fig.add_subplot(
                grid_dim[0], grid_dim[1], i * grid_dim[1] + j + 1, projection="3d"
            )
            if context_grid_name is not None:
                plt.title(context_grid_name[i][j])
            ax.scatter(
                embedding[f_obs, 0],
                embedding[f_obs, 1],
                embedding[f_obs, 2],
                c=context_grid[i][j][f_obs],
                cmap=plt.get_cmap("jet"),
                s=1,
            )


def show_dUQ_refinement(
    UQ,
    y=None,
    d=0,
    f_obs=None,
    max_cut_A=0.99,
    q_Eratio=2,
    E_cut_in_var_nominal=False,
    A_res_in_var_atypic=False,
):
    if isinstance(UQ, tuple):
        UQ = np.array(UQ)

    if f_obs is None:
        f_obs = np.arange(UQ.shape[1])

    var_A, var_E = UQ
    extremum_var_TOT, ndUQ_ratio = UQ_proc.get_extremum_var_TOT_and_ndUQ_ratio(
        UQ,
        min_cut=0,
        max_cut=max_cut_A,
        var_min=0,
        var_max=None,
        factor=1,
        q_var=1,
        q_Eratio=q_Eratio,
        mode_multidim=True,
        E_cut_in_var_nominal=E_cut_in_var_nominal,
        A_res_in_var_atypic=A_res_in_var_atypic,
    )

    var_A_cut, var_E_res = UQ_proc.split_var_dUQ(
        UQ,
        q_var=1,
        q_var_e=1,
        ndUQ_ratio=ndUQ_ratio,
        E_cut_in_var_nominal=E_cut_in_var_nominal,
        A_res_in_var_atypic=A_res_in_var_atypic,
        extremum_var_TOT=extremum_var_TOT,
    )

    var_A_res = var_A - var_A_cut
    var_E_cut = var_E - var_E_res

    val = 0
    if y is not None:
        val = 1

    fig, ax = plt.subplots(3 + val, 1, sharex=True, figsize=(20, 5))
    if val == 1:
        ax[0].plot(y[f_obs, d: d + 1], label="true_val")
    ax[0 + val].plot(var_A[f_obs, d: d + 1], label="row_var_A")
    ax[0 + val].plot(var_A_cut[f_obs, d: d + 1], label="refined_var_A")
    ax[0 + val].legend()
    ax[1 + val].plot(var_E[f_obs, d: d + 1], label="row_var_E")
    ax[1 + val].plot(var_E_res[f_obs, d: d + 1], label="refined_var_E")
    ax[1 + val].legend()
    ratio = var_E[f_obs, d: d + 1] / var_A[f_obs, d: d + 1]
    ax[2 + val].plot(ratio / ratio.std(), label="row_ratio")
    refined_ratio = (var_A_res[f_obs, d: d + 1] + var_E_res[f_obs, d: d + 1]) / (
        var_A_cut[f_obs, d: d + 1] + var_E_cut[f_obs, d: d + 1]
    )
    ax[2 + val].plot(refined_ratio / refined_ratio.std(), label="refined_ratio")
    ax[2 + val].legend()
    print("yaya")
    return (fig, ax)
