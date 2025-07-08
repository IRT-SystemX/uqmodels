"""
Visualization module.
"""

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.colors import LinearSegmentedColormap

import uqmodels.postprocessing.UQ_processing as UQ_proc
from uqmodels.utils import compute_born, propagate


def provide_cmap(mode="bluetored"):
    """Generate a bluetored or a cyantopurple cutsom cmap

    Args:
        mode (str, optional):Values: bluetored' or 'cyantopurple '

    return:
       Colormap matplotlib
    """
    if mode == "bluetored":
        bluetored = [
            [0.0, (0, 0, 90)],
            [0.05, (5, 5, 120)],
            [0.1, (20, 20, 150)],
            [0.15, (20, 20, 190)],
            [0.2, (40, 40, 220)],
            [0.25, (70, 70, 255)],
            [0.33, (100, 100, 255)],
            [0.36, (180, 180, 255)],
            [0.4, (218, 218, 255)],
            [0.45, (245, 245, 255)],
            [0.5, (255, 253, 253)],
            [0.55, (255, 245, 245)],
            [0.6, (255, 218, 218)],
            [0.63, (255, 200, 200)],
            [0.66, (255, 160, 160)],
            [0.7, (255, 110, 110)],
            [0.75, (255, 70, 70)],
            [0.8, (230, 40, 40)],
            [0.85, (200, 20, 20)],
            [0.9, (180, 10, 10)],
            [0.95, (150, 5, 5)],
            [1.0, (130, 0, 0)],
        ]
        bluetored_cmap = LinearSegmentedColormap.from_list(
            "bluetored", [np.array(i[1]) / 255 for i in bluetored], N=255
        )
        return bluetored_cmap
    elif mode == "cyantopurple":
        cyantopurple = [
            [0.0, (25, 255, 255)],
            [0.05, (20, 250, 250)],
            [0.1, (20, 230, 230)],
            [0.15, (20, 220, 220)],
            [0.2, (15, 200, 200)],
            [0.25, (10, 170, 170)],
            [0.3, (10, 140, 140)],
            [0.36, (5, 80, 80)],
            [0.4, (5, 50, 50)],
            [0.45, (0, 30, 30)],
            [0.5, (0, 0, 0)],
            [0.55, (30, 0, 30)],
            [0.6, (59, 0, 50)],
            [0.64, (80, 0, 80)],
            [0.7, (140, 0, 140)],
            [0.75, (170, 0, 170)],
            [0.8, (200, 40, 200)],
            [0.85, (220, 20, 220)],
            [0.9, (240, 10, 240)],
            [0.95, (250, 5, 250)],
            [1.0, (255, 0, 255)],
        ]
    else:
        raise NameError

        cyantopurple_cmap = LinearSegmentedColormap.from_list(
            "cyantopurple", [np.array(i[1]) / 255 for i in cyantopurple], N=255
        )
        return cyantopurple_cmap


plt.rcParams["figure.figsize"] = [8, 8]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["legend.fontsize"] = 15


def dim_1d_check(y):
    """Reshape (n,1) 2D array to (n) 1D array else do nothing"""
    if not isinstance(y, type(None)):
        if not isinstance(y, tuple):
            if len(y.shape) == 2:
                if y.shape[1] == 1:
                    return y.reshape(-1)
        else:
            return (dim_1d_check(y[0]), dim_1d_check(y[1]))
    return y


def aux_plot_pred(ax, x, y, pred):
    ax.plot(x, y, ls="dotted", color="black", linewidth=0.9, alpha=1)
    ax.plot(
        x,
        pred,
        "-",
        color="darkgreen",
        alpha=1,
        linewidth=0.7,
        zorder=-4,
        label="Prediction",
    )

    ax.scatter(x, y, c="black", s=10, marker="x", linewidth=1, label="Observation")


def aux_plot_anom(ax, x, y):
    ax.scatter(
        x, y, linewidth=1, marker="x", c="magenta", s=25, label='"Abnormal" real demand'
    )


def aux_plot_PIs(
    ax,
    x,
    list_PIs,
    list_alpha_PIs,
    list_colors_PIs=None,
    list_alpha_fig_PIs=None,
    list_label_PIs=None,
):
    """Plot PIs enveloppe on ax suplot

    Args:
        ax (_type_): ax suplot
        x (_type_): x_scale
        list_PIs (_type_): list of PIs ordered from minumum to maximum: ex [PI_low_1,PI_low_2,PI_high_2,PI_high_1]
        list_alpha (_type_): List of alpha of PIs
        list_colors_PIs (list, optional): List of color by pair. Defaults to None -> use grey.
        list_alpha_PIs (_type_, optional): List of color by pair. Defaults to None -> 0.2 as transparancy
    """
    n_couple = int(len(list_PIs) / 2)

    for i in range(n_couple):
        color = None
        if list_colors_PIs is not None:
            color = list_colors_PIs[i]
        else:
            color = None

        if list_label_PIs is None:
            label = (
                "Predictive interval: "
                + str(list_alpha_PIs[n_couple - i] - list_alpha_PIs[i])
                + "%"
            )
        else:
            label = list_label_PIs[i]

        alpha = 0.1
        if list_alpha_fig_PIs is not None:
            alpha = list_alpha_fig_PIs[i]

        ax.plot(x, list_PIs[i], color=color, ls="dotted", lw=1.2)
        ax.plot(x, list_PIs[n_couple - i], color=color, ls="dotted", lw=1.2)
        ax.fill_between(
            x,
            list_PIs[i],
            list_PIs[n_couple - i],
            color=list_colors_PIs[i],
            alpha=alpha,
            label=label,
        )


def aux_plot_conf_score(ax, x, pred, confidence_lvl, label, mode_res=False):
    if mode_res:
        pred = pred - pred

    for i in range(0, int(1 + confidence_lvl.max())):
        mask = i == confidence_lvl
        ax.scatter(
            x[mask],
            pred[mask],
            c=confidence_lvl[mask],
            marker="D",
            s=14,
            edgecolors="black",
            linewidth=0.2,
            cmap=plt.get_cmap("RdYlGn_r", len(label) + 1),
            vmin=0,
            vmax=int(confidence_lvl.max()),
            label=label[i],
            zorder=10 + i,
        )


def plot_pi(
    y,
    y_pred,
    y_pred_lower,
    y_pred_upper,
    mode_res=False,
    f_obs=None,
    X=None,
    size=(12, 2),
    name=None,
    show_plot=True,
):
    y = dim_1d_check(y)
    y_pred = dim_1d_check(y_pred)
    y_pred_upper = dim_1d_check(y_pred_upper)
    y_pred_lower = dim_1d_check(y_pred_lower)

    if isinstance(f_obs, type(None)):
        f_obs = np.arange(len(y))

    plt.figure(figsize=(size[0], size[1]))

    if not isinstance(y_pred, type(None)):
        if mode_res:
            y = y - y_pred
            y_pred_upper = y_pred_upper - y_pred
            y_pred_lower = y_pred_lower - y_pred
            y_pred = y_pred - y_pred

        plt.plot(y_pred[f_obs], "black", label="Prediction")

    if name is not None:
        plt.title(name)

    plt.plot(
        y[f_obs],
        "darkgreen",
        marker="X",
        markersize=2,
        linewidth=0,
        label="Observation (inside PI)",
        zorder=20,
    )

    plt.xlabel("X")
    plt.ylabel("Y")

    if not isinstance(y_pred_upper, type(None)):
        anom = (y[f_obs] > y_pred_upper[f_obs]) | (y[f_obs] < y_pred_lower[f_obs])

        plt.plot(
            np.arange(len(f_obs))[anom],
            y[f_obs][anom],
            color="red",
            marker="o",
            markersize=2,
            linewidth=0,
            label="Observation (outside PI)",
            zorder=20,
        )

        plt.plot(y_pred_upper[f_obs], "--", color="blue", linewidth=1, alpha=0.7)
        plt.plot(y_pred_lower[f_obs], "--", color="blue", linewidth=1, alpha=0.7)
        plt.fill_between(
            x=np.arange(len(f_obs)),
            y1=y_pred_upper[f_obs],
            y2=y_pred_lower[f_obs],
            alpha=0.2,
            fc="b",
            ec="None",
            label="Prediction Interval",
        )
    plt.legend(loc="best")
    if show_plot:
        plt.show()


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


def aux_fill_area(context, **kwargs):
    if "list_name_subset" in kwargs.keys():
        list_name_subset = kwargs["list_name_subset"]


def uncertainty_plot(
    y,
    output,
    context=None,
    size=(15, 5),
    f_obs=None,
    name="UQplot",
    mode_res=False,
    born=None,
    born_bis=None,
    dim=0,
    confidence_lvl=None,
    list_percent=[0.8, 0.9, 0.99, 0.999, 1],
    env=[0.95, 0.65],
    type_UQ="old",
    show_plot=True,
    with_colorbar=False,
    **kwarg,
):
    if f_obs is None:
        f_obs = np.arange(len(y))

    ind_ctx = None
    if "ind_ctx" in kwarg.keys():
        ind_ctx = kwarg["ind_ctx"]

    split_ctx = -1
    if "split_ctx" in kwarg.keys():
        split_ctx = kwarg["split_ctx"]

    ylim = None
    if "ylim" in kwarg.keys():
        ylim = kwarg["ylim"]

    if "compare_deg" in kwarg.keys():
        kwarg["compare_deg"]
    else:
        pass

    min_A, min_E = 0.000001, 0.000001
    if "var_min" in kwarg.keys():
        min_A, min_E = kwarg["var_min"]

    if output is not None:
        if type_UQ == "old":
            pred, var_A, var_E = output

        elif type_UQ == "var_A&E":
            pred, (var_A, var_E) = output

            var_E[var_E < min_E] = min_E
            var_A[var_A < min_A] = min_A

            # Post-processig PIs naifs:
            # Post-processing epistemics naifs.

        var_E[var_E < min_E] = min_E
        var_A[var_A < min_A] = min_A

    only_data = False
    if "list_name_subset" in kwarg.keys():
        list_name_subset = kwarg["list_name_subset"]

    if "only_data" in kwarg.keys():
        only_data = kwarg["only_data"]
        if only_data:
            name = "Data"

    f_obs_full = np.copy(f_obs)
    n_ctx = 1
    if isinstance(dim, int):
        dim = [dim]

    if split_ctx > -1:
        if ind_ctx is None:
            list_ctx_ = list(set(context[f_obs, split_ctx]))
        else:
            list_ctx_ = ind_ctx
        n_ctx = len(list_ctx_)

    fig, axs = plt.subplots(len(dim), n_ctx, sharex=True, figsize=size)
    label = None

    for n, d in enumerate(dim):
        for n_fig in range(n_ctx):
            ax = axs
            if len(dim) > 1:
                ax = ax[n]
            if n_ctx > 1:
                ax = ax[n_fig]

            if split_ctx > -1:
                f_obs = f_obs_full[context[f_obs_full, split_ctx] == list_ctx_[n_fig]]
            if only_data:
                ax.scatter(
                    f_obs,
                    y[f_obs, d],
                    c="black",
                    s=10,
                    marker="x",
                    linewidth=1,
                    label="observation",
                )

                ax.plot(
                    f_obs,
                    y[f_obs, d],
                    ls=":",
                    color="darkgreen",
                    alpha=1,
                    linewidth=0.7,
                    zorder=-4,
                )
                if ylim is not None:
                    ax.set_ylim(ylim[0], ylim[1])
                else:
                    (y.min(), y.max())

            else:
                born_ = None
                if born:
                    born_ = born[0][f_obs, d], born[1][f_obs, d]

                born_bis_ = None
                if born_bis:
                    born_bis_ = born_bis[0][f_obs, d], born_bis[1][f_obs, d]

                x = np.arange(len(y))
                if "x" in kwarg.keys():
                    x = kwarg["x"]

                if confidence_lvl is None:
                    confidence_lvl, params_ = UQ_proc.compute_Epistemic_score(
                        (var_A, var_E),
                        type_UQ="var_A&E",
                        pred=pred,
                        list_percent=list_percent,
                        params_=None,
                    )

                aux_plot_confiance(
                    ax=ax,
                    y=y[f_obs, d],
                    pred=pred[f_obs, d],
                    var_A=var_A[f_obs, d],
                    var_E=var_E[f_obs, d],
                    born=born_,
                    born_bis=born_bis_,
                    env=env,
                    x=x[f_obs],
                    mode_res=mode_res,
                    **kwarg,
                )

                label = [str(i) for i in list_percent]
                label.append(">1")

                aux_plot_conf_score(
                    ax,
                    x[f_obs],
                    pred[f_obs, d],
                    confidence_lvl[f_obs, d],
                    label=label,
                    mode_res=mode_res,
                )

            if "ctx_attack" in kwarg.keys():
                y[f_obs, d]
                if ylim is None:
                    ylim = (y.min(), y.max())
                dim_ctx, ctx_val = kwarg["ctx_attack"]
                if ctx_val == -1:
                    list_ctx = list(set(context[f_obs, dim_ctx]))
                    color = plt.get_cmap("jet", len(list_name_subset))
                    list_color = [color(i) for i in range(3)]
                    for n, i in enumerate(list_ctx):
                        ax.fill_between(
                            f_obs,
                            ylim[0],
                            ylim[1],
                            where=context[f_obs, dim_ctx] == i,
                            color=list_color[int(i)],
                            alpha=0.2,
                        )
                else:
                    ax.fill_between(
                        f_obs,
                        y.min(),
                        y.max(),
                        where=context[f_obs, dim_ctx] == 1,
                        color="yellow",
                        alpha=0.2,
                    )

            if n_fig != 0:
                ax.set_yticklabels([])

            if "ctx_attack" in kwarg.keys():
                color = plt.get_cmap("jet", len(list_name_subset))
                for n, i in enumerate(range(len(list_name_subset))):
                    ax.fill_between(
                        [],
                        ylim[0],
                        ylim[1],
                        where=[],
                        label=list_name_subset[int(i)],
                        color=color(i),
                        alpha=0.08,
                    )
    plt.suptitle(name)
    plt.subplots_adjust(
        wspace=0.03, hspace=0.03, left=0.1, bottom=0.22, right=0.90, top=0.8
    )
    plt.legend(frameon=True, ncol=6, fontsize=10, bbox_to_anchor=(0.5, 0, 0.38, -0.11))
    # plt.xlim(0, 8400)

    if label is not None:
        cmap = [plt.get_cmap("RdYlGn_r", 7)(i) for i in np.arange(len(label))]

        list_percent
        bounds = np.concatenate(
            [[0], np.cumsum(np.abs(np.array(list_percent) - 1) + 0.1)]
        )
        bounds = 10 * bounds / bounds.max()

        if with_colorbar:
            cmap = mpl.colors.ListedColormap(cmap)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            color_ls = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            cbar1 = plt.colorbar(
                color_ls,
                pad=0.20,
                fraction=0.10,
                shrink=0.5,
                anchor=(0.2, 0.0),
                orientation="horizontal",
                spacing="proportional",
            )
            cbar1.set_label("Confidence_lvl", fontsize=14)

            ticks = (bounds + np.roll(bounds, -1)) / 2
            ticks[-1] = 10

            cbar1.set_ticks(ticks)
            cbar1.set_ticklabels(label, fontsize=12)
    plt.tight_layout()
    if show_plot:
        plt.show()
    return


def aux_plot_confiance(
    ax,
    y,
    pred,
    var_A,
    var_E,
    born=None,
    born_bis=None,
    ylim=None,
    split_values=-1,
    x=None,
    mode_res=False,
    min_A=0.08,
    min_E=0.02,
    env=[0.95, 0.68],
    **kwarg,
):
    if x is None:
        x = np.arange(len(y))

    if mode_res:
        y = y - pred
        if born is not None:
            born = born[0] - pred, born[1] - pred
        pred = pred * 0

    y_lower_A, y_upper_A = compute_born(pred, np.sqrt(var_A), 0.045)
    y_lower_E, y_upper_E = compute_born(pred, np.sqrt(var_E / (var_A + var_E)), 0.045)
    y_lower, y_upper = compute_born(pred, np.sqrt(var_A + var_E * 0), 0.045)
    y_lower_N, y_upper_N = compute_born(pred, np.sqrt(var_A + var_E), 0.045)

    ind_A = np.sqrt(var_A)
    ind_E = np.sqrt(var_E)
    ind_E[ind_E < min_E] = min_E
    ind_A[ind_A < min_A] = min_A

    if born:
        anom_score = (y < born[0]) | (y > born[1])
        flag_anom = anom_score > 0

    else:
        anom_score = (np.abs(y - pred) + 0 * ind_E) / (
            2 * np.sqrt(np.power(ind_E, 2) + np.power(ind_A, 2))
        )
        flag_anom = anom_score > 1

    aux_plot_pred(ax, x, y, pred)
    aux_plot_anom(ax, x[flag_anom], y[flag_anom])

    if born:
        aux_plot_PIs(
            ax,
            x,
            [born[0], born[1]],
            list_alpha_PIs=[0.025, 0.975],
            list_colors_PIs=["green"],
            list_label_PIs=["Normal_limit"],
        )
        ax.fill_between(
            x,
            born[1],
            y,
            where=propagate(y > born[1], 0, sym=True),
            facecolor="red",
            alpha=0.8,
            label="Anomaly",
            interpolate=True,
            zorder=-10,
        )

        ax.fill_between(
            x,
            born[0],
            y,
            where=propagate(y < born[0], 0),
            interpolate=True,
            facecolor="red",
            alpha=0.8,
            zorder=-10,
        )

    else:
        aux_plot_PIs(
            ax,
            x,
            [y_lower, y_upper],
            list_alpha_PIs=[0.025, 0.975],
            list_colors_PIs=["green"],
            list_label_PIs=["2σAleatoric PIs (95%)"],
            list_alpha_fig_PIs=[0.2,0.2],
        )

        aux_plot_PIs(
            ax,
            x,
            [y_lower_N, y_upper_N],
            list_alpha_PIs=[0.16, 0.84],
            list_colors_PIs=["darkblue"],
            list_alpha_fig_PIs=[0.1,0.1],
            list_label_PIs=["2σTotal PIs(95%)"],
        )

    if born_bis:
        aux_plot_PIs(
            ax,
            x,
            [born_bis[0], born_bis[1]],
            list_alpha_PIs=[0.025, 0.975],
            list_colors_PIs=["teal"],
            list_alpha_fig_PIs=[0.1,0.1],
            list_label_PIs=["Normal_limit"],
        )

    if False:
        aux_plot_PIs(
            ax,
            x,
            [y_lower_A, y_upper_A],
            list_alpha_PI=[0.025, 0.975],
            list_colors_PIs=["blue"],
            list_alpha_fig_PIs=[0.2,0.2],
            list_label_PIs=["Var_A"],
        )

        aux_plot_PIs(
            ax,
            x,
            [y_lower_E, y_upper_E],
            [0.025, 0.975],
            list_colors_PIs=["red"],
            list_alpha_PIs=None,
            list_label_PIs=["Var_E"],
        )

    if ylim is None:
        ylim_ = min(y.min(), y_lower.min()), max(y.max(), y_upper.max())
        ylim_ = ylim_[0] + np.abs(ylim_[0] * 0.05), ylim_[1] - np.abs(ylim_[1] * 0.05)
    else:
        ylim_ = ylim[0], ylim[1]

    ax.set_ylim(ylim_[0], ylim_[1])
    ax.set_xlim(-0.5 + x.min(), x.max() + 0.5)
    # ax.legend(ncol=7)


def plot_anom_matrice(
    score,
    score2=None,
    f_obs=None,
    true_label=None,
    data=None,
    x=None,
    vmin=-3,
    vmax=3,
    cmap=None,
    list_anom_ind=None,
    figsize=(15, 6),
    grid_spec=None,
    x_date=False,
    show_plot=True,
    setup=None
):
    """Plot score_anomalie matrice and true label if there is.
    Args:
        score (_type_): Anomaly score matrice or list of Anomaly matrix
        f_obs (_type_, optional): mask_focus
        true_label (_type_, optional): True label or None
        vmin (int, optional): _description_. Defaults to -3.
        vmax (int, optional): _description_. Defaults to 3.
        cmap (_type_, optional): _description_. Defaults to None.
        figsize (tuple, optional): _description_. Defaults to (15, 6).
    """
    if isinstance(score, list):
        len_score = len(score[0])
        dim_score = [score_.shape[-1] for score_ in score]
        n_score = len(score)
    else:
        score = [score]
        len_score = len(score)
        dim_score = [score[0].shape[-1]]
        n_score = 1

    if f_obs is None:
        f_obs = np.arange(len_score)

    if cmap is None:
        cmap = provide_cmap("bluetored")

    x_flag = True
    if x is None:
        x_flag = False
        x = np.arange(len_score)
        list_extend = [None for dim_score_ in dim_score]

    else:
        d0 = mdates.date2num(x[f_obs][0])
        d1 = mdates.date2num(x[f_obs][-1])
        list_extend = [[d0, d1, 0, dim_score_] for dim_score_ in dim_score]

    n_fig = 3 + n_score
    if true_label is None:
        n_fig -= 1

    if score2 is None:
        n_fig -= 1

    if data is None:
        n_fig -= 1

    sharey = False
    if (
        (n_score == 1)
        and (true_label is not None)
        and (dim_score[0] == true_label.shape)
    ):
        sharey = True

    if grid_spec is None:
        grid_spec = np.ones(n_fig)

    fig, ax = plt.subplots(
        n_fig,
        1,
        sharex=True,
        sharey=sharey,
        gridspec_kw={"height_ratios": grid_spec},
        figsize=figsize,
    )

    if n_fig == 1:
        ax.set_title("score")

        ax.imshow(
            score[0][f_obs].T[::-1],
            cmap=provide_cmap("bluetored"),
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            extent=list_extend[0],
            interpolation=None,
        )

        if(setup) is not None:
            (n_chan,n_sensor) = setup

            for i in range(n_chan* n_sensor):
                a0.hlines(i, 0, len(f_obs), color="grey", lw=0.5)

            for i in range(n_sensor):
                a0.hlines(i * n_chan,0, len(f_obs), color="black", lw=1)


    else:
        ind_ax = -1
        for n, score_ in enumerate(score):
            ind_ax += 1
            ax[ind_ax].set_title("score")
            ax[ind_ax].imshow(
                score_[f_obs].T[::-1],
                cmap=provide_cmap("bluetored"),
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
                extent=list_extend[n],
                interpolation="None",
            )

        if score2 is not None:
            ind_ax += 1
            ax[ind_ax].set_title("score")
            ax[ind_ax].imshow(
                score2[f_obs].T[::-1],
                cmap=provide_cmap("bluetored"),
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
                extent=list_extend[0],
                interpolation="None",
            )

        if true_label is not None:
            ind_ax += 1
            ax[ind_ax].set_title("True_anom")
            ax[ind_ax].imshow(
                true_label[f_obs].T[::-1],
                cmap="Reds",
                aspect="auto",
                extent=list_extend[0],
                interpolation="None",
            )

        if data is not None:
            ind_ax += 1
            ax[ind_ax].set_title("data")
            colors = []
            for i in range(data.shape[1]):
                colors.append(plt.get_cmap("Greens", data.shape[1])(i))
            if list_anom_ind is not None:
                for n, anom_ind in enumerate(list_anom_ind):
                    colors[anom_ind] = plt.get_cmap("Reds", len(list_anom_ind) + 4)(
                        n + 4
                    )

            for i in range(dim_score[0]):
                ax[ind_ax].plot(x[f_obs], data[f_obs, i], color=colors[i], lw=0.9)

            for i in range(dim_score[0]):
                mask = np.abs(score[0])[f_obs, i] > 1
                ax[ind_ax].scatter(
                    x[f_obs][mask],
                    data[f_obs, i][mask],
                    color="red",
                    marker="x",
                    s=1,
                    zorder=10,
                )

        if (true_label is not None) & (data is not None):
            for i in range(data.shape[1]):
                mask = true_label[f_obs, i] > 0
                ax[ind_ax].scatter(x[f_obs][mask], data[f_obs, i][mask], color="purple")
        if x_flag:
            ax[ind_ax].xaxis_date()
        if x_date:
            ax[ind_ax].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))

    fig.tight_layout()
    if show_plot:
        plt.show()


# Display of data curve with mean and variance.
def plot_var(
    Y,
    data_full,
    variance,
    impact_anom=None,
    anom=None,
    f_obs=None,
    dim=(400, 20, 3),
    g=0,
    res_flag=False,
    fig_s=(20, 3),
    title=None,
    ylim=None,
):
    def add_noise(data, noise_mult, noise_add):
        signal = (data * (1 + noise_mult)) + noise_add
        return signal

    dim_n, dim_t, dim_g = dim
    anom_pred = (np.abs(impact_anom).sum(axis=-1) > 0).astype(int) - (anom > 0).astype(
        int
    )

    if anom_pred.sum() < 1:
        anom_pred[0] = 1
        anom_pred[-1] = 1

    step = g
    plt.figure(figsize=fig_s)
    plt.title(title)
    # norm = Y.mean(axis=0)
    if anom_pred.sum() < 1:
        anom_pred[0] = 1
        anom_pred[-1] = 1

    ni = [100, 98, 95, 80, 50]
    color_full = [
        (0.5, 0.0, 0.5),
        (0.8, 0, 0),
        (0.8, 0.6, 0),
        (0, 0.8, 0),
        (0, 0.4, 0),
        (0, 0.8, 0),
        (0.8, 0.6, 0),
        (0.8, 0, 0),
        (0.5, 0.0, 0.5),
    ]
    color_full2 = [
        (0.5, 0.0, 0.5),
        (0.8, 0, 0),
        (0.8, 0.6, 0),
        (0, 0.8, 0),
        (0, 0.4, 0),
        (0, 0.4, 0),
        (0, 0.8, 0),
        (0.8, 0.6, 0),
        (0.8, 0, 0),
        (0.5, 0.0, 0.5),
    ]

    per_list = [0.01, 1, 2.5, 10, 25, 75, 90, 97.5, 99, 99.99]
    per = []

    res = data_full * 0
    if res_flag:
        res = data_full

    for i in per_list:
        per.append(
            add_noise(
                data_full - res,
                0,
                scipy.stats.norm.ppf((i / 100), 0, np.sqrt(variance)),
            ),
        )

    for i in range(len(per) - 1):
        plt.fill_between(
            np.arange(len(f_obs)),
            per[i][f_obs, step],
            per[i + 1][f_obs, step],
            color=color_full[i],
            alpha=0.20,
        )
    for i in range(len(per)):
        plt.plot(
            np.arange(len(f_obs)),
            per[i][f_obs, step],
            color=color_full2[i],
            linewidth=0.5,
            alpha=0.40,
        )
        if i > 4:
            plt.fill_between(
                [],
                [],
                [],
                color=color_full2[i],
                label=str(ni[9 - i]) + "% Coverage",
                alpha=0.20,
            )

    plt.plot(
        Y[f_obs, step] - res[f_obs, step],
        label="Series",
        color="black",
        linewidth=1.5,
        marker="o",
        ms=3,
    )
    flag = impact_anom[f_obs, step] != 0

    plt.plot(
        np.arange(len(f_obs))[flag],
        Y[f_obs, step][flag] - res[f_obs, step][flag],
        label="Anom",
        color="red",
        ls="",
        marker="X",
        ms=10,
        alpha=0.8,
    )

    if False:
        f_anom = np.repeat(np.abs(impact_anom)[f_obs, step] > 0, 7)
        for i in range(8):
            f_anom += np.roll(f_anom, i - 4)
        f_anom = f_anom > 0
        plt.fill_between(
            np.arange(len(f_obs) * 7) / 7 - (3 / 7),
            -1000,
            Y[f_obs, step].max() * 1.2,
            where=f_anom,
            facecolor="blue",
            alpha=0.2,
            label="Anomaly",
            zorder=-10,
        )

    if ylim is None:
        plt.ylim(per[0][f_obs, step].min() * 1.05, per[-1][f_obs, step].max() * 1.05)
    else:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(ncol=7, fontsize=14)
    plt.xlim(0, len(f_obs))
    plt.tight_layout()
    plt.show()
    return (per, per_list)


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
        ax[0].plot(y[f_obs, d : d + 1], label="true_val")
    ax[0 + val].plot(var_A[f_obs, d : d + 1], label="row_var_A")
    ax[0 + val].plot(var_A_cut[f_obs, d : d + 1], label="refined_var_A")
    ax[0 + val].legend()
    ax[1 + val].plot(var_E[f_obs, d : d + 1], label="row_var_E")
    ax[1 + val].plot(var_E_res[f_obs, d : d + 1], label="refined_var_E")
    ax[1 + val].legend()
    ratio = var_E[f_obs, d : d + 1] / var_A[f_obs, d : d + 1]
    ax[2 + val].plot(ratio / ratio.std(), label="row_ratio")
    refined_ratio = (var_A_res[f_obs, d : d + 1] + var_E_res[f_obs, d : d + 1]) / (
        var_A_cut[f_obs, d : d + 1] + var_E_cut[f_obs, d : d + 1]
    )
    ax[2 + val].plot(refined_ratio / refined_ratio.std(), label="refined_ratio")
    ax[2 + val].legend()
    print("yaya")
    return (fig, ax)
