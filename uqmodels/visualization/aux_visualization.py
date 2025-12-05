import matplotlib.pyplot as plt
import numpy as np
from uqmodels.utils import compute_born, propagate, _merge_config, _merge_nested

import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap


def aux_adjust_axes(ax, x, y_list, ylim=None, x_lim=None, margin=0.05, x_margin=0.5):
    """
    Adjust x/y axis limits based on data and optional explicit limits.

    Parameters
    ----------
    ax : Axes
    x : array-like
    y_list : array-like or list of array-like
        One or several y-series to consider for limits.
    ylim : tuple or None
        If not None, force (ymin, ymax).
    x_lim : tuple or None
        If not None, force (xmin, xmax).
    margin : float
        Relative margin applied to inferred y-limits (ignored if ylim is set).
    x_margin : float
        Margin added around min/max of x (ignored if x_lim is set).
    """
    x = np.asarray(x)

    if not isinstance(y_list, (list, tuple)):
        y_list = [y_list]

    y_min = min(np.asarray(y).min() for y in y_list)
    y_max = max(np.asarray(y).max() for y in y_list)

    # Y limits
    if ylim is None:
        y_low = y_min - abs(y_min * margin)
        y_high = y_max + abs(y_max * margin)
    else:
        y_low, y_high = ylim

    ax.set_ylim(y_low, y_high)

    # X limits
    if x_lim is None:
        x_low = x.min() - x_margin
        x_high = x.max() + x_margin
    else:
        x_low, x_high = x_lim

    ax.set_xlim(x_low, x_high)


DEFAULT_PLOT_PRED_CONFIG = {
    "truth_line": {  # line for y (true / observed curve)
        "ls": "dotted",
        "color": "black",
        "linewidth": 0.9,
        "alpha": 1.0,
    },
    "pred_line": {   # line for pred
        "linestyle": "-",  # alias to show it works even with different key
        "color": "darkgreen",
        "alpha": 1.0,
        "linewidth": 0.7,
        "zorder": -4,
        "label": "Prediction",
    },
    "obs_scatter": {  # scatter for observations as points
        "c": "black",
        "s": 10,
        "marker": "x",
        "linewidth": 1,
        "label": "Observation",
    },
}

DEFAULT_LINE_CONFIG = {
    "color": "black",
    "linestyle": "-",
    "linewidth": 1.0,
    "marker": None,
    "markersize": None,
    "label": None,
    "zorder": None,
}


def aux_plot_line(ax, x, y, config=None):
    """
    Plot a line (or markers only) on an Axes with configurable style.

    Parameters
    ----------
    ax : Axes
    x, y : array-like
    config : dict, optional
        Keys: color, linestyle, linewidth, marker, markersize, label, zorder.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    cfg = DEFAULT_LINE_CONFIG.copy()
    if config is not None:
        cfg.update({k: v for k, v in config.items() if v is not None})

    # Filtrer les None pour ne pas polluer ax.plot
    kwargs = {k: v for k, v in cfg.items() if v is not None}
    return ax.plot(x, y, **kwargs)


DEFAULT_PLOT_PRED_CONFIG = {
    "truth_line": {   # line for y (true / observed curve)
        "color": "black",
        "linestyle": "dotted",
        "linewidth": 0.9,
        "alpha": 1.0,
        "zorder": -5,
        "label": None,
    },
    "pred_line": {    # line for pred
        "color": "darkgreen",
        "linestyle": "-",
        "linewidth": 0.7,
        "alpha": 1.0,
        "zorder": -4,
        "label": "Prediction",
    },
    "obs_scatter": {  # observations as points
        "color": "black",
        "marker": "x",
        "markersize": 4,
        "linestyle": "none",
        "linewidth": 1.0,
        "alpha": 1.0,
        "zorder": 10,
        "label": "Observation",
    },
}


def aux_plot_pred(ax, x, y, pred, config=None):
    """
    Plot observations and predictions on a given Axes with optional style config.

    Parameters
    ----------
    ax : Axes
    x, y, pred : array-like
        Coordinates, observations, and predictions.
    config : dict, optional
        Style overrides for truth line, prediction line, and observation scatter.
        Keys: "truth_line", "pred_line", "obs_scatter".
    """
    x = np.asarray(x)
    y = np.asarray(y)
    pred = np.asarray(pred)

    cfg = _merge_config(DEFAULT_PLOT_PRED_CONFIG, config)

    # Courbe "vérité terrain"
    aux_plot_line(ax, x, y, config=cfg["truth_line"])

    # Courbe de prédiction
    aux_plot_line(ax, x, pred, config=cfg["pred_line"])

    # Points d'observation
    aux_plot_line(ax, x, y, config=cfg["obs_scatter"])


DEFAULT_PLOT_ANOM_CONFIG = {
    "anom_scatter": {
        "linewidth": 1,
        "marker": "x",
        "c": "magenta",
        "s": 25,
        "label": '"Abnormal" real demand',
    }
}

DEFAULT_PLOT_ANOM_CONFIG = {
    "anom_scatter": {
        "color": "magenta",
        "marker": "x",
        "markersize": 5,
        "linestyle": "none",
        "linewidth": 1.0,
        "alpha": 1.0,
        "zorder": 15,
        "label": '"Abnormal" observation',
    }
}


def aux_plot_anom(ax, x, y, config=None):
    """
    Plot anomalous observations on an Axes using aux_plot_line.

    Parameters
    ----------
    ax : Axes
    x, y : array-like
        Coordinates and anomalous values.
    config : dict, optional
        Style overrides for anomalous points.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    cfg = _merge_config(DEFAULT_PLOT_ANOM_CONFIG, config)
    aux_plot_line(ax, x, y, config=cfg["anom_scatter"])


DEFAULT_FILL_AREA_CONFIG = {
    "color": None,
    "alpha": 0.2,
    "label": None,
}

DEFAULT_FILL_BETWEEN_CONFIG = {
    "color": None,
    "facecolor": None,
    "alpha": 0.2,
    "label": None,
    "interpolate": False,
    "zorder": None,
}


def aux_fill_between(ax, x, y1, y2, where=None, config=None):
    """
    Wrapper around ax.fill_between with configurable style.

    Parameters
    ----------
    ax : Axes
    x, y1, y2 : array-like
    where : array-like or None
        Boolean mask for conditional fill.
    config : dict, optional
        Style overrides (color, facecolor, alpha, label, interpolate, zorder).
    """
    x = np.asarray(x)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    kwargs = DEFAULT_FILL_BETWEEN_CONFIG.copy()
    if config is not None:
        kwargs.update({k: v for k, v in config.items() if v is not None})

    return ax.fill_between(
        x,
        y1,
        y2,
        where=where,
        **kwargs,
    )


def aux_fill_area(ax, x, env_bot, env_top, config=None):
    """
    Fill an envelope between two curves on an Axes.

    Parameters
    ----------
    ax : Axes
    x, env_bot, env_top : array-like
        Coordinates and envelope bounds.
    config : dict, optional
        Style overrides (color, alpha, label).
    """
    x = np.asarray(x)
    env_bot = np.asarray(env_bot)
    env_top = np.asarray(env_top)

    final_cfg = DEFAULT_FILL_AREA_CONFIG.copy()
    if config is not None:
        final_cfg.update({k: v for k, v in config.items() if v is not None})

    return ax.fill_between(
        x,
        env_bot,
        env_top,
        color=final_cfg["color"],
        alpha=final_cfg["alpha"],
        label=final_cfg["label"],
    )


DEFAULT_PI_PLOT_CONFIG = {
    "line": {      # style des lignes de bornes
        "ls": "dotted",
        "lw": 1.2,
        "color": None,
    },
    "fill": {      # style du remplissage
        "color": None,
        "alpha": 0.2,
    },
}


def aux_plot_PIs(
    ax,
    x,
    list_PIs,
    list_alpha_PIs,
    list_colors_PIs=["lightblue", "lightgreen"],
    list_alpha_fig_PIs=[0.3, 0.15],
    list_label_PIs=None,
    config=None,
):
    """
    Plot multiple prediction interval envelopes on an Axes.

    Parameters
    ----------
    ax : Axes
    x : array-like
    list_PIs : list of array-like
        [low_1, ..., low_k, high_k, ..., high_1].
    list_alpha_PIs : list of float
        Quantile levels for each bound.
    list_colors_PIs : list, optional
        Per-interval colors overriding config.
    list_alpha_fig_PIs : list, optional
        Per-interval alphas overriding config.
    list_label_PIs : list, optional
        Per-interval labels.
    config : dict, optional
        Global style config: {"line": {...}, "fill": {...}}.
    """
    n_bounds = len(list_PIs)
    n_couple = n_bounds // 2

    # Merge global config with defaults
    line_cfg = DEFAULT_PI_PLOT_CONFIG["line"].copy()
    fill_cfg = DEFAULT_PI_PLOT_CONFIG["fill"].copy()
    if config is not None:
        if "line" in config:
            line_cfg.update(config["line"])
        if "fill" in config:
            fill_cfg.update(config["fill"])

    # Defaults for per-interval overrides
    if list_colors_PIs is None:
        list_colors_PIs = [None] * n_couple
    if list_alpha_fig_PIs is None:
        list_alpha_fig_PIs = [fill_cfg.get("alpha", 0.2)] * n_couple

    for i in range(n_couple):
        low = list_PIs[i]
        high = list_PIs[-(i + 1)]

        color = list_colors_PIs[i] if list_colors_PIs[i] is not None else fill_cfg.get("color", None)
        alpha = list_alpha_fig_PIs[i]

        if list_label_PIs is None:
            coverage = (list_alpha_PIs[-(i + 1)] - list_alpha_PIs[i]) * 100
            label = f"Predictive interval: {coverage:.0f}%"
        else:
            label = list_label_PIs[i]

        # Lignes de bornes
        line_kwargs = line_cfg.copy()
        if color is not None:
            line_kwargs["color"] = color
        ax.plot(x, low, **line_kwargs)
        ax.plot(x, high, **line_kwargs)

        # Remplissage via le helper
        fill_kwargs = fill_cfg.copy()
        if color is not None:
            fill_kwargs["color"] = color
        fill_kwargs["alpha"] = alpha
        fill_kwargs["label"] = label

        aux_fill_area(ax, x, low, high, config=fill_kwargs)


DEFAULT_CONF_SCORE_CONFIG = {
    "marker": "D",
    "s": 14,
    "edgecolors": "black",
    "linewidth": 0.2,
    "cmap": "RdYlGn_r",
    "zorder_base": 10,
}


def aux_plot_conf_score(ax, x, pred, confidence_lvl, label, mode_res=False, config=None):
    """
    Plot confidence scores as colored markers on an Axes.

    Parameters
    ----------
    ax : Axes
    x, pred : array-like
        Coordinates and predictions.
    confidence_lvl : array-like
        Discrete confidence levels (int).
    label : list of str
        Legend labels per confidence level.
    mode_res : bool, default False
        If True, plot residual scores around zero.
    config : dict, optional
        Style overrides (marker, s, edgecolors, linewidth, cmap, zorder_base).
    """
    x = np.asarray(x)
    pred = np.asarray(pred)
    confidence_lvl = np.asarray(confidence_lvl)

    if mode_res:
        pred = pred - pred

    cfg = DEFAULT_CONF_SCORE_CONFIG.copy()
    if config is not None:
        cfg.update({k: v for k, v in config.items() if v is not None})

    max_conf = int(confidence_lvl.max())
    cmap = plt.get_cmap(cfg["cmap"], len(label) + 1)

    for i in range(0, max_conf + 1):
        mask = (confidence_lvl == i)
        if not mask.any():
            continue

        ax.scatter(
            x[mask],
            pred[mask],
            c=confidence_lvl[mask],
            marker=cfg["marker"],
            s=cfg["s"],
            edgecolors=cfg["edgecolors"],
            linewidth=cfg["linewidth"],
            cmap=cmap,
            vmin=0,
            vmax=max_conf,
            label=label[i],
            zorder=cfg["zorder_base"] + i,
        )


DEFAULT_CONFIDENCE_PLOT_CONFIG = {
    "pred": None,              # config -> aux_plot_pred
    "anom": None,              # config -> aux_plot_anom
    "pis_born": None,          # config -> aux_plot_PIs (born)
    "pis_aleatoric": None,     # config -> aux_plot_PIs (aleatoric PI)
    "pis_total": None,         # config -> aux_plot_PIs (total PI)
    "pis_born_bis": None,      # config -> aux_plot_PIs (born_bis)
    "anom_fill_upper": {       # config -> aux_fill_between (born upper)
        "facecolor": "red",
        "alpha": 0.8,
        "label": "Anomaly",
        "interpolate": True,
        "zorder": -10,
    },
    "anom_fill_lower": {       # config -> aux_fill_between (born lower)
        "facecolor": "red",
        "alpha": 0.8,
        "interpolate": True,
        "zorder": -10,
    },
    "axes": {                  # config -> aux_adjust_axes
        "margin": 0.05,
        "x_margin": 0.5,
    },
}


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
    config=None,
    **kwarg,
):
    """
    Plot prediction, uncertainty intervals and anomaly regions on an Axes.
    """
    y = np.asarray(y)
    pred = np.asarray(pred)
    var_A = np.asarray(var_A)
    var_E = np.asarray(var_E)

    if x is None:
        x = np.arange(len(y))
    else:
        x = np.asarray(x)

    # Config globale fusionnée
    cfg = _merge_nested(DEFAULT_CONFIDENCE_PLOT_CONFIG, config or {})

    # Mode résidus
    if mode_res:
        y = y - pred
        if born is not None:
            born = (born[0] - pred, born[1] - pred)
        pred = pred * 0.0

    # Indicateurs tronqués
    ind_A = np.sqrt(var_A)
    ind_E = np.sqrt(var_E)
    ind_E[ind_E < min_E] = min_E
    ind_A[ind_A < min_A] = min_A

    # PIs (aleatoric et total)
    y_lower, y_upper = compute_born(pred, np.sqrt(var_A + var_E * 0.0), 0.045)
    y_lower_N, y_upper_N = compute_born(pred, np.sqrt(var_A + var_E), 0.045)

    # Anomalies
    if born is not None:
        anom_mask = (y < born[0]) | (y > born[1])
    else:
        anom_score = np.abs(y - pred) / (2.0 * np.sqrt(ind_E**2 + ind_A**2))
        anom_mask = anom_score > 1.0

    # Prédiction + anomalies
    aux_plot_pred(ax, x, y, pred, config=cfg["pred"])
    aux_plot_anom(ax, x[anom_mask], y[anom_mask], config=cfg["anom"])

    # Cas avec bornes explicites
    if born is not None:
        aux_plot_PIs(
            ax,
            x,
            [born[0], born[1]],
            list_alpha_PIs=[0.025, 0.975],
            list_colors_PIs=["green"],
            list_label_PIs=["Normal_limit"],
            config=cfg["pis_born"],
        )

        aux_fill_between(
            ax,
            x,
            born[1],
            y,
            where=propagate(y > born[1], 0, sym=True),
            config=cfg["anom_fill_upper"],
        )

        aux_fill_between(
            ax,
            x,
            born[0],
            y,
            where=propagate(y < born[0], 0),
            config=cfg["anom_fill_lower"],
        )

    # Cas sans bornes explicites : PIs théoriques
    else:
        aux_plot_PIs(
            ax,
            x,
            [y_lower, y_upper],
            list_alpha_PIs=[0.025, 0.975],
            list_colors_PIs=["green"],
            list_label_PIs=["2σAleatoric PIs (95%)"],
            list_alpha_fig_PIs=[0.2],
            config=cfg["pis_aleatoric"],
        )

        aux_plot_PIs(
            ax,
            x,
            [y_lower_N, y_upper_N],
            list_alpha_PIs=[0.16, 0.84],
            list_colors_PIs=["darkblue"],
            list_alpha_fig_PIs=[0.1],
            list_label_PIs=["2σTotal PIs(95%)"],
            config=cfg["pis_total"],
        )

    # Bornes supplémentaires
    if born_bis is not None:
        aux_plot_PIs(
            ax,
            x,
            [born_bis[0], born_bis[1]],
            list_alpha_PIs=[0.025, 0.975],
            list_colors_PIs=["teal"],
            list_alpha_fig_PIs=[0.1],
            list_label_PIs=["Normal_limit"],
            config=cfg["pis_born_bis"],
        )

    # Ajustement axes
    axes_cfg = cfg["axes"]
    aux_adjust_axes(ax, x, [y, y_lower], ylim=ylim, margin=axes_cfg["margin"], x_margin=axes_cfg["x_margin"])


# Auxiliar function related to matplot

def aux_norm_score_inputs(score, f_obs=None, cmap=None):
    """
    Normalise score en liste, infère len_score, dim_score, n_score et f_obs.
    """
    if isinstance(score, list):
        len_score = len(score[0])
        dim_score = [score_.shape[-1] for score_ in score]
        n_score = len(score)
        score_list = score
    else:
        score_list = [score]
        len_score = len(score)
        dim_score = [score_list[0].shape[-1]]
        n_score = 1

    if f_obs is None:
        f_obs = np.arange(len_score)
    else:
        f_obs = np.asarray(f_obs)

    if cmap is None:
        cmap = provide_cmap("bluetored")

    return score_list, len_score, dim_score, n_score, f_obs, cmap


def aux_prepare_x_extent(x, f_obs, dim_score):
    """
    Prépare x (échelle) et la liste d'extent pour imshow.
    Retourne x, x_flag (datetime ou non), list_extent.
    """
    if x is None:
        x_flag = False
        x = np.arange(len(f_obs))
        list_extent = [None for _ in dim_score]
    else:
        x_flag = True
        x = np.asarray(x)
        d0 = mdates.date2num(x[f_obs][0])
        d1 = mdates.date2num(x[f_obs][-1])
        list_extent = [[d0, d1, 0, dim] for dim in dim_score]

    return x, x_flag, list_extent


def aux_compute_layout_params(n_score, dim_score, true_label, score2, data, grid_spec=None):
    """
    Calcule n_fig, sharey, grid_spec.
    """
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
        # Même condition que ton code initial (même si peu naturelle)
        sharey = True

    if grid_spec is None:
        grid_spec = np.ones(n_fig)

    return n_fig, sharey, grid_spec


def aux_create_score_figure(n_fig, sharey, grid_spec, figsize):
    """
    Crée la figure et les axes pour la matrice d'anomalies.
    """
    fig, ax = plt.subplots(
        n_fig,
        1,
        sharex=True,
        sharey=sharey,
        gridspec_kw={"height_ratios": grid_spec},
        figsize=figsize,
    )
    return fig, ax


def aux_plot_score_matrix(ax, score_mat, f_obs, extent, vmin, vmax, cmap, title="score"):
    """
    Affiche une matrice de score via imshow.
    """
    ax.set_title(title)
    ax.imshow(
        score_mat[f_obs].T[::-1],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        extent=extent,
        interpolation=None,
    )


def aux_overlay_setup_grid(ax, setup, n_points):
    """
    Superpose la grille channels/sensors sur une matrice de score.
    setup = (n_chan, n_sensor)
    """
    if setup is None:
        return
    n_chan, n_sensor = setup
    for i in range(n_chan * n_sensor):
        ax.hlines(i, 0, n_points, color="grey", lw=0.5)
    for i in range(n_sensor):
        ax.hlines(i * n_chan, 0, n_points, color="black", lw=1)


def aux_plot_true_label_matrix(ax, true_label, f_obs, extent):
    """
    Affiche la matrice de labels vrais.
    """
    ax.set_title("True_anom")
    ax.imshow(
        true_label[f_obs].T[::-1],
        cmap="Reds",
        aspect="auto",
        extent=extent,
        interpolation=None,
    )


def aux_build_data_colors(data, list_anom_ind=None):
    """
    Construit la palette de couleurs par canal, en surlignant éventuellement
    certains indices anormaux.
    """
    n_chan = data.shape[1]
    base_cmap = plt.get_cmap("Greens", n_chan)
    colors = [base_cmap(i) for i in range(n_chan)]

    if list_anom_ind is not None:
        red_cmap = plt.get_cmap("Reds", len(list_anom_ind) + 4)
        for n, anom_ind in enumerate(list_anom_ind):
            colors[anom_ind] = red_cmap(n + 4)

    return colors


def aux_plot_data_timeseries(ax, x, data, f_obs, dim, colors, lw=0.9):
    """
    Trace les séries temporelles multicanal.
    """
    for i in range(dim):
        ax.plot(x[f_obs], data[f_obs, i], color=colors[i], lw=lw)


def aux_overlay_score_anoms_on_data(ax, x, data, score, f_obs, dim, threshold=1.0):
    """
    Superpose les points où |score| > threshold sur les séries de données.
    """
    for i in range(dim):
        mask = np.abs(score)[f_obs, i] > threshold
        ax.scatter(
            x[f_obs][mask],
            data[f_obs, i][mask],
            color="red",
            marker="x",
            s=1,
            zorder=10,
        )


def aux_overlay_true_label_on_data(ax, x, data, true_label, f_obs, color="purple"):
    """
    Superpose les labels vrais sur les séries temporelles.
    """
    for i in range(data.shape[1]):
        mask = true_label[f_obs, i] > 0
        ax.scatter(x[f_obs][mask], data[f_obs, i][mask], color=color)


def aux_format_time_axis(ax, x_flag, x_date):
    """
    Configure l'axe des x comme temporel et éventuellement applique un formatter.
    """
    if x_flag:
        ax.xaxis_date()
    if x_date:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))


def aux_finalize_figure(fig, show_plot=True):
    """
    Finalise la figure (tight_layout + show optionnel).
    """
    fig.tight_layout()
    if show_plot:
        plt.show()


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


def _get_panel_ax(axs, n_dim, n_ctx, idx_dim, idx_ctx):
    """Sélectionne l'Axes correct dans la grille axs (len(dim) x n_ctx)."""
    if n_dim == 1 and n_ctx == 1:
        return axs
    if n_dim == 1:
        return axs[idx_ctx]
    if n_ctx == 1:
        return axs[idx_dim]
    return axs[idx_dim, idx_ctx]
