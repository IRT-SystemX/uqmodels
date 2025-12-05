"""
Visualization module.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import uqmodels.postprocessing.UQ_processing as UQ_proc
from uqmodels.check import dim_1d_check
import uqmodels.visualization.aux_visualization as auxvisu
from uqmodels.visualization.aux_visualization import provide_cmap  # noqa: F401
from uqmodels.visualization.old_visualisation import plot_prediction_interval, plot_sorted_pi  # noqa: F401
from uqmodels.visualization.old_visualisation import visu_latent_space, show_dUQ_refinement  # noqa: F401

plt.rcParams["figure.figsize"] = [8, 8]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["legend.fontsize"] = 15


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
    config=None,
    ylim=None,
    xlim=None,
    **kwargs,
):
    """
    Plot prediction intervals (PI) together with observations and predictions.

    Displays observed values, predicted values, and their prediction interval
    (upper/lower bounds). Optionally plots residuals instead of absolute values
    (mode_res=True). Observations falling outside the PI are highlighted.
    Parameter f_obs selects which observation indices to display.
    """

    # Sous-configs avec defaults
    cfg_pred = (config.get("prediction") if config else {}) or {}
    cfg_inside = (config.get("inside") if config else {}) or {}
    cfg_outside = (config.get("outside") if config else {}) or {}
    cfg_pi = (config.get("pi") if config else {}) or {}
    cfg_axes = (config.get("axes") if config else {}) or {}

    # Coercions
    y = dim_1d_check(y)
    if y_pred is not None:
        y_pred = dim_1d_check(y_pred)
    if y_pred_upper is not None:
        y_pred_upper = dim_1d_check(y_pred_upper)
    if y_pred_lower is not None:
        y_pred_lower = dim_1d_check(y_pred_lower)

    # x-scale
    if X is None:
        x = np.arange(len(y))
    else:
        x = dim_1d_check(X)

    # Indices observés
    if f_obs is None:
        f_obs = np.arange(len(y))
    else:
        f_obs = np.asarray(f_obs)

    x_plot = x[f_obs]

    fig, ax = plt.subplots(figsize=size)

    # Prediction (+ mode résidus)
    if y_pred is not None:
        if mode_res:
            y = y - y_pred
            if y_pred_upper is not None:
                y_pred_upper = y_pred_upper - y_pred
            if y_pred_lower is not None:
                y_pred_lower = y_pred_lower - y_pred
            y_pred = y_pred * 0.0

        pred_cfg = {
            "color": cfg_pred.get("color", "black"),
            "linestyle": cfg_pred.get("linestyle", cfg_pred.get("ls", "-")),
            "linewidth": cfg_pred.get("linewidth", cfg_pred.get("lw", 1.0)),
            "label": cfg_pred.get("label", "Prediction"),
            "zorder": cfg_pred.get("zorder", None),
        }
        auxvisu.aux_plot_line(ax, x_plot, y_pred[f_obs], config=pred_cfg)

    if name is not None:
        ax.set_title(name)

    # Observations (inside PI)
    inside_cfg = {
        "color": cfg_inside.get("color", "darkgreen"),
        "marker": cfg_inside.get("marker", "X"),
        "markersize": cfg_inside.get("markersize", 2),
        "linestyle": "none",
        "label": cfg_inside.get("label", "Observation (inside PI)"),
        "zorder": cfg_inside.get("zorder", 20),
    }
    auxvisu.aux_plot_line(ax, x_plot, y[f_obs], config=inside_cfg)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # PI + anomalies
    if y_pred_upper is not None and y_pred_lower is not None:
        anom = (y[f_obs] > y_pred_upper[f_obs]) | (y[f_obs] < y_pred_lower[f_obs])

        # Observations outside PI
        if np.any(anom):
            outside_cfg = {
                "color": cfg_outside.get("color", "red"),
                "marker": cfg_outside.get("marker", "o"),
                "markersize": cfg_outside.get("markersize", 2),
                "linestyle": "none",
                "label": cfg_outside.get("label", "Observation (outside PI)"),
                "zorder": cfg_outside.get("zorder", 21),
            }
            auxvisu.aux_plot_line(ax, x_plot[anom], y[f_obs][anom], config=outside_cfg)

        # Bornes du PI
        pi_line_cfg = {
            "color": cfg_pi.get("color", "blue"),
            "linestyle": cfg_pi.get("linestyle", cfg_pi.get("ls", "--")),
            "linewidth": cfg_pi.get("linewidth", cfg_pi.get("lw", 1)),
            "zorder": cfg_pi.get("zorder_line", None),
        }
        auxvisu.aux_plot_line(ax, x_plot, y_pred_upper[f_obs], config=pi_line_cfg)
        auxvisu.aux_plot_line(ax, x_plot, y_pred_lower[f_obs], config=pi_line_cfg)

        # Zone PI
        auxvisu.aux_fill_area(
            ax,
            x_plot,
            y_pred_lower[f_obs],
            y_pred_upper[f_obs],
            config={
                "color": cfg_pi.get("fill_color", cfg_pi.get("color", "blue")),
                "alpha": cfg_pi.get("alpha_fill", 0.2),
                "label": cfg_pi.get("label", "Prediction Interval"),
            },
        )

    # Données à considérer pour les limites
    y_for_axes = [y[f_obs]]
    if y_pred_lower is not None and y_pred_upper is not None:
        y_for_axes.extend([y_pred_lower[f_obs], y_pred_upper[f_obs]])
    if y_pred is not None:
        y_for_axes.append(y_pred[f_obs])

    # Gestion des axes via helper (inclut x_lim)
    auxvisu.aux_adjust_axes(
        ax,
        x_plot,
        y_for_axes,
        ylim=ylim,
        x_lim=xlim,
        margin=cfg_axes.get("margin", 0.05),
        x_margin=cfg_axes.get("x_margin", 0.5),
    )

    ax.legend(loc="best")
    if show_plot:
        plt.show()

# ---------- Orchestrateur principal ----------


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
    setup=None,
):
    """
    Visualize anomaly score matrices and optional ground-truth labels or data.

    This function plots one or several anomaly score matrices (e.g., per model or
    per transformation), an optional secondary anomaly score matrix, optional
    ground-truth anomaly labels, and optional multichannel time series data.
    It supports contextual segmentation, date-based x-axes, sensor/channel
    structural overlays, and anomaly highlighting. The function preserves its
    original API while delegating rendering to modular helpers.

    Parameters
    ----------
    score : array-like or list of array-like
        Primary anomaly score matrix or list of matrices.
        Each matrix must be of shape (n_samples, n_features).
    score2 : array-like, optional
        Secondary anomaly score matrix of shape (n_samples, n_features).
    f_obs : array-like, optional
        Indices of samples to visualize; defaults to all.
    true_label : array-like, optional
        Ground-truth anomaly labels of shape (n_samples, n_features).
    data : array-like, optional
        Multichannel time series of shape (n_samples, n_features), used for
        overlaying raw data and score-based anomaly markers.
    x : array-like, optional
        X-axis values. If None, integer indices are used. If datetime-like,
        the function automatically switches to a date axis.
    vmin, vmax : float, default=(-3, 3)
        Color limits for the anomaly score colormap.
    cmap : Colormap, optional
        Colormap for score matrices. If None, a default diverging map is used.
    list_anom_ind : list of int, optional
        Indices of features/sensors to highlight in the time-series panel.
    figsize : tuple, default=(15, 6)
        Figure size in inches.
    grid_spec : array-like, optional
        Height ratios for subplot layout. If None, all subplots have equal height.
    x_date : bool, default=False
        If True, the x-axis is formatted as a date axis (dd/mm HH:MM).
    show_plot : bool, default=True
        Whether to display the resulting figure.
    setup : tuple, optional
        Tuple (n_channel_per_sensor, n_sensor) enabling structural overlays
        (horizontal grid lines) on score matrices for multi-sensor setups.

    Notes
    -----
    - The function supports:
        * multiple score matrices displayed in stacked subplots,
        * contextual slicing when `x` contains datetime values,
        * ground-truth anomaly maps,
        * multichannel data with anomaly highlighting,
        * optional highlighting of anomalous sensor indices.
    - Rendering is internally modularized via helper functions to improve
      clarity and maintainability, while keeping the public API identical.

    Returns
    -------
    None
        The function creates the figure and optionally displays it.
    """

    # 1) Normalisation des entrées score / f_obs / cmap
    score_list, len_score, dim_score, n_score, f_obs, cmap = (
        auxvisu.aux_norm_score_inputs(score, f_obs=f_obs, cmap=cmap)
    )

    # 2) Préparation de x et des extents pour imshow
    x, x_flag, list_extent = auxvisu.aux_prepare_x_extent(x, f_obs, dim_score)

    # 3) Paramètres de layout
    n_fig, sharey, grid_spec = auxvisu.aux_compute_layout_params(
        n_score, dim_score, true_label, score2, data, grid_spec=grid_spec
    )

    # 4) Création figure / axes
    fig, ax = auxvisu.aux_create_score_figure(n_fig, sharey, grid_spec, figsize)

    # 5) Cas simple : une seule figure (n_fig == 1)
    if n_fig == 1:
        # un seul score, pas de true_label / score2 / data
        auxvisu.aux_plot_score_matrix(
            ax,
            score_list[0],
            f_obs,
            list_extent[0],
            vmin,
            vmax,
            cmap,
            title="score",
        )
        auxvisu.aux_overlay_setup_grid(ax, setup, len(f_obs))

        auxvisu.aux_finalize_figure(fig, show_plot=show_plot)
        return

    # 6) Cas général : plusieurs panneaux
    ind_ax = -1

    # 6.1: matrices de score (un axe par score principal)
    for n, score_ in enumerate(score_list):
        ind_ax += 1
        auxvisu.aux_plot_score_matrix(
            ax[ind_ax],
            score_,
            f_obs,
            list_extent[n],
            vmin,
            vmax,
            cmap,
            title="score",
        )

    # 6.2: seconde matrice de score
    if score2 is not None:
        ind_ax += 1
        auxvisu.aux_plot_score_matrix(
            ax[ind_ax],
            score2,
            f_obs,
            list_extent[0],
            vmin,
            vmax,
            cmap,
            title="score",
        )

    # 6.3: matrice de labels
    if true_label is not None:
        ind_ax += 1
        auxvisu.aux_plot_true_label_matrix(
            ax[ind_ax],
            true_label,
            f_obs,
            list_extent[0],
        )

    # 6.4: données temporelles + anomalies
    last_ax = None
    if data is not None:
        ind_ax += 1
        last_ax = ax[ind_ax]
        last_ax.set_title("data")

        colors = auxvisu.aux_build_data_colors(data, list_anom_ind=list_anom_ind)
        auxvisu.aux_plot_data_timeseries(
            last_ax,
            x,
            data,
            f_obs,
            dim_score[0],
            colors,
            lw=0.9,
        )
        # anomalies basées sur score[0]
        auxvisu.aux_overlay_score_anoms_on_data(
            last_ax,
            x,
            data,
            np.abs(score_list[0]),
            f_obs,
            dim_score[0],
            threshold=1.0,
        )

    # 6.5: overlay des labels vrais sur les données
    if (true_label is not None) and (data is not None) and (last_ax is not None):
        auxvisu.aux_overlay_true_label_on_data(last_ax, x, data, true_label, f_obs)

    # 6.6: axe temps (sur le dernier axe utilisé)
    if last_ax is None:
        # si pas de data, on applique au dernier axe utilisé pour les matrices
        last_ax = ax[ind_ax]
    auxvisu.aux_format_time_axis(last_ax, x_flag=x_flag, x_date=x_date)

    # 7) Grille éventuelle sur le premier axe (comme avant, uniquement dans le cas n_fig==1)
    #    -> si tu veux aussi la grille quand n_fig > 1, tu peux l’activer ici
    # if setup is not None:
    #     aux_overlay_setup_grid(ax[0], setup, len(f_obs))

    # 8) Finalisation figure
    auxvisu.aux_finalize_figure(fig, show_plot=show_plot)


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
    """
    Visualize uncertainty diagnostics for multivariate predictive models.

    This function plots observations, predictions, prediction intervals,
    aleatoric/epistemic uncertainty contributions, confidence-level scores,
    optional anomaly bounds, and context-based segmentations. It supports
    multi-output signals, multiple contextual partitions, residual mode, and
    both full UQ views and data-only views. The function preserves the original
    API and integrates with modular visualization helpers (aux_*).

    Parameters
    ----------
    y : array-like
        Ground-truth observations of shape (n_samples, n_dim).
    output : tuple or None
        UQ model outputs. Either (pred, var_A, var_E) or (pred, (var_A, var_E)),
        depending on `type_UQ`. Set to None in data-only mode.
    context : array-like, optional
        Context matrix used for splitting the plot by contextual dimension or
        highlighting contextual regions.
    size : tuple, default=(15, 5)
        Figure size in inches.
    f_obs : array-like, optional
        Indices of samples to display; defaults to all.
    name : str, default="UQplot"
        Figure suptitle.
    mode_res : bool, default=False
        If True, plot residuals instead of raw values.
    born : tuple of array-like, optional
        Lower and upper anomaly bounds for each dimension.
    born_bis : tuple of array-like, optional
        Secondary set of anomaly bounds.
    dim : int or list of int, default=0
        Target output dimensions to visualize.
    confidence_lvl : array-like, optional
        Precomputed confidence-level matrix. If None, it is computed internally.
    list_percent : list of float, default=[0.8, 0.9, 0.99, 0.999, 1]
        Confidence thresholds used to compute epistemic confidence levels.
    env : list of float, default=[0.95, 0.65]
        Default uncertainty envelopes for plotting.
    type_UQ : {"old", "var_A&E"}, default="old"
        Format specification of `output`.
    show_plot : bool, default=True
        Whether to display the final figure.
    with_colorbar : bool, default=False
        Whether to add a confidence-level colorbar.
    **kwarg :
        Additional parameters, including:
        - "ind_ctx": context values to include,
        - "split_ctx": context dimension used for splitting subplots,
        - "ylim": vertical limits,
        - "var_min": minimum (var_A, var_E) values,
        - "only_data": disable UQ & plot observations only,
        - "x": explicit x-axis values,
        - "ctx_attack": tuple defining contextual highlight rules,
        - "list_name_subset": labels for contextual annotations.

    Notes
    -----
    - This function acts as a high-level orchestrator and delegates rendering
      to modular visualization helpers (aux_plot_confiance, aux_plot_conf_score,
      aux_plot_line, aux_fill_between, etc.).
    - The input API is preserved for backward compatibility.

    Returns
    -------
    None
        The function creates a figure and optionally displays it.
    """

    if f_obs is None:
        f_obs = np.arange(len(y))
    f_obs = np.asarray(f_obs)

    # --- Extraction des options depuis kwarg (API inchangée) ---
    ind_ctx = kwarg.get("ind_ctx", None)
    split_ctx = kwarg.get("split_ctx", -1)
    ylim = kwarg.get("ylim", None)
    # compare_deg était lu mais jamais utilisé -> no-op conservé implicitement

    # Bornes mini sur var_A, var_E
    min_A, min_E = kwarg.get("var_min", (1e-6, 1e-6))

    # Only data / subset de contexte (pour ctx_attack)
    only_data = kwarg.get("only_data", False)
    list_name_subset = kwarg.get("list_name_subset", None)
    ctx_attack = kwarg.get("ctx_attack", None)

    if only_data:
        name = "Data"

    # Support x
    if "x" in kwarg:
        x = kwarg["x"]
    else:
        x = np.arange(len(y))
    x = np.asarray(x)

    # --- Préparation de pred, var_A, var_E selon type_UQ ---
    pred = var_A = var_E = None
    if output is not None:
        if type_UQ == "old":
            pred, var_A, var_E = output
        elif type_UQ == "var_A&E":
            pred, (var_A, var_E) = output
        else:
            raise ValueError(f"Unknown type_UQ '{type_UQ}'.")

        var_A = np.asarray(var_A)
        var_E = np.asarray(var_E)
        var_E[var_E < min_E] = min_E
        var_A[var_A < min_A] = min_A

    # --- Dimensions & contextes ---
    f_obs_full = np.copy(f_obs)

    if isinstance(dim, int):
        dim_list = [dim]
    else:
        dim_list = list(dim)
    n_dim = len(dim_list)

    n_ctx = 1
    list_ctx_ = [None]
    if split_ctx > -1 and context is not None:
        if ind_ctx is None:
            list_ctx_ = list(set(context[f_obs, split_ctx]))
        else:
            list_ctx_ = ind_ctx
        n_ctx = len(list_ctx_)

    # --- Figure et axes ---
    fig, axs = plt.subplots(n_dim, n_ctx, sharex=True, figsize=size)

    # --- Confidence level : si non fourni, calcul global une fois ---
    if (confidence_lvl is None) and (output is not None):
        confidence_lvl, _ = UQ_proc.compute_Epistemic_score(
            (var_A, var_E),
            type_UQ="var_A&E",
            pred=pred,
            list_percent=list_percent,
            params_=None,
        )
    # sinon : on utilise la matrice fournie en argument

    label = None  # servira pour la colorbar éventuelle

    # --- Boucle principale : dimensions x contextes ---
    for idx_dim, d in enumerate(dim_list):
        for idx_ctx in range(n_ctx):
            ax = auxvisu._get_panel_ax(axs, n_dim, n_ctx, idx_dim, idx_ctx)

            # Filtrage par contexte si demandé
            if split_ctx > -1 and context is not None:
                mask_ctx = context[f_obs_full, split_ctx] == list_ctx_[idx_ctx]
                f_obs_ctx = f_obs_full[mask_ctx]
            else:
                f_obs_ctx = f_obs_full

            if f_obs_ctx.size == 0:
                continue

            # Mode "only_data" : pas d'UQ, juste la série + scatter
            if only_data:
                # scatter
                auxvisu.aux_plot_line(
                    ax,
                    f_obs_ctx,
                    y[f_obs_ctx, d],
                    config={
                        "color": "black",
                        "marker": "x",
                        "markersize": 10,
                        "linestyle": "none",
                        "linewidth": 1.0,
                        "label": "observation",
                    },
                )
                # ligne pointillée
                auxvisu.aux_plot_line(
                    ax,
                    f_obs_ctx,
                    y[f_obs_ctx, d],
                    config={
                        "color": "darkgreen",
                        "linestyle": ":",
                        "linewidth": 0.7,
                        "alpha": 1.0,
                        "zorder": -4,
                    },
                )
                if ylim is not None:
                    ax.set_ylim(ylim[0], ylim[1])

            else:
                # Préparation bornes par dimension
                born_ = None
                if born is not None:
                    born_ = (born[0][f_obs_ctx, d], born[1][f_obs_ctx, d])

                born_bis_ = None
                if born_bis is not None:
                    born_bis_ = (born_bis[0][f_obs_ctx, d], born_bis[1][f_obs_ctx, d])

                # Appel au helper UQ principal
                auxvisu.aux_plot_confiance(
                    ax=ax,
                    y=y[f_obs_ctx, d],
                    pred=pred[f_obs_ctx, d],
                    var_A=var_A[f_obs_ctx, d],
                    var_E=var_E[f_obs_ctx, d],
                    born=born_,
                    born_bis=born_bis_,
                    env=env,
                    x=x[f_obs_ctx],
                    mode_res=mode_res,
                    **kwarg,
                )

                # Scores de confiance & legends
                if confidence_lvl is not None:
                    label = [str(i) for i in list_percent]
                    label.append(">1")
                    auxvisu.aux_plot_conf_score(
                        ax,
                        x[f_obs_ctx],
                        pred[f_obs_ctx, d],
                        confidence_lvl[f_obs_ctx, d],
                        label=label,
                        mode_res=mode_res,
                    )

            # Overlay de contexte "attaque" si demandé
            if ctx_attack is not None:
                # ctx_attack = (dim_ctx, ctx_val)
                dim_ctx, ctx_val = ctx_attack
                ylim_local = ylim
                if ylim_local is None:
                    ylim_local = (y.min(), y.max())

                if ctx_val == -1:
                    # coloration par catégories de contexte
                    list_ctx = list(set(context[f_obs_ctx, dim_ctx]))
                    if list_name_subset is None:
                        list_name_subset = list_ctx
                    cmap_ctx = plt.get_cmap("jet", len(list_name_subset))
                    for i in list_ctx:
                        mask_ctx_val = context[f_obs_ctx, dim_ctx] == i
                        auxvisu.aux_fill_between(
                            ax,
                            f_obs_ctx,
                            np.full_like(f_obs_ctx, ylim_local[0], dtype=float),
                            np.full_like(f_obs_ctx, ylim_local[1], dtype=float),
                            where=mask_ctx_val,
                            config={
                                "color": cmap_ctx(int(i)),
                                "alpha": 0.2,
                            },
                        )
                else:
                    # un seul contexte mis en évidence
                    mask_ctx_val = context[f_obs_ctx, dim_ctx] == 1
                    auxvisu.aux_fill_between(
                        ax,
                        f_obs_ctx,
                        np.full_like(f_obs_ctx, ylim_local[0], dtype=float),
                        np.full_like(f_obs_ctx, ylim_local[1], dtype=float),
                        where=mask_ctx_val,
                        config={
                            "color": "yellow",
                            "alpha": 0.2,
                        },
                    )

            # Masquer les ticks y pour les contextes > 0 (comme avant)
            if idx_ctx != 0:
                ax.set_yticklabels([])

            # Légende "fantôme" pour les ctx_attack (list_name_subset)
            if ctx_attack is not None and list_name_subset is not None:
                ylim_local = ylim
                if ylim_local is None:
                    ylim_local = (y.min(), y.max())
                cmap_ctx = plt.get_cmap("jet", len(list_name_subset))
                for i in range(len(list_name_subset)):
                    auxvisu.aux_fill_between(
                        ax,
                        np.array([]),
                        np.array([]),
                        np.array([]),
                        where=None,
                        config={
                            "color": cmap_ctx(i),
                            "alpha": 0.08,
                            "label": list_name_subset[int(i)],
                        },
                    )

    # --- Mise en forme globale : titre / layout / légende / colorbar ---
    plt.suptitle(name)
    plt.subplots_adjust(
        wspace=0.03, hspace=0.03, left=0.1, bottom=0.22, right=0.90, top=0.8
    )
    plt.legend(frameon=True, ncol=6, fontsize=10, bbox_to_anchor=(0.5, 0, 0.38, -0.11))

    # Colorbar pour les niveaux de confiance
    if label is not None:
        cmap_vals = [plt.get_cmap("RdYlGn_r", 7)(i) for i in np.arange(len(label))]
        bounds = np.concatenate(
            [[0], np.cumsum(np.abs(np.array(list_percent) - 1) + 0.1)]
        )
        bounds = 10 * bounds / bounds.max()

        if with_colorbar:
            cmap_cb = mpl.colors.ListedColormap(cmap_vals)
            norm = mpl.colors.BoundaryNorm(bounds, cmap_cb.N)
            color_ls = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_cb)
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


# Display of data curve with mean and variance.


def aux_get_var_color_sets():
    """
    Return the color sets used for percentile envelope visualization
    in `plot_var`.

    Returns
    -------
    color_full : list of tuple
        Colors for filled regions between percentile curves.
    color_full2 : list of tuple
        Colors for percentile boundary lines.
    """
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

    return color_full, color_full2


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
    """
    Plot empirical variance envelopes around a univariate time series.

    This function builds a set of percentile-based envelopes from the provided
    variance and overlays them on the original (or residual) series together
    with anomaly markers. It visualizes how the variance translates into
    coverage levels for a given component of a multivariate signal.

    Parameters
    ----------
    Y : array-like
        Ground-truth series of shape (n_samples, n_dim).
    data_full : array-like
        Reference series used to construct the envelopes, same shape as Y.
    variance : array-like
        Point-wise variance of shape (n_samples, n_dim) for the selected
        component.
    impact_anom : array-like, optional
        Anomaly impact indicator of shape (n_samples, n_dim). Non-zero entries
        are flagged as anomalies.
    anom : array-like, optional
        Unused placeholder kept for backward compatibility.
    f_obs : array-like, optional
        Indices of samples to visualize. If None, all samples are used.
    dim : tuple, default=(400, 20, 3)
        Unused placeholder describing (n_samples, n_time, n_groups). Kept for
        backward compatibility.
    g : int, default=0
        Index of the dimension (component) to plot.
    res_flag : bool, default=False
        If True, envelopes are computed around `data_full - data_full`
        (i.e. residuals), otherwise around `data_full`.
    fig_s : tuple, default=(20, 3)
        Figure size in inches.
    title : str, optional
        Figure title.
    ylim : tuple, optional
        Manual y-axis limits (ymin, ymax). If None, limits are inferred from
        the outer envelopes.

    Returns
    -------
    per : list of np.ndarray
        List of envelope curves (one array per percentile in `per_list`).
    per_list : list of float
        Percentile levels used to build the envelopes.
    """
    import scipy.stats as st

    def add_noise(data, noise_mult, noise_add):
        return (data * (1 + noise_mult)) + noise_add

    # Indices observés
    if f_obs is None:
        f_obs = np.arange(Y.shape[0])
    f_obs = np.asarray(f_obs)

    step = g  # dimension à tracer

    # Figure / axe
    fig, ax = plt.subplots(figsize=fig_s)
    if title is not None:
        ax.set_title(title)

    # Série de base : résidus ou données brutes
    res = data_full * 0
    if res_flag:
        res = data_full

    # Définitions des niveaux et des couleurs
    ni = [100, 98, 95, 80, 50]
    color_full, color_full2 = aux_get_var_color_sets()

    per_list = [0.01, 1, 2.5, 10, 25, 75, 90, 97.5, 99, 99.99]
    per = []

    # Construction des enveloppes (percentiles)
    for p in per_list:
        noise = st.norm.ppf(p / 100.0, loc=0.0, scale=np.sqrt(variance))
        per.append(add_noise(data_full - res, 0.0, noise))

    x_idx = np.arange(len(f_obs))

    # Zones entre percentiles
    for i in range(len(per) - 1):
        auxvisu.aux_fill_area(
            ax,
            x_idx,
            per[i][f_obs, step],
            per[i + 1][f_obs, step],
            config={
                "color": color_full[i],
                "alpha": 0.20,
                "label": None,
            },
        )

    # Courbes des percentiles + prototypes de légende
    for i in range(len(per)):
        auxvisu.aux_plot_line(
            ax,
            x_idx,
            per[i][f_obs, step],
            config={
                "color": color_full2[i],
                "linewidth": 0.5,
                "alpha": 0.40,
            },
        )
        if i > 4:
            # Handles de légende "fantômes" pour les niveaux de couverture
            auxvisu.aux_fill_area(
                ax,
                np.array([]),
                np.array([]),
                np.array([]),
                config={
                    "color": color_full2[i],
                    "alpha": 0.20,
                    "label": f"{ni[9 - i]}% Coverage",
                },
            )

    # Série observée
    series = Y[f_obs, step] - res[f_obs, step]
    auxvisu.aux_plot_line(
        ax,
        x_idx,
        series,
        config={
            "color": "black",
            "linewidth": 1.5,
            "marker": "o",
            "markersize": 3,
            "label": "Series",
        },
    )

    # Anomalies (impact_anom)
    if impact_anom is not None:
        flag = impact_anom[f_obs, step] != 0
        if flag.any():
            auxvisu.aux_plot_line(
                ax,
                x_idx[flag],
                series[flag],
                config={
                    "color": "red",
                    "marker": "X",
                    "markersize": 10,
                    "linestyle": "none",
                    "label": "Anom",
                    "alpha": 0.8,
                },
            )

    # Gestion des axes (en s'appuyant sur aux_adjust_axes)
    y_for_axes = [per[0][f_obs, step], per[-1][f_obs, step], series]
    auxvisu.aux_adjust_axes(
        ax,
        x_idx,
        y_for_axes,
        ylim=ylim,
        x_lim=(0, len(f_obs)),
        margin=0.05,
        x_margin=0.0,
    )

    ax.legend(ncol=7, fontsize=14)
    fig.tight_layout()
    plt.show()

    return per, per_list
