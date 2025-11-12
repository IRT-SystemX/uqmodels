import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple, Optional
from matplotlib.ticker import FormatStrFormatter
from abc import ABC, abstractmethod
from abench.store.store import read,write
import abench.store.api as api
from abench.store.api import extract_tabular_data
from typing import Any, Dict, Iterable, List, Tuple, Union, Optional

# Encapsulated visualitation format :


def print_agg_result(storing,agg_name,component_name,list_metrics):
    dictperf_agg = api.get_dictperf(storing,agg_name=agg_name)
    dictperf_agg_component = dictperf_agg[component_name]     
    for metric in list_metrics:
        metric_name = metric.name
        dict_res = dictperf_agg_component[metric_name]

        print(component_name,
            metric_name,
            "Train",
            np.round(dict_res['mean_train'], 3),
            "±",
            np.round(dict_res['std_train'], 3),
            "| TEST",
            np.round(dict_res['mean_test'], 3),
            "±",
            np.round(dict_res['std_test'], 3),
        )



def scatter_result(
    Metrics_performance,
    candidates,
    perf_data,
    colors,
    xlim=None,
    ylim=None,
    names=None,
    figsize=(15, 15),
):
    """[summary]

    Args:
        Metrics_performance ([type]): [description]
        candidates ([type]): [description]
        perf_data ([type]): [description] sub dictionary of dict_perf containing a specific aggregation.
        colors ([type]): [description]
    """

    # Filter candidates to only those that exist in the data
    valid_candidates = [c for c in candidates if c in perf_data]
    if len(valid_candidates) != len(candidates):
        missing = set(candidates) - set(valid_candidates)
        print(f"Warning: Missing performance data for candidates: {missing}")

    if names is None:
        names = valid_candidates

    X_loc = np.zeros((len(valid_candidates), len(Metrics_performance) * 2))

    for n, candidate in enumerate(valid_candidates):
        for nn, m in enumerate(Metrics_performance):
            try:
                perf = perf_data[candidate][m]
                X_loc[n][nn * 2] = perf['mean_test']
                X_loc[n][nn * 2 + 1] = perf['std_test']
            except KeyError as e:
                print(f"Warning: Missing metric {m} for candidate {candidate}")
                X_loc[n][nn * 2] = np.nan
                X_loc[n][nn * 2 + 1] = np.nan

    # Filter out any NaN values
    valid_indices = ~np.isnan(X_loc[:, 0])
    X_loc = X_loc[valid_indices]
    valid_candidates = [c for i, c in enumerate(valid_candidates) if valid_indices[i]]
    names = [n for i, n in enumerate(names) if valid_indices[i]] if names else None

    if len(X_loc) == 0:
        raise ValueError("No valid performance data available for plotting")

    x = X_loc[:, 0]
    y = X_loc[:, 2]
    x_error = X_loc[:, 1] / 2
    y_error = X_loc[:, 3] / 2

    fig, ax0 = plt.subplots(nrows=1, sharex=True, figsize=figsize)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    for i in range(len(X_loc)):
        color = colors[i]
        ax0.scatter(x[i], y[i], label=names[i], color=color)
        ax0.errorbar(
            x[i],
            y[i],
            xerr=x_error[i],
            yerr=y_error[i],
            fmt="none",
            lw=1,
            color=color,
            capsize=2,
        )

    if xlim:
        ax0.set_xlim(xlim[0], xlim[1])
    if ylim:
        ax0.set_ylim(ylim[0], ylim[1])

    ax0.set_xlabel(Metrics_performance[0])
    ax0.set_ylabel(Metrics_performance[1])
    ax0.legend(loc=0, ncol=2, fontsize=8)
    plt.show()


def barplot_result(
    Metrics_performance,
    candidates,
    dict_perf,
    colors,
    xlim=None,
    ylim=None,
    names=None,
    target=(None, None),
    figsize=(15, 15),
    save_path=None,
    loc=None,
):
    """[summary]

    Args:
        Metrics_performance ([type]): [description]
        candidates ([type]): [description]
        dict_perf ([type]): [description]
        colors ([type]): [description]
    """

    if names is None:
        names = candidates

    X_loc = np.zeros((len(candidates), len(Metrics_performance) * 2))
    print(xlim)
    for n, candidate in enumerate(candidates):
        for nn, m in enumerate(Metrics_performance):
            perf = dict_perf[candidate][m]
            if not (loc is None):
                X_loc[n][nn * 2] = perf[2, loc]
                X_loc[n][nn * 2 + 1] = perf[3, loc]
            else:
                X_loc[n][nn * 2] = np.mean(perf[2])
                X_loc[n][nn * 2 + 1] = np.mean(perf[3])

    y = np.arange(len(candidates))[::-1]

    plt.figure(figsize=figsize)
    plt.style.use("seaborn-whitegrid")
    for k, metric in enumerate(Metrics_performance):
        ax = plt.subplot(1, len(Metrics_performance), k + 1)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        plt.title(metric, fontsize=24)
        if k == 0:
            plt.yticks(y, names, fontsize=18)
        else:
            plt.yticks(y, [])

        for n, candidate in enumerate(candidates):
            plt.errorbar(
                X_loc[n, k * 2],
                y[n],
                xerr=X_loc[n, k * 2 + 1] / 2,
                fmt="ok",
                lw=3,
                marker="d",
                markersize=12,
                capsize=10,
                color=colors[n],
            )
        if not (target[0] is None):
            plt.vlines(
                target[0],
                y.min() - 0.5,
                y.max() + 0.5,
                color="red",
                ls="--",
                label="target",
            )
        plt.xticks(fontsize=16)
        if not (xlim is None):
            if not (xlim[k] is None):
                plt.xlim(xlim[k][0], xlim[k][1])
    plt.tight_layout()
    plt.yticks(y, [], fontsize=22)
    if not (save_path is None):
        plt.savefig(save_path)
    plt.show()


def barplot_ctx(
    Metrics_performance,
    candidates,
    dict_perf,
    colors,
    xlim=None,
    ylim=None,
    names=None,
    list_names_ctx=None,
    target=(None, None),
    figshape=None,
    figsize=(15, 15),
    save_path=None,
):
    """Perform multi barplot visualisation"""

    if figshape == None:
        carre = int(np.ceil(np.sqrt(len(Metrics_performance))))
        figshape = (carre, carre)

    if names is None:
        names = candidates

    plt.figure(figsize=figsize)
    plt.style.use("seaborn-whitegrid")
    for nn, metrics in enumerate(Metrics_performance):
        ax = plt.subplot(figshape[0], figshape[1], nn + 1)

        perf_res = dict_perf[candidates[0]][metrics][0]
        if type(perf_res) in [list, np.ndarray]:
            n_ctx = len(perf_res)
        else:
            n_ctx = 1

        y = np.arange(n_ctx)
        set_off_ctx = 0.5 / len(candidates)
        X_loc = np.zeros((len(candidates), 2, n_ctx))
        for n, candidate in enumerate(candidates):
            perf = dict_perf[candidate][metrics]
            X_loc[n][0] = perf[2]
            X_loc[n][1] = perf[3]
        for n, candidate in enumerate(candidates):
            for i_ctx in range(n_ctx):
                if i_ctx == 0:
                    plt.bar(
                        y[i_ctx] + set_off_ctx * (n),
                        X_loc[n, 0, i_ctx],
                        width=set_off_ctx * 0.9,
                        color=colors[n],
                        label=candidate,
                    )
                else:
                    plt.bar(
                        y[i_ctx] + set_off_ctx * (n),
                        X_loc[n, 0, i_ctx],
                        width=set_off_ctx * 0.9,
                        color=colors[n],
                    )

                plt.errorbar(
                    y[i_ctx] + set_off_ctx * (n),
                    X_loc[n, 0, i_ctx],
                    yerr=X_loc[n, 1, i_ctx] / 2,
                    fmt="ok",
                    lw=2,
                    marker="d",
                    markersize=1,
                    capsize=5,
                    zorder=10,
                    color="black",
                )
        if list_names_ctx is None:
            names_ctx = y
        else:
            names_ctx = list_names_ctx[nn]
        if nn == 0:
            leg = ax.legend(loc=0, fontsize=10, framealpha=0.5)
            frame = leg.get_frame()
            frame.set_facecolor("gray")
        # plt.yticks(y, [], fontsize=20)
        ax.set_xticks(y + set_off_ctx, names_ctx, fontsize=14)
        ax.set_ylabel(metrics, fontsize=14)

        if not (ylim is None):
            if not (ylim[0] is None):
                ax.set_ylim(ylim[k][0], ylim[k][1])

    if not (target[0] is None):
        plt.vlines(
            target[0],
            y.min() - 0.2,
            y.max() + 1.2,
            color="red",
            ls="--",
            label="target",
        )

    if not (save_path is None):
        plt.savefig(save_path)
    plt.tight_layout()
    plt.show()


def barplot_result_ctx(
    Metrics_performance,
    candidates,
    dict_perf,
    colors,
    xlim=None,
    ylim=None,
    names=None,
    target=(None, None),
    figsize=(15, 15),
    save_path=None,
):
    """[summary]

    Args:
        Metrics_performance ([type]): [description]
        candidates ([type]): [description]
        dict_perf ([type]): [description]
        colors ([type]): [description]
    """

    if names is None:
        names = candidates

    n_ctx = int(len(Metrics_performance) / 2)

    set_off_ctx = 0.5 / n_ctx

    X_loc = np.zeros((len(candidates), len(Metrics_performance) * 2))

    for n, candidate in enumerate(candidates):
        for nn, m in enumerate(Metrics_performance):
            perf = dict_perf[candidate][m]
            X_loc[n][nn * 2] = perf[2]
            X_loc[n][nn * 2 + 1] = perf[3]

    y = np.arange(len(candidates))[::-1]

    plt.figure(figsize=figsize)
    plt.style.use("seaborn-whitegrid")
    plt.subplot(1, 2, 1)
    plt.title("PINAW (sharpness)", fontsize=28)
    name_ctx = ["All", "low-var", "mid-var", "high-var"]
    for n, candidate in enumerate(candidates):
        for m in range(n_ctx):
            if n == 0:
                plt.errorbar(
                    X_loc[:, m * 2][n],
                    y[n] + set_off_ctx * (m),
                    xerr=X_loc[:, m * 2 + 1][n] / 2,
                    fmt="ok",
                    lw=2,
                    marker="d",
                    markersize=1,
                    capsize=5,
                    color=colors[n],
                    label=name_ctx[m],
                )
            else:
                plt.errorbar(
                    X_loc[:, m * 2][n],
                    y[n] + set_off_ctx * (m),
                    xerr=X_loc[:, m * 2 + 1][n] / 2,
                    fmt="ok",
                    lw=2,
                    marker="d",
                    markersize=1,
                    capsize=5,
                    color=colors[n],
                )

    if not (target[0] is None):
        plt.vlines(
            target[0],
            y.min() - 0.2,
            y.max() + 1.2,
            color="red",
            ls="--",
            label="target",
        )
    plt.yticks(y, names, fontsize=24)
    plt.xlabel("← best", fontsize=24)
    plt.tight_layout()
    plt.xticks(fontsize=15)
    plt.legend(loc=1, fontsize=15)
    plt.ylim(y.min() - 0.2, y.max() + 0.5)

    plt.subplot(1, 2, 2)
    plt.title("PICP (Coverage)", fontsize=24)
    for n, candidate in enumerate(candidates):
        for m in range(n_ctx):
            ind = m + n_ctx
            plt.errorbar(
                X_loc[n, ind * 2],
                y[n] + set_off_ctx * (m),
                xerr=X_loc[n, ind * 2 + 1] / 2,
                fmt="ok",
                lw=2,
                marker="d",
                markersize=6,
                capsize=6,
                color=colors[m],
            )
    if not (target[1] is None):
        plt.vlines(
            target[1],
            y.min() - 0.2,
            y.max() + 1.2,
            color="red",
            ls="--",
            label="target",
        )

    plt.xticks(fontsize=20)
    plt.xlabel("best →", fontsize=24)
    plt.tight_layout()
    plt.ylim(y.min() - 0.2, y.max() + 0.5)
    plt.yticks(y, [], fontsize=20)
    if not (save_path is None):
        plt.savefig(save_path)
    plt.show()

def print_flexible_latex_table(data_dict, caption="Metrics Table", label="tab:metrics"):
    """
    Print a LaTeX tabular from a nested dict.
    Supports:
    - 1 or 2-level row keys (strings or tuples)
    - 1 or 2-level column keys (strings or tuples)
    """

    # Detect row levels
    row_keys = list(data_dict.keys())
    row_level = 2 if isinstance(row_keys[0], tuple) else 1

    # Extract and flatten column keys
    all_columns = set()
    for v in data_dict.values():
        all_columns.update(v.keys())

    col_level = 2 if isinstance(next(iter(all_columns)), tuple) else 1

    # Sort and organize column headers
    if col_level == 1:
        columns = sorted(all_columns)
        col_headers_top = ["\\textbf{Group}"] * row_level + [f"\\textbf{{{c}}}" for c in columns]
        col_headers_bottom = None
    else:
        columns = sorted(all_columns)
        level1 = [c[0] for c in columns]
        col_headers_top = ["\\textbf{Group}"] * row_level + [f"\\multicolumn{{1}}{{c}}{{\\textbf{{{c}}}}}" for c in level1]
        try:
            level2 = [c[1] for c in columns]
        except:
            level2 = [c[0] for c in columns]
        col_headers_bottom = [""] * row_level + [f"\\textbf{{{c}}}" for c in level2]
        
    

    print("\\begin{table}[h!]")
    print("\\centering")
    print(f"\\begin{{tabular}}{{{'|c'* (row_level + len(columns))}|}}")
    print("\\hline")

    # Top header
    print(" & ".join(col_headers_top) + " \\\\")
    if col_headers_bottom:
        print("\\hline")
        print(" & ".join(col_headers_bottom) + " \\\\")

    print("\\hline")

    # Rows
    for row_key in row_keys:
        if row_level == 1:
            row_header = [f"\\textbf{{{row_key}}}"]
        else:
            row_header = [f"\\textbf{{{row_key[0]}}}", f"{row_key[1]}"]

        row_values = []
        for col in columns:
            val = data_dict[row_key].get(col, "")
            if(type(val) in [tuple,np.ndarray]):
                row_values.append(str(val[0])+'$\pm$'+str(val[1]))
            else:
                row_values.append(str(val))
        print(" & ".join(row_header + row_values) + " \\\\")
        print("\\hline")

    print("\\end{tabular}")
    print(f"\\caption{{{caption}}}")
    print(f"\\label{{{label}}}")
    print("\\end{table}")


def plot_model_metrics(
    perf_dict: dict[str, dict[str, tuple[float, float]]],
    *,
    metric_order: list[str] = None,
    figsize: tuple[int, int] = (10, 6),
    bar_width: float = 0.8,
    capsize: float = 0.15,
    legend_fontsize: int = 12,
    legend_title_fontsize: int = 13,
    xrotation: int = 30,           # <-- NEW: rotation angle in degrees
) -> None:
    """
    Grouped bar-plot (mean ± std) with boxed legend and rotated x-labels.

    Parameters
    ----------
    perf_dict : {"Model": {"metric": (mean, std)}}
        Nested performance dictionary.
    metric_order : list[str], optional
        Explicit order of metrics on the x-axis.
    figsize, bar_width, capsize
        Usual styling parameters.
    legend_fontsize / legend_title_fontsize
        Font sizes for the legend.
    xrotation : int
        Angle (degrees) for metric labels on the x-axis.
    """
    # ---------- tidy DataFrame -----------------------------------------
    records = [
        {"model": m, "metric": k, "mean": mean, "std": std}
        for m, metrics in perf_dict.items()
        for k, (mean, std) in metrics.items()
    ]
    df = pd.DataFrame(records)

    if metric_order is not None:
        df["metric"] = pd.Categorical(df["metric"],
                                      categories=metric_order,
                                      ordered=True)

    # ---------- plot ----------------------------------------------------
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        x="metric", y="mean", hue="model", data=df,
        errorbar=None, width=bar_width,
    )

    # manual ±std error bars
    for bar, (_, row) in zip(ax.patches, df.iterrows()):
        x_center = bar.get_x() + bar.get_width() / 2
        ax.errorbar(
            x=x_center, y=row["mean"], yerr=row["std"],
            fmt="none", ecolor="black", elinewidth=1,
            capsize=capsize * bar_width * len(df["model"].unique()),
        )

    # axis labels & title
    ax.set_ylabel("Performance (mean ± std)")
    ax.set_xlabel("")
    ax.set_title("Model performance per metric")

    # ---------- rotate x-tick labels -----------------------------------
    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation=xrotation,
                       ha="right",
                       rotation_mode="anchor")

    # ---------- boxed legend -------------------------------------------
    legend = ax.legend(
        title="Model",
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
        frameon=True, fancybox=True, framealpha=1, edgecolor="black",
        loc="upper left", bbox_to_anchor=(1.02, 1.0)
    )
    legend.get_frame().set_linewidth(1)

    plt.tight_layout()
    plt.show()

    ####################################
    # New Generic visualisation function 
    ####################################

    from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Helpers ----------

def _union_models_in_order(metrics_to_df: Dict[str, pd.DataFrame]) -> list:
    """Return a stable union of model names (columns) across all metric dataframes."""
    models = []
    for df in metrics_to_df.values():
        for c in df.columns:
            if c not in models:
                models.append(c)
    return models

def _center_by_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Row-center scores to remove per-dataset difficulty (subtract row mean)."""
    row_means = df.mean(axis=1, skipna=True)
    return df.sub(row_means, axis=0)

def _std_per_model(df: pd.DataFrame) -> pd.Series:
    """Sample standard deviation per model across datasets (rows)."""
    return df.std(axis=0, ddof=1, skipna=True)

def _ci_per_model(df: pd.DataFrame, alpha: float = 0.05) -> pd.Series:
    """Two-sided Confidence Interval half-width per model for the mean across datasets."""
    n = df.notna().sum(axis=0)
    s = _std_per_model(df)
    se = s / np.sqrt(n.replace(0, np.nan))
    try:
        from scipy.stats import t
        crit = t.ppf(1 - alpha/2, df=(n - 1))
    except Exception:
        crit = 1.96 if abs(alpha - 0.05) < 1e-9 else 1.96
    return crit * se

def _error_bars(
    df: pd.DataFrame,
    error_type: str = "std",
    alpha: float = 0.05,
    normalize_error: str = "none"
) -> pd.Series:
    """Compute error bars per model according to requested mode and normalization."""
    if normalize_error not in {"none", "dataset_centered"}:
        raise ValueError("normalize_error must be 'none' or 'dataset_centered'.")
    work = df if normalize_error == "none" else _center_by_dataset(df)

    if error_type.lower() == "std":
        return _std_per_model(work)
    elif error_type.lower() == "ci":
        return _ci_per_model(work, alpha=alpha)
    else:
        raise ValueError("error_type must be 'std' or 'ci'.")

# ---------- Main Function ----------

def plot_grouped_bars_with_errors(
    metrics_to_df: Dict[str, pd.DataFrame],
    figsize: Tuple[int, int] = (12, 6),
    capsize: int = 5,
    rotation: int = 0,
    legend_rotation: int = 0,
    ylim: Optional[Tuple[float, float]] = None,
    title: str = "Grouped barplot",
    y_label: str = "Score",
    cmap=plt.cm.tab10,
    bar_gap: float = 0.15,
    normalize_error: str = "none",
    error_type: str = "std",
    alpha: float = 0.05,
    legend_alias: Optional[Dict[str, str]] = None,
    swap_axes: bool = False  # NEW PARAMETER
):
    """
    Create a grouped barplot from {metric_name: DataFrame}.

    Parameters
    ----------
    swap_axes : bool
        - False (default): Groups = metrics, sub-bars = models.
        - True : Groups = models, sub-bars = metrics (swap roles).
    """
    if not metrics_to_df:
        raise ValueError("metrics_to_df is empty.")

    metrics = list(metrics_to_df.keys())
    models_order = _union_models_in_order(metrics_to_df)

    # Precompute means and error bars
    means_per_metric = {}
    errs_per_metric = {}
    for m in metrics:
        df = metrics_to_df[m].copy().reindex(columns=models_order)
        means_per_metric[m] = df.mean(axis=0, skipna=True)
        errs_per_metric[m]  = _error_bars(df, error_type, alpha, normalize_error)

    # ---- Handle swapping ----
    if not swap_axes:
        # Default: X = metrics, group by metric
        groups = metrics
        subgroups = models_order
        colors = cmap(np.linspace(0, 1, len(subgroups)))

        def get_height(metric, model):
            return means_per_metric[metric].get(model, np.nan)

        def get_error(metric, model):
            return errs_per_metric[metric].get(model, np.nan)
    else:
        # Swap: X = models, group by model
        groups = models_order
        subgroups = metrics
        colors = cmap(np.linspace(0, 1, len(subgroups)))

        # Reorganize data: treat models as top-level group
        def get_height(model, metric):
            return means_per_metric[metric].get(model, np.nan)

        def get_error(model, metric):
            return errs_per_metric[metric].get(model, np.nan)

    # ---- Plot geometry ----
    n_groups = len(groups)
    n_subgroups = len(subgroups)
    group_width = 1.0 - bar_gap
    bar_width = group_width / max(n_subgroups, 1)
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw bars
    for j, sg in enumerate(subgroups):
        offsets = (j - (n_subgroups - 1) / 2) * bar_width
        xpos = x + offsets
        heights = []
        yerrs = []
        for g in groups:
            if not swap_axes:
                heights.append(get_height(g, sg))
                yerrs.append(get_error(g, sg))
            else:
                heights.append(get_height(g, sg))  # swapped roles
                yerrs.append(get_error(g, sg))

        ax.bar(
            xpos,
            heights,
            width=bar_width,
            yerr=yerrs,
            capsize=capsize,
            label=legend_alias.get(sg, sg) if legend_alias else sg,
            color=colors[j],
            edgecolor="black",
            linewidth=0.7,
        )

    # ---- Labels ----
    # X tick labels
    xtick_labels = [legend_alias.get(g, g) if legend_alias else g for g in groups]
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=rotation, ha="center")

    # Y label
    ax.set_ylabel(y_label)

    # Title
    mode_str = "std" if error_type == "std" else f"CI {int((1-alpha)*100)}%"
    norm_str = "raw" if normalize_error == "none" else "dataset-centered"
    ax.set_title(f"{title}\nError bars = {mode_str} ({norm_str})")

    if ylim:
        ax.set_ylim(*ylim)

    # Legend
    legend = ax.legend(title="Metrics" if swap_axes else "Models", frameon=False)
    for text in legend.get_texts():
        text.set_rotation(legend_rotation)

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig, ax


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def plot_grid_mean_std(
    means,
    stds=None,
    *,
    # --- couleurs / échelle ---
    cmap="viridis",
    vmin=None, vmax=None,
    color_scale="mean",   # "mean" | "mean_pm_std" | "custom"
    std_mult=1.0,
    nan_color=(0.9, 0.9, 0.9, 1.0),
    show_colorbar=True,
    colorbar_pad=0.02,
    colorbar_label=None,
    # --- mise en page ---
    figsize_per_cell=(2.2, 2.2),
    suptitle=None, 
    suptitle_fontsize=12,
    # --- 4 listes de noms ---
    meta_col_headers=None,   # len = g_l (étiquettes X pour la 1re ligne de matrices)
    meta_row_headers=None,   # len = g_h (étiquettes Y pour la 1re colonne de matrices)
    col_labels=None,         # len = l   (labels X internes)
    row_labels=None,         # len = h   (labels Y internes)
    axis_label_fontsize=11,
    axis_label_pad=6,
    # --- ticks ---
    show_axis_on="outer",    # "outer" | "all" | "none"
    tick_fontsize=8,
    xtick_rotation=0,
    ytick_rotation=0,
    # --- annotations numériques ---
    annotate=True,
    value_fmt="{:.2f}",
    annotate_fontsize=8,
    vmin_mirror=False):
    """
    Affiche une grille (g_h x g_l) de heatmaps des moyennes (et stds optionnels).

    Nomenclature :
      - meta_row_headers[i] -> ylabel sur axes[i, 0] (1re colonne de la grille)
      - meta_col_headers[j] -> xlabel sur axes[0, j] (1re ligne   de la grille)
      - col_labels          -> ticks X internes (taille l)
      - row_labels          -> ticks Y internes (taille h)
    """
    means = np.asarray(means)
    if means.ndim != 4:
        raise ValueError("`means` doit avoir la forme (g_h, g_l, h, l).")
    if stds is not None:
        stds = np.asarray(stds)
        if stds.shape != means.shape:
            raise ValueError("`stds` doit avoir la même forme que `means` ou être None.")

    g_h, g_l, h, l = means.shape

    # validations
    if meta_col_headers is not None and len(meta_col_headers) != g_l:
        raise ValueError("`meta_col_headers` doit avoir longueur g_l.")
    if meta_row_headers is not None and len(meta_row_headers) != g_h:
        raise ValueError("`meta_row_headers` doit avoir longueur g_h.")
    if col_labels is not None and len(col_labels) != l:
        raise ValueError("`col_labels` doit avoir longueur l.")
    if row_labels is not None and len(row_labels) != h:
        raise ValueError("`row_labels` doit avoir longueur h.")

    # échelle commune
    if color_scale == "mean":
        global_vmin, global_vmax = np.nanmin(means), np.nanmax(means)
    elif color_scale == "mean_pm_std":
        if stds is None:
            raise ValueError("color_scale='mean_pm_std' requiert `stds`.")
        low, high = means - std_mult * stds, means + std_mult * stds
        global_vmin, global_vmax = np.nanmin(low), np.nanmax(high)
    elif color_scale == "custom":
        if vmin is None or vmax is None:
            raise ValueError("Avec 'custom', fournir vmin et vmax.")
        global_vmin, global_vmax = float(vmin), float(vmax)
    else:
        raise ValueError("color_scale ∈ {'mean','mean_pm_std','custom'}.")

    if not np.isfinite(global_vmin) or not np.isfinite(global_vmax):
        raise ValueError("Échelle de couleurs non définissable (NaN/inf partout ?).")

    # figure / axes
    fig_w = max(4.0, g_l * figsize_per_cell[0])
    fig_h = max(3.0, g_h * figsize_per_cell[1])
    fig, axes = plt.subplots(g_h, g_l, figsize=(fig_w, fig_h), squeeze=False, constrained_layout=True)

    if(vmin_mirror):
        global_vmin= -global_vmax

    norm = Normalize(vmin=global_vmin, vmax=global_vmax)
    cmap_obj = plt.cm.get_cmap(cmap).copy()
    cmap_obj.set_bad(nan_color)

    if colorbar_label is None:
        colorbar_label = "Mean" if color_scale != "mean_pm_std" else f"Mean (échelle: mean ± {std_mult}·std)"

    # tracé
    for i in range(g_h):
        for j in range(g_l):
            ax = axes[i, j]
            ax.imshow(means[i, j], cmap=cmap_obj, norm=norm, aspect="auto", origin="upper")
            ax.set_xticks(np.arange(l))
            ax.set_yticks(np.arange(h))

            # ticks visibles ?
            show_x = show_axis_on == "all" or (show_axis_on == "outer" and i == g_h - 1)
            show_y = show_axis_on == "all" or (show_axis_on == "outer" and j == 0)

            if show_x and col_labels is not None:
                ax.set_xticklabels(col_labels, fontsize=tick_fontsize, rotation=xtick_rotation)
            elif show_x:
                ax.set_xticklabels(np.arange(l), fontsize=tick_fontsize, rotation=xtick_rotation)
            else:
                ax.set_xticklabels([])

            if show_y and row_labels is not None:
                ax.set_yticklabels(row_labels, fontsize=tick_fontsize, rotation=ytick_rotation)
            elif show_y:
                ax.set_yticklabels(np.arange(h), fontsize=tick_fontsize, rotation=ytick_rotation)
            else:
                ax.set_yticklabels([])

            # --- NOUVEAU : meta labels sur axes ---
            if j == 0 and meta_row_headers is not None:
                ax.set_ylabel(str(meta_row_headers[i]), fontsize=axis_label_fontsize, labelpad=axis_label_pad)
            if i == 0 and meta_col_headers is not None:
                ax.set_xlabel(str(meta_col_headers[j]), fontsize=axis_label_fontsize, labelpad=axis_label_pad)
                ax.xaxis.set_label_position('top') 
            # annotations numériques
            if annotate:
                m_ij = means[i, j]
                s_ij = stds[i, j] if stds is not None else None
                for y in range(h):
                    for x in range(l):
                        m_val = m_ij[y, x]
                        if np.isnan(m_val):
                            text = "NaN"
                        else:
                            if s_ij is None or np.isnan(s_ij[y, x]):
                                text = f"{value_fmt.format(m_val)}"
                            else:
                                text = f"{value_fmt.format(m_val)}±{value_fmt.format(s_ij[y, x])}"
                        ax.text(x, y, text, ha="center", va="center", fontsize=annotate_fontsize)

    # colorbar commune
    if show_colorbar:
        sm = ScalarMappable(norm=norm, cmap=cmap_obj)
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), pad=colorbar_pad)
        cbar.set_label(colorbar_label)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=suptitle_fontsize)

    return fig, axes