"""
Result synthesis and contextual visualization utilities.

This module orchestrates the end-to-end transformation of benchmark outputs
into contextual result visualizations.

Processing architecture
-----------------------

    raw benchmark metrics results (at aggregated level)
            |
            v
    contextual grid aggregation
            |
            v
    KPIView resolution and expansion
            |
            v
    recursive ResultView evaluation
            |
            v
    visualization adaptation
            |
            v
    contextual grid rendering

KPIViews may represent direct metrics, derived multi-source KPIs, comparisons
against a reference dimension value, or combinations of these constructs.

Main responsibilities:
- aggregate raw benchmark outputs into structured result grids;
- define result dimension metadata;
- orchestrate KPIView resolution and ResultView computation;
- adapt generic N-dimensional results to the 2-D contextual representation
  expected by the grid renderer;
- render and save contextual result visualizations.

KPI specification, recursive numerical evaluation, and numerical operations
are delegated respectively to `kpi_view`, `result_engine`, and
`result_operations`.
"""

import numpy as np

from abench.visu.visu import plot_grid_mean_std
from .result_engine import compute_result_view
from .kpi_view import (normalize_kpi_view,get_kpi_view_sources,resolve_kpi_view,expand_kpi_view,make_kpi_view, make_kpi_views)

import abench.store.api as api

DEFAULT_DIMENSION_MAP = {
    "experiment": 0,
    "component": 1}

def normalize_context_dimensions(context_dimensions=None):
    """Normalize contextual dimension specifications."""
    if context_dimensions is None:
        return {}

    return {
        name: np.atleast_1d(labels).tolist()
        for name, labels in context_dimensions.items()
    }

def compute_grid_results(
    dict_perf,
    list_component_name,
    list_metrics,
    dict_experiments,
    context_dimensions,
    n_REPET,
):
    """
    Aggregate benchmark outputs into structured contextual result grids.

    Output shape
    ------------
    experiment x component x context_1 x ... x context_N x statistic
    """
    context_dimensions = normalize_context_dimensions(context_dimensions)

    context_shape = tuple(
        len(labels)
        for labels in context_dimensions.values()
    )

    raw_shape = (
        n_REPET,
        *context_shape,
    )

    result_shape = (
        len(dict_experiments),
        len(list_component_name),
        *context_shape,
        2,
    )

    expected_size = int(np.prod(raw_shape))

    grid_results = {
        metric: np.empty(result_shape, dtype=float)
        for metric in list_metrics
    }

    for experiment_idx, (
        experiment_name,
        experiment_plan,
    ) in enumerate(dict_experiments.items()):

        metric_results = api.extract_benchmark_tables(
            dict_perf,
            experiment_plan=experiment_plan,
            list_components_name=list_component_name,
            metrics=list_metrics,
            agg_name="no-agg",
        )

        for metric in list_metrics:

            for component_idx, component_name in enumerate(
                list_component_name
            ):
                values = np.asarray(
                    metric_results[metric][component_name],
                    dtype=float,
                )

                if values.size != expected_size:
                    raise ValueError(
                        "Invalid result size for "
                        f"metric='{metric}', "
                        f"component='{component_name}', "
                        f"experiment='{experiment_name}'. "
                        f"Expected {expected_size} values "
                        f"for shape {raw_shape}, "
                        f"received {values.size}."
                    )

                values = values.reshape(raw_shape)

                grid_results[metric][experiment_idx, component_idx, ..., 0] = (values.mean(axis=0))
                grid_results[metric][experiment_idx, component_idx, ..., 1] = (values.std(axis=0))

    return grid_results

def _prepare_grid_for_plot(
    means,
    stds,
    context_dimensions,
):
    """
    Adapt generic result arrays to the 4-D layout expected by
    `plot_grid_mean_std`.

    Renderer layout:
        meta_row x meta_col x row_context x col_context
    """
    means = np.asarray(means)
    stds = None if stds is None else np.asarray(stds)

    context_dimensions = context_dimensions or {}
    context_names = list(context_dimensions)

    # --------------------------------------------------------------
    # Global result: no contextual dimension
    # (experiment, component)
    # -> (experiment, component, 1, 1)
    # --------------------------------------------------------------
    if len(context_names) == 0:

        means = means[..., np.newaxis, np.newaxis]

        if stds is not None:
            stds = stds[..., np.newaxis, np.newaxis]

        row_labels = [""]
        col_labels = [""]

    # --------------------------------------------------------------
    # One contextual dimension
    # (experiment, component, ctx)
    # -> (experiment, component, ctx, 1)
    # --------------------------------------------------------------
    elif len(context_names) == 1:

        means = means[..., np.newaxis]

        if stds is not None:
            stds = stds[..., np.newaxis]

        row_labels = context_dimensions[context_names[0]]
        col_labels = [""]

    # --------------------------------------------------------------
    # Two contextual dimensions
    # Already compatible with renderer
    # --------------------------------------------------------------
    elif len(context_names) == 2:

        row_labels = context_dimensions[context_names[0]]
        col_labels = context_dimensions[context_names[1]]

    else:
        raise ValueError(
            "plot_cond_grid_results currently supports at most "
            "two displayed contextual dimensions."
        )

    return means, stds, row_labels, col_labels

def plot_cond_grid_results(
    dict_perf,
    list_component_name,
    list_kpi_views,
    dict_experiments,
    context_dimensions=None,
    n_REPET=3,
    save_name='figure',
    figsize_per_cell=(5, 2),
    with_std=False,
    value_fmt="{:.3f}",
    **kwargs):
    """
    Plot contextual cross-validation results as hierarchical metric grids.

    Numerical view construction is delegated to `compute_result_view`.
    This keeps plotting independent from result selection and transformation.
    """
    if not context_dimensions:
        context_dimensions= {}
    
    context_names = list(context_dimensions)

    if len(context_names) == 0:
        row_labels = [""]
        col_labels = [""]

    elif len(context_names) == 1:
        row_labels = context_dimensions[context_names[0]]
        col_labels = [""]

    else:
        row_labels = context_dimensions[context_names[0]]
        col_labels = context_dimensions[context_names[1]]

        if len(context_dimensions) > 2:
            raise ValueError(
                "plot_CondGridResults currently supports at most "
                "two displayed contextual dimensions. "
                "Use selection to reduce higher-dimensional results "
                "before plotting."
            )
    
    experiment_names = np.asarray(list(dict_experiments.keys()))
    dimension_labels = {
        "experiment": np.asarray(list(dict_experiments.keys())),
        "component": np.asarray(list_component_name),
        **{
            name: np.asarray(labels)
         for name, labels in context_dimensions.items()
        },
    }

    kpi_views = [normalize_kpi_view(view)for view in list_kpi_views]
    # Determine which raw metrics must actually be computed.
    required_metrics = get_kpi_view_sources(kpi_views)
    grid_results = compute_grid_results(
        dict_perf=dict_perf,
        list_component_name=list_component_name,
        list_metrics=required_metrics,
        dict_experiments=dict_experiments,
        context_dimensions=context_dimensions,
        n_REPET=n_REPET,
    )
    
    dimension_map = {"experiment": 0,"component": 1}
    for idx, context_name in enumerate(context_dimensions):
        dimension_map[context_name] = idx + 2

    for kpi_view in kpi_views:
        kpi_view = resolve_kpi_view(kpi_view,dimension_labels)
        result_views = expand_kpi_view(kpi_view)
        for result_view in result_views:

            result = compute_result_view(
                dict_results=grid_results,
                result_view=result_view,
                dimension_map=dimension_map,
                with_std=with_std,
            )

            plot_means, plot_stds, row_labels, col_labels = (
                _prepare_grid_for_plot(
                    means=result["means"],
                    stds=result["stds"],
                    context_dimensions=context_dimensions,
                )
)

            # ----------------------------------------------------------
            # Recover labels corresponding to retained dimensions
            # ----------------------------------------------------------

            experiment_indices = result["indices"]["experiment"]
            component_indices = result["indices"]["component"]

            meta_row_headers = experiment_names[experiment_indices]

            meta_col_headers = np.asarray(
                list_component_name
            )[component_indices]

            # ----------------------------------------------------------
            # Plot configuration
            # ----------------------------------------------------------


            style = result["style"]
            style = result.get("style", {})
            current_cmap = style.get("cmap", "RdYlGn_r")
            vmin = style.get("vmin")
            vmax = style.get("vmax")
            vmin_mirror = style.get("vmin_mirror", False)

            # ----------------------------------------------------------
            # Rendering
            # ----------------------------------------------------------

            fig, axes = plot_grid_mean_std(
                means=plot_means,
                stds=plot_stds,
                cmap=current_cmap,
                meta_col_headers=meta_col_headers,
                meta_row_headers=meta_row_headers,
                col_labels=col_labels,
                row_labels=row_labels,
                value_fmt=value_fmt,
                figsize_per_cell=figsize_per_cell,
                vmin=vmin,
                vmax=vmax,
                vmin_mirror=vmin_mirror,
                **kwargs,
            )

            fig.savefig(
                save_name + result["name"]
            )

#Aliais
computeGridResults = compute_grid_results
plot_CondGridResults = plot_cond_grid_results