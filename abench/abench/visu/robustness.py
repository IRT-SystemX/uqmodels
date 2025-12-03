import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from fnmatch import fnmatch
from collections.abc import Iterable
from typing import Callable, List, Optional, Union, Sequence
from abench.utils import (Extract_dict,apply_mask,apply_mask_along_dim,build_ctx_mask,build_sets,unique_with_nan)
from abench.store import api
from abench.visu.robustness2D import compute_indicator_surface,_build_2d_bins_from_meta,_filter_data_2d
import numpy as np

def filter_indicator_data(
    meta_values,
    indicator_values,
    context: np.ndarray | None = None,
    ctx_constraints: list | None = None,
):
    """
    Applique :
      - éventuel filtrage par contexte via build_ctx_mask + apply_mask
      - nettoyage des NaN / inf

    Retourne des vecteurs 1D filtrés et nettoyés.
    """

    meta_values = np.asarray(meta_values)
    indicator_values = np.asarray(indicator_values)

    # Optionnel : filtrage par contexte
    if context is not None and ctx_constraints is not None:
        context = np.asarray(context)
        if context.shape[0] != meta_values.shape[0]:
            raise ValueError(
                f"context.shape[0] ({context.shape[0]}) "
                f"≠ meta_values.shape[0] ({meta_values.shape[0]})"
            )

        ctx_mask = build_ctx_mask(context, ctx_constraints)
        meta_values = apply_mask(meta_values, ctx_mask)
        indicator_values = apply_mask(indicator_values, ctx_mask)

    # Nettoyage NaN / inf
    valid = np.isfinite(meta_values) & np.isfinite(indicator_values)
    meta_values = apply_mask(meta_values, valid)
    indicator_values = apply_mask(indicator_values, valid)

    if meta_values.size == 0:
        raise ValueError("Aucune donnée valide après filtrage / nettoyage.")

    return meta_values, indicator_values

def build_bins_from_meta(meta_values: np.ndarray, n_bins: int):
    """
    Construit une grille de bins réguliers à partir des valeurs de métadata.

    Retour
    ------
    bin_edges : np.ndarray, shape (n_bins + 1,)
    bin_centers : np.ndarray, shape (n_bins,)
    """
    meta_min = float(np.quantile(meta_values,0.01))
    meta_max = float(np.quantile(np.max(meta_values),0.99))

    if meta_min == meta_max:
        raise ValueError("Toutes les métadatas sont égales : pas de binning possible.")

    bin_edges = np.linspace(meta_min, meta_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_edges, bin_centers

def aggregate_indicator_by_bins(
    meta_values: np.ndarray,
    indicator_values: np.ndarray,
    bin_edges: np.ndarray,
    agg: str = "mean",
):
    """
    Agrège indicator_values dans les bins définis par bin_edges.

    Retour
    ------
    y_binned : np.ndarray, shape (n_bins,)
        Valeurs agrégées (mean/median) par bin.
    """

    n_bins = len(bin_edges) - 1
    y_binned = np.full(n_bins, np.nan, dtype=float)

    # Indice de bin pour chaque échantillon
    bin_indices = np.digitize(meta_values, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    for b in range(n_bins):
        mask_b = bin_indices == b
        if not np.any(mask_b):
            continue
        vals = apply_mask(indicator_values, mask_b)

        if agg == "median":
            y_binned[b] = np.median(vals)
        else:
            y_binned[b] = np.mean(vals)

    return y_binned

def interpolate_curve_from_bins(
    bin_centers: np.ndarray,
    y_binned: np.ndarray,
    grid_size: int,
    meta_min: float | None = None,
    meta_max: float | None = None,
):
    """
    Construit une courbe interpolée à partir de points (bin_centers, y_binned).

    Si meta_min / meta_max ne sont pas fournis, ils sont dérivés de bin_centers.
    """

    bin_centers = np.asarray(bin_centers)
    y_binned = np.asarray(y_binned)

    valid = np.isfinite(y_binned)
    x_valid = bin_centers[valid]
    y_valid = y_binned[valid]

    if x_valid.size < 2:
        raise ValueError(
            "Pas assez de bins non vides pour interpoler (au moins 2 nécessaires)."
        )

    if meta_min is None:
        meta_min = float(np.min(bin_centers))
    if meta_max is None:
        meta_max = float(np.max(bin_centers))

    x_grid = np.linspace(meta_min, meta_max, grid_size)
    y_interp = np.interp(x_grid, x_valid, y_valid)

    return x_grid, y_interp

def compute_indicator_curve(
    indicator_values,
    meta_values,
    n_bins: int = 20,
    grid_size: int = 200,
    agg: str = "mean",
    context: np.ndarray | None = None,
    ctx_constraints: list | None = None,
):
    """
    Pipeline complet :
      1) filtrage / masking via filter_indicator_data
      2) construction des bins via build_bins_from_meta
      3) agrégation par bin via aggregate_indicator_by_bins
      4) interpolation de la courbe via interpolate_curve_from_bins
    """

    # 1) filtrage / masking / nettoyage
    meta_values_f, indicator_values_f = filter_indicator_data(
        meta_values=meta_values,
        indicator_values=indicator_values,
        context=context,
        ctx_constraints=ctx_constraints,
    )

    # 2) bins
    bin_edges, bin_centers = build_bins_from_meta(meta_values_f, n_bins=n_bins)

    # 3) agrégation
    y_binned = aggregate_indicator_by_bins(
        meta_values=meta_values_f,
        indicator_values=indicator_values_f,
        bin_edges=bin_edges,
        agg=agg,
    )

    # 4) interpolation
    x_grid, y_interp = interpolate_curve_from_bins(
        bin_centers=bin_centers,
        y_binned=y_binned,
        grid_size=grid_size,
        meta_min=float(bin_edges[0]),
        meta_max=float(bin_edges[-1]),
    )

    return x_grid, y_interp, bin_centers, y_binned

def aggregate_indicator_curves(
    curves,
    ci: float = 0.95,
    use_sem: bool = True,
    max_nan_ratio_per_curve: float = 1.0,
):
    """
    Agrège plusieurs courbes/surfaces d'indicateur et calcule une bande
    d'incertitude (IC de la moyenne ou dispersion).

    Paramètres
    ----------
    curves :
        Cas 1D :
            list of (x_grid, y_values)
              - x_grid  : (M,)
              - y_values: (M,)
        Cas 2D :
            list of ((centers_x, centers_y), Z)
              - centers_x : (Bx,)
              - centers_y : (By,)
              - Z         : (Bx, By)

    ci : float
        Niveau de confiance (≈ normal, 0.90 / 0.95 / 0.99 typiquement).
    use_sem : bool
        Si True : IC de la moyenne (± z * std / sqrt(n)).
        Si False : bande de dispersion (± z * std).
    max_nan_ratio_per_curve : float in [0,1]
        Si < 1, élimine les courbes/surfaces dont la proportion de NaN
        dépasse ce seuil.

    Retour
    ------
    Cas 1D :
        x_ref     : (M,)
        y_mean    : (M,)
        y_lower   : (M,)
        y_upper   : (M,)
        y_all     : (K_eff, M)

    Cas 2D :
        (centers_x_ref, centers_y_ref),
        Z_mean   : (Bx, By)
        Z_lower  : (Bx, By)
        Z_upper  : (Bx, By)
        Z_all    : (K_eff, Bx, By)
    """
    if not curves:
        raise ValueError("La liste 'curves' est vide.")

    # On regarde la dimension de la première série de valeurs
    first_x_or_centers, first_values = curves[0]
    values = np.asarray(first_values)

    if values.ndim == 1:
        # 1D : courbes
        return _aggregate_indicator_curves_1d(
            curves=curves,
            ci=ci,
            use_sem=use_sem,
            max_nan_ratio_per_curve=max_nan_ratio_per_curve,
        )
    elif values.ndim == 2:
        # 2D : surfaces
        return _aggregate_indicator_curves_2d(
            curves=curves,
            ci=ci,
            use_sem=use_sem,
            max_nan_ratio_per_curve=max_nan_ratio_per_curve,
        )
    else:
        raise ValueError(
            "Les valeurs doivent être 1D (courbes) ou 2D (surfaces). "
            f"Dimension reçue : {values.ndim}"
        )

def _aggregate_indicator_curves_1d(
    curves,
    ci: float,
    use_sem: bool,
    max_nan_ratio_per_curve: float,
):
    """
    Agrégation de courbes 1D y(x).

    curves : list of (x_grid, y_values)
    """
    # Grille de référence
    ref_x = np.asarray(curves[0][0])
    M = ref_x.shape[0]
    K = len(curves)

    y_all = np.full((K, M), np.nan, dtype=float)

    # On réinterpole chaque courbe sur ref_x si besoin
    for k, (x_k, y_k) in enumerate(curves):
        x_k = np.asarray(x_k)
        y_k = np.asarray(y_k)

        if x_k.shape[0] == M and np.allclose(x_k, ref_x, rtol=1e-6, atol=1e-8):
            y_all[k, :] = y_k
        else:
            y_all[k, :] = np.interp(ref_x, x_k, y_k, left=np.nan, right=np.nan)

    # Filtrage des courbes trop "NaN"
    if max_nan_ratio_per_curve < 1.0:
        nan_ratio = np.mean(~np.isfinite(y_all), axis=1)
        keep = nan_ratio <= max_nan_ratio_per_curve
        if not np.any(keep):
            raise ValueError(
                "Toutes les courbes ont été filtrées (max_nan_ratio_per_curve trop strict)."
            )
        y_all = y_all[keep, :]

    # Statistiques point à point
    y_mean = np.nanmean(y_all, axis=0)
    y_std = np.nanstd(y_all, axis=0, ddof=1)
    n_valid = np.sum(np.isfinite(y_all), axis=0)

    # Coefficient z (approx normale)
    if ci >= 0.989:
        z = 2.575  # ~99%
    elif ci >= 0.94:
        z = 1.96   # ~95%
    elif ci >= 0.89:
        z = 1.645  # ~90%
    else:
        z = 1.96 * (ci / 0.95)

    if use_sem:
        with np.errstate(divide="ignore", invalid="ignore"):
            y_sem = y_std / np.sqrt(n_valid)
        delta = z * y_sem
    else:
        delta = z * y_std

    y_lower = y_mean - delta
    y_upper = y_mean + delta

    # Invalider les points mal supportés
    y_mean[n_valid < 1] = np.nan
    y_lower[n_valid < 2] = np.nan
    y_upper[n_valid < 2] = np.nan

    return ref_x, y_mean, y_lower, y_upper, y_all

def _aggregate_indicator_curves_2d(
    curves,
    ci: float,
    use_sem: bool,
    max_nan_ratio_per_curve: float,
):
    """
    Agrégation de surfaces 2D Z(x, y).

    curves : list of ((centers_x, centers_y), Z)
        centers_x : (Bx,)
        centers_y : (By,)
        Z         : (Bx, By)
    """
    # Référence : centres et shape de la première surface
    (centers_x_ref, centers_y_ref), Z0 = curves[0]
    centers_x_ref = np.asarray(centers_x_ref)
    centers_y_ref = np.asarray(centers_y_ref)
    Z0 = np.asarray(Z0)

    Bx, By = Z0.shape
    K = len(curves)

    Z_all = np.full((K, Bx, By), np.nan, dtype=float)

    for k, (centers_k, Z_k) in enumerate(curves):
        cx_k, cy_k = centers_k
        cx_k = np.asarray(cx_k)
        cy_k = np.asarray(cy_k)
        Z_k = np.asarray(Z_k)

        # On impose pour l'instant que les grilles sont identiques
        if Z_k.shape != (Bx, By):
            raise ValueError(
                f"Shape de Z_k {Z_k.shape} incompatible avec la référence {(Bx, By)}."
            )
        if not (np.allclose(cx_k, centers_x_ref) and np.allclose(cy_k, centers_y_ref)):
            raise ValueError(
                "Les grilles (centers_x, centers_y) doivent être identiques "
                "pour toutes les surfaces (sinon prévoir une réinterpolation)."
            )

        Z_all[k, :, :] = Z_k

    # Filtrage des surfaces trop "NaN"
    if max_nan_ratio_per_curve < 1.0:
        nan_ratio = np.mean(~np.isfinite(Z_all), axis=(1, 2))
        keep = nan_ratio <= max_nan_ratio_per_curve
        if not np.any(keep):
            raise ValueError(
                "Toutes les surfaces ont été filtrées (max_nan_ratio_per_curve trop strict)."
            )
        Z_all = Z_all[keep, :, :]

    # Statistiques cellule par cellule
    Z_mean = np.nanmean(Z_all, axis=0)          # (Bx, By)
    Z_std = np.nanstd(Z_all, axis=0, ddof=1)    # (Bx, By)
    n_valid = np.sum(np.isfinite(Z_all), axis=0)

    # Coefficient z (approx normale)
    if ci >= 0.989:
        z = 2.575  # ~99%
    elif ci >= 0.94:
        z = 1.96   # ~95%
    elif ci >= 0.89:
        z = 1.645  # ~90%
    else:
        z = 1.96 * (ci / 0.95)

    if use_sem:
        with np.errstate(divide="ignore", invalid="ignore"):
            Z_sem = Z_std / np.sqrt(n_valid)
        delta = z * Z_sem
    else:
        delta = z * Z_std

    Z_lower = Z_mean - delta
    Z_upper = Z_mean + delta

    # Invalider les cellules mal supportées
    Z_mean[n_valid < 1] = np.nan
    Z_lower[n_valid < 2] = np.nan
    Z_upper[n_valid < 2] = np.nan

    return (centers_x_ref, centers_y_ref), Z_mean, Z_lower, Z_upper, Z_all

def plot_indicator_curve(
    x_grid,
    y_interp: np.ndarray | None = None,
    bin_centers: np.ndarray | None = None,
    y_binned: np.ndarray | None = None,
    ax=None,
    show: bool = True,
    label_curve: str = "Courbe interpolée",
    label_points: str = "Bins agrégés",
    # agrégation + incertitude
    y_lower: np.ndarray | None = None,
    y_upper: np.ndarray | None = None,
    label_band: str = "Bande d'incertitude",
    alpha_band: float = 0.2,
    # courbes individuelles de CV (1D uniquement)
    y_all: np.ndarray | None = None,
    show_individual: bool = False,
    alpha_individual: float = 0.3,
    lw_individual: float = 1.0,
):
    """
    Fonction générique de visualisation d'indicateur conditionnel.

    Cas 1D (courbe) :
    -----------------
    x_grid      : np.ndarray (M,)
    y_interp    : np.ndarray (M,)
    y_lower     : np.ndarray (M,) ou None
    y_upper     : np.ndarray (M,) ou None
    bin_centers : np.ndarray (B,) ou None
    y_binned    : np.ndarray (B,) ou None
    y_all       : np.ndarray (K, M) ou None (courbes individuelles)

    -> Trace courbe + bande d'incertitude + points de bins + courbes individuelles.

    Cas 2D (surface) :
    ------------------
    x_grid   : tuple/list (centers_x, centers_y)
               centers_x : np.ndarray (Bx,)
               centers_y : np.ndarray (By,)
    y_interp : np.ndarray (Bx, By)   (valeur moyenne)
    y_lower  : np.ndarray (Bx, By)   (borne inférieure IC, requis)
    y_upper  : np.ndarray (Bx, By)   (borne supérieure IC, requis)
    ax       : None ou séquence de 2 axes matplotlib

    -> Trace deux heatmaps :
        - valeur moyenne (y_interp)
        - incertitude (y_upper - y_lower)
    """
    # Cas dégradé : pas de y_interp -> on reste en 1D (comportement historique)
    if y_interp is None:
        return _plot_indicator_curve_1d(
            x_grid=x_grid,
            y_interp=None,
            bin_centers=bin_centers,
            y_binned=y_binned,
            ax=ax,
            show=show,
            label_curve=label_curve,
            label_points=label_points,
            y_lower=y_lower,
            y_upper=y_upper,
            label_band=label_band,
            alpha_band=alpha_band,
            y_all=y_all,
            show_individual=show_individual,
            alpha_individual=alpha_individual,
            lw_individual=lw_individual,
        )

    y_interp = np.asarray(y_interp)

    # 1D : même logique qu'avant
    if y_interp.ndim == 1:
        return _plot_indicator_curve_1d(
            x_grid=x_grid,
            y_interp=y_interp,
            bin_centers=bin_centers,
            y_binned=y_binned,
            ax=ax,
            show=show,
            label_curve=label_curve,
            label_points=label_points,
            y_lower=y_lower,
            y_upper=y_upper,
            label_band=label_band,
            alpha_band=alpha_band,
            y_all=y_all,
            show_individual=show_individual,
            alpha_individual=alpha_individual,
            lw_individual=lw_individual,
        )

    # 2D : affichage de deux heatmaps (valeur + incertitude)
    elif y_interp.ndim == 2:
        return _plot_indicator_surface_2d(
            centers=x_grid,
            Z_mean=y_interp,
            Z_lower=y_lower,
            Z_upper=y_upper,
            ax=ax,
            show=show,
        )

    else:
        raise ValueError(
            "y_interp doit être 1D (courbe) ou 2D (surface). "
            f"Dimension reçue : {y_interp.ndim}"
        )


# ---------------------------------------------------------------------
# Sous-fonction : cas 1D (courbe + bande d'incertitude)
# ---------------------------------------------------------------------
def _plot_indicator_curve_1d(
    x_grid: np.ndarray,
    y_interp: np.ndarray | None = None,
    bin_centers: np.ndarray | None = None,
    y_binned: np.ndarray | None = None,
    ax=None,
    show: bool = True,
    label_curve: str = "Courbe interpolée",
    label_points: str = "Bins agrégés",
    # agrégation + incertitude
    y_lower: np.ndarray | None = None,
    y_upper: np.ndarray | None = None,
    label_band: str = "Bande d'incertitude",
    alpha_band: float = 0.2,
    # courbes individuelles de CV
    y_all: np.ndarray | None = None,
    show_individual: bool = False,
    alpha_individual: float = 0.3,
    lw_individual: float = 1.0,
):
    if ax is None:
        fig, ax = plt.subplots()

    x_grid = np.asarray(x_grid)

    # Courbes individuelles (folds de CV)
    if show_individual and y_all is not None:
        for k in range(y_all.shape[0]):
            ax.plot(
                x_grid,
                y_all[k, :],
                linewidth=lw_individual,
                alpha=alpha_individual,
            )

    # Bande d'incertitude
    if y_lower is not None and y_upper is not None:
        y_lower = np.asarray(y_lower)
        y_upper = np.asarray(y_upper)
        finite = np.isfinite(y_lower) & np.isfinite(y_upper)
        ax.fill_between(
            x_grid,
            y_lower,
            y_upper,
            where=finite,
            alpha=alpha_band,
            label=label_band,
        )

    # Courbe principale
    if y_interp is not None:
        ax.plot(x_grid, y_interp, label=label_curve)

    # Points agrégés (bins)
    if bin_centers is not None and y_binned is not None:
        bin_centers = np.asarray(bin_centers)
        y_binned = np.asarray(y_binned)
        valid_bins = np.isfinite(y_binned)
        ax.scatter(
            bin_centers[valid_bins],
            y_binned[valid_bins],
            marker="o",
            label=label_points,
        )

    ax.set_xlabel("Métadata")
    ax.set_ylabel("Indicateur")
    ax.legend()
    ax.grid(True)

    if show:
        plt.show()

    return ax


# ---------------------------------------------------------------------
# Sous-fonction : cas 2D (deux heatmaps : valeur + incertitude)
# ---------------------------------------------------------------------
def _plot_indicator_surface_2d(
    centers,
    Z_mean: np.ndarray,
    Z_lower: np.ndarray | None = None,
    Z_upper: np.ndarray | None = None,
    ax=None,
    show: bool = True,
    title_mean: str = "Indicateur moyen",
    title_unc: str = "Incertitude (largeur IC)",
):
    """
    centers : (centers_x, centers_y)
        centers_x : (Bx,)
        centers_y : (By,)
    Z_mean  : (Bx, By)
    Z_lower : (Bx, By), requis pour l'incertitude
    Z_upper : (Bx, By), requis pour l'incertitude

    ax :
        - None : création de fig, axes = plt.subplots(1, 2)
        - séquence de 2 axes : (ax_mean, ax_unc)
    """
    if not (isinstance(centers, (tuple, list)) and len(centers) == 2):
        raise ValueError(
            "Pour le cas 2D, 'centers' doit être un tuple/list (centers_x, centers_y)."
        )

    centers_x = np.asarray(centers[0])
    centers_y = np.asarray(centers[1])
    Z_mean = np.asarray(Z_mean)

    if Z_mean.shape != (centers_x.size, centers_y.size):
        raise ValueError(
            f"Shape de Z_mean {Z_mean.shape} incompatible avec "
            f"centers_x ({centers_x.size}) et centers_y ({centers_y.size})."
        )

    if Z_lower is None or Z_upper is None:
        raise ValueError(
            "Pour le cas 2D avec visualisation de l'incertitude, "
            "Z_lower et Z_upper doivent être fournis."
        )

    Z_lower = np.asarray(Z_lower)
    Z_upper = np.asarray(Z_upper)

    if Z_lower.shape != Z_mean.shape or Z_upper.shape != Z_mean.shape:
        raise ValueError("Z_lower, Z_upper et Z_mean doivent avoir la même shape.")

    # Mesure d'incertitude : largeur de la bande
    Z_unc = Z_upper - Z_lower

    # Gestion des axes
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax_mean, ax_unc = axes
    else:
        # on accepte un iterable de 2 axes
        if isinstance(ax, (tuple, list, np.ndarray)) and len(ax) == 2:
            ax_mean, ax_unc = ax
        else:
            raise ValueError(
                "Pour le cas 2D, 'ax' doit être None ou une séquence de 2 axes (ax_mean, ax_unc)."
            )

    # Utilitaire interne pour computed extent
    def _compute_extent_from_centers(c):
        c = np.asarray(c)
        if c.size < 2:
            delta = 0.5 if c.size == 0 else (0.5 * (abs(c[0]) + 1.0))
            return c[0] - delta, c[0] + delta
        step = c[1] - c[0]
        return c[0] - step / 2.0, c[-1] + step / 2.0

    xmin, xmax = _compute_extent_from_centers(centers_x)
    ymin, ymax = _compute_extent_from_centers(centers_y)

    # Convention d'affichage : transpose pour avoir (y, x)
    Z_mean_disp = Z_mean.T  # (By, Bx)
    Z_unc_disp = Z_unc.T

    # Heatmap valeur moyenne
    im_mean = ax_mean.imshow(
        Z_mean_disp,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        aspect="auto",
    )
    ax_mean.set_title(title_mean)
    ax_mean.set_xlabel("Méta 1")
    ax_mean.set_ylabel("Méta 2")
    plt.colorbar(im_mean, ax=ax_mean, label="Indicateur")

    # Heatmap incertitude
    im_unc = ax_unc.imshow(
        Z_unc_disp,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        aspect="auto",
    )
    ax_unc.set_title(title_unc)
    ax_unc.set_xlabel("Méta 1")
    ax_unc.set_ylabel("Méta 2")
    plt.colorbar(im_unc, ax=ax_unc, label="Largeur IC")

    if show:
        plt.show()
    return (ax_mean, ax_unc)

def plot_indicator_curve_from_AB_results(storing,component_name,Exp_plan,metric,dict_sets_config):
    curves = []
    mode='1D'
    for Train_set,Test_set_list in Exp_plan.items():
        for set_name in Test_set_list:
            X,y,output,context,metadata=api.get_data_and_output(storing,component_name=component_name,trainset_name=Train_set,set_name=set_name)
            res = metric.compute(y,output,context,{})

            list_keys = ['context_mask','context_dim_mask','context_variable_ids']
            context_mask,context_dim_mask,context_variable_ids = Extract_dict(dict_sets_config,list_keys=list_keys)
            context_filtered = apply_mask_along_dim(context, context_mask, context_dim_mask)
            print(context_filtered.shape)
            if(len(context_variable_ids[0])==1):
                mode = '1D'
                x_grid, y_interp, bin_centers, y_binned = compute_indicator_curve(
                    res,
                    context_filtered[:,context_variable_ids[0]].astype(float).reshape(-1),
                    n_bins=30,
                    grid_size=300,
                    agg="mean",
                )
                curves.append((x_grid, y_interp))
            else:
                meta_2d = context_filtered[:,context_variable_ids[0]].astype(float).reshape(-1,2)
                
                if(mode=='1D'):
                    meta_x, meta_y, res = _filter_data_2d(meta_2d, res)
                    n_bins_x = 20
                    n_bins_y = 20
                    bin_edges_x, bin_edges_y, bin_centers_x, bin_centers_y = _build_2d_bins_from_meta(meta_x, meta_y, n_bins_x, n_bins_y)
 
                                        
                mode = '2D'
                output_arg = compute_indicator_surface(res,
                                                       meta_2d,
                                                       n_bins_x=n_bins_x, 
                                                       n_bins_y=n_bins_y,
                                                       agg="mean",
                                                       interpolate=False,
                                                       bin_edges_x=bin_edges_x,
                                                       bin_edges_y=bin_edges_y)
                if(len(output_arg)==6):
                    X_grid, Y_grid, Z_grid, bin_x, bin_y, Z_binned= output_arg
            
                    # On stocke au format attendu par aggregate_indicator_curves en 2D :
                    # ((centers_x, centers_y), Z)
                    curves.append(((bin_x, bin_y), Z_binned))
                else:
                    centers_x_interp, centers_y_interp, Z_binned = output_arg
                    curves.append(((centers_x_interp, centers_y_interp), Z_binned))


    # 2) Agrégation des courbes sur les folds + bande d'incertitude
    if(mode=='1D'):
        x_grid_agg, y_mean, y_lower, y_upper, y_all = aggregate_indicator_curves(
            curves,
            ci=0.95,     # IC 95%
            use_sem=True # IC de la moyenne; False pour dispersion brute)
            )
    else:
        (centers_x_ref, centers_y_ref), Z_mean, Z_lower, Z_upper, Z_all = aggregate_indicator_curves(
        curves,
        ci=0.95,               # niveau de confiance
        use_sem=True,          # IC de la moyenne ; False => dispersion brute
        max_nan_ratio_per_curve=1.0)
        x_grid_agg = (centers_x_ref, centers_y_ref)
        y_mean = Z_mean
        y_lower = Z_lower
        y_upper = Z_upper
        y_all = Z_all



    plot_indicator_curve(
        x_grid_agg,
        y_interp=y_mean,
        y_lower=y_lower,
        y_upper=y_upper,
        label_curve="Moyenne CV",
        label_band="IC 95% (moyenne)",
        # pour visualiser aussi la variabilité des folds :
        y_all=y_all,
        show_individual=False)