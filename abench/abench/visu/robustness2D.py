import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from fnmatch import fnmatch
from collections.abc import Iterable
from typing import Any, Dict, Callable, List, Tuple, Hashable, Optional, Union, Sequence
from abench.utils import (apply_mask, build_ctx_mask,build_sets)
from abench.store import api

def _filter_data_2d(meta_2d, indicator):
    """
    Nettoie les données :
      - conversion en np.ndarray
      - suppression des lignes avec NaN / inf sur meta_x, meta_y ou indicator
    """
    meta_2d = np.asarray(meta_2d)
    indicator = np.asarray(indicator)

    if meta_2d.ndim != 2 or meta_2d.shape[1] != 2:
        raise ValueError("meta_2d doit être de shape (N, 2).")

    if indicator.shape[0] != meta_2d.shape[0]:
        raise ValueError("indicator et meta_2d doivent avoir la même longueur N.")

    meta_x = meta_2d[:, 0]
    meta_y = meta_2d[:, 1]

    mask_valid = (
        np.isfinite(meta_x) &
        np.isfinite(meta_y) &
        np.isfinite(indicator)
    )

    meta_x = meta_x[mask_valid]
    meta_y = meta_y[mask_valid]
    indicator = indicator[mask_valid]

    if meta_x.size == 0:
        raise ValueError("Aucune donnée valide après nettoyage.")

    return meta_x, meta_y, indicator


def _build_2d_bins_from_meta(
    meta_x: np.ndarray,
    meta_y: np.ndarray,
    n_bins_x: int,
    n_bins_y: int,
):
    """
    Construit les bords et centres de bins en x et en y.
    """
    x_min, x_max = float(np.quantile(meta_x,0.02)), float(np.quantile(meta_x,0.98))
    y_min, y_max = float(np.quantile(meta_y,0.02)), float(np.quantile(meta_y,0.98))

    if x_min == x_max:
        raise ValueError("Toutes les valeurs meta_x sont identiques.")
    if y_min == y_max:
        raise ValueError("Toutes les valeurs meta_y sont identiques.")

    bin_edges_x = np.linspace(x_min, x_max, n_bins_x + 1)
    bin_edges_y = np.linspace(y_min, y_max, n_bins_y + 1)

    bin_centers_x = 0.5 * (bin_edges_x[:-1] + bin_edges_x[1:])
    bin_centers_y = 0.5 * (bin_edges_y[:-1] + bin_edges_y[1:])

    return bin_edges_x, bin_edges_y, bin_centers_x, bin_centers_y

def _aggregate_indicator_by_bins_2d(
    meta_x: np.ndarray,
    meta_y: np.ndarray,
    indicator: np.ndarray,
    bin_edges_x: np.ndarray,
    bin_edges_y: np.ndarray,
    agg: str = "mean",
):
    """
    Agrège indicator dans les bins 2D définis par bin_edges_x, bin_edges_y.

    Retour
    ------
    Z_binned : np.ndarray, shape (n_bins_x, n_bins_y)
        Moyenne / médiane de indicator par bin (NaN si bin vide).
    """
    n_bins_x = len(bin_edges_x) - 1
    n_bins_y = len(bin_edges_y) - 1

    Z = np.full((n_bins_x, n_bins_y), np.nan, dtype=float)

    # indices de bin pour chaque point
    ix = np.digitize(meta_x, bin_edges_x) - 1
    iy = np.digitize(meta_y, bin_edges_y) - 1

    ix = np.clip(ix, 0, n_bins_x - 1)
    iy = np.clip(iy, 0, n_bins_y - 1)

    # on parcourt les bins (brut mais clair; vectorisable si besoin)
    for i in range(n_bins_x):
        mask_x = ix == i
        if not np.any(mask_x):
            continue
        for j in range(n_bins_y):
            mask = mask_x & (iy == j)
            if not np.any(mask):
                continue
            vals = indicator[mask]
            if agg == "median":
                Z[i, j] = np.median(vals)
            else:
                Z[i, j] = np.mean(vals)

    return Z

def _interpolate_surface_from_bins(
    bin_centers_x: np.ndarray,
    bin_centers_y: np.ndarray,
    Z_binned: np.ndarray,
    grid_size_x: int,
    grid_size_y: int,
):
    """
    Interpolation 2D simple (bilinéaire) à partir de la surface binned.

    Retour
    ------
    X_grid, Y_grid, Z_grid : np.ndarray (grid_size_y, grid_size_x)
    """
    # grille cible en x / y
    x_grid = np.linspace(bin_centers_x.min(), bin_centers_x.max(), grid_size_x)
    y_grid = np.linspace(bin_centers_y.min(), bin_centers_y.max(), grid_size_y)

    # 1) interpolation en x pour chaque ligne de y
    Z_interp_x = np.empty((Z_binned.shape[0], grid_size_x))
    for i in range(Z_binned.shape[0]):
        row = Z_binned[i, :]
        mask = np.isfinite(row)
        if mask.sum() < 2:
            Z_interp_x[i, :] = np.nan
            continue
        Z_interp_x[i, :] = np.interp(x_grid, bin_centers_y[mask], row[mask])

    # 2) interpolation en y pour chaque colonne de x
    Z_grid = np.empty((grid_size_y, grid_size_x))
    for j in range(grid_size_x):
        col = Z_interp_x[:, j]
        mask = np.isfinite(col)
        if mask.sum() < 2:
            Z_grid[:, j] = np.nan
            continue
        Z_grid[:, j] = np.interp(y_grid, bin_centers_x[mask], col[mask])

    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    return X_grid, Y_grid, Z_grid

def compute_indicator_surface(
    indicator,
    meta_2d,
    n_bins_x: int = 20,
    n_bins_y: int = 20,
    agg: str = "mean",
    interpolate: bool = False,
    grid_size_x: int | None = None,
    grid_size_y: int | None = None,
    bin_edges_x: np.ndarray | None = None,
    bin_edges_y: np.ndarray | None = None,
):
    """
    Si bin_edges_x / bin_edges_y sont fournis, on les utilise tels quels,
    sinon on les construit à partir des données du fold.
    """
def compute_indicator_surface(
    indicator,
    meta_2d,
    n_bins_x: int = 20,
    n_bins_y: int = 20,
    agg: str = "mean",
    interpolate: bool = False,
    grid_size_x: int | None = None,
    grid_size_y: int | None = None,
    bin_edges_x: np.ndarray | None = None,
    bin_edges_y: np.ndarray | None = None,
):
    # 1) filtrage / nettoyage
    meta_x, meta_y, indicator = _filter_data_2d(meta_2d, indicator)

    # -----------------------------
    # 2) bins 2D : deux cas
    # -----------------------------
    if bin_edges_x is None or bin_edges_y is None:
        # COMPORTEMENT HISTORIQUE : bins calculés à partir des données du fold
        bin_edges_x, bin_edges_y, bin_centers_x, bin_centers_y = _build_2d_bins_from_meta(
            meta_x, meta_y, n_bins_x, n_bins_y
        )
    else:
        # CAS "OPTION 1" : bins imposés globalement
        bin_edges_x = np.asarray(bin_edges_x)
        bin_edges_y = np.asarray(bin_edges_y)

        n_bins_x = bin_edges_x.size - 1
        n_bins_y = bin_edges_y.size - 1

        bin_centers_x = 0.5 * (bin_edges_x[:-1] + bin_edges_x[1:])
        bin_centers_y = 0.5 * (bin_edges_y[:-1] + bin_edges_y[1:])


    # 3) agrégation dans les bins 2D
    Z_binned = _aggregate_indicator_by_bins_2d(
        meta_x, meta_y, indicator,
        bin_edges_x, bin_edges_y,
        agg=agg,
    )

    if not interpolate:
        return bin_centers_x, bin_centers_y, Z_binned

    # 4) interpolation optionnelle
    if grid_size_x is None:
        grid_size_x = n_bins_x
    if grid_size_y is None:
        grid_size_y = n_bins_y

    X_grid, Y_grid, Z_grid = _interpolate_surface_from_bins(
        bin_centers_x, bin_centers_y, Z_binned,
        grid_size_x, grid_size_y,
    )

    return X_grid, Y_grid, Z_grid, bin_centers_x, bin_centers_y, Z_binned