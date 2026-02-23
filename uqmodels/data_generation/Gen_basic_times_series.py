from dataclasses import dataclass, field
import numpy as np
import scipy.stats
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
# from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Sequence, Optional
from uqmodels.utils import cut, base_cos_freq
from uqmodels.preprocessing.preprocessing import rolling_statistics

rng = np.random.RandomState(42)


def attack_mean(y: np.ndarray,
                loc: np.ndarray,
                dim: Sequence[int] = (0,),
                f: float = 1.0) -> np.ndarray:
    """
    Ajoute une dérive progressive sur la moyenne sur les indices `loc`.
    """
    y = np.copy(y)
    len_ = len(loc)
    for d in dim:
        for t in np.arange(len_):
            y[loc[t], d] += f * np.sqrt(min(t, len_ - t) / len_)
    return y


def attack_var(y: np.ndarray,
               loc: np.ndarray,
               dim: Sequence[int] = (0,),
               f: float = 1.0) -> np.ndarray:
    """
    Augmente (ou diminue) la variance sur une fenêtre d'indices `loc`.
    """
    y = np.copy(y)
    len_ = len(loc)
    ext_min = scipy.stats.norm.ppf(0.15, 0, f * 0.2)
    ext_max = scipy.stats.norm.ppf(0.85, 0, f * 0.2)
    for d in dim:
        noise = rng.normal(0, f * 0.2, len_)
        noise = np.maximum(np.minimum(noise, ext_max), ext_min)
        y[loc, d] += noise
    return y


def attack_spike(y: np.ndarray,
                 loc: np.ndarray,
                 dim: Sequence[int] = (0,),
                 f: float = 1.0) -> np.ndarray:
    """
    Ajoute un spike ponctuel sur le milieu de la fenêtre `loc`.
    """
    y = np.copy(y)
    len_ = len(loc)
    idx = loc[int(len_ / 2)]
    for d in dim:
        y[idx, d] += f
    return y


@dataclass
class AttackSpec:
    """
    Spécification d'une attaque à appliquer sur un signal.

    Parameters
    ----------
    func : Callable
        Fonction d'attaque (e.g. attack_mean, attack_var, attack_spike).
    loc : np.ndarray
        Indices sur lesquels appliquer l'attaque.
    kwargs : dict
        Arguments additionnels à passer à la fonction (e.g. f=1.1, dim=[0]).
    """
    func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    loc: np.ndarray
    kwargs: Dict[str, Any] = field(default_factory=dict)


def apply_attacks(y: np.ndarray,
                  attacks: Sequence[AttackSpec]) -> np.ndarray:
    """
    Applique séquentiellement une liste d'attaques sur le signal y.
    """
    y_out = np.copy(y)
    for spec in attacks:
        y_out = spec.func(y_out, spec.loc, **spec.kwargs)
    return y_out


def core_gen(
    N: int = 10000,
    freq: float = 100.0,
    r1: float = -0.31,
    r2: float = 5.1,
    r3: float = 1.3,
    r4: float = 0.4,
    seed: int = 42,
    train_ratio: float = 0.7,
    name: str = "Unnoised ML-task",
    # attaques sur y et Z
    attacks_y: Optional[Sequence[AttackSpec]] = None,
    attacks_z: Optional[Sequence[AttackSpec]] = None,
    # fonctions externes
    cut_func: Callable[[np.ndarray, float, float], np.ndarray] = cut,
    base_cos_freq_func: Callable[[np.ndarray, Sequence[float]], np.ndarray] = base_cos_freq,
        rolling_statistics_func: Callable[..., pd.DataFrame] = rolling_statistics):
    """
    Version enrichie : retourne aussi
      - y_no_obs : y et Z perturbés, sans statistiques
      - y_old    : y et Z non perturbés, sans statistiques
    """

    if cut_func is None:
        raise ValueError("cut_func doit être fourni (par ex. `cut`).")
    if base_cos_freq_func is None:
        raise ValueError("base_cos_freq_func doit être fourni (par ex. `base_cos_freq`).")
    if rolling_statistics_func is None:
        raise ValueError("rolling_statistics_func doit être fourni (par ex. `rolling_statistics`).")

    # RNG
    local_rng = np.random.RandomState(seed)

    # -----------------------------
    # 1. Grille temporelle
    # -----------------------------
    X = np.arange(0, freq, freq / N)

    # -----------------------------
    # 2. Signaux propres
    # -----------------------------
    y_mean = np.cos(X * math.pi + r1) + np.cos(r2 * X * math.pi)
    Z_mean = np.power(np.sin(X * math.pi + r3), 3) + np.cos(r4 * X * math.pi)

    # -----------------------------
    # 3. Bruit
    # -----------------------------
    Y_noise = cut_func(local_rng.normal(0, 1.5, N), 0.25, 0.75) * (0.02 + np.abs(y_mean) / 5)
    Z_noise = cut_func(local_rng.normal(0, 1.2, N), 0.25, 0.75) * (0.02 + np.abs(1 - Z_mean) / 5)

    y_clean = cut_func(y_mean + Y_noise, 0.01, 0.99).reshape(-1, 1)
    Z_clean = cut_func(Z_mean + Z_noise, 0.01, 0.99).reshape(-1, 1)

    # Non perturbés (référence)
    old_y = np.copy(y_clean)
    old_Z = np.copy(Z_clean)

    # -----------------------------
    # 4. Attaques
    # -----------------------------
    if attacks_y is None:
        attacks_y = [
            AttackSpec(attack_mean, np.arange(9100, 9120), dict(f=-1.1)),
            AttackSpec(attack_spike, np.arange(9501, 9502), dict(f=1.55)),
            AttackSpec(attack_var, np.arange(9750, 9770), dict(f=2.5)),
        ]

    if attacks_z is None:
        attacks_z = [
            AttackSpec(attack_var, np.arange(9100, 9120), dict(f=2.0)),
            AttackSpec(attack_spike, np.arange(9770, 9771), dict(f=-0.6)),
        ]

    y_attacked = apply_attacks(y_clean, attacks_y)
    Z_attacked = apply_attacks(Z_clean, attacks_z)

    # -----------------------------
    # NOUVEAU : sorties brutes pour visualisation
    # -----------------------------
    y_no_obs = np.concatenate([y_attacked, Z_attacked], axis=1)
    y_old = np.concatenate([old_y, old_Z], axis=1)

    # -----------------------------
    # 5. Rolling statistics
    # -----------------------------
    mean_var_10 = rolling_statistics_func(
        pd.DataFrame(np.roll(y_attacked, 0)),
        10, 1,
        ['mean', 'std', 'extremum'],
        ['mean', 'std', 'extremum']
    ).replace(np.nan, 0).values[:, :1]

    var_and_ext = rolling_statistics_func(
        pd.DataFrame(np.roll(y_attacked, 0)),
        30, 1,
        ['mean', 'std', 'extremum'],
        ['mean', 'std', 'extremum']
    ).replace(np.nan, 0).values[:, 1:]

    Z_mean_var_10 = rolling_statistics_func(
        pd.DataFrame(np.roll(Z_attacked, 0)),
        10, 1,
        ['mean', 'std', 'extremum'],
        ['mean', 'std', 'extremum']
    ).replace(np.nan, 0).values[:, :1]

    Z_var_and_ext = rolling_statistics_func(
        pd.DataFrame(np.roll(Z_attacked, 0)),
        30, 1,
        ['mean', 'std', 'extremum'],
        ['mean', 'std', 'extremum']
    ).replace(np.nan, 0).values[:, 1:]

    # --- Stats sur signaux NON perturbés (old_y / old_Z) ---

    old_mean_var_10 = rolling_statistics_func(
        pd.DataFrame(np.roll(old_y, 0)),
        10, 1,
        ['mean', 'std', 'extremum'],
        ['mean', 'std', 'extremum']
    ).replace(np.nan, 0).values[:, :1]

    old_var_and_ext = rolling_statistics_func(
        pd.DataFrame(np.roll(old_y, 0)),
        30, 1,
        ['mean', 'std', 'extremum'],
        ['mean', 'std', 'extremum']
    ).replace(np.nan, 0).values[:, 1:]

    old_Z_mean_var_10 = rolling_statistics_func(
        pd.DataFrame(np.roll(old_Z, 0)),
        10, 1,
        ['mean', 'std', 'extremum'],
        ['mean', 'std', 'extremum']
    ).replace(np.nan, 0).values[:, :1]

    old_Z_var_and_ext = rolling_statistics_func(
        pd.DataFrame(np.roll(old_Z, 0)),
        30, 1,
        ['mean', 'std', 'extremum'],
        ['mean', 'std', 'extremum']
    ).replace(np.nan, 0).values[:, 1:]

    # --- y_old = même format que y_target mais sur données non perturbées ---
    y_old = np.concatenate(
        [old_mean_var_10, old_var_and_ext,
         old_Z_mean_var_10, old_Z_var_and_ext],
        axis=1
    )

    # cible ML
    y_target = np.concatenate(
        [mean_var_10, var_and_ext, Z_mean_var_10, Z_var_and_ext],
        axis=1
    )

    y_old = np.concatenate([
        old_mean_var_10,
        old_var_and_ext,
        old_Z_mean_var_10,
        old_Z_var_and_ext], axis=1)

    # -----------------------------
    # 6. Features contextuelles
    # -----------------------------
    feat1 = base_cos_freq_func(X, [0.5, 2, 0.5 * r2, 2 * r2])
    feat2 = base_cos_freq_func(X, [0.5, 2, 0.5 * r4, 2 * r4])
    features = np.concatenate([feat1, feat2], axis=1)

    features = StandardScaler().fit_transform(features)

    # -----------------------------
    # 7. Train/test + state
    # -----------------------------
    state = np.zeros(N)
    train_mask = np.arange(N) < int(train_ratio * N)

    # -----------------------------
    # 8. RETOUR enrichi
    # -----------------------------
    return {
        "X": features,
        "Y": y_target,
        "Context": state,
        "train": train_mask,
        "test": np.invert(train_mask),
        "split": train_mask,
        "aux": {"y_no_obs": y_no_obs, "y_old": y_old}
    }


def generate_default(dict_params=dict()):
    dict_data = core_gen(**dict_params)
    return dict_data
