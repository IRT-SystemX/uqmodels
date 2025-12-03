# test.py


"""
Tests for normalization behavior of statistical test wrappers.

These tests check that:
- normalized statistics are consistent with their theoretical formulas,
- raw and normalized versions share the same p-value (when applicable),
- normalization produces finite, dimensionless values on simple toy data.
"""

import numpy as np
import sys
sys.path.insert(1, '../')
from scipy import stats as sstats
from abench.stats.schema import SCHEMA_REGISTRY

import numpy as np

# Adapte ce import à ton projet
# from ton_module_stats import SCHEMA_REGISTRY


# -------------------------------------------------------------------
# 1) mw_contrast : test fonctionnel
# -------------------------------------------------------------------

def test_mw_contrast_functional():
    rng = np.random.default_rng(0)
    n = 200

    # Deux groupes (ex: altered vs normal)
    mask_group1 = np.zeros(n, dtype=bool)
    mask_group1[: n // 2] = True
    mask_group2 = ~mask_group1

    # --- Cas quasi-H0 : ref et cmp très proches ---
    ref_null = rng.normal(0.0, 1.0, size=n)
    cmp_null = ref_null + rng.normal(0.0, 0.05, size=n)  # petite perturbation

    v1_null, v2_null = SCHEMA_REGISTRY["mw_contrast"]["fn"](
        sample_ref=ref_null,
        sample_cmp=cmp_null,
        mask_group1=mask_group1,
        mask_group2=mask_group2,
    )

    # Sous H0, contrastes modérés
    assert abs(v1_null) < 2.0
    assert abs(v2_null) < 2.0

    # --- Cas effet positif : cmp plus "abîmé" sur group1 uniquement ---
    ref = rng.normal(0.0, 1.0, size=n)
    cmp = ref.copy()
    # On dégrade cmp uniquement sur group1 (par ex. attaque)
    cmp[mask_group1] = cmp[mask_group1] - 1.0

    v1, v2 = SCHEMA_REGISTRY["mw_contrast"]["fn"](
        sample_ref=ref,
        sample_cmp=cmp,
        mask_group1=mask_group1,
        mask_group2=mask_group2,
    )

    # Δ = ref - cmp est plus grand sur group1 → v1 > 0
    assert v1 > 0.0
    assert abs(v1) > 2.5

    # cmp sépare mieux les groupes que ref → v2 > 0
    assert v2 > 0.0
    assert abs(v2) > 1.5

def test_mw_contrast_control_vs_experiment():
    # Motif de base sur 8 points (4 dans group1, 4 dans group2)
    ref_block = np.array([
        10.0, 11.0, 12.0, 13.0,   # group1
        10.0, 11.0, 12.0, 13.0,   # group2
    ])

    # ---------------------------
    # 1) Condition "contrôle"
    #    Δ identique dans les deux groupes
    # ---------------------------
    delta_ctrl_block = np.array([
        1.0, 1.0, 1.0, 1.0,       # group1
        1.0, 1.0, 1.0, 1.0,       # group2
    ])

    # ---------------------------
    # 2) Condition "expérience"
    #    Δ plus fort dans group1 que dans group2
    # ---------------------------
    delta_exp_block = np.array([
        2.0, 2.0, 2.0, 2.0,       # group1
        1.0, 1.0, 1.0, 1.0,       # group2
    ])

    # On répète le motif pour augmenter le n
    k = 50
    ref_ctrl = np.tile(ref_block, k)
    ref_exp  = np.tile(ref_block, k)

    delta_ctrl = np.tile(delta_ctrl_block, k)
    delta_exp  = np.tile(delta_exp_block, k)

    cmp_ctrl = ref_ctrl - delta_ctrl
    cmp_exp  = ref_exp  - delta_exp

    n = ref_ctrl.shape[0]

    # Construction des masques groupes (en cohérence avec le motif)
    mask_group1_block = np.array([True, True, True, True, False, False, False, False])
    mask_group1 = np.tile(mask_group1_block, k)
    mask_group2 = ~mask_group1

    # ---------------------------
    # Contrôle
    # ---------------------------
    v1_ctrl, v2_ctrl = SCHEMA_REGISTRY["mw_contrast"]["fn"](
        sample_ref=ref_ctrl,
        sample_cmp=cmp_ctrl,
        mask_group1=mask_group1,
        mask_group2=mask_group2,
    )

    # Δ identiques dans les deux groupes → pas de contraste
    assert abs(v1_ctrl) < 1e-3
    assert abs(v2_ctrl) < 1e-3

    # ---------------------------
    # Expérience
    # ---------------------------
    v1_exp, v2_exp = SCHEMA_REGISTRY["mw_contrast"]["fn"](
        sample_ref=ref_exp,
        sample_cmp=cmp_exp,
        mask_group1=mask_group1,
        mask_group2=mask_group2,
    )

    # Ici, Δ(group1) > Δ(group2) strictement → changement d'ordre clair
    # -> v1_exp doit être nettement positif
    assert v1_exp > 0.0
    assert abs(v1_exp) > 2.0

    # Pour v2 : dans le contrôle, ni ref ni cmp ne séparent les groupes ;
    # dans l'expérience, cmp_exp sépare fortement group1 vs group2.
    assert v2_exp > 0.0
    assert abs(v2_exp) > abs(v2_ctrl) + 1e-2

# -------------------------------------------------------------------
# 2) wilcoxon_paired_subset : test fonctionnel
# -------------------------------------------------------------------

def test_wilcoxon_paired_subset_functional():
    rng = np.random.default_rng(1)
    n = 80

    mask_group = np.zeros(n, dtype=bool)
    mask_group[: n // 2] = True

    # --- Cas quasi-H0 : ref ≈ cmp ---
    base = rng.normal(0.0, 1.0, size=n)
    ref_null = base + rng.normal(0.0, 0.02, size=n)
    cmp_null = base + rng.normal(0.0, 0.02, size=n)

    z_null = SCHEMA_REGISTRY["wilcoxon_paired_subset"]["fn"](
        sample_ref=ref_null,
        sample_cmp=cmp_null,
        mask_group=mask_group,
    )

    assert abs(z_null) < 2.0

    # --- Cas effet positif : ref > cmp sur le subset ---
    base = rng.normal(0.0, 1.0, size=n)
    ref = base + 0.5
    cmp = base

    z = SCHEMA_REGISTRY["wilcoxon_paired_subset"]["fn"](
        sample_ref=ref,
        sample_cmp=cmp,
        mask_group=mask_group,
    )

    assert z > 0.0
    assert abs(z) > 2.0


# -------------------------------------------------------------------
# 3) mw_subset : test fonctionnel
# -------------------------------------------------------------------

def test_mw_subset_functional():
    rng = np.random.default_rng(2)
    n = 200

    mask_group = np.zeros(n, dtype=bool)
    mask_group[: n // 2] = True

    # --- Cas quasi-H0 : ref ≈ cmp sur le subset ---
    ref_null = rng.normal(0.0, 1.0, size=n)
    cmp_null = rng.normal(0.0, 1.0, size=n)

    z_null = SCHEMA_REGISTRY["mw_subset"]["fn"](
        sample_ref=ref_null,
        sample_cmp=cmp_null,
        mask_group=mask_group,
    )

    assert abs(z_null) < 2.0

    # --- Cas effet positif : ref > cmp sur le subset ---
    ref = rng.normal(0.0, 1.0, size=n)
    cmp = ref.copy()
    cmp[mask_group] = cmp[mask_group] - 1.0  # cmp plus petit sur le subset

    z = SCHEMA_REGISTRY["mw_subset"]["fn"](
        sample_ref=ref,
        sample_cmp=cmp,
        mask_group=mask_group,
    )

    # Comme le schema est paramétré avec alternative="greater"
    # et que ref > cmp, on attend un Z nettement positif.
    assert z > 0.0
    assert abs(z) > 2.5

# -------------------------------------------------------------------
# 3) mw_subset : test fonctionnel
# -------------------------------------------------------------------

def test_wasserstein_contrast_control_vs_experiment():
    # Motif de base sur 8 points (4 dans group1, 4 dans group2)
    ref_block = np.array([
        0.0, 1.0, 2.0, 3.0,   # group1
        0.0, 1.0, 2.0, 3.0,   # group2
    ])

    # ---------------------------
    # 1) Condition "contrôle"
    #    Δ identique dans les deux groupes
    # ---------------------------
    delta_ctrl_block = np.array([
        1.0, 1.0, 1.0, 1.0,   # group1
        1.0, 1.0, 1.0, 1.0,   # group2
    ])

    # ---------------------------
    # 2) Condition "expérience"
    #    Δ plus fort dans group1 que dans group2
    # ---------------------------
    delta_exp_block = np.array([
        2.0, 2.0, 2.0, 2.0,   # group1
        0.0, 0.0, 0.0, 0.0,   # group2
    ])

    # On répète le motif pour stabiliser les distances
    k = 50
    ref_ctrl = np.tile(ref_block, k)
    ref_exp  = np.tile(ref_block, k)

    delta_ctrl = np.tile(delta_ctrl_block, k)
    delta_exp  = np.tile(delta_exp_block, k)

    cmp_ctrl = ref_ctrl - delta_ctrl
    cmp_exp  = ref_exp  - delta_exp

    n = ref_ctrl.shape[0]

    # Masques de groupes
    mask_group1_block = np.array([True, True, True, True, False, False, False, False])
    mask_group1 = np.tile(mask_group1_block, k)
    mask_group2 = ~mask_group1

    # ---------------------------
    # Contrôle
    # ---------------------------
    v1_ctrl, v2_ctrl = SCHEMA_REGISTRY["wass_contrast"]["fn"](
        sample_ref=ref_ctrl,
        sample_cmp=cmp_ctrl,
        mask_group1=mask_group1,
        mask_group2=mask_group2,
    )

    # Sous contrôle, delta identiques dans les deux groupes,
    # et ref/cmp ont la même structure de groupe -> contrastes ~0
    assert abs(v1_ctrl) < 1e-6
    assert abs(v2_ctrl) < 1e-6

    # ---------------------------
    # Expérience
    # ---------------------------
    v1_exp, v2_exp = SCHEMA_REGISTRY["wass_contrast"]["fn"](
        sample_ref=ref_exp,
        sample_cmp=cmp_exp,
        mask_group1=mask_group1,
        mask_group2=mask_group2,
    )

    # Ici, delta(group1) >> delta(group2) => distance entre deltas > 0
    assert v1_exp > 0.0
    assert v1_exp > abs(v1_ctrl) + 1e-3

    # cmp_exp rend les distributions de groupe différentes alors que
    # ref_ctrl/exp ne séparaient pas les groupes -> v2_exp > 0
    assert v2_exp > 0.0
    assert v2_exp > abs(v2_ctrl) + 1e-3