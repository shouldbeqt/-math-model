"""Тесты точной кинематики compute_kinematics_exact."""

import numpy as np

from src.yashin_kinematics import compute_kinematics, compute_kinematics_exact


def _geometry():
    return {
        "l_pole": 2.6, "bk": 1.6, "r": 0.71, "R0": 2.8, "ak": 2.963,
        "l1": 1.77, "l2": 5.0, "S0_m": 3.5,
    }


def test_exact_kinematics_stroke():
    geom = _geometry()
    kin = compute_kinematics_exact(geom, n_samples=100, period_seconds=11.0)
    assert kin.kinematics_mode == "exact"
    np.testing.assert_allclose(kin.S.max(), 3.5, rtol=0.02)
    assert kin.S.min() >= -0.01


def test_exact_velocity_asymmetric():
    """Точная v(t) не обязана быть чистой синусоидой."""
    geom = _geometry()
    kin_h = compute_kinematics(geom, 200, 11.0)
    kin_e = compute_kinematics_exact(geom, 200, 11.0)
    # численный gradient даёт небольшую асимметрию forward/backward
    v_h_pos = kin_h.v[kin_h.v > 0.01]
    v_e_pos = kin_e.v[kin_e.v > 0.01]
    assert abs(np.mean(v_h_pos) - np.mean(v_e_pos)) / np.mean(v_h_pos) > 0.001
