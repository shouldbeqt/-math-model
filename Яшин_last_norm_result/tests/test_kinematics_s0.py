"""Тест задания S₀ константой в geometry.S0_m."""

import numpy as np

from src.yashin_kinematics import compute_geometry_angles, compute_kinematics, resolve_s0


def test_resolve_s0_from_formula():
    geometry = {"ak": 1.6, "l_pole": 2.6, "bk": 1.6, "r": 1.2, "R0": 2.8}
    _, _, _, s0_formula = compute_geometry_angles(geometry)
    s0, source = resolve_s0(geometry, s0_formula)
    assert source == "formula"
    assert s0 == s0_formula


def test_resolve_s0_from_config():
    geometry = {"S0_m": 3.5, "ak": 1.6, "l_pole": 2.6, "bk": 1.6, "r": 1.2, "R0": 2.8}
    _, _, _, s0_formula = compute_geometry_angles(geometry)
    s0, source = resolve_s0(geometry, s0_formula)
    assert source == "config"
    assert s0 == 3.5
    assert s0 != s0_formula


def test_compute_kinematics_constant_s0():
    geometry = {
        "S0_m": 3.5,
        "l1": 3.0,
        "l2": 2.5,
        "r": 1.2,
        "R0": 2.8,
        "l_pole": 2.6,
        "ak": 1.6,
        "bk": 1.6,
    }
    kin = compute_kinematics(geometry, n_samples=100, period_seconds=10.0)
    assert kin.s0_source == "config"
    assert kin.S0 == 3.5
    assert np.isclose(kin.S.max(), 3.5, atol=1e-9)
    assert np.isclose(kin.S.min(), 0.0, atol=1e-9)
