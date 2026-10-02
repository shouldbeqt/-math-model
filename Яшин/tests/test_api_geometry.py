"""Тесты API RP 11L / Spec 11E четырёхзвенной кинематики."""

import numpy as np

from src.kinematics.api_geometry import (
    ApiUnitGeometry,
    compute_kinematics_api_geometry,
    kinematics_from_config,
    stroke_displacement,
    stroke_fraction,
)
from src.yashin_kinematics import compute_kinematics_simple


def test_c640_geometry_stroke_order():
    kin = compute_kinematics_api_geometry(3.5, 11.0, 200)
    assert kin.kinematics_mode == "api_geometry"
    assert kin.S[0] < 0.05 * kin.S0
    assert kin.S.max() > 0.9 * kin.S0
    assert np.all(np.isfinite(kin.v))
    assert np.all(np.isfinite(kin.a))
    assert kin.v.max() > 0
    assert kin.v.min() < 0


def test_stroke_displacement_accepts_acpr():
    geom = ApiUnitGeometry.c640_default()
    theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    s = stroke_displacement(theta, geom.A, geom.C, geom.P, geom.R, s0_m=3.5)
    assert s.min() >= -1e-6
    assert s.max() <= 3.5 * 1.01


def test_api_vs_harmonic_different_shape():
    s0, period, n = 3.534, 11.32, 57
    h = compute_kinematics_simple(s0, period, n)
    a = compute_kinematics_api_geometry(s0, period, n)
    rmse = float(np.sqrt(np.mean((h.S - a.S) ** 2)))
    assert rmse > 0.05


def test_kinematics_mode_router():
    kin = kinematics_from_config({"mode": "harmonic"}, 3.5, 10.0, 50)
    assert kin.kinematics_mode == "harmonic_simple"
    kin_api = kinematics_from_config({"mode": "api_geometry"}, 3.5, 10.0, 50)
    assert kin_api.kinematics_mode == "api_geometry"


def test_stroke_fraction_bounds():
    geom = ApiUnitGeometry.c640_default()
    theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    pr = stroke_fraction(theta, geom)
    assert pr.min() >= -1e-6
    assert pr.max() <= 1.0 + 1e-6
