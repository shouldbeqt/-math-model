"""Тесты инерционной силы F_inertia = m_eq · a."""

import numpy as np
import yaml

from src.dynamics.inertia import compute_inertia_force, resolve_m_eq
from src.yashin_mvp import process_cycle_mvp


def test_compute_inertia_force():
    a = np.array([1.0, -2.0, 0.5])
    f = compute_inertia_force(a, m_eq=100.0)
    np.testing.assert_allclose(f, [100.0, -200.0, 50.0])


def test_m_eq_zero_disables_inertia():
    a = np.array([1.0, -1.0])
    f = compute_inertia_force(a, m_eq=0.0)
    np.testing.assert_allclose(f, [0.0, 0.0])


def test_resolve_m_eq_from_config():
    assert resolve_m_eq({"dynamics": {"m_eq": 250.0}}) == 250.0
    assert resolve_m_eq({"m_eq": 10.0}) == 10.0
    assert resolve_m_eq({}) == 0.0


def test_mvp_includes_inertia_in_total_force():
    with open("config_well16427.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["model"] = {"mode": "mvp"}
    cfg["pump_load"] = {"mode": "none"}
    cfg["dynamics"] = {"m_eq": 500.0}
    i = 25 + 5 * np.sin(np.linspace(0, 2 * np.pi, 57))
    res = process_cycle_mvp(i, 11.32, cfg)
    np.testing.assert_allclose(
        res["f"],
        res["f_surface"] + res["f_elastic"] + res["f_fluid"] + res["f_inertia"] + res["f_friction"],
        rtol=1e-9,
    )
    np.testing.assert_allclose(res["f_inertia"], 500.0 * res["kinematics"].a)
    assert res["m_eq"] == 500.0
