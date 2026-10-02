"""Тесты упругой силы по Гуку F = E A ΔL / L."""

import numpy as np
import yaml

from src.physics.elastic import (
    compute_elastic_force,
    compute_elastic_force_from_config,
    elongation_from_stroke,
    is_elastic_enabled,
    rod_cross_section_area,
)
from src.yashin_mvp import process_cycle_mvp


def test_hooke_formula():
    A = rod_cross_section_area(0.02)
    f = compute_elastic_force(0.01, E=2.1e11, rod_length=1000.0, A=A)
    np.testing.assert_allclose(f, 2.1e11 * A * 0.01 / 1000.0)


def test_elastic_disabled_is_zero():
    s = np.linspace(0, 3.5, 20)
    f = compute_elastic_force_from_config(s, {"elastic": {"enabled": False}})
    np.testing.assert_allclose(f, 0.0)
    assert not is_elastic_enabled({"elastic": {"enabled": False}})


def test_elastic_enabled_from_stroke():
    s = np.array([0.0, 1.0, 2.0, 3.5])
    cfg = {
        "elastic": {
            "enabled": True,
            "E": 2.1e11,
            "rod_length": 1000.0,
            "rod_diameter": 0.022,
        }
    }
    f = compute_elastic_force_from_config(s, cfg)
    A = rod_cross_section_area(0.022)
    expected = 2.1e11 * A * elongation_from_stroke(s) / 1000.0
    np.testing.assert_allclose(f, expected)
    assert f[0] == 0.0
    assert f[-1] > f[0]


def test_mvp_adds_elastic_term():
    with open("config_well16427.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["wave_model"] = False
    cfg["pump_load"] = {"mode": "none"}
    cfg["dynamics"] = {"m_eq": 0.0}
    cfg["elastic"] = {
        "enabled": True,
        "E": 2.1e11,
        "rod_length": 1000.0,
        "rod_diameter": 0.022,
    }
    i = 25 + 5 * np.sin(np.linspace(0, 2 * np.pi, 40))
    res = process_cycle_mvp(i, 10.0, cfg)
    assert res["elastic_enabled"] is True
    assert np.any(res["f_elastic"] != 0)
    np.testing.assert_allclose(
        res["f"],
        res["f_surface"] + res["f_elastic"] + res["f_fluid"] + res["f_inertia"] + res["f_friction"],
        rtol=1e-9,
    )
