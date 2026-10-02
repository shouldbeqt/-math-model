"""Тест гидравлической добавки pump_load."""

import numpy as np

from src.yashin_power_balance import get_pump_load_force


def test_pump_force_profile_downstroke():
    s = np.array([0.0, 1.75, 3.5, 1.75, 0.0])
    v = np.array([0.1, 0.5, 0.1, -0.5, -0.1])
    cfg = {"mode": "force_profile", "F0_down_n": 20000, "F1_down_n": 0, "F_up_n": 0}
    f = get_pump_load_force(s, v, 3.5, cfg)
    assert f[v > 0].sum() == 0
    np.testing.assert_allclose(f[v < 0], 20000)


def test_pump_force_offset():
    s = np.linspace(0, 3.5, 10)
    v = np.ones(10)
    v[5:] = -1
    cfg = {"mode": "force_offset", "F_down_n": 1000, "F_up_n": 500}
    f = get_pump_load_force(s, v, 3.5, cfg)
    np.testing.assert_allclose(f[:5], -500)
    np.testing.assert_allclose(f[5:], 1000)
