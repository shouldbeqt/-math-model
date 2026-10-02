"""Тесты волновой модели колонны штанг (Gibbs)."""

from pathlib import Path

import numpy as np
import yaml

from src.physics.rod_wave import (
    RodWaveParameters,
    is_wave_model_enabled,
    plot_head_vs_surface_force,
    plot_wave_propagation,
    save_displacement_field,
    solve_rod_wave,
)
from src.yashin_mvp import process_cycle_mvp


def test_rod_wave_parameters_area_and_c():
    p = RodWaveParameters.from_config({
        "L": 1000.0, "E": 2.1e11, "rho": 7850.0,
        "diameter": 0.02, "nu": 0.1, "n_nodes": 21,
    })
    np.testing.assert_allclose(p.A, np.pi * 0.02 ** 2 / 4.0)
    np.testing.assert_allclose(p.c, np.sqrt(2.1e11 / 7850.0))
    np.testing.assert_allclose(p.dx, 1000.0 / 20.0)


def test_solve_rod_wave_shapes_and_surface_bc():
    n_t = 40
    t = np.linspace(0, 8.0, n_t, endpoint=False)
    dt = t[1] - t[0]
    s = 1.5 * (1.0 - np.cos(2 * np.pi * t / 8.0))
    v = np.gradient(s, dt)
    a = np.gradient(v, dt)
    f_head = 20000.0 + 5000.0 * np.sin(2 * np.pi * t / 8.0)
    params = RodWaveParameters.from_config({
        "L": 800.0, "diameter": 0.022, "nu": 0.8, "n_nodes": 25, "g": 0.0,
    })
    res = solve_rod_wave(s, v, a, f_head, dt, params)
    assert res.u.shape == (25, n_t)
    np.testing.assert_allclose(res.u[0, :], s, rtol=1e-9)
    assert np.all(np.isfinite(res.f_surface))
    assert res.f_surface.shape == (n_t,)


def test_wave_model_flag_off_keeps_legacy():
    with open("config_well16427.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["wave_model"] = False
    cfg["pump_load"] = {"mode": "none"}
    cfg["dynamics"] = {"m_eq": 0.0}
    i = 25 + 5 * np.sin(np.linspace(0, 2 * np.pi, 40))
    res = process_cycle_mvp(i, 10.0, cfg)
    assert res["wave_model"] is False
    assert res["wave_result"] is None
    np.testing.assert_allclose(res["f_surface"], res["f_dyn"])
    np.testing.assert_allclose(
        res["f"],
        res["f_surface"] + res["f_elastic"] + res["f_fluid"] + res["f_inertia"] + res["f_friction"],
    )


def test_wave_model_on_returns_surface_force(tmp_path: Path):
    with open("config_well16427.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["wave_model"] = True
    cfg["pump_load"] = {"mode": "none"}
    cfg["dynamics"] = {"m_eq": 0.0}
    cfg["rod_wave"] = {
        "L": 1000.0, "E": 2.1e11, "rho": 7850.0,
        "diameter": 0.022, "nu": 1.0, "n_nodes": 21, "g": 0.0, "save_u": True,
    }
    i = 25 + 5 * np.sin(np.linspace(0, 2 * np.pi, 40))
    res = process_cycle_mvp(i, 10.0, cfg)
    assert res["wave_model"] is True
    assert res["wave_result"] is not None
    np.testing.assert_allclose(res["f_surface"], res["wave_result"].f_surface)
    np.testing.assert_allclose(
        res["f"],
        res["f_surface"] + res["f_elastic"] + res["f_fluid"] + res["f_inertia"] + res["f_friction"],
    )
    assert is_wave_model_enabled(cfg)
    p = save_displacement_field(res["wave_result"], tmp_path / "u.npz")
    assert p.exists()
    plot_wave_propagation(res["wave_result"], tmp_path / "wave.png")
    plot_head_vs_surface_force(res["wave_result"], tmp_path / "forces.png")
    assert (tmp_path / "wave.png").exists()
    assert (tmp_path / "forces.png").exists()
