"""Тесты MVP-модели."""

import numpy as np
import yaml

from src.motor_model import compute_power_poly2, torque_from_current_simple
from src.yashin_inverse import compute_rod_load_simple
from src.yashin_kinematics import compute_kinematics_simple
from src.yashin_power_balance import get_parasitic_power_constant
from src.yashin_mvp import process_cycle_mvp


def test_kinematics_simple_one_param():
    kin = compute_kinematics_simple(3.5, 11.0, 100)
    assert kin.kinematics_mode == "harmonic_simple"
    np.testing.assert_allclose(kin.S.max(), 3.5, rtol=1e-6)


def test_poly2_power():
    p = compute_power_poly2(np.array([10.0, 20.0]), 1000, 50, -1)
    np.testing.assert_allclose(p, [1400, 1600])
    p3 = compute_power_poly2(np.array([10.0]), 1000, 50, -1, c=0.5)
    np.testing.assert_allclose(p3, [1900.0])


def test_mvp_process_cycle():
    with open("config_well16427.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["model"] = {"mode": "mvp"}
    i = 25 + 5 * np.sin(np.linspace(0, 2 * np.pi, 57))
    cfg.setdefault("pump_load", {"mode": "none"})
    res = process_cycle_mvp(i, 11.32, cfg)
    assert len(res["s"]) == 57
    assert np.all(np.isfinite(res["f"]))
