"""
Synthetic self-test (Раздел К ТЗ):
прямая модель (2.27) → инверсия (2.32) → восстановление P_ТПШ.
"""

import numpy as np
import yaml

from src.motor_model import torque_from_current
from src.yashin_inverse import compute_rod_load
from src.yashin_kinematics import compute_kinematics
from src.yashin_power_balance import compute_direct_power, compute_parasitic_powers

CONFIG_DEMO = "config_demo.yaml"
RTOL = 0.05


def _load_demo_config() -> dict:
    with open(CONFIG_DEMO, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _synthetic_rod_load(t: np.ndarray, period: float) -> np.ndarray:
    return 12000.0 + 4000.0 * np.sin(2 * np.pi * t / period)


def test_synthetic_roundtrip():
    config = _load_demo_config()
    n_samples = 100
    period = 8.0
    kinematics = compute_kinematics(config["geometry"], n_samples, period)
    p_true = _synthetic_rod_load(kinematics.t, period)

    p_measured = compute_direct_power(
        p_true, kinematics, config["geometry"], config["masses"], config["motor"],
    )
    p_delta, p_r = compute_parasitic_powers(
        kinematics, config["geometry"], config["masses"], config["motor"],
    )
    inverse = compute_rod_load(
        p_measured, p_delta, p_r, kinematics.v, config["geometry"], config["motor"],
    )
    p_recovered = inverse.p_tpsh

    valid = ~inverse.v_near_zero & np.isfinite(p_recovered)
    assert np.sum(valid) > n_samples * 0.8

    rel_err = np.abs(p_recovered[valid] - p_true[valid]) / np.max(np.abs(p_true))
    assert float(np.max(rel_err)) < RTOL


def test_motor_kloss_produces_finite_torque():
    config = _load_demo_config()
    i = np.linspace(15, 35, 50)
    result = torque_from_current(i, config["motor"], config["geometry"]["r"], method="kloss")
    assert np.all(np.isfinite(result.M_ed))
    assert np.all(result.M_ed > 0)


def test_motor_linear_produces_finite_torque():
    config = _load_demo_config()
    i = np.linspace(15, 35, 50)
    result = torque_from_current(i, config["motor"], config["geometry"]["r"], method="linear")
    assert np.all(np.isfinite(result.M_ed))
