"""
Согласованность прямой (2.27) и обратной (2.32) модели.
P(t) через compute_direct_power → compute_rod_load → P_ТПШ с точностью 1e-6.
"""

import numpy as np
import yaml

from src.yashin_inverse import compute_rod_load
from src.yashin_kinematics import compute_kinematics
from src.yashin_power_balance import compute_direct_power, compute_parasitic_powers

CONFIG_DEMO = "config_demo.yaml"
ATOL = 1e-6


def _load_demo_config() -> dict:
    with open(CONFIG_DEMO, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_forward_backward_consistency():
    config = _load_demo_config()
    geometry = config["geometry"]
    masses = config["masses"]
    motor = config["motor"]

    n_samples = 120
    period = 10.0
    kinematics = compute_kinematics(geometry, n_samples, period)
    t = kinematics.t

    # произвольный P_ТПШ(t)
    p_true = 15000.0 + 5000.0 * np.sin(2 * np.pi * t / period) + 2000.0 * np.cos(4 * np.pi * t / period)

    # прямая модель (2.27), согласованная с (2.30)–(2.31)
    p_measured = compute_direct_power(p_true, kinematics, geometry, masses, motor)
    p_delta, p_r = compute_parasitic_powers(kinematics, geometry, masses, motor)

    inverse = compute_rod_load(p_measured, p_delta, p_r, kinematics.v, geometry, motor)
    p_recovered = inverse.p_tpsh

    valid = ~inverse.v_near_zero & np.isfinite(p_recovered)
    assert np.sum(valid) > n_samples * 0.9

    err = np.max(np.abs(p_recovered[valid] - p_true[valid]))
    assert err < ATOL, f"max |P_recovered - P_true| = {err:.2e} >= {ATOL}"


def test_decomposed_equals_direct():
    """P_XX − Pδ − PR + coeff·P_ТПШ·v == compute_direct_power."""
    config = _load_demo_config()
    geometry = config["geometry"]
    masses = config["masses"]
    motor = config["motor"]

    kinematics = compute_kinematics(geometry, 80, 9.0)
    p_true = np.full(80, 18000.0)

    p_direct = compute_direct_power(p_true, kinematics, geometry, masses, motor)
    p_delta, p_r = compute_parasitic_powers(kinematics, geometry, masses, motor)
    coeff = geometry["l1"] / (geometry["l2"] * motor["eta_ed"] * motor["eta_per"])
    p_decomposed = motor["P_xx_w"] - p_delta - p_r + coeff * p_true * kinematics.v

    assert np.allclose(p_direct, p_decomposed, atol=ATOL)
