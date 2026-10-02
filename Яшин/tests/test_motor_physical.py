"""Тесты физических моделей двигателя: equivalent_circuit, kloss_modified."""

import numpy as np
import yaml

from src.motor_model import (
    estimate_equivalent_circuit_params,
    estimate_p_const_loss,
    torque_from_current,
    torque_from_current_equivalent_circuit,
    torque_from_current_kloss_modified,
)

CONFIG_DEMO = "config_demo.yaml"


def _motor():
    with open(CONFIG_DEMO, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["motor"], cfg["geometry"]["r"]


def test_estimate_equivalent_circuit_positive():
    motor, _ = _motor()
    ec = estimate_equivalent_circuit_params(motor)
    for k in ("R1_ohm", "R2_p_ohm", "X1_ohm", "X2_p_ohm", "Xm_ohm"):
        assert ec[k] > 0, k


def test_equivalent_circuit_finite():
    motor, r = _motor()
    i = np.linspace(20, 40, 50)
    res = torque_from_current_equivalent_circuit(i, motor, r)
    assert np.all(np.isfinite(res.M_ed))
    assert np.all(np.isfinite(res.P))
    assert np.all(res.M_crank > 0)


def test_kloss_modified_finite():
    motor, r = _motor()
    i = np.linspace(20, 40, 50)
    res = torque_from_current_kloss_modified(i, motor, r)
    assert np.all(np.isfinite(res.M_ed))
    assert np.all(res.M_ed >= 0)


def test_torque_from_current_dispatches_new_methods():
    motor, r = _motor()
    i = np.linspace(20, 40, 30)
    for method in ("equivalent_circuit", "kloss_modified"):
        res = torque_from_current(i, motor, r, method=method)
        assert np.all(np.isfinite(res.M_crank))


def test_p_const_loss_estimate():
    motor, _ = _motor()
    p = estimate_p_const_loss(motor)
    assert p >= 0
