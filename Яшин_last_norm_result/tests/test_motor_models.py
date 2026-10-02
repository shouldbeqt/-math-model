"""Unit-тесты новых моделей I→P и бенчмарк метрик."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.motor_model import (
    available_methods,
    compute_power_from_current,
    default_u_eff,
    torque_from_current,
)

CONFIG_DEMO = "config_demo.yaml"
CONFIG_WELL = "config_well16427.yaml"
BENCHMARK_JSON = Path("output/motor_models_benchmark.json")


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("method", ["affine", "poly2", "ui", "ui_bias"])
def test_new_power_models_finite(method: str):
    config = _load(CONFIG_DEMO)
    motor = {**config["motor"], "P_a_w_per_a": 400, "P_b_w_per_a2": -10, "P_bias_w": 800}
    i = np.linspace(15, 35, 50)
    result = torque_from_current(
        i, motor, config["geometry"]["r"], method=method,
    )
    assert np.all(np.isfinite(result.P))
    assert np.all(np.isfinite(result.M_ed))


def test_affine_power_formula():
    motor = {"P_xx_w": 1000, "P_a_w_per_a": 50, "P_bias_w": 200}
    p = compute_power_from_current(np.array([10.0, 20.0]), motor, "affine")
    np.testing.assert_allclose(p, [700, 1200])


def test_ui_default_u_eff():
    motor = {"U_nom_v": 380, "cos_phi_nom": 0.86, "P_xx_w": 0}
    u = default_u_eff(motor)
    assert 500 < u < 600
    p = compute_power_from_current(np.array([10.0]), motor, "ui")
    np.testing.assert_allclose(p, [u * 10])


def test_available_methods_includes_new():
    methods = available_methods()
    for m in ("affine", "poly2", "ui", "ui_bias"):
        assert m in methods


def test_motor_models_benchmark_runs():
    """Интеграционный бенчмарк: все модели, метрики записаны в JSON."""
    import sys
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.benchmark_motor_models import run_benchmark

    out = run_benchmark(Path("config.yaml"))
    assert len(out["models"]) == len(available_methods())
    assert BENCHMARK_JSON.exists()
    data = json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))
    for row in data["models"]:
        assert row["metrics"]["rmse_f"] > 0
        assert row["metrics"]["mape_f"] >= 0
