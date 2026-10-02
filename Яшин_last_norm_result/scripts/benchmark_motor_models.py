"""Бенчмарк всех моделей I→P: калибровка + метрики на скв. 16427."""

from __future__ import annotations

import copy
import json
import logging
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_dynamogram import read_reference_dynamogram
from src.io_tokogram import read_tokogram
from src.motor_model import available_methods, default_u_eff
from src.pipeline import _cycle_mape, _dead_mask, process_cycle
from src.segmentation import segment_all_cycles, segment_tokogram
from src.validation import validate_dynamograms

CONFIG_PATH = ROOT / "config.yaml"
OUT_PATH = ROOT / "output" / "motor_models_benchmark.json"

MODEL_META = {
    "kloss": {"kind": "physical", "formula": "Kloss M(s(I)), P=ωM/(η_ЭДη_пер)+P_XX"},
    "kloss_modified": {"kind": "physical", "formula": "Kloss − M_loss, M_loss=P_const/ω_sync"},
    "linear": {"kind": "physical", "formula": "M=k(I−I_XX), P=ωM/(η_ЭДη_пер)+P_XX"},
    "equivalent_circuit": {"kind": "physical", "formula": "Г-схема: I(s), M_em(s), P_mech"},
    "quadratic": {"kind": "regression", "formula": "P=P_XX+aI+bI²+cI³"},
    "poly2": {"kind": "regression", "formula": "P=P_XX+aI+bI²"},
    "affine": {"kind": "regression", "formula": "P=aI+b"},
    "ui": {"kind": "physical", "formula": "P=U_eff·I"},
    "ui_bias": {"kind": "physical", "formula": "P=U_eff·I+P_bias"},
}


@contextmanager
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in ("src.pipeline", "src.yashin_inverse", "src.io_tokogram"):
            logging.getLogger(name).setLevel(logging.ERROR)
        yield


def _pick_cycle(tok_current, config, dt, ref_s, ref_f, val_cfg):
    seg = segment_tokogram(tok_current, dt_seconds=dt)
    all_cycles = segment_all_cycles(
        tok_current, seg,
        prominence_fraction=config["segmentation"]["prominence_fraction"],
    )
    period_tol = seg.period_samples or 56.0
    candidates = [
        c for c in all_cycles
        if 0.85 * period_tol <= c.length_samples <= 1.15 * period_tol
    ] or all_cycles

    best_cyc, best_mape = candidates[0], float("inf")
    for cyc in candidates:
        with _quiet():
            res = process_cycle(cyc.current, None, config, dt_seconds=dt)
            mape = _cycle_mape(res, ref_s, ref_f, val_cfg)
        if mape < best_mape:
            best_mape, best_cyc = mape, cyc
    return best_cyc, seg


def _metrics(config, cycle, dt, ref_s, ref_f, val_cfg) -> dict:
    with _quiet():
        result = process_cycle(cycle.current, None, config, dt_seconds=dt)
    dead = _dead_mask(result)
    m = validate_dynamograms(
        result["s"], result["f"], ref_s, ref_f,
        n_resample_points=val_cfg.get("n_resample_points", 200),
        calc_force_unit="N",
        ref_force_unit="kgf",
        v_near_zero=dead,
    )
    return {
        "rmse_f": float(m.rmse_f),
        "mape_f": float(m.mape_f),
        "area_diff_percent": float(m.area_diff_percent),
        "s0_m": float(result["kinematics"].S0),
    }


def _score(m: dict) -> float:
    s = m["rmse_f"]
    if m["mape_f"] > 10:
        s += 50 * (m["mape_f"] - 10)
    return s


def _calibrate_method(
    base: dict,
    method: str,
    cycle,
    dt,
    ref_s,
    ref_f,
    val_cfg,
) -> tuple[dict, dict]:
    """Подбор свободных параметров модели (если нужно)."""
    cfg = copy.deepcopy(base)
    cfg["current_to_torque"] = {"method": method}
    motor = cfg["motor"]

    if method in ("kloss", "linear", "quadratic", "kloss_modified", "equivalent_circuit"):
        return cfg, {}

    def eval_params(extra: dict) -> float:
        c = copy.deepcopy(cfg)
        c["motor"].update(extra)
        try:
            return _score(_metrics(c, cycle, dt, ref_s, ref_f, val_cfg))
        except Exception:
            return 1e12

    fitted: dict = {}

    def run_de(obj, bounds, x0, maxiter=40):
        res = differential_evolution(
            obj, bounds, x0=x0, maxiter=maxiter, popsize=8,
            seed=42, polish=True, tol=0.01,
        )
        return res.x

    if method == "affine":
        x0 = np.array([
            float(motor.get("P_a_w_per_a", 400)),
            float(motor.get("P_xx_w", 3000)),
        ])
        bounds = [(80, 900), (500, 12000)]

        def obj(x):
            return eval_params({"P_a_w_per_a": x[0], "P_bias_w": x[1]})

        x = run_de(obj, bounds, x0)
        fitted = {"P_a_w_per_a": float(x[0]), "P_bias_w": float(x[1])}

    elif method == "poly2":
        x0 = np.array([
            float(motor.get("P_a_w_per_a", 400)),
            float(motor.get("P_b_w_per_a2", 0)),
        ])
        bounds = [(80, 900), (-150, 50)]

        def obj(x):
            return eval_params({"P_a_w_per_a": x[0], "P_b_w_per_a2": x[1]})

        x = run_de(obj, bounds, x0)
        fitted = {"P_a_w_per_a": float(x[0]), "P_b_w_per_a2": float(x[1])}

    elif method == "ui":
        u0 = float(motor.get("U_eff_v") or default_u_eff(motor))
        x0 = np.array([u0])
        bounds = [(120, 650)]

        def obj(x):
            return eval_params({"U_eff_v": x[0]})

        x = run_de(obj, bounds, x0, maxiter=30)
        fitted = {"U_eff_v": float(x[0])}

    elif method == "ui_bias":
        u0 = float(motor.get("U_eff_v") or default_u_eff(motor))
        x0 = np.array([u0, float(motor.get("P_xx_w", 3000))])
        bounds = [(120, 650), (500, 12000)]

        def obj(x):
            return eval_params({"U_eff_v": x[0], "P_bias_w": x[1]})

        x = run_de(obj, bounds, x0)
        fitted = {"U_eff_v": float(x[0]), "P_bias_w": float(x[1])}

    cfg["motor"].update(fitted)
    return cfg, fitted


def run_benchmark(config_path: Path = CONFIG_PATH) -> dict:
    logging.getLogger().setLevel(logging.ERROR)
    with open(config_path, encoding="utf-8") as f:
        base = yaml.safe_load(f)

    dt = float(base["tokogram"]["dt_seconds"])
    val_cfg = base.get("validation", {})

    with _quiet():
        tok = read_tokogram(base["paths"]["tokogram_xlsx"], dt_seconds=dt)
    ref = read_reference_dynamogram(base["paths"]["reference_dynamogram_txt"])

    cycle, _ = _pick_cycle(tok.current_a, base, dt, ref.s_m, ref.f_force, val_cfg)

    results: list[dict] = []
    for method in available_methods():
        print(f"=== {method} ===", flush=True)
        cfg, fitted = _calibrate_method(
            base, method, cycle, dt, ref.s_m, ref.f_force, val_cfg,
        )
        met = _metrics(cfg, cycle, dt, ref.s_m, ref.f_force, val_cfg)
        row = {
            "method": method,
            **MODEL_META[method],
            "fitted_params": fitted,
            "metrics": met,
        }
        results.append(row)
        print(
            f"  RMSE={met['rmse_f']:.1f} N  MAPE={met['mape_f']:.2f}%  "
            f"areaΔ={met['area_diff_percent']:.1f}%",
            flush=True,
        )
        if fitted:
            print(f"  fitted: {fitted}", flush=True)

    results.sort(key=lambda r: r["metrics"]["rmse_f"])
    out = {
        "config": str(config_path),
        "s0_m": base["geometry"].get("S0_m"),
        "cycle_start": int(cycle.start),
        "cycle_length": int(cycle.length_samples),
        "models": results,
        "best_method": results[0]["method"],
    }
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUT_PATH}", flush=True)
    return out


if __name__ == "__main__":
    run_benchmark()
