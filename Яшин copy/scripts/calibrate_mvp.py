"""Калибровка MVP по ветвям S-F (физическая форма петли)."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_dynamogram import read_reference_dynamogram
from src.io_tokogram import read_tokogram
from src.pipeline import _dead_mask, process_cycle
from src.segmentation import segment_all_cycles, segment_tokogram
from src.validation import branch_force_metrics, validate_dynamograms

CONFIG_PATH = ROOT / "config_well16427.yaml"


def _pick_cycle(tok_current, config, dt, ref_s, ref_f):
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

    best_cyc, best_score = candidates[0], float("inf")
    for cyc in candidates:
        res = process_cycle(cyc.current, None, config, dt_seconds=dt)
        br = branch_force_metrics(
            res["s"], res["f"], ref_s, ref_f, v_calc=res["kinematics"].v,
        )
        score = br["rmse_f_downstroke"] + br["rmse_f_upstroke"]
        if score < best_score:
            best_score, best_cyc = score, cyc
    return best_cyc


def _metrics(config, cycle, dt, ref_s, ref_f):
    res = process_cycle(cycle.current, None, config, dt_seconds=dt)
    dead = _dead_mask(res)
    m = validate_dynamograms(
        res["s"], res["f"], ref_s, ref_f,
        n_resample_points=200,
        calc_force_unit="N", ref_force_unit="kgf", v_near_zero=dead,
        v_calc=res["kinematics"].v,
    )
    br = branch_force_metrics(
        res["s"], res["f"], ref_s, ref_f, v_calc=res["kinematics"].v,
    )
    return {**br, "rmse_f": m.rmse_f, "mape_f": m.mape_f, "area_diff": m.area_diff_percent}


def _objective_score(met: dict, ref_ratio: float = 2.29) -> float:
    ratio_err = abs(met["branch_ratio_calc"] - ref_ratio) if np.isfinite(met["branch_ratio_calc"]) else 10.0
    return (
        1.5 * met["rmse_f_downstroke"]
        + 0.8 * met["rmse_f_upstroke"]
        + 8000 * ratio_err
        + 50 * met["area_diff"] / 100.0
        + 0.02 * met["mape_f"] * met["rmse_f"]
    )


def _apply_x(cfg: dict, x: np.ndarray) -> None:
    cfg["motor"]["P0_w"] = x[0]
    cfg["motor"]["P_a_w_per_a"] = x[1]
    cfg["motor"]["P_b_w_per_a2"] = x[2]
    cfg["motor"]["P_c_w_per_a3"] = x[3]
    cfg["inverse"]["k_tf"] = x[4]
    cfg["inverse"]["k_v"] = x[4]
    cfg["parasitic"] = {
        "mode": "harmonic2",
        "P0": x[5],
        "P1": x[6],
        "phi": x[7],
        "P2": x[8],
        "phi2": x[9],
    }
    cfg["pump_load"] = {
        "mode": "force_profile",
        "F0_down_n": x[10],
        "F1_down_n": x[11],
        "F_up_n": x[12],
    }
    cfg["tokogram"]["i_phase_shift_samples"] = int(round(x[13]))
    cfg["kinematics"]["S0_m"] = x[14]


def calibrate(apply: bool = False, maxiter: int = 120) -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    base["model"] = {"mode": "mvp"}
    base.setdefault("inverse", {})
    base["inverse"]["method"] = "torque_factor"
    base["inverse"]["interpolate_dead_points"] = True
    base.setdefault("pump_load", {"mode": "force_profile"})
    base.setdefault("tokogram", {})["i_phase_shift_samples"] = 0

    dt = float(base["tokogram"]["dt_seconds"])
    tok = read_tokogram(base["paths"]["tokogram_xlsx"], dt_seconds=dt)
    ref = read_reference_dynamogram(base["paths"]["reference_dynamogram_txt"])
    cycle = _pick_cycle(tok.current_a, base, dt, ref.s_m, ref.f_force)
    ref_ratio = branch_force_metrics(
        ref.s_m, ref.f_force * 9.80665, ref.s_m, ref.f_force,
        v_calc=None, calc_force_unit="N", ref_force_unit="kgf",
    )["branch_ratio_ref"]

    motor = base["motor"]
    par = base.get("parasitic", {})
    pump = base.get("pump_load", {})
    inv = base["inverse"]

    bounds = [
        (2800, 3800),      # P0
        (80, 500),         # a
        (-60, 5),          # b
        (-2, 3),           # c
        (0.22, 0.55),      # k_tf
        (-2000, 3000),     # P0 par
        (0, 5000),         # P1 par
        (-np.pi, np.pi),   # phi
        (-2000, 2000),     # P2 par
        (-np.pi, np.pi),   # phi2
        (5000, 35000),     # F0_down
        (-5000, 15000),    # F1_down
        (0, 8000),         # F_up
        (-5, 5),           # i_phase_shift
        (3.3, 3.6),        # S0
    ]
    x0 = np.array([
        motor.get("P0_w", 3266),
        motor.get("P_a_w_per_a", 200),
        motor.get("P_b_w_per_a2", -20),
        motor.get("P_c_w_per_a3", 0.5),
        inv.get("k_tf", 0.35),
        par.get("P0", 0),
        par.get("P1", 2000),
        par.get("phi", 0),
        par.get("P2", 0),
        par.get("phi2", 0),
        pump.get("F0_down_n", 20000),
        pump.get("F1_down_n", 0),
        pump.get("F_up_n", 0),
        base["tokogram"].get("i_phase_shift_samples", 0),
        base["kinematics"].get("S0_m", 3.5),
    ])
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

    best: dict = {"score": np.inf}

    def objective(x):
        cfg = copy.deepcopy(base)
        _apply_x(cfg, x)
        try:
            met = _metrics(cfg, cycle, dt, ref.s_m, ref.f_force)
        except Exception:
            return 1e12
        score = _objective_score(met, ref_ratio)
        if score < best["score"]:
            best.update(score=score, x=x.copy(), **met)
            print(
                f"F↓RMSE={met['rmse_f_downstroke']:.0f} F↑RMSE={met['rmse_f_upstroke']:.0f} "
                f"F↓/F↑={met['branch_ratio_calc']:.2f} MAPE={met['mape_f']:.1f}% "
                f"areaΔ={met['area_diff']:.0f}%",
                flush=True,
            )
        return score

    print("Baseline:", _metrics(base, cycle, dt, ref.s_m, ref.f_force))
    print(f"Ref F↓/F↑={ref_ratio:.2f}")
    differential_evolution(
        objective, bounds, x0=x0, maxiter=maxiter, popsize=15, seed=42,
        polish=True, mutation=(0.5, 1.2), recombination=0.8,
    )

    x = best["x"]
    result = {
        "kinematics": {"S0_m": round(float(x[14]), 4)},
        "motor": {
            "P0_w": round(float(x[0]), 2),
            "P_a_w_per_a": round(float(x[1]), 4),
            "P_b_w_per_a2": round(float(x[2]), 4),
            "P_c_w_per_a3": round(float(x[3]), 4),
        },
        "inverse": {
            "method": "torque_factor",
            "k_tf": round(float(x[4]), 4),
            "k_v": round(float(x[4]), 4),
            "interpolate_dead_points": True,
        },
        "parasitic": {
            "mode": "harmonic2",
            "P0": round(float(x[5]), 2),
            "P1": round(float(x[6]), 2),
            "phi": round(float(x[7]), 4),
            "P2": round(float(x[8]), 2),
            "phi2": round(float(x[9]), 4),
        },
        "pump_load": {
            "mode": "force_profile",
            "F0_down_n": round(float(x[10]), 1),
            "F1_down_n": round(float(x[11]), 1),
            "F_up_n": round(float(x[12]), 1),
        },
        "tokogram": {"i_phase_shift_samples": int(round(x[13]))},
        "metrics": {
            "rmse_f_up": round(best["rmse_f_upstroke"], 1),
            "rmse_f_dn": round(best["rmse_f_downstroke"], 1),
            "branch_ratio": round(best["branch_ratio_calc"], 3),
            "mape_f": round(best["mape_f"], 2),
            "area_diff": round(best["area_diff"], 1),
        },
    }
    out = ROOT / "output" / "calibration_mvp.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nBest:", json.dumps(result, indent=2))

    if apply:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["kinematics"].update(result["kinematics"])
        cfg["motor"].update(result["motor"])
        cfg["inverse"].update(result["inverse"])
        cfg["parasitic"] = result["parasitic"]
        cfg["pump_load"] = result["pump_load"]
        cfg.setdefault("tokogram", {})["i_phase_shift_samples"] = result["tokogram"]["i_phase_shift_samples"]
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
        print(f"Updated {CONFIG_PATH}")

    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true")
    p.add_argument("--maxiter", type=int, default=120)
    calibrate(apply=p.parse_args().apply, maxiter=p.parse_args().maxiter)
