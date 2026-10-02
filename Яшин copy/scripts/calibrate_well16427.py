"""Подбор параметров config_well16427 при фиксированном S0_m=3.5."""

from __future__ import annotations

import argparse
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
from src.pipeline import _cycle_mape, _dead_mask, process_cycle
from src.segmentation import segment_all_cycles, segment_tokogram
from src.validation import validate_dynamograms

CONFIG_PATH = ROOT / "config_well16427.yaml"

# Критерии ранней остановки (можно переопределить через CLI)
DEFAULT_TARGET_RMSE = 1200.0   # N
DEFAULT_TARGET_MAPE = 6.0      # %
DEFAULT_PATIENCE = 25          # поколений DE без улучшения score
DEFAULT_MAX_EVALS = 400


@contextmanager
def _quiet_pipeline():
    """Подавляет шумные предупреждения при массовых прогонах оптимизации."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loggers = (
            "src.pipeline",
            "src.yashin_inverse",
            "src.io_tokogram",
            "src.yashin_kinematics",
            "src.yashin_power_balance",
        )
        prev = {name: logging.getLogger(name).level for name in loggers}
        for name in loggers:
            logging.getLogger(name).setLevel(logging.ERROR)
        try:
            yield
        finally:
            for name, level in prev.items():
                logging.getLogger(name).setLevel(level)


class EarlyStopState:
    """Состояние ранней остановки (проверяется в callback DE)."""

    def __init__(
        self,
        target_rmse: float,
        target_mape: float,
        patience: int,
        max_evals: int,
    ):
        self.target_rmse = target_rmse
        self.target_mape = target_mape
        self.patience = patience
        self.max_evals = max_evals
        self.generation = 0
        self.generations_without_improve = 0
        self.last_best_score = np.inf
        self.reason: str | None = None
        self.halt = False

    def note_generation(self, evaluator: "Evaluator") -> bool:
        """Вызывается в callback DE после каждого поколения."""
        self.generation += 1
        best = evaluator.best
        rmse = best.get("rmse_f", np.inf)
        mape = best.get("mape_f", np.inf)
        score = best.get("score", np.inf)

        if rmse <= self.target_rmse and mape <= self.target_mape:
            self.reason = f"цель достигнута: RMSE={rmse:.1f} N, MAPE={mape:.2f}%"
            self.halt = True
            return True

        if score < self.last_best_score - 1.0:
            self.last_best_score = score
            self.generations_without_improve = 0
        else:
            self.generations_without_improve += 1

        if self.generations_without_improve >= self.patience:
            self.reason = (
                f"нет улучшения {self.patience} поколений "
                f"(лучший RMSE={rmse:.1f}, MAPE={mape:.2f}%)"
            )
            self.halt = True
            return True
        return False


class Evaluator:
    """Кэширует сегментацию и рабочий цикл; оценивает метрики на одном цикле ≈1·T."""

    def __init__(self, base: dict, dt: float, tok_current, ref_s, ref_f, val_cfg: dict):
        self.base = base
        self.dt = dt
        self.ref_s = ref_s
        self.ref_f = ref_f
        self.val_cfg = val_cfg
        self.eval_count = 0
        self.max_evals = DEFAULT_MAX_EVALS
        self.best: dict = {"score": np.inf}
        self.halt = False

        seg_cfg = base.get("segmentation", {})
        with _quiet_pipeline():
            self.seg = segment_tokogram(tok_current, dt_seconds=dt)
            all_cycles = segment_all_cycles(
                tok_current, self.seg,
                prominence_fraction=seg_cfg.get("prominence_fraction", 0.3),
            )
        period_tol = self.seg.period_samples or 56.0
        candidates = [
            c for c in all_cycles
            if 0.85 * period_tol <= c.length_samples <= 1.15 * period_tol
        ] or all_cycles

        best_mape = float("inf")
        self.fixed_cycle = candidates[0]
        for cyc in candidates:
            with _quiet_pipeline():
                res = process_cycle(cyc.current, None, base, dt_seconds=dt)
                mape = _cycle_mape(res, ref_s, ref_f, val_cfg)
            if mape < best_mape:
                best_mape = mape
                self.fixed_cycle = cyc

        print(
            f"Рабочий цикл: start={self.fixed_cycle.start}, "
            f"n={self.fixed_cycle.length_samples}, baseline MAPE={best_mape:.2f}%",
            flush=True,
        )

    def metrics(self, config: dict) -> dict:
        with _quiet_pipeline():
            result = process_cycle(
                self.fixed_cycle.current, None, config, dt_seconds=self.dt,
            )
        dead = _dead_mask(result)
        m = validate_dynamograms(
            result["s"], result["f"], self.ref_s, self.ref_f,
            n_resample_points=self.val_cfg.get("n_resample_points", 200),
            calc_force_unit="N",
            ref_force_unit="kgf",
            v_near_zero=dead,
        )
        return {"rmse_f": m.rmse_f, "mape_f": m.mape_f, "area_diff": m.area_diff_percent}

    def objective(self, x: np.ndarray) -> float:
        if self.halt:
            return float(self.best.get("score", 1e12))

        self.eval_count += 1
        if self.eval_count > self.max_evals:
            self.halt = True
            return float(self.best.get("score", 1e12))

        cfg = _apply_x(self.base, x)
        try:
            met = self.metrics(cfg)
        except Exception:
            return 1e12

        score = met["rmse_f"]
        if met["mape_f"] > 10:
            score += 50 * (met["mape_f"] - 10)
        if met["mape_f"] > 50:
            score += 5000

        if score < self.best["score"]:
            self.best.update(score=score, x=x.copy(), **met)
            print(
                f"[{self.eval_count}] RMSE={met['rmse_f']:.1f} "
                f"MAPE={met['mape_f']:.2f}% areaΔ={met['area_diff']:.1f}%",
                flush=True,
            )
        return score


def _apply_x(config: dict, x: np.ndarray) -> dict:
    cfg = copy.deepcopy(config)
    (
        cfg["motor"]["P_xx_w"],
        cfg["motor"]["P_a_w_per_a"],
        cfg["motor"]["P_b_w_per_a2"],
        cfg["motor"]["P_c_w_per_a3"],
        cfg["masses"]["G"],
        cfg["masses"]["Q_delta"],
        cfg["masses"]["Q_R"],
        cfg["geometry"]["l1"],
        cfg["geometry"]["l2"],
    ) = x
    cfg["geometry"]["S0_m"] = 3.5
    return cfg


def _apply_to_yaml(config_path: Path, cfg_opt: dict) -> None:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k in ("P_xx_w", "P_a_w_per_a", "P_b_w_per_a2", "P_c_w_per_a3"):
        cfg["motor"][k] = round(float(cfg_opt["motor"][k]), 4 if k != "P_xx_w" else 2)
    for k in ("G", "Q_delta", "Q_R"):
        cfg["masses"][k] = round(float(cfg_opt["masses"][k]), 2)
    for k in ("l1", "l2"):
        cfg["geometry"][k] = round(float(cfg_opt["geometry"][k]), 4)
    cfg["geometry"]["S0_m"] = 3.5
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Калибровка config_well16427 при S0_m=3.5")
    p.add_argument("--target-rmse", type=float, default=DEFAULT_TARGET_RMSE)
    p.add_argument("--target-mape", type=float, default=DEFAULT_TARGET_MAPE)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                   help="Поколений DE без улучшения score")
    p.add_argument("--max-evals", type=int, default=DEFAULT_MAX_EVALS)
    p.add_argument("--maxiter", type=int, default=60)
    p.add_argument("--popsize", type=int, default=10)
    p.add_argument("--apply", action="store_true",
                   help="Записать лучшие параметры в config_well16427.yaml и config.yaml")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.getLogger().setLevel(logging.ERROR)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        base = yaml.safe_load(f)

    dt = float(base["tokogram"]["dt_seconds"])
    with _quiet_pipeline():
        tok = read_tokogram(base["paths"]["tokogram_xlsx"], dt_seconds=dt)
    ref = read_reference_dynamogram(base["paths"]["reference_dynamogram_txt"])
    val_cfg = base.get("validation", {})

    x0 = np.array([
        base["motor"]["P_xx_w"],
        base["motor"]["P_a_w_per_a"],
        base["motor"]["P_b_w_per_a2"],
        base["motor"]["P_c_w_per_a3"],
        base["masses"]["G"],
        base["masses"]["Q_delta"],
        base["masses"]["Q_R"],
        base["geometry"]["l1"],
        base["geometry"]["l2"],
    ])

    bounds = [
        (500, 8000),
        (50, 1200),
        (-200, 200),
        (-5, 5),
        (200, 2000),
        (50, 800),
        (500, 4000),
        (0.8, 4.0),
        (1.5, 5.0),
    ]

    evalr = Evaluator(base, dt, tok.current_a, ref.s_m, ref.f_force, val_cfg)
    evalr.max_evals = args.max_evals
    stop_state = EarlyStopState(
        target_rmse=args.target_rmse,
        target_mape=args.target_mape,
        patience=args.patience,
        max_evals=args.max_evals,
    )

    print("Baseline:")
    b0 = evalr.metrics(base)
    print(f"  RMSE={b0['rmse_f']:.1f} MAPE={b0['mape_f']:.2f}%")
    print(
        f"Stop: RMSE≤{args.target_rmse}, MAPE≤{args.target_mape}%, "
        f"patience={args.patience} gen, max_evals={args.max_evals}",
        flush=True,
    )

    def callback(_xk, _convergence=0.0) -> bool:
        if evalr.halt and evalr.eval_count > evalr.max_evals and not stop_state.reason:
            stop_state.reason = f"достигнут лимит оценок ({evalr.max_evals})"
        if stop_state.note_generation(evalr):
            evalr.halt = True
            print(f"\n⏹ Early stop: {stop_state.reason}", flush=True)
            return True
        if evalr.halt:
            if not stop_state.reason:
                stop_state.reason = f"достигнут лимит оценок ({evalr.max_evals})"
            print(f"\n⏹ Early stop: {stop_state.reason}", flush=True)
            return True
        return False

    result = differential_evolution(
        evalr.objective,
        bounds,
        x0=x0,
        maxiter=args.maxiter,
        popsize=args.popsize,
        mutation=(0.5, 1.2),
        recombination=0.8,
        seed=42,
        polish=not stop_state.halt,
        workers=1,
        updating="deferred",
        tol=0.01,
        callback=callback,
    )

    x_best = evalr.best.get("x", result.x)
    cfg_opt = _apply_x(base, x_best)
    final = evalr.metrics(cfg_opt)

    print("\n=== Best ===")
    print(f"RMSE F = {final['rmse_f']:.2f} N")
    print(f"MAPE F = {final['mape_f']:.2f} %")
    print(f"Δ area = {final['area_diff']:.2f} %")
    print(f"Evaluations: {evalr.eval_count}")
    if stop_state.reason:
        print(f"Stop reason: {stop_state.reason}")
    print("\nParameters:")
    for k in ("P_xx_w", "P_a_w_per_a", "P_b_w_per_a2", "P_c_w_per_a3"):
        print(f"  motor.{k}: {cfg_opt['motor'][k]:.4f}")
    for k in ("G", "Q_delta", "Q_R"):
        print(f"  masses.{k}: {cfg_opt['masses'][k]:.2f}")
    for k in ("l1", "l2"):
        print(f"  geometry.{k}: {cfg_opt['geometry'][k]:.4f}")

    out = {
        "motor": {k: round(float(cfg_opt["motor"][k]), 4 if k != "P_xx_w" else 2)
                  for k in ("P_xx_w", "P_a_w_per_a", "P_b_w_per_a2", "P_c_w_per_a3")},
        "masses": {k: round(float(cfg_opt["masses"][k]), 2) for k in ("G", "Q_delta", "Q_R")},
        "geometry": {k: round(float(cfg_opt["geometry"][k]), 4) for k in ("l1", "l2")},
        "metrics": {k: round(float(final[k]), 4) for k in final},
        "stop_reason": stop_state.reason,
        "evaluations": evalr.eval_count,
    }
    out_path = ROOT / "output" / "calibration_s0_35.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {out_path}")

    if args.apply:
        for path in (CONFIG_PATH, ROOT / "config.yaml"):
            _apply_to_yaml(path, cfg_opt)
            print(f"Updated {path}")


if __name__ == "__main__":
    main()
