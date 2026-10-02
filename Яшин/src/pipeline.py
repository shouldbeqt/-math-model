"""Полный расчётный конвейер (Раздел Г ТЗ)."""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.io_dynamogram import read_reference_dynamogram
from src.io_tokogram import read_tokogram
from src.motor_model import torque_from_current
from src.segmentation import (
    CycleSlice,
    SegmentationResult,
    plot_segmentation,
    segment_all_cycles,
    segment_tokogram,
)
from src.model_summary import write_model_summary
from src.physics.rod_wave import (
    is_wave_model_enabled,
    plot_head_vs_surface_force,
    plot_wave_propagation,
    save_displacement_field,
)
from src.validation import (
    plot_dynamogram_overlay,
    plot_force_vs_phase,
    plot_metrics_summary,
    save_metrics,
    split_calc_strokes_for_plot,
    split_reference_branches,
    strip_dead_points,
    validate_dynamograms,
    _apply_v_near_zero_mask,
    _sanitize_force,
)
from src.yashin_inverse import compute_rod_load, compute_rod_load_tf
from src.yashin_kinematics import compute_kinematics_for_config
from src.yashin_mvp import process_cycle_mvp

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

NULL_VALUES = {None, "null", "None", ""}


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_null(value: Any) -> bool:
    return value in NULL_VALUES


def validate_config(config: dict, demo: bool = False) -> None:
    if demo:
        return
    mode = config.get("model", {}).get("mode", "full")
    if mode == "mvp":
        missing_mvp: list[str] = []
        if _is_null(config.get("kinematics", {}).get("S0_m")):
            missing_mvp.append("kinematics.S0_m")
        motor = config.get("motor", {})
        for key in ("P0_w", "P_a_w_per_a", "P_b_w_per_a2", "eta_ed", "eta_per", "n_red", "n_kp", "s_nom"):
            if key not in motor and key == "P0_w" and motor.get("P_xx_w") is not None:
                continue
            if _is_null(motor.get(key)) and not (key == "P0_w" and motor.get("P_xx_w")):
                missing_mvp.append(f"motor.{key}")
        if _is_null(config.get("inverse", {}).get("k_v")) and _is_null(
            config.get("inverse", {}).get("k_tf")
        ):
            missing_mvp.append("inverse.k_v или inverse.k_tf")
        if _is_null(config.get("tokogram", {}).get("dt_seconds")):
            missing_mvp.append("tokogram.dt_seconds")
        if missing_mvp:
            raise ValueError("MVP config: незаполнено:\n  " + "\n  ".join(missing_mvp))
        return

    missing: list[str] = []
    for key, val in config.get("geometry", {}).items():
        if key == "S0_m":
            continue  # опционально: константа хода вместо (2.19)
        if _is_null(val):
            missing.append(f"geometry.{key}")
    for key, val in config.get("masses", {}).items():
        if _is_null(val):
            missing.append(f"masses.{key}")
    for key in ("P_nom_kw", "I_nom_a", "n_red", "n_kp", "P_xx_w", "s_nom"):
        if _is_null(config.get("motor", {}).get(key)):
            missing.append(f"motor.{key}")
    if _is_null(config.get("tokogram", {}).get("dt_seconds")):
        missing.append("tokogram.dt_seconds")
    if missing:
        raise ValueError(
            "config.yaml содержит незаполненные параметры (null):\n  "
            + "\n  ".join(missing)
        )


def _resolve_motor(motor: dict, i_cycle: np.ndarray) -> tuple[dict, bool]:
    """I_xx_a: только из конфига; валидация относительно I_nom."""
    i_xx_estimated = False
    i_nom = float(motor.get("I_nom_a") or 45.0)

    if _is_null(motor.get("I_xx_a")):
        raise ValueError(
            "motor.I_xx_a не задан. Укажите паспортный ток холостого хода "
            "(для 22 кВт типично 13–18 А, не min(I_cycle))."
        )

    i_xx = float(motor["I_xx_a"])
    if i_xx > 0.5 * i_nom:
        warnings.warn(
            f"motor.I_xx_a={i_xx:.1f} А > 50% I_nom={i_nom:.1f} А — "
            "завышенный I_XX может давать отрицательную P(I) в зоне малых токов",
            stacklevel=2,
        )
        logger.warning("I_xx=%.2f А подозрительно высок относительно I_nom=%.2f А", i_xx, i_nom)

    return motor, i_xx_estimated


def process_cycle(
    i_cycle: np.ndarray,
    period_seconds: float | None,
    config: dict,
    dt_seconds: float | None = None,
) -> dict:
    """Расчёт одного цикла: I(t) → динамограмма (S, F)."""
    n = len(i_cycle)
    if dt_seconds is not None and dt_seconds > 0:
        period = n * float(dt_seconds)
    elif period_seconds is not None and period_seconds > 0:
        period = float(period_seconds)
    else:
        raise ValueError("process_cycle: задайте dt_seconds или period_seconds")

    if config.get("model", {}).get("mode", "full") == "mvp":
        return process_cycle_mvp(i_cycle, period, config)

    return _process_cycle_full(i_cycle, period, config, dt_seconds)


def _process_cycle_full(
    i_cycle: np.ndarray,
    period: float,
    config: dict,
    dt_seconds: float | None,
) -> dict:
    """Полная физическая модель (deprecated для production, см. model.mode=mvp)."""
    motor = config["motor"]
    geometry = config["geometry"]
    masses = config["masses"]
    method = config.get("current_to_torque", {}).get("method", "kloss")
    inv_cfg = config.get("inverse", {})
    n = len(i_cycle)

    motor, _ = _resolve_motor(motor, i_cycle)

    from src.yashin_power_balance import compute_parasitic_powers

    motor_result = torque_from_current(
        i_cycle, motor, crank_radius_m=float(geometry["r"]), method=method,
    )
    kinematics = compute_kinematics_for_config(
        geometry, n_samples=n, period_seconds=period,
        kinematics_cfg=config.get("kinematics"),
    )
    p_delta, p_r = compute_parasitic_powers(kinematics, geometry, masses, motor)

    inv_method = inv_cfg.get("method", "torque_factor")
    omega_crank = 2.0 * np.pi / period

    if inv_method == "torque_factor":
        moment_src = inv_cfg.get("moment_source", "net_power")
        coeff = float(geometry["l1"]) / (
            float(geometry["l2"]) * float(motor["eta_ed"]) * float(motor["eta_per"])
        )
        if moment_src == "m_crank":
            m_for_tf = motor_result.M_crank
            tf_scale = 1.0
        elif moment_src == "net_power":
            p_xx = float(motor["P_xx_w"])
            m_for_tf = (motor_result.P - p_xx + p_delta + p_r) / omega_crank
            tf_scale = coeff
        else:
            raise ValueError(
                f"inverse.moment_source={moment_src!r}; допустимо 'm_crank' или 'net_power'"
            )
        inverse = compute_rod_load_tf(
            m_for_tf,
            kinematics.v,
            omega_crank,
            geometry=geometry,
            v_zero_threshold_fraction=inv_cfg.get("v_zero_threshold_fraction", 0.01),
            tf_min=float(inv_cfg.get("tf_min", 1e-6)),
            interpolate=inv_cfg.get("interpolate_dead_points", True),
            tf_scale=tf_scale,
        )
    elif inv_method == "power_balance":
        inverse = compute_rod_load(
            motor_result.P, p_delta, p_r, kinematics.v, geometry, motor,
            v_zero_threshold_fraction=inv_cfg.get("v_zero_threshold_fraction", 0.01),
        )
    else:
        raise ValueError(
            f"inverse.method={inv_method!r}; допустимо 'torque_factor' или 'power_balance'"
        )

    f_rod = inverse.p_tpsh
    f_abs_max = float(np.nanmax(np.abs(f_rod))) if np.any(np.isfinite(f_rod)) else 0.0
    if f_abs_max > 50000.0:
        warnings.warn(
            f"Аномальная сила на штоке: max|F|={f_abs_max:.0f} N (> 50000 N)",
            stacklevel=2,
        )
        logger.warning("Аномальная сила: max|F|=%.0f N", f_abs_max)

    return {
        "s": kinematics.S,
        "f": f_rod,
        "v_near_zero": inverse.v_near_zero,
        "kinematics": kinematics,
        "motor_result": motor_result,
        "period_seconds": period,
    }


def _dead_mask(result: dict, fraction: float = 0.08) -> np.ndarray:
    v = result["kinematics"].v
    v_max = float(np.max(np.abs(v)))
    return result["v_near_zero"] | (np.abs(v) < fraction * v_max)


def _cycle_mape(
    result: dict,
    ref_s: np.ndarray,
    ref_f: np.ndarray,
    val_cfg: dict,
) -> float:
    dead = _dead_mask(result)
    m = validate_dynamograms(
        result["s"], result["f"], ref_s, ref_f,
        n_resample_points=val_cfg.get("n_resample_points", 200),
        calc_force_unit="N",
        ref_force_unit="kgf",
        v_near_zero=dead,
        scale_stroke_to_ref=val_cfg.get("scale_stroke_to_ref", False),
        v_calc=result["kinematics"].v,
    )
    return m.mape_f


def _select_best_cycle(
    tokogram_current: np.ndarray,
    seg: SegmentationResult,
    config: dict,
    dt: float,
    ref_s: np.ndarray,
    ref_f: np.ndarray,
    val_cfg: dict,
    prominence_fraction: float,
) -> tuple[dict, int]:
    """Выбор полного цикла ≈1·T с минимальным MAPE."""
    all_cycles = segment_all_cycles(tokogram_current, seg, prominence_fraction=prominence_fraction)
    period_tol = seg.period_samples or 56.0
    candidates = [
        c for c in all_cycles
        if 0.85 * period_tol <= c.length_samples <= 1.15 * period_tol
    ]
    if not candidates:
        candidates = all_cycles

    best_result: dict | None = None
    best_idx = seg.cycle_index
    best_mape = float("inf")

    for cyc in candidates:
        res = process_cycle(cyc.current, None, config, dt_seconds=dt)
        mape = _cycle_mape(res, ref_s, ref_f, val_cfg)
        if mape < best_mape:
            best_mape = mape
            best_result = res
            best_idx = cyc.start

    if best_result is None:
        best_result = process_cycle(seg.cycle_current, None, config, dt_seconds=dt)
    logger.info("Выбран цикл [%d], MAPE=%.2f%%", best_idx, best_mape)
    return best_result, best_idx


def _plot_all_cycles_overlay(
    cycles_data: list[dict],
    ref_s: np.ndarray,
    ref_f: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    s_up, f_up, s_dn, f_dn = split_reference_branches(ref_s, ref_f)
    ax.plot(s_up, f_up, "b-", linewidth=1.5, label="Эталон ↑", zorder=10)
    ax.plot(s_dn, f_dn, "b--", linewidth=1.2, label="Эталон ↓", zorder=10)
    for i, cd in enumerate(cycles_data):
        v = cd["kinematics"].v
        dead = cd["v_near_zero"]
        su, fu, sd, fd = split_calc_strokes_for_plot(cd["s"], cd["f"], v, dead)
        fu_kgf = fu / 9.80665
        fd_kgf = fd / 9.80665
        ax.plot(su, fu_kgf, "-", linewidth=0.9, alpha=0.7, label=f"Цикл {i:02d} ↑" if i == 0 else None)
        if len(su) and len(sd):
            ax.plot(
                np.r_[su[-1], sd], np.r_[fu_kgf[-1], fd_kgf],
                "--", linewidth=0.9, alpha=0.7,
                label=f"Цикл {i:02d} ↓" if i == 0 else None,
            )
            ax.plot([sd[-1], su[0]], [fd_kgf[-1], fu_kgf[0]], "-", linewidth=0.9, alpha=0.7)
        elif len(sd):
            ax.plot(sd, fd_kgf, "--", linewidth=0.9, alpha=0.7, label=f"Цикл {i:02d} ↓" if i == 0 else None)
    ax.set_xlabel("S, м")
    ax.set_ylabel("F, kgf")
    ax.set_title("Все расчётные динамограммы vs эталон (↑/↓, |v|≈0 — хорда)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_pipeline(config_path: str | Path, demo: bool = False) -> dict:
    config = load_config(config_path)
    validate_config(config, demo=demo)

    paths = config["paths"]
    output_dir = Path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dt = config.get("tokogram", {}).get("dt_seconds")
    tokogram = read_tokogram(paths["tokogram_xlsx"], dt_seconds=dt)
    if tokogram.dt_seconds is None:
        raise ValueError("tokogram.dt_seconds не задан — невозможен расчёт")
    dt = float(tokogram.dt_seconds)

    seg_cfg = config.get("segmentation", {})
    cycle_index_cfg = seg_cfg.get("cycle_index", -1)
    seg_cycle_index = -1 if isinstance(cycle_index_cfg, str) else cycle_index_cfg

    seg = segment_tokogram(
        tokogram.current_a,
        dt_seconds=dt,
        smoothing_window=seg_cfg.get("smoothing_window", 7),
        smoothing_polyorder=seg_cfg.get("smoothing_polyorder", 2),
        spm_min=seg_cfg.get("spm_min", 3.0),
        spm_max=seg_cfg.get("spm_max", 12.0),
        prominence_fraction=seg_cfg.get("prominence_fraction", 0.3),
        period_search_margin=seg_cfg.get("period_search_margin", 0.3),
        refine_by_xcorr=seg_cfg.get("refine_by_xcorr", False),
        cycle_index=seg_cycle_index,
    )

    n_cycles = max(0, len(seg.boundaries) - 1)
    period_seconds = seg.period_seconds or 1.0

    print(f"\n=== Сегментация ===")
    print(f"Сегментов (впадина→впадина): {n_cycles}")
    print(f"Полных циклов (1 токовый пик): {len(seg.full_cycle_indices)}")
    print(f"Период: {seg.period_samples:.1f} отсч. = {period_seconds:.2f} с ({60/period_seconds:.2f} SPM)")

    plot_segmentation(tokogram.current_a, seg, output_dir / "segmentation.png", dt_seconds=dt)

    ref = read_reference_dynamogram(paths["reference_dynamogram_txt"])
    val_cfg = config.get("validation", {})
    is_mvp = config.get("model", {}).get("mode", "full") == "mvp"
    method = "mvp" if is_mvp else config.get("current_to_torque", {}).get("method", "kloss")
    i_xx_estimated = False

    process_all = isinstance(cycle_index_cfg, str) and cycle_index_cfg == "all"
    cycles_data: list[dict] = []

    if process_all:
        all_cycles = segment_all_cycles(
            tokogram.current_a, seg,
            prominence_fraction=seg_cfg.get("prominence_fraction", 0.3),
        )
        print(f"Режим 'all': обрабатывается {len(all_cycles)} полных циклов")
        for cyc in all_cycles:
            cd = process_cycle(cyc.current, None, config, dt_seconds=dt)
            cycles_data.append(cd)
            s_plot, f_plot = strip_dead_points(cd["s"], cd["f"], cd["v_near_zero"])
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(s_plot, f_plot / 9.80665, "r-")
            ax.set_xlabel("S, м")
            ax.set_ylabel("F, kgf")
            ax.set_title(f"Расчётная динамограмма, цикл {cyc.index:02d}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / f"dynamogram_cycle_{cyc.index:02d}.png", dpi=120)
            plt.close(fig)

        _plot_all_cycles_overlay(cycles_data, ref.s_m, ref.f_force, output_dir / "dynamogram_all_cycles.png")
        # для метрик — последний полный цикл
        result = cycles_data[-1] if cycles_data else process_cycle(seg.cycle_current, None, config, dt_seconds=dt)
    elif isinstance(cycle_index_cfg, str) and cycle_index_cfg == "best":
        if config.get("model", {}).get("mode", "full") != "mvp":
            motor_cfg, i_xx_estimated = _resolve_motor(config["motor"], seg.cycle_current)
            config = {**config, "motor": motor_cfg}
        result, best_start = _select_best_cycle(
            tokogram.current_a, seg, config, dt, ref.s_m, ref.f_force, val_cfg,
            prominence_fraction=seg_cfg.get("prominence_fraction", 0.3),
        )
        seg = SegmentationResult(
            boundaries=seg.boundaries,
            period_samples=seg.period_samples,
            period_seconds=result["period_seconds"],
            smoothed=seg.smoothed,
            cycle_index=seg.cycle_index,
            cycle_start=best_start,
            cycle_end=best_start + len(result["motor_result"].P),
            cycle_current=tokogram.current_a[
                best_start: best_start + len(result["motor_result"].P)
            ],
            full_cycle_indices=seg.full_cycle_indices,
        )
        cycles_data = [result]
    else:
        if config.get("model", {}).get("mode", "full") != "mvp":
            motor_cfg, i_xx_estimated = _resolve_motor(config["motor"], seg.cycle_current)
            config = {**config, "motor": motor_cfg}
        result = process_cycle(seg.cycle_current, None, config, dt_seconds=dt)
        cycles_data = [result]

    s_calc = result["s"]
    f_calc = result["f"]
    v_near_zero = result["v_near_zero"]
    v = result["kinematics"].v
    dead_mask = _dead_mask(result, fraction=0.08)

    metrics = validate_dynamograms(
        s_calc, f_calc, ref.s_m, ref.f_force,
        n_resample_points=val_cfg.get("n_resample_points", 200),
        calc_force_unit="N",
        ref_force_unit="kgf",
        v_near_zero=dead_mask,
        scale_stroke_to_ref=val_cfg.get("scale_stroke_to_ref", False),
        v_calc=v,
    )

    s_calc_plot, f_calc_plot_n = strip_dead_points(s_calc, f_calc, dead_mask)
    f_calc_plot = f_calc_plot_n / 9.80665

    extra = {
        "method": method,
        "dt_seconds": dt,
        "dt_source": tokogram.dt_source,
        "spm": round(60.0 / result["period_seconds"], 2),
        "n_cycles": n_cycles,
        "n_full_cycles": len(seg.full_cycle_indices),
        "cycle_index": seg.cycle_index,
        "I_xx_estimated": i_xx_estimated,
        "S0_m": float(result["kinematics"].S0),
        "S0_source": result["kinematics"].s0_source,
        "S_range_m": [float(s_calc.min()), float(s_calc.max())],
        "F_calc_range_N": [
            float(np.nanmin(_apply_v_near_zero_mask(f_calc, dead_mask))),
            float(np.nanmax(_apply_v_near_zero_mask(f_calc, dead_mask))),
        ],
        "F_calc_plot_kgf": [
            float(np.nanmin(f_calc_plot)),
            float(np.nanmax(f_calc_plot)),
        ],
    }
    save_metrics(metrics, output_dir / "metrics.json", extra=extra)

    plot_dynamogram_overlay(
        s_calc, f_calc, ref.s_m, ref.f_force, dead_mask,
        output_dir / "dynamogram_overlay.png",
        v_calc=v,
        calc_force_unit="N",
        ref_force_unit="kgf",
    )
    plot_force_vs_phase(
        s_calc_plot, f_calc_plot, ref.s_m, ref.f_force,
        val_cfg.get("n_resample_points", 200),
        output_dir / "force_vs_phase.png",
    )
    plot_metrics_summary(metrics, extra, output_dir / "metrics_summary.png")
    summary_path = write_model_summary(config, output_dir, metrics)
    logger.info("Сводка модели: %s", summary_path)

    wave_result = result.get("wave_result")
    if is_wave_model_enabled(config) and wave_result is not None:
        rw_cfg = config.get("rod_wave") or {}
        if rw_cfg.get("save_u", True):
            u_path = save_displacement_field(wave_result, output_dir / "rod_wave_u.npz")
            logger.info("Поле u(x,t): %s", u_path)
        plot_wave_propagation(wave_result, output_dir / "rod_wave_propagation.png")
        plot_head_vs_surface_force(wave_result, output_dir / "rod_wave_forces.png")
        extra["wave_model"] = True
        extra["rod_wave_c"] = float(wave_result.params.c)
        extra["rod_wave_L"] = float(wave_result.params.L)
        save_metrics(metrics, output_dir / "metrics.json", extra=extra)

    print("\n=== Предупреждения (Раздел Ж ТЗ) ===")
    if tokogram.dt_source != "file":
        print(f"⚠ [Ж.1] dt={dt} с — ДОПУЩЕНИЕ (не указано в файле/диссертации)")
    if tokogram.timestamp and ref.timestamp:
        print(
            f"⚠ [Ж.3] Метки времени не совпадают: {tokogram.timestamp} vs {ref.timestamp}. "
            f"Сравнение — форма кривой, не один цикл."
        )
    if i_xx_estimated:
        print("⚠ I_xx — оценка min(I_cycle), не паспортный ток ХХ (см. metrics.json)")
    if extra.get("S0_source") == "config":
        print(
            f"⚠ S₀ = {extra['S0_m']:.3f} м — из geometry.S0_m (константа), "
            "не по формуле (2.19) a_k·δ_max"
        )

    print(f"\n=== Кинематика / силы ===")
    s0_note = " (конфиг)" if extra.get("S0_source") == "config" else " (2.19)"
    print(f"S₀ = {extra['S0_m']:.3f} м{s0_note}, S(t) ∈ [{extra['S_range_m'][0]:.3f}, {extra['S_range_m'][1]:.3f}] м")
    print(f"F_calc ∈ [{extra['F_calc_range_N'][0]:.1f}, {extra['F_calc_range_N'][1]:.1f}] N")

    print(f"\n=== Метрики ===")
    print(f"RMSE F(τ) = {metrics.rmse_f:.2f} N")
    print(f"MAPE F(τ) = {metrics.mape_f:.2f} %")
    if metrics.rmse_f_upstroke is not None:
        print(f"RMSE F↑ (S-F) = {metrics.rmse_f_upstroke:.0f} N")
        print(f"RMSE F↓ (S-F) = {metrics.rmse_f_downstroke:.0f} N")
        print(
            f"F↓/F↑: расч={metrics.branch_ratio_calc:.2f}, "
            f"эталон={metrics.branch_ratio_ref:.2f}"
        )
    print(f"Δ площади петли = {metrics.area_diff_percent:.2f} %")

    return {"metrics": metrics, "segmentation": seg, "extra": extra}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Расчёт динамограммы по токограмме (модель Яшина)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args(argv)

    config_path = "config_well16427.yaml" if args.demo else args.config
    demo = args.demo or config_path.endswith("demo.yaml") or config_path.endswith("well16427.yaml")
    try:
        run_pipeline(config_path, demo=demo)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
