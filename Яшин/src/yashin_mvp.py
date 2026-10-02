"""MVP-модель Яшина: калибруемые параметры + гидравлическая петля."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.dynamics.inertia import compute_inertia_force, resolve_m_eq
from src.kinematics.api_geometry import kinematics_from_config
from src.motor_model import torque_from_current_simple
from src.physics.elastic import compute_elastic_force_from_config, is_elastic_enabled
from src.physics.rod_wave import (
    RodWaveParameters,
    is_wave_model_enabled,
    resolve_friction_force,
    solve_rod_wave,
)
from src.yashin_inverse import compute_rod_load_simple, compute_rod_load_tf
from src.yashin_power_balance import get_parasitic_power, get_pump_load_force

logger = logging.getLogger(__name__)


def _motor_mvp_params(motor: dict) -> dict[str, float]:
    """P0, a, b, c из MVP- или legacy-ключей motor."""
    p0 = motor.get("P0_w", motor.get("P_xx_w", 3000.0))
    return {
        "P0": float(p0),
        "a": float(motor.get("P_a_w_per_a", 400.0)),
        "b": float(motor.get("P_b_w_per_a2", 0.0)),
        "c": float(motor.get("P_c_w_per_a3", 0.0)),
        "eta_ed": float(motor.get("eta_ed", 0.88)),
        "eta_per": float(motor.get("eta_per", 0.95)),
        "n_red": float(motor.get("n_red", 33.0)),
        "n_kp": float(motor.get("n_kp", 3.0)),
    }


def _omega_sync(motor: dict) -> float:
    rpm = float(motor.get("n_sync_rpm", 1500))
    return 2.0 * np.pi * rpm / 60.0


def _align_current_phase(i_cycle: np.ndarray, config: dict) -> np.ndarray:
    """Циклический сдвиг I(t) относительно кинематики (калибруемый, отсчёты)."""
    shift = int(config.get("tokogram", {}).get("i_phase_shift_samples", 0))
    if shift == 0:
        return i_cycle
    return np.roll(np.asarray(i_cycle, dtype=float), shift)


def process_cycle_mvp(
    i_cycle: np.ndarray,
    period: float,
    config: dict,
) -> dict[str, Any]:
    """
    Упрощённый цикл:
      F_head → F_surface (wave) или F_dyn;
      F_elastic = E·A·ΔL/L (elastic.enabled);
      F = F_surface + F_elastic + F_fluid + F_inertia + F_friction.
    """
    kin_cfg = config.get("kinematics", {})
    s0 = float(kin_cfg.get("S0_m", 3.5))
    n = len(i_cycle)
    motor = config["motor"]
    inv_cfg = config.get("inverse", {})
    par_cfg = config.get("parasitic", {})
    pump_cfg = config.get("pump_load", {})
    m_eq = resolve_m_eq(config)
    dt = period / n

    i_aligned = _align_current_phase(i_cycle, config)
    kinematics = kinematics_from_config(kin_cfg, s0, period, n)
    motor_result = torque_from_current_simple(
        i_aligned,
        _motor_mvp_params(motor),
        omega_sync=_omega_sync(motor),
        s_nom=float(motor.get("s_nom", 0.03)),
    )
    p_par = get_parasitic_power(kinematics.t, period, par_cfg)
    p0 = float(motor.get("P0_w", motor.get("P_xx_w", 0.0)))
    p_net = motor_result.P - p0 + p_par

    inv_method = inv_cfg.get("method", "torque_factor")
    omega_crank = 2.0 * np.pi / period

    if inv_method == "torque_factor":
        k_tf = float(inv_cfg.get("k_tf", inv_cfg.get("k_v", 0.35)))
        m_net = p_net / omega_crank
        inverse = compute_rod_load_tf(
            m_net,
            kinematics.v,
            omega_crank,
            v_zero_threshold_fraction=inv_cfg.get("v_zero_threshold_fraction", 0.01),
            tf_min=float(inv_cfg.get("tf_min", 1e-6)),
            interpolate=inv_cfg.get("interpolate_dead_points", True),
            tf_scale=k_tf,
        )
    elif inv_method == "power_balance":
        inverse = compute_rod_load_simple(
            p_net,
            kinematics.v,
            k_v=float(inv_cfg.get("k_v", 0.35)),
            epsilon=float(inv_cfg.get("epsilon", 0.02)),
        )
    else:
        raise ValueError(
            f"inverse.method={inv_method!r}; допустимо 'torque_factor' или 'power_balance'"
        )

    f_head = inverse.p_tpsh
    f_fluid = get_pump_load_force(kinematics.S, kinematics.v, s0, pump_cfg)
    f_inertia = compute_inertia_force(kinematics.a, m_eq)
    f_friction = resolve_friction_force(n, config)
    f_elastic = compute_elastic_force_from_config(kinematics.S, config)

    wave_result = None
    if is_wave_model_enabled(config):
        wave_params = RodWaveParameters.from_config(config.get("rod_wave"))
        wave_result = solve_rod_wave(
            kinematics.S,
            kinematics.v,
            kinematics.a,
            f_head,
            dt,
            wave_params,
        )
        f_surface = wave_result.f_surface
        f_dyn = f_head
    else:
        f_dyn = f_head
        f_surface = f_dyn

    f_rod = f_surface + f_elastic + f_fluid + f_inertia + f_friction

    v = kinematics.v
    if np.any(v > 0) and np.any(v < 0):
        ratio = float(np.mean(f_rod[v < 0]) / max(np.mean(f_rod[v > 0]), 1e-6))
        logger.info(
            "MVP: F↑=%.0f N F↓=%.0f N F↓/F↑=%.2f wave=%s elastic=%s m_eq=%.0f kg",
            float(np.mean(f_rod[v > 0])),
            float(np.mean(f_rod[v < 0])),
            ratio,
            is_wave_model_enabled(config),
            is_elastic_enabled(config),
            m_eq,
        )

    return {
        "s": kinematics.S,
        "f": f_rod,
        "f_dyn": f_dyn,
        "f_head": f_head,
        "f_surface": f_surface,
        "f_elastic": f_elastic,
        "f_fluid": f_fluid,
        "f_inertia": f_inertia,
        "f_friction": f_friction,
        "v_near_zero": inverse.v_near_zero,
        "kinematics": kinematics,
        "motor_result": motor_result,
        "period_seconds": period,
        "p_parasitic": p_par,
        "p_net": p_net,
        "m_eq": m_eq,
        "wave_result": wave_result,
        "wave_model": is_wave_model_enabled(config),
        "elastic_enabled": is_elastic_enabled(config),
        "dt": dt,
    }
