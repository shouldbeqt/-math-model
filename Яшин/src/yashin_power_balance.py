"""Баланс мощности: формулы (2.23)–(2.31), dissertation Яшin_расчет.pdf."""

from __future__ import annotations

import numpy as np

from src.yashin_kinematics import KinematicsResult

G = 9.80665  # м/с² — перевод кгс → Н


def _mass_to_newtons(mass_kgf: float, masses: dict) -> float:
    """кгс → Н для (2.30)–(2.31), если masses.convert_to_newton: true."""
    if masses.get("convert_to_newton", False):
        return float(mass_kgf) * G
    return float(mass_kgf)


def compute_equivalent_weight(masses: dict) -> float:
    # формула (2.28), см. dissertation Яшин_расчет.pdf
    return float(masses["G"]) + float(masses["Q_delta"])


def compute_center_of_mass(masses: dict, geometry: dict) -> float:
    """L — расстояние ЦТ сборки балансира до оси, формула (2.29)."""
    g_w = float(masses["G"])
    q_delta = float(masses["Q_delta"])
    l_cm = float(masses["L_center_of_mass"])

    # формула (2.29), dissertation Яшин_расчет.pdf, с.62:
    # L = (G·(b_k − a_k) + 2·Qδ·L_cm) / (2·(Qδ + G))
    bk = float(geometry["bk"])
    ak = float(geometry["ak"])
    return (g_w * (bk - ak) + 2.0 * q_delta * l_cm) / (2.0 * (q_delta + g_w))


def _beam_coeff(geometry: dict, motor: dict) -> float:
    """Множитель l1/(l2·η_ЭД·η_пер.) из (2.27)/(2.32)."""
    l1 = float(geometry["l1"])
    l2 = float(geometry["l2"])
    eta_ed = float(motor["eta_ed"])
    eta_per = float(motor["eta_per"])
    return l1 / (l2 * eta_ed * eta_per)


def compute_parasitic_powers(
    kinematics: KinematicsResult,
    geometry: dict,
    masses: dict,
    motor: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pδ(t) — формула (2.31), PR(t) — формула (2.30).
    l_б.1 → geometry.l1, l_б.2 → geometry.l2, G_б → masses.G, G_б.2 → masses.Q_R.

    .. deprecated::
        Для MVP используйте ``get_parasitic_power`` (constant/harmonic).
    """
    eta_ed = float(motor["eta_ed"])
    eta_per = float(motor["eta_per"])

    l1 = float(geometry["l1"])   # l_б.1
    l2 = float(geometry["l2"])   # l_б.2
    # кгс → Н (формулы (2.30)–(2.31), dissertation Яшин_расчет.pdf)
    g_w = _mass_to_newtons(masses["G"], masses)
    q_r = _mass_to_newtons(masses["Q_R"], masses)

    omega = kinematics.omega
    t = kinematics.t
    s = kinematics.S
    a = kinematics.a
    delta_max = kinematics.delta_max
    l_delta = compute_center_of_mass(masses, geometry)
    period = kinematics.period_seconds

    # формула (2.30), dissertation Яшин_расчet.pdf, с.63:
    # G_б.2 — вес противовеса, кгс; sin(2πωt) ≡ sin(2π·t/T) при ω = 2π/T
    p_r = omega * l2 / (eta_ed * eta_per) * q_r * np.sin(2 * np.pi * t / period)

    # формула (2.31), см. dissertation Яшin_расчет.pdf, с.63:
    # P_δ(t) = G_б·L·ω·l_б.2 / (l_1·η_ЭД·η_пер.) · (cos(δ_max/2 − S/l_1) − L·a/(l_2·g))
    p_delta = (
        g_w
        * l_delta
        * omega
        * l2
        / (l1 * eta_ed * eta_per)
        * (np.cos(delta_max / 2 - s / l1) - l_delta * a / (l2 * G))
    )

    return p_delta, p_r


def compute_direct_power(
    p_tpsh: np.ndarray,
    kinematics: KinematicsResult,
    geometry: dict,
    masses: dict,
    motor: dict,
) -> np.ndarray:
    """
    Прямая модель (2.27), алгебраически согласованная с (2.30)–(2.32):
    P(t) = P_XX − P_δ(t) − P_R(t) + l1/(l2·η_ЭД·η_пер.) · P_ТПШ(t) · v(t)
    """
    p_xx = float(motor["P_xx_w"])
    coeff = _beam_coeff(geometry, motor)
    p_delta, p_r = compute_parasitic_powers(kinematics, geometry, masses, motor)
    # формула (2.27) / (2.32), см. dissertation Яшин_расчет.pdf
    return p_xx - p_delta - p_r + coeff * p_tpsh * kinematics.v


def get_parasitic_power_constant(t: np.ndarray, p_const: float) -> np.ndarray:
    """MVP: P_parasitic = const [Вт]."""
    return np.full_like(np.asarray(t, dtype=float), float(p_const), dtype=float)


def get_parasitic_power_harmonic(
    t: np.ndarray,
    period: float,
    p0: float,
    p1: float,
    phi: float,
    p2: float = 0.0,
    phi2: float = 0.0,
) -> np.ndarray:
    """MVP: P_par = P₀ + P₁·sin(ωt+φ) + P₂·sin(2ωt+φ₂)."""
    t = np.asarray(t, dtype=float)
    omega = 2.0 * np.pi / float(period)
    base = float(p0) + float(p1) * np.sin(omega * t + float(phi))
    if abs(p2) > 1e-9:
        base = base + float(p2) * np.sin(2.0 * omega * t + float(phi2))
    return base


def get_pump_load_force(
    s: np.ndarray,
    v: np.ndarray,
    s0: float,
    pump_cfg: dict,
) -> np.ndarray:
    """
    Гидравлическая добавка к нагрузке штока [N].

    На спуске (v<0) насос переносит столб жидкости — нагрузка выше, чем по
    одному балансу мощности (2.32). На подъёме (v>0) — опциональная поправка.
    """
    mode = pump_cfg.get("mode", "none")
    if mode in ("none", None, ""):
        return np.zeros_like(s, dtype=float)

    s = np.asarray(s, dtype=float)
    v = np.asarray(v, dtype=float)
    up = v > 0
    dn = v < 0
    f_add = np.zeros_like(s, dtype=float)

    if mode == "force_offset":
        f_add[dn] += float(pump_cfg.get("F_down_n", 0.0))
        f_add[up] -= float(pump_cfg.get("F_up_n", 0.0))
        return f_add

    if mode == "force_profile":
        s0_safe = max(float(s0), 1e-6)
        profile = float(pump_cfg.get("F0_down_n", 0.0)) + float(pump_cfg.get("F1_down_n", 0.0)) * s / s0_safe
        f_add[dn] += profile[dn]
        f_add[up] -= float(pump_cfg.get("F_up_n", 0.0))
        return f_add

    raise ValueError(
        f"pump_load.mode={mode!r}; допустимо 'none', 'force_offset' или 'force_profile'"
    )


def get_parasitic_power(t: np.ndarray, period: float, par_cfg: dict) -> np.ndarray:
    """Диспетчер MVP-паразитных мощностей."""
    mode = par_cfg.get("mode", "constant")
    if mode == "constant":
        return get_parasitic_power_constant(t, float(par_cfg.get("P_const", 0.0)))
    if mode in ("harmonic", "harmonic2"):
        return get_parasitic_power_harmonic(
            t, period,
            float(par_cfg.get("P0", 0.0)),
            float(par_cfg.get("P1", 0.0)),
            float(par_cfg.get("phi", 0.0)),
            float(par_cfg.get("P2", 0.0)),
            float(par_cfg.get("phi2", 0.0)),
        )
    raise ValueError(f"parasitic.mode={mode!r}; допустимо 'constant', 'harmonic' или 'harmonic2'")


def equivalent_power_from_motor(
    m_crank: np.ndarray,
    omega_ed: np.ndarray,
    motor: dict,
) -> np.ndarray:
    """
    Модифицированная (2.23): P(t) = ω_ЭД·M_кр(t) / (η_ЭД·η_пер.) + P_XX
    с M_кр из motor_model (замена измеренной ваттмограммы).
    """
    eta_ed = float(motor["eta_ed"])
    eta_per = float(motor["eta_per"])
    p_xx = float(motor["P_xx_w"])
    return omega_ed * m_crank / (eta_ed * eta_per) + p_xx
