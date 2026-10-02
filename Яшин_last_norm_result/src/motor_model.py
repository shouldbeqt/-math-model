"""Модель перехода ток → момент/мощность (Раздел А ТЗ)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq

logger = logging.getLogger(__name__)

# kloss, linear — из ТЗ (методы 2–3)
# quadratic, poly2, affine — регрессии P(I)
# ui, ui_bias — физические P = U·I (+ bias)
Method = Literal[
    "kloss", "kloss_modified", "linear",
    "equivalent_circuit",
    "quadratic", "poly2", "affine",
    "ui", "ui_bias",
]

POWER_METHODS = frozenset({"quadratic", "poly2", "affine", "ui", "ui_bias"})

# Метод 1 (P = √3·U·I·cosφ) реализован как ui / ui_bias с U_eff = √3·U·cosφ.


@dataclass
class MotorTorqueResult:
    """Результат расчёта момента и мощности из тока."""

    M_ed: np.ndarray       # момент на валу ЭД, Н·м
    omega_ed: np.ndarray   # угловая скорость ротора ЭД, рад/с
    s: np.ndarray          # скольжение
    P: np.ndarray          # эквивалентная потребляемая мощность, Вт
    M_crank: np.ndarray    # момент на кривошипе M_кр = M_ed * n_red * n_kp, Н·м


def available_methods() -> list[str]:
    """Все поддерживаемые method для current_to_torque."""
    return [
        "kloss", "kloss_modified", "linear",
        "equivalent_circuit",
        "quadratic", "poly2", "affine",
        "ui", "ui_bias",
    ]


def default_u_eff(motor: dict) -> float:
    """U_eff = √3·U_ном·cosφ_ном — эффективное линейное напряжение для P = U·I."""
    u_nom = float(motor["U_nom_v"])
    cos_phi = float(motor.get("cos_phi_nom", 0.86))
    return float(np.sqrt(3.0) * u_nom * cos_phi)


def _motor_nominals(motor: dict) -> tuple[float, float, float, float, float, float]:
    """Извлекает паспортные параметры двигателя."""
    p_nom_w = float(motor["P_nom_kw"]) * 1000.0
    i_nom = float(motor["I_nom_a"])
    n_sync = float(motor["n_sync_rpm"])
    s_nom = float(motor["s_nom"])
    lambda_max = float(motor["lambda_max"])
    k_i_start = float(motor["k_I_start"])

    omega_sync = 2 * np.pi * n_sync / 60.0
    m_nom = p_nom_w / (omega_sync * (1 - s_nom) + 1e-12)
    m_max = lambda_max * m_nom
    s_k = s_nom * (lambda_max + np.sqrt(lambda_max**2 - 1))
    i_start = k_i_start * i_nom

    return omega_sync, m_nom, m_max, s_k, s_nom, i_start


def _kloss_slip_from_current(
    i: np.ndarray,
    i_nom: float,
    i_xx: float,
    i_start: float,
    s_nom: float,
    s_k: float,
) -> np.ndarray:
    """Обратная зависимость s(I) через аппроксимацию характеристики I(s)."""

    def i_of_s(s: float) -> float:
        if s <= 0:
            return i_xx
        if s <= s_k:
            return i_xx + (i_start - i_xx) * (s / s_k) ** 2
        return i_xx + (i_nom - i_xx) * (s / s_nom)

    def invert(i_target: float) -> float:
        if i_target <= i_xx:
            return 0.0
        s_lo, s_hi = 1e-6, min(0.99, max(s_k * 2, s_nom * 3))
        try:
            return brentq(lambda s: i_of_s(s) - i_target, s_lo, s_hi)
        except ValueError:
            return s_nom

    return np.array([invert(float(x)) for x in i], dtype=float)


def estimate_equivalent_circuit_params(motor: dict) -> dict[str, float]:
    """
    Оценка параметров Г-образной схемы замещения АД из паспортных данных.

    R1, R2', X1, X2', Xm [Ом] — грубая оценка для отсутствующих паспортных данных.
    """
    u_ph = float(motor["U_nom_v"]) / np.sqrt(3.0)
    i_nom = float(motor["I_nom_a"])
    i_xx = float(motor.get("I_xx_a") or 0.3 * i_nom)
    i_start = float(motor["k_I_start"]) * i_nom
    s_nom = float(motor["s_nom"])
    cos_phi = float(motor.get("cos_phi_nom", 0.86))

    z_nom = u_ph / max(i_nom, 1e-6)
    r1 = 0.35 * z_nom * max(1.0 - cos_phi, 0.05)
    x1 = np.sqrt(max(z_nom**2 - r1**2, 1e-12)) * 0.45

    z_start = u_ph / max(i_start, 1e-6)
    r2 = max(z_start * 0.55 - r1, 0.02 * z_nom)
    x2 = np.sqrt(max(z_start**2 - (r1 + r2) ** 2, (0.15 * z_nom) ** 2))
    xm = max(6.0 * (x1 + x2), 0.5 * z_nom)

    # уточнение R2' по номинальному скольжению
    r2 = max(r2, s_nom * u_ph / max(i_nom, 1e-6) * 0.5)

    return {
        "R1_ohm": float(r1),
        "R2_p_ohm": float(r2),
        "X1_ohm": float(x1),
        "X2_p_ohm": float(x2),
        "Xm_ohm": float(xm),
    }


def _resolve_equivalent_circuit(motor: dict) -> dict[str, float]:
    """Параметры схемы замещения: из motor.equivalent_circuit или оценка."""
    ec = motor.get("equivalent_circuit") or {}
    estimated = estimate_equivalent_circuit_params(motor)
    null = {None, "null", "None", ""}

    def _val(key: str) -> float:
        v = ec.get(key)
        return float(v) if v not in null else estimated[key]

    return {
        "R1_ohm": _val("R1_ohm"),
        "R2_p_ohm": _val("R2_p_ohm"),
        "X1_ohm": _val("X1_ohm"),
        "X2_p_ohm": _val("X2_p_ohm"),
        "Xm_ohm": _val("Xm_ohm"),
    }


def _z_eq_slip(s: float, r1: float, x1: float, r2: float, x2: float, xm: float) -> complex:
    if s <= 0:
        s = 1e-9
    z2s = r2 / s + 1j * x2
    zxm = 1j * xm
    z_rotor = (z2s * zxm) / (z2s + zxm)
    return r1 + 1j * x1 + z_rotor


def _current_slip(s: float, u_ph: float, r1: float, x1: float, r2: float, x2: float, xm: float) -> float:
    z = _z_eq_slip(s, r1, x1, r2, x2, xm)
    return float(abs(u_ph / z))


def _slip_from_current_ec(
    i: np.ndarray,
    u_ph: float,
    ec: dict[str, float],
    i_xx: float,
    s_nom: float,
) -> np.ndarray:
    """Обратная задача s(I) для схемы замещения."""

    def invert(i_target: float) -> float:
        if i_target <= i_xx:
            return 0.0
        s_lo, s_hi = 1e-6, min(0.99, max(0.2, s_nom * 5))

        def f(s: float) -> float:
            return _current_slip(s, u_ph, ec["R1_ohm"], ec["X1_ohm"],
                                 ec["R2_p_ohm"], ec["X2_p_ohm"], ec["Xm_ohm"]) - i_target

        try:
            return brentq(f, s_lo, s_hi)
        except ValueError:
            return s_nom

    return np.array([invert(float(x)) for x in i], dtype=float)


def _m_em_slip(
    s: float,
    u_ph: float,
    r1: float,
    x1: float,
    r2: float,
    x2: float,
    omega_sync: float,
) -> float:
    """M_em(s) = (3·U_ф²·R2'/s) / (ω_sync·[(R1 + R2'/s)² + (X1 + X2')²])."""
    if s <= 0:
        s = 1e-9
    rp = r2 / s
    num = 3.0 * u_ph**2 * rp
    den = omega_sync * ((r1 + rp) ** 2 + (x1 + x2) ** 2)
    return float(num / max(den, 1e-12))


def estimate_p_const_loss(motor: dict, r1: float | None = None) -> float:
    """P_const_loss ≈ 3·I_xx²·R1 + P_mech_const."""
    null = {None, "null", "None", ""}
    if motor.get("P_const_loss_w") not in null:
        return float(motor["P_const_loss_w"])
    i_xx = float(motor.get("I_xx_a") or 0.0)
    if r1 is None:
        r1 = estimate_equivalent_circuit_params(motor)["R1_ohm"]
    p_mech = float(motor.get("P_mech_const_w") or 0.0)
    return float(3.0 * i_xx**2 * r1 + p_mech)


def torque_from_current_equivalent_circuit(
    current_a: np.ndarray,
    motor: dict,
    crank_radius_m: float,
) -> MotorTorqueResult:
    """
    I(t) → M(t) через Г-образную схему замещения АД.
    P_mech = M_em · ω_sync · (1 − s).
    """
    current_a = np.asarray(current_a, dtype=float)
    eta_ed = float(motor["eta_ed"])
    eta_per = float(motor["eta_per"])
    n_red = float(motor["n_red"])
    n_kp = float(motor["n_kp"])
    p_xx = float(motor["P_xx_w"])
    i_xx = float(motor.get("I_xx_a") or np.min(current_a))
    u_ph = float(motor["U_nom_v"]) / np.sqrt(3.0)

    omega_sync, _, _, _, s_nom, _ = _motor_nominals(motor)
    ec = _resolve_equivalent_circuit(motor)

    s = _slip_from_current_ec(current_a, u_ph, ec, i_xx, s_nom)
    m_em = np.array([
        _m_em_slip(float(si), u_ph, ec["R1_ohm"], ec["X1_ohm"],
                   ec["R2_p_ohm"], ec["X2_p_ohm"], omega_sync)
        for si in s
    ])
    omega_ed = omega_sync * (1.0 - s)
    p_mech = m_em * omega_ed
    p = p_mech / (eta_ed * eta_per) + p_xx
    m_crank = m_em * n_red * n_kp

    return MotorTorqueResult(
        M_ed=m_em,
        omega_ed=omega_ed,
        s=s,
        P=p,
        M_crank=m_crank,
    )


def torque_from_current_kloss_modified(
    current_a: np.ndarray,
    motor: dict,
    crank_radius_m: float,
) -> MotorTorqueResult:
    """
    Модифицированный Клосс: M_em = 2·M_max/(s/s_k + s_k/s) − M_loss.
    M_loss = P_const_loss / ω_sync.
    """
    current_a = np.asarray(current_a, dtype=float)
    eta_ed = float(motor["eta_ed"])
    eta_per = float(motor["eta_per"])
    n_red = float(motor["n_red"])
    n_kp = float(motor["n_kp"])
    p_xx = float(motor["P_xx_w"])
    i_nom = float(motor["I_nom_a"])
    i_xx = float(motor.get("I_xx_a") or np.min(current_a))

    omega_sync, m_nom, m_max, s_k, s_nom, i_start = _motor_nominals(motor)
    p_const = estimate_p_const_loss(motor)
    m_loss = p_const / max(omega_sync, 1e-12)

    s = _kloss_slip_from_current(current_a, i_nom, i_xx, i_start, s_nom, s_k)
    m_kloss = 2.0 * m_max / (s / s_k + s_k / np.maximum(s, 1e-9))
    m_ed = np.maximum(m_kloss - m_loss, 0.0)

    omega_ed = omega_sync * (1.0 - s)
    p = m_ed * omega_ed / (eta_ed * eta_per) + p_xx
    m_crank = m_ed * n_red * n_kp

    return MotorTorqueResult(
        M_ed=m_ed,
        omega_ed=omega_ed,
        s=s,
        P=p,
        M_crank=m_crank,
    )


def compute_power_from_current(
    current_a: np.ndarray,
    motor: dict,
    method: str,
) -> np.ndarray:
    """
    P(t) из I(t) для регрессионных и физических моделей мощности.

    Модели:
      quadratic — P = P_XX + a·I + b·I² + c·I³
      poly2       — P = P_XX + a·I + b·I²
      affine      — P = a·I + b
      ui          — P = U_eff·I
      ui_bias     — P = U_eff·I + P_bias
    """
    i = np.asarray(current_a, dtype=float)
    p_xx = float(motor["P_xx_w"])

    if method == "quadratic":
        a = float(motor.get("P_a_w_per_a", 400.0))
        b = float(motor.get("P_b_w_per_a2", 0.0))
        c = float(motor.get("P_c_w_per_a3", 0.0))
        return p_xx + a * i + b * i**2 + c * i**3

    if method == "poly2":
        a = float(motor.get("P_a_w_per_a", 400.0))
        b = float(motor.get("P_b_w_per_a2", 0.0))
        return p_xx + a * i + b * i**2

    if method == "affine":
        a = float(motor.get("P_a_w_per_a", 400.0))
        b = float(motor.get("P_bias_w", p_xx))
        return a * i + b

    if method == "ui":
        u_eff = float(motor.get("U_eff_v") or default_u_eff(motor))
        return u_eff * i

    if method == "ui_bias":
        u_eff = float(motor.get("U_eff_v") or default_u_eff(motor))
        bias = float(motor.get("P_bias_w", 0.0))
        return u_eff * i + bias

    raise ValueError(f"compute_power_from_current: method={method!r} не является P(I)-моделью")


def _result_from_power(
    current_a: np.ndarray,
    p: np.ndarray,
    motor: dict,
    omega_sync: float,
    s_nom: float,
    n_red: float,
    n_kp: float,
    p_xx: float,
    eta_ed: float,
) -> MotorTorqueResult:
    """P(t) → M_ЭД, ω_ЭД, M_кр (как для quadratic: ω ≈ const, M из избыточной мощности)."""
    omega_ed = np.full_like(current_a, omega_sync * (1.0 - s_nom))
    m_ed = (p - p_xx) * eta_ed / np.maximum(omega_ed, 1e-6)
    s = np.full_like(current_a, s_nom)
    return MotorTorqueResult(
        M_ed=m_ed,
        omega_ed=omega_ed,
        s=s,
        P=p,
        M_crank=m_ed * n_red * n_kp,
    )


def torque_from_current(
    current_a: np.ndarray,
    motor: dict,
    crank_radius_m: float,
    method: Method = "kloss",
) -> MotorTorqueResult:
    """
    I(t) → M(t), ω_ЭД(t) → P(t), M_кр(t).

    method="kloss": Метод 2 (скольжение + формула Клосса).
    method="kloss_modified": Клосс с вычетом M_loss (потери в стали).
    method="equivalent_circuit": Г-образная схема замещения АД.
    method="linear": Метод 3 (линейная аппроксимация момента).
    method="quadratic"|"poly2"|"affine": регрессии P(I).
    method="ui"|"ui_bias": физические P = U·I (+ bias).
    """
    current_a = np.asarray(current_a, dtype=float)

    if method == "equivalent_circuit":
        return torque_from_current_equivalent_circuit(current_a, motor, crank_radius_m)
    if method == "kloss_modified":
        return torque_from_current_kloss_modified(current_a, motor, crank_radius_m)

    eta_ed = float(motor["eta_ed"])
    eta_per = float(motor["eta_per"])
    n_red = float(motor["n_red"])
    n_kp = float(motor["n_kp"])
    p_xx = float(motor["P_xx_w"])
    i_nom = float(motor["I_nom_a"])
    i_xx = float(motor.get("I_xx_a") or np.min(current_a))

    omega_sync, m_nom, m_max, s_k, s_nom, i_start = _motor_nominals(motor)

    if method in POWER_METHODS:
        p = compute_power_from_current(current_a, motor, method)
        return _result_from_power(
            current_a, p, motor, omega_sync, s_nom, n_red, n_kp, p_xx, eta_ed,
        )

    if method == "kloss":
        s = _kloss_slip_from_current(current_a, i_nom, i_xx, i_start, s_nom, s_k)
        m_ed = 2.0 * m_max / (s / s_k + s_k / np.maximum(s, 1e-9))
    elif method == "linear":
        k = m_nom / max(i_nom - i_xx, 1e-6)
        m_ed = k * (current_a - i_xx)
        s = np.full_like(current_a, s_nom)
    else:
        raise ValueError(
            f"Неизвестный method={method!r}; допустимо: {', '.join(available_methods())}"
        )

    omega_ed = omega_sync * (1.0 - s)
    p = m_ed * omega_ed / (eta_ed * eta_per) + p_xx
    m_crank = m_ed * n_red * n_kp

    return MotorTorqueResult(
        M_ed=m_ed,
        omega_ed=omega_ed,
        s=s,
        P=p,
        M_crank=m_crank,
    )


def compute_power_poly2(
    current_a: np.ndarray,
    p0: float,
    a: float,
    b: float,
    c: float = 0.0,
) -> np.ndarray:
    """MVP: P(I) = P₀ + a·I + b·I² + c·I³ [Вт]."""
    i = np.asarray(current_a, dtype=float)
    return float(p0) + float(a) * i + float(b) * i**2 + float(c) * i**3


def torque_from_current_simple(
    current_a: np.ndarray,
    motor_params: dict,
    omega_sync: float,
    s_nom: float,
) -> MotorTorqueResult:
    """
    MVP: P(I)=P₀+aI+bI²(+cI³) → M_кр. Без паспортных I_xx/s_nom аппроксимаций.
    """
    current_a = np.asarray(current_a, dtype=float)
    p0 = float(motor_params["P0"])
    a = float(motor_params["a"])
    b = float(motor_params["b"])
    c = float(motor_params.get("c", 0.0))
    eta_ed = float(motor_params["eta_ed"])
    eta_per = float(motor_params["eta_per"])
    n_red = float(motor_params["n_red"])
    n_kp = float(motor_params["n_kp"])

    p = compute_power_poly2(current_a, p0, a, b, c)
    omega_ed = float(omega_sync) * (1.0 - float(s_nom))
    eta_total = eta_ed * eta_per
    m_ed = (p - p0) * eta_total / max(omega_ed, 1e-6)
    m_crank = m_ed * n_red * n_kp

    return MotorTorqueResult(
        M_ed=m_ed,
        omega_ed=np.full_like(current_a, omega_ed),
        s=np.full_like(current_a, s_nom),
        P=p,
        M_crank=m_crank,
    )
