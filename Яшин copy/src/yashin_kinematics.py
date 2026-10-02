"""Кинематика станка-качалки: формулы (2.16)–(2.22), dissertation Яшин_расчет.pdf."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

NULL_S0 = {None, "null", "None", ""}


@dataclass
class KinematicsResult:
    """Кинематические величины одного цикла."""

    gamma_0: float
    gamma_1: float
    delta_max: float
    S0: float
    S: np.ndarray
    v: np.ndarray
    a: np.ndarray
    omega: float  # 2π/T, рад/с
    t: np.ndarray
    period_seconds: float
    s0_source: str = "formula"  # "formula" (2.19) | "config"
    kinematics_mode: str = "harmonic"  # harmonic (2.20–2.22) | exact (2.16–2.19)


def resolve_s0(geometry: dict, s0_formula: float) -> tuple[float, str]:
    """
    Ход S₀: из geometry.S0_m (константа) или по формуле (2.19) a_k·δ_max.

    Если S0_m задан в конфиге — используется напрямую (паспорт/эталон скважины),
    углы γ₀, γ₁, δ_max всё равно считаются для (2.30)–(2.31).
    """
    s0_cfg = geometry.get("S0_m")
    if s0_cfg not in NULL_S0:
        s0 = float(s0_cfg)
        if s0 <= 0:
            raise ValueError(f"geometry.S0_m должен быть > 0, получено {s0}")
        return s0, "config"
    return float(s0_formula), "formula"


def compute_geometry_angles(geometry: dict) -> tuple[float, float, float, float]:
    """
    (2.16)–(2.19): углы γ₀, γ₁, δ_max и ход S₀.

    В (2.16)–(2.17) по PDF (с.60, рис. 2.13):
      l   — вертикальное расстояние до полюса (geometry.l_pole),
      l₂  — длина заднего плеча балансира b_k (geometry.bk),
      R₀  — расстояние до полюса (geometry.R0),
      r   — длина кривошипа (geometry.r).
    """
    l_pole = float(geometry["l_pole"])
    bk = float(geometry["bk"])
    r = float(geometry["r"])
    r0 = float(geometry["R0"])
    ak = float(geometry["ak"])

    # формула (2.16), см. dissertation Яшин_расчет.pdf, с.60
    arg1 = (l_pole**2 + (bk + r) ** 2 - r0**2) / (2 * l_pole * bk)
    gamma_1 = np.arccos(np.clip(arg1, -1.0, 1.0))

    # формула (2.17), см. dissertation Яшин_расчет.pdf, с.60
    arg0 = (l_pole**2 + (bk - r) ** 2 - r0**2) / (2 * l_pole * bk)
    gamma_0 = np.arccos(np.clip(arg0, -1.0, 1.0))

    # формула (2.18): δ_max = γ₁ − γ₀; при r>0 → γ₁ < γ₀, берём модуль для положительного хода
    delta_max = abs(gamma_1 - gamma_0)

    # формула (2.19), см. dissertation Яшин_расчet.pdf, с.60
    s0 = ak * delta_max

    return gamma_0, gamma_1, delta_max, s0


def compute_kinematics(
    geometry: dict,
    n_samples: int,
    period_seconds: float,
) -> KinematicsResult:
    """
    S(t), v(t), a(t) по (2.20)–(2.22).

    ОГРАНИЧЕНИЕ МОДЕЛИ (унаследовано от Яшина): ω = 2π/T постоянна за цикл,
    тогда как motor_model.torque_from_current вычисляет переменную ω_ЭД(t) через s(t).
    Это допущение исходной модели (2.20)–(2.22); согласование с ω_ЭД(t) не требуется.
    """
    gamma_0, gamma_1, delta_max, s0 = compute_geometry_angles(geometry)
    s0, s0_source = resolve_s0(geometry, s0)
    if s0_source == "config":
        logger.info("S₀ = %.4f м — из geometry.S0_m (константа конфига)", s0)

    omega = 2 * np.pi / period_seconds
    t = np.linspace(0, period_seconds, n_samples, endpoint=False)
    phase = 2 * np.pi * t / period_seconds

    # формула (2.20), см. dissertation Яшин_расчет.pdf
    s = (s0 / 2.0) * (1.0 - np.cos(phase))

    # формула (2.21), см. dissertation Яшин_расчет.pdf
    v = (omega * s0 / 2.0) * np.sin(phase)

    # формула (2.22), см. dissertation Яшин_расчет.pdf
    a = (omega**2 * s0 / 2.0) * np.cos(phase)

    return KinematicsResult(
        gamma_0=gamma_0,
        gamma_1=gamma_1,
        delta_max=delta_max,
        S0=s0,
        S=s,
        v=v,
        a=a,
        omega=omega,
        t=t,
        period_seconds=period_seconds,
        s0_source=s0_source,
        kinematics_mode="harmonic",
    )


def compute_kinematics_exact(
    geometry: dict,
    n_samples: int,
    period_seconds: float,
) -> KinematicsResult:
    """
    Точная кинематика по (2.16)–(2.19): δ(t) из геометрии балансира, S = a_k·δ(t).

    v(t), a(t) — численное дифференцирование np.gradient (не гармоника (2.21)).
    При geometry.S0_m — масштабирование хода к заданному S₀.
    """
    gamma_0, gamma_1, delta_max, s0_formula = compute_geometry_angles(geometry)
    s0, s0_source = resolve_s0(geometry, s0_formula)
    ak = float(geometry["ak"])

    omega = 2 * np.pi / period_seconds
    t = np.linspace(0, period_seconds, n_samples, endpoint=False)
    dt = period_seconds / n_samples
    phase = omega * t

    g_min = min(gamma_0, gamma_1)
    g_max = max(gamma_0, gamma_1)
    delta_t = g_min + (g_max - g_min) * 0.5 * (1.0 - np.cos(phase))
    s_raw = ak * (delta_t - g_min)

    s_max = float(np.max(s_raw))
    if s_max > 1e-12:
        s = (s0 / s_max) * s_raw if s0_source == "config" else s_raw
    else:
        s = s_raw

    v = np.gradient(s, dt)
    a = np.gradient(v, dt)

    if s0_source == "config":
        logger.info("S₀ = %.4f м — из geometry.S0_m (точная кинематика)", s0)
    logger.info(
        "Точная кинематика: δ_max=%.4f рад, |v|_max=%.4f м/с",
        delta_max, float(np.max(np.abs(v))),
    )

    return KinematicsResult(
        gamma_0=gamma_0,
        gamma_1=gamma_1,
        delta_max=delta_max,
        S0=s0,
        S=s,
        v=v,
        a=a,
        omega=omega,
        t=t,
        period_seconds=period_seconds,
        s0_source=s0_source,
        kinematics_mode="exact",
    )


def compute_kinematics_for_config(
    geometry: dict,
    n_samples: int,
    period_seconds: float,
    kinematics_cfg: dict | None = None,
) -> KinematicsResult:
    """
    Выбор harmonic / exact по config.kinematics.method.

    .. deprecated::
        Для production используйте ``compute_kinematics_simple`` (MVP, model.mode=mvp).
    """
    mode = (kinematics_cfg or {}).get("method", "harmonic")
    if mode == "exact":
        return compute_kinematics_exact(geometry, n_samples, period_seconds)
    if mode == "harmonic":
        return compute_kinematics(geometry, n_samples, period_seconds)
    raise ValueError(f"kinematics.method={mode!r}; допустимо 'harmonic' или 'exact'")


def compute_kinematics_simple(
    s0: float,
    period_seconds: float,
    n_samples: int,
) -> KinematicsResult:
    """
    MVP: гармоническая кинематика (2.20)–(2.22) с одним параметром S₀.

    Не использует геометрию СК (l1, l2, r, …).
    """
    s0 = float(s0)
    omega = 2 * np.pi / period_seconds
    t = np.linspace(0, period_seconds, n_samples, endpoint=False)
    phase = omega * t

    s = (s0 / 2.0) * (1.0 - np.cos(phase))
    v = (s0 * omega / 2.0) * np.sin(phase)
    a = (s0 * omega**2 / 2.0) * np.cos(phase)

    return KinematicsResult(
        gamma_0=0.0,
        gamma_1=0.0,
        delta_max=0.0,
        S0=s0,
        S=s,
        v=v,
        a=a,
        omega=omega,
        t=t,
        period_seconds=period_seconds,
        s0_source="config",
        kinematics_mode="harmonic_simple",
    )
