"""Обратный расчёт усилия на штоке: формула (2.32), dissertation Яшин_расчет.pdf."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InverseResult:
    """Результат инверсии (2.32) или TF-метода."""

    p_tpsh: np.ndarray
    v_near_zero: np.ndarray  # bool-маска точек с |v| ≈ 0 или |TF| ≈ 0
    tf: np.ndarray | None = None  # фактор крутящего момента [м], если TF-метод


def _dead_point_mask(
    v: np.ndarray,
    tf: np.ndarray | None,
    v_zero_threshold_fraction: float,
    tf_min: float,
) -> np.ndarray:
    v_max = float(np.max(np.abs(v)))
    v_threshold = v_zero_threshold_fraction * v_max
    v_near_zero = np.abs(v) < v_threshold
    if tf is not None:
        v_near_zero = v_near_zero | (np.abs(tf) < tf_min)
    return v_near_zero


logger = logging.getLogger(__name__)

TF_INTERP_MIN_POINTS = 2


def _interpolate_masked(f: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Линейная интерполяция F в точках mask (мёртвые точки хода)."""
    f = np.asarray(f, dtype=float).copy()
    bad = mask | ~np.isfinite(f)
    if not np.any(bad):
        return f
    if np.all(bad):
        return f
    idx = np.arange(len(f))
    good = ~bad
    f[bad] = np.interp(idx[bad], idx[good], f[good])
    return f


def _interpolate_by_velocity_sign(
    f: np.ndarray,
    v: np.ndarray,
    bad: np.ndarray,
) -> np.ndarray:
    """
    Интерполяция отдельно на участках v>0 и v<0.

    Не соединяет ветви через v≈0 (ВМТ/НМТ), где F скачет из-за деления на v.
    """
    f = np.asarray(f, dtype=float).copy()
    v = np.asarray(v, dtype=float)
    bad = bad | ~np.isfinite(f)
    if not np.any(bad):
        return f
    for branch in (v > 0, v < 0):
        seg = branch & bad
        if not np.any(seg):
            continue
        idx = np.arange(len(f))
        good = branch & ~bad & np.isfinite(f)
        if np.sum(good) < TF_INTERP_MIN_POINTS:
            continue
        f[seg] = np.interp(idx[seg], idx[good], f[good])
    return f


def compute_rod_load_tf(
    m_crank: np.ndarray,
    v_sh: np.ndarray,
    omega_crank: float,
    geometry: dict | None = None,
    v_zero_threshold_fraction: float = 0.01,
    tf_min: float = 1e-6,
    interpolate: bool = True,
    tf_scale: float = 1.0,
) -> InverseResult:
    """
    Усилие на штоке через фактор крутящего момента (Torque Factor):

      TF(θ) = tf_scale · v_шток / ω_кр   [м]
      F_шток = M / TF

    moment_source=m_crank: M = M_кр, tf_scale=1.
    moment_source=net_power: M = (P−P_XX+Pδ+PR)/ω, tf_scale=l1/(l2·η_ЭД·η_пер.) — экв. (2.32).
    """
    v_sh = np.asarray(v_sh, dtype=float)
    m_crank = np.asarray(m_crank, dtype=float)
    omega = float(omega_crank)
    scale = float(tf_scale)
    if omega <= 0:
        raise ValueError(f"omega_crank должен быть > 0, получено {omega}")

    tf = scale * v_sh / omega
    tf_abs = np.abs(tf)
    tf_safe = np.where(tf_abs >= tf_min, tf, np.nan)
    f_rod = m_crank / tf_safe

    v_near_zero = _dead_point_mask(v_sh, tf, v_zero_threshold_fraction, tf_min)
    # Расширяем маску: ±1 отсчёт от смены знака v (ВМТ/НМТ), чтобы не склеивать ветви
    sign_change = np.zeros(len(v_sh), dtype=bool)
    if len(v_sh) > 2:
        sgn = np.sign(v_sh)
        sgn[sgn == 0] = np.nan
        for i in range(1, len(v_sh) - 1):
            if np.isfinite(sgn[i - 1]) and np.isfinite(sgn[i + 1]) and sgn[i - 1] != sgn[i + 1]:
                sign_change[i - 1 : i + 2] = True
    v_near_zero = v_near_zero | sign_change

    if interpolate and np.any(v_near_zero):
        f_rod = _interpolate_by_velocity_sign(f_rod, v_sh, v_near_zero)
        still_bad = v_near_zero & ~np.isfinite(f_rod)
        if np.any(still_bad):
            f_rod = _interpolate_masked(f_rod, still_bad)

    tf_min_val = float(np.nanmin(tf_abs[tf_abs >= tf_min])) if np.any(tf_abs >= tf_min) else 0.0
    tf_max_val = float(np.nanmax(tf_abs))
    logger.info(
        "TF-метод: |TF| ∈ [%.4g, %.4g] м; мёртвых точек %d/%d",
        tf_min_val, tf_max_val, int(np.sum(v_near_zero)), len(v_sh),
    )
    if geometry is not None:
        logger.debug(
            "TF geometry: l1=%.3f l2=%.3f r=%.3f",
            float(geometry.get("l1", 0)), float(geometry.get("l2", 0)),
            float(geometry.get("r", 0)),
        )

    return InverseResult(p_tpsh=f_rod, v_near_zero=v_near_zero, tf=tf)


def compute_rod_load(
    p_measured: np.ndarray,
    p_delta: np.ndarray,
    p_r: np.ndarray,
    v: np.ndarray,
    geometry: dict,
    motor: dict,
    v_zero_threshold_fraction: float = 0.01,
) -> InverseResult:
    """
    P_ТПШ(t) по формуле (2.32):
    P_ТПШ = (P − P_XX + Pδ + PR) / (l1/(l2·η_ЭД·η_пер.) · v)
    """
    eta_ed = float(motor["eta_ed"])
    eta_per = float(motor["eta_per"])
    p_xx = float(motor["P_xx_w"])
    l1 = float(geometry["l1"])
    l2 = float(geometry["l2"])

    denom_factor = l1 / (l2 * eta_ed * eta_per)
    v_denom = denom_factor * v

    v_max = np.max(np.abs(v))
    threshold = v_zero_threshold_fraction * v_max
    v_near_zero = np.abs(v) < threshold

    if np.any(v_near_zero):
        n_bad = int(np.sum(v_near_zero))
        warnings.warn(
            f"Формула (2.32): {n_bad} точек с |v| < {threshold:.4f} м/с "
            f"({v_zero_threshold_fraction*100:.0f}% от max|v|) — "
            f"результат P_ТПШ вблизи мёртвых точек недостоверен",
            stacklevel=2,
        )
        logger.warning(
            "Мёртвые точки хода: %d/%d отсчётов (|v| < %.4g м/с)",
            n_bad,
            len(v),
            threshold,
        )

    # формула (2.32), см. dissertation Яшин_расчет.pdf
    numerator = p_measured - p_xx + p_delta + p_r
    v_safe = np.where(np.abs(v_denom) < 1e-9, np.nan, v_denom)
    p_tpsh = numerator / v_safe

    return InverseResult(p_tpsh=p_tpsh, v_near_zero=v_near_zero)


def compute_rod_load_simple(
    p_net: np.ndarray,
    v: np.ndarray,
    k_v: float,
    epsilon: float = 0.02,
) -> InverseResult:
    """
    MVP: F = P_net / (k_v · v_reg), v_reg = √(v² + ε²).

    k_v заменяет l1/(l2·η_ЭД·η_пер); ε устраняет сингулярность в мёртвых точках.
    """
    p_net = np.asarray(p_net, dtype=float)
    v = np.asarray(v, dtype=float)
    kv = float(k_v)
    eps = float(epsilon)
    if kv <= 0:
        raise ValueError(f"k_v должен быть > 0, получено {kv}")

    v_reg = np.sqrt(v**2 + eps**2)
    f_rod = p_net / (kv * v_reg)
    v_near_zero = np.abs(v) < eps

    return InverseResult(p_tpsh=f_rod, v_near_zero=v_near_zero, tf=v_reg / max(kv, 1e-12))
