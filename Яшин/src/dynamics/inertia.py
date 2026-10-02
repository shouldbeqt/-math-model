"""Инерционная сила штока: F_inertia = m_eq · a."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_inertia_force(
    a: np.ndarray,
    m_eq: float,
) -> np.ndarray:
    """
    F_inertia = m_eq · a  [N].

    m_eq — эффективная масса приведённая к полированному штоку [kg].
    a — ускорение штока [m/s²].
    """
    a = np.asarray(a, dtype=float)
    m = float(m_eq)
    if m == 0.0:
        return np.zeros_like(a)
    return m * a


def resolve_m_eq(config: dict[str, Any] | None) -> float:
    """
    Эффективная масса из конфигурации.

    Ищет dynamics.m_eq, затем верхнеуровневый m_eq.
    По умолчанию 0 (инерция отключена).
    """
    if not config:
        return 0.0
    dyn = config.get("dynamics") or {}
    if dyn.get("m_eq") is not None:
        return float(dyn["m_eq"])
    if config.get("m_eq") is not None:
        return float(config["m_eq"])
    return 0.0
