"""Упругая нагрузка колонны штанг по закону Гука: F = E A (ΔL / L)."""

from __future__ import annotations

from typing import Any

import numpy as np


def rod_cross_section_area(rod_diameter: float) -> float:
    """A = π D² / 4."""
    d = float(rod_diameter)
    return float(np.pi * d * d / 4.0)


def compute_elastic_force(
    delta_L: np.ndarray | float,
    E: float,
    rod_length: float,
    rod_diameter: float | None = None,
    A: float | None = None,
) -> np.ndarray:
    """
    F_elastic = E · A · (ΔL / L)  [N].

    Parameters
    ----------
    delta_L : текущее удлинение колонны [м]
    E : модуль Юнга [Па]
    rod_length : длина колонны L [м]
    rod_diameter : диаметр штанг D [м] (если A не задана)
    A : площадь сечения [м²]; иначе A = π D² / 4
    """
    delta = np.asarray(delta_L, dtype=float)
    L = float(rod_length)
    if L <= 0:
        raise ValueError(f"rod_length должен быть > 0, получено {L}")
    if A is None:
        if rod_diameter is None:
            raise ValueError("Нужно задать rod_diameter или A")
        A = rod_cross_section_area(rod_diameter)
    else:
        A = float(A)
    return float(E) * A * (delta / L)


def elongation_from_stroke(s: np.ndarray) -> np.ndarray:
    """
    ΔL(t) от кинематики: удлинение относительно нижней мёртвой точки.

    ΔL = S(t) − min(S)  ≥ 0 за цикл.
    """
    s = np.asarray(s, dtype=float)
    return s - float(np.min(s))


def is_elastic_enabled(config: dict[str, Any] | None) -> bool:
    if not config:
        return False
    return bool((config.get("elastic") or {}).get("enabled", False))


def compute_elastic_force_from_config(
    s: np.ndarray,
    config: dict[str, Any] | None,
    delta_L: np.ndarray | None = None,
) -> np.ndarray:
    """
    F_elastic из config.elastic; при enabled=False — нули.

    Параметры конфига: E, rod_length, rod_diameter [, A].
    """
    s = np.asarray(s, dtype=float)
    n = len(s)
    if not is_elastic_enabled(config):
        return np.zeros(n, dtype=float)

    el = (config or {}).get("elastic") or {}
    E = float(el.get("E", 2.1e11))
    L = float(el.get("rod_length", el.get("L", 1500.0)))
    D = el.get("rod_diameter", el.get("diameter"))
    A = el.get("A")
    if A is not None:
        A = float(A)
    elif D is not None:
        D = float(D)
    else:
        D = 0.022

    if delta_L is None:
        delta_L = elongation_from_stroke(s)
    return compute_elastic_force(delta_L, E, L, rod_diameter=D, A=A)
