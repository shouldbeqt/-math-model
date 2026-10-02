"""
Wave Propagation — модель колонны штанг Гиббса.

Решается 1D уравнение:
  u_tt + ν u_t = c² u_xx + g

Граничные условия:
  u(0, t) = S(t)                 (поверхность / полированный шток)
  E A u_x(L, t) = F_head(t)      (низ колонны)

Численно: неявная конечно-разностная схема 2-го порядка по x,
трёхдиагональная система на каждом шаге (scipy.linalg.solve_banded).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_banded

logger = logging.getLogger(__name__)

G_DEFAULT = 9.80665


@dataclass
class RodWaveParameters:
    """Параметры упругой колонны штанг (Gibbs)."""

    L: float
    E: float
    rho: float
    diameter: float
    A: float
    nu: float
    n_nodes: int
    g: float = G_DEFAULT

    @property
    def c(self) -> float:
        """Скорость продольной волны c = √(E/ρ)."""
        return float(np.sqrt(self.E / self.rho))

    @property
    def dx(self) -> float:
        """Шаг сетки dx = L / (N − 1)."""
        return self.L / (self.n_nodes - 1)

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None = None) -> RodWaveParameters:
        """Сборка из config.rod_wave (A = π D²/4)."""
        cfg = cfg or {}
        d = float(cfg.get("diameter", 0.022))
        a_cfg = cfg.get("A")
        area = float(a_cfg) if a_cfg is not None else float(np.pi * d * d / 4.0)
        n = int(cfg.get("n_nodes", 51))
        if n < 3:
            raise ValueError(f"rod_wave.n_nodes должен быть ≥ 3, получено {n}")
        return cls(
            L=float(cfg.get("L", 1500.0)),
            E=float(cfg.get("E", 2.1e11)),
            rho=float(cfg.get("rho", 7850.0)),
            diameter=d,
            A=area,
            nu=float(cfg.get("nu", 0.5)),
            n_nodes=n,
            g=float(cfg.get("g", G_DEFAULT)),
        )


@dataclass
class RodWaveResult:
    """Результат волнового расчёта по колонне."""

    x: np.ndarray          # [n_nodes]
    t: np.ndarray          # [n_times]
    u: np.ndarray          # [n_nodes, n_times]
    epsilon: np.ndarray    # [n_nodes, n_times]
    f_elastic: np.ndarray  # [n_nodes, n_times]
    f_surface: np.ndarray  # [n_times]
    f_head: np.ndarray     # [n_times]
    params: RodWaveParameters


def _surface_derivative(u: np.ndarray, dx: float) -> np.ndarray:
    """du/dx|_(x=0) — односторонняя 2-го порядка: (−3u0 + 4u1 − u2)/(2 dx)."""
    return (-3.0 * u[0] + 4.0 * u[1] - u[2]) / (2.0 * dx)


def _strain_field(u: np.ndarray, dx: float) -> np.ndarray:
    """ε = ∂u/∂x по x для всех узлов (центральная / односторонняя на краях)."""
    eps = np.empty_like(u)
    eps[0] = (-3.0 * u[0] + 4.0 * u[1] - u[2]) / (2.0 * dx)
    eps[-1] = (3.0 * u[-1] - 4.0 * u[-2] + u[-3]) / (2.0 * dx)
    eps[1:-1] = (u[2:] - u[:-2]) / (2.0 * dx)
    return eps


def solve_rod_wave(
    s: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    f_head: np.ndarray,
    dt: float,
    params: RodWaveParameters,
) -> RodWaveResult:
    """
    Неявная конечно-разностная схема для u_tt + ν u_t = c² u_xx + g.

    Входы:
      s, v, a — кинематика поверхности [м, м/с, м/с²]
      f_head  — сила на головке (нижнее ГУ) [N]
      dt      — шаг по времени [с]
    """
    del a  # API-совместимость с кинематикой
    s = np.asarray(s, dtype=float)
    v = np.asarray(v, dtype=float)
    f_head = np.asarray(f_head, dtype=float)
    n_t = len(s)
    if len(f_head) != n_t or len(v) != n_t:
        raise ValueError("s, v, f_head должны иметь одинаковую длину")
    if dt <= 0:
        raise ValueError(f"dt должен быть > 0, получено {dt}")

    n = params.n_nodes
    dx = params.dx
    c2 = params.c ** 2
    nu = params.nu
    g = params.g
    ea = params.E * params.A
    r = c2 * dt * dt / (dx * dx)
    alpha = 1.0 + 0.5 * nu * dt + 2.0 * r
    beta = 1.0 - 0.5 * nu * dt

    x = np.linspace(0.0, params.L, n)
    t = np.arange(n_t, dtype=float) * dt
    u = np.zeros((n, n_t), dtype=float)

    # IC: u(x,0)=S(0); u(x,−dt) ≈ S(0) − dt·v(0)
    u[:, 0] = s[0]
    u_nm1 = np.full(n, float(s[0] - dt * v[0]), dtype=float)
    u_n = u[:, 0].copy()

    # Неизвестные: узлы 1 … n−1
    m = n - 1
    ab = np.zeros((3, m), dtype=float)
    # scipy banded: ab[0,j]=a[j-1,j] (upper), ab[1]=diag, ab[2,j]=a[j+1,j] (lower of eq j+1)
    ab[0, 1:] = -r
    ab[1, :] = alpha
    if m >= 3:
        ab[2, : m - 2] = -r
    if m >= 2:
        ab[2, m - 2] = -2.0 * r  # нижнее ГУ через ghost-узел

    rhs = np.zeros(m, dtype=float)
    g_term = g * dt * dt
    ghost_scale = 2.0 * r * dx / ea

    for k in range(1, n_t):
        s_k = float(s[k])
        f_k = float(f_head[k])
        # RHS для узлов 1..n-2 (индексы unknowns 0..m-2)
        rhs[: m - 1] = 2.0 * u_n[1:n - 1] - beta * u_nm1[1:n - 1] + g_term
        rhs[0] += r * s_k  # Дирихле u_0 = S
        # узел n-1 + ghost для Neumann
        rhs[-1] = 2.0 * u_n[-1] - beta * u_nm1[-1] + g_term + ghost_scale * f_k

        u_int = solve_banded((1, 1), ab.copy(), rhs, overwrite_ab=True, check_finite=False)
        u_np1 = np.empty(n, dtype=float)
        u_np1[0] = s_k
        u_np1[1:] = u_int

        u[:, k] = u_np1
        u_nm1 = u_n
        u_n = u_np1

    epsilon = np.empty_like(u)
    f_elastic = np.empty_like(u)
    f_surface = np.empty(n_t, dtype=float)
    for k in range(n_t):
        epsilon[:, k] = _strain_field(u[:, k], dx)
        f_elastic[:, k] = ea * epsilon[:, k]
        f_surface[k] = ea * _surface_derivative(u[:, k], dx)

    logger.info(
        "Rod wave: L=%.0f m, c=%.0f m/s, N=%d, F_surf∈[%.0f, %.0f] N",
        params.L, params.c, n,
        float(np.nanmin(f_surface)), float(np.nanmax(f_surface)),
    )

    return RodWaveResult(
        x=x, t=t, u=u, epsilon=epsilon, f_elastic=f_elastic,
        f_surface=f_surface, f_head=f_head.copy(), params=params,
    )


def save_displacement_field(result: RodWaveResult, path: str | Path) -> Path:
    """Сохранить u(x,t) как .npz (массив + оси x, t)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        u=result.u,
        x=result.x,
        t=result.t,
        epsilon=result.epsilon,
        f_elastic=result.f_elastic,
        f_surface=result.f_surface,
        f_head=result.f_head,
        L=result.params.L,
        E=result.params.E,
        c=result.params.c,
    )
    return path


def plot_wave_propagation(
    result: RodWaveResult,
    output_path: str | Path,
    n_snapshots: int = 8,
) -> None:
    """Профили u(x) в несколько моментов цикла — распространение волны."""
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    n_t = result.u.shape[1]
    idxs = np.linspace(0, n_t - 1, min(n_snapshots, n_t), dtype=int)
    for i in idxs:
        ax.plot(result.x, result.u[:, i], label=f"t={result.t[i]:.2f} с")
    ax.set_xlabel("x вдоль колонны, м")
    ax.set_ylabel("u(x), м")
    ax.set_title("Распространение продольной волны по колонне штанг")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_head_vs_surface_force(
    result: RodWaveResult,
    output_path: str | Path,
) -> None:
    """Сравнение F_head(t) и F_surface(t)."""
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(result.t, result.f_head / 9.80665, "b-", label="F_head (вход)")
    ax.plot(result.t, result.f_surface / 9.80665, "r--", label="F_surface (упругая)")
    ax.set_xlabel("t, с")
    ax.set_ylabel("F, kgf")
    ax.set_title("Сила на головке vs упругая поверхностная сила")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def resolve_friction_force(n: int, config: dict[str, Any] | None) -> np.ndarray:
    """F_friction из config.friction (по умолчанию 0)."""
    fr = (config or {}).get("friction") or {}
    mode = fr.get("mode", "none")
    if mode in (None, "none", "off", False):
        return np.zeros(n, dtype=float)
    if mode == "constant":
        return np.full(n, float(fr.get("F_friction_n", 0.0)), dtype=float)
    raise ValueError(f"friction.mode={mode!r}; допустимо 'none' или 'constant'")


def is_wave_model_enabled(config: dict[str, Any] | None) -> bool:
    """wave_model: True в корне конфига или в model.wave_model / rod_wave.enabled."""
    if not config:
        return False
    if config.get("wave_model") is True:
        return True
    if (config.get("model") or {}).get("wave_model") is True:
        return True
    if (config.get("rod_wave") or {}).get("enabled") is True:
        return True
    return False
