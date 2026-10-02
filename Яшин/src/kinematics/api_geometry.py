"""
Кинематика станка-качалки по API Spec 11E / Svinos (четырёхзвенник Class I).

S(θ) = PR(θ) · S₀,  PR = (ψ_B − ψ(θ)) / (ψ_B − ψ_T),
где ψ(θ) — угол walking beam, θ — угол кривошипа от 12 ч (по часовой).

v = dS/dt,  a = d²S/dt² при ω = 2π/T (постоянная, как в гармонической модели).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.yashin_kinematics import KinematicsResult

logger = logging.getLogger(__name__)

INCH_TO_M = 0.0254

# Lufkin C-640D-305-120, API geometry table (inches), R = R3 (120" stroke)
C640D_305_120_IN: dict[str, float] = {
    "A_in": 155.0,
    "C_in": 111.09,
    "I_in": 111.0,
    "P_in": 133.50,
    "H_in": 234.0,
    "G_in": 96.0,
    "R_in": 30.0,
    "unit": "C-640D-305-120",
}


@dataclass(frozen=True)
class ApiUnitGeometry:
    """Размеры четырёхзвенника API 11E (Class I), метры."""

    A: float
    C: float
    P: float
    R: float
    I: float
    H: float
    G: float
    name: str = "custom"

    @property
    def K(self) -> float:
        return float(np.hypot(self.H - self.G, self.I))

    @classmethod
    def from_inches(
        cls,
        A: float,
        C: float,
        P: float,
        R: float,
        I: float,
        H: float,
        G: float,
        name: str = "custom",
    ) -> ApiUnitGeometry:
        return cls(
            A=A * INCH_TO_M,
            C=C * INCH_TO_M,
            P=P * INCH_TO_M,
            R=R * INCH_TO_M,
            I=I * INCH_TO_M,
            H=H * INCH_TO_M,
            G=G * INCH_TO_M,
            name=name,
        )

    @classmethod
    def c640_default(cls) -> ApiUnitGeometry:
        g = C640D_305_120_IN
        return cls.from_inches(
            g["A_in"], g["C_in"], g["P_in"], g["R_in"],
            g["I_in"], g["H_in"], g["G_in"],
            name=str(g["unit"]),
        )


@dataclass(frozen=True)
class _LinkageAngles:
    alpha: float
    gamma: float
    psi_b: float
    psi_t: float
    stroke_geom: float


def _clip_acos(x: float | np.ndarray) -> float | np.ndarray:
    return np.arccos(np.clip(x, -1.0, 1.0))


def _linkage_constants(geom: ApiUnitGeometry) -> _LinkageAngles:
    """Углы ψ_B, ψ_T и вспомогательные α, γ (Svinos, API 11E Appendix B)."""
    k = geom.K
    r, p, c = geom.R, geom.P, geom.C
    gamma = float(_clip_acos(((r + p) ** 2 + k ** 2 - c ** 2) / (2.0 * (r + p) * k)))
    alpha = float(np.arcsin(np.clip(geom.I / k, -1.0, 1.0)) - gamma)
    psi_b = float(_clip_acos((c ** 2 + k ** 2 - (p + r) ** 2) / (2.0 * c * k)))
    psi_t = float(_clip_acos((c ** 2 + k ** 2 - (p - r) ** 2) / (2.0 * c * k)))
    stroke_geom = geom.A * (psi_b - psi_t)
    return _LinkageAngles(alpha=alpha, gamma=gamma, psi_b=psi_b, psi_t=psi_t, stroke_geom=stroke_geom)


def beam_angle_psi(theta: np.ndarray, geom: ApiUnitGeometry, lc: _LinkageAngles | None = None) -> np.ndarray:
    """
    Угол walking beam ψ(θ), рад.

    θ — угол кривошипа от 12 ч, по часовой [rad].
    """
    lc = lc or _linkage_constants(geom)
    theta = np.asarray(theta, dtype=float)
    k, r, p, c = geom.K, geom.R, geom.P, geom.C
    theta2 = 2.0 * np.pi - theta + lc.alpha
    l_vec = np.sqrt(k ** 2 + r ** 2 - 2.0 * k * r * np.cos(theta2))
    l_vec = np.maximum(l_vec, 1e-12)
    cos_beta = (l_vec ** 2 + k ** 2 - r ** 2) / (2.0 * k * l_vec)
    beta = _clip_acos(cos_beta)
    beta = np.where((theta2 > 0) & (theta2 < np.pi), beta, -beta)
    cos_psi = (c ** 2 + l_vec ** 2 - p ** 2) / (2.0 * c * l_vec)
    return _clip_acos(cos_psi) + beta


def stroke_fraction(
    theta: np.ndarray,
    geom: ApiUnitGeometry,
    lc: _LinkageAngles | None = None,
) -> np.ndarray:
    """PR(θ) ∈ [0, 1] — безразмерное положение штока (API polished-rod position)."""
    lc = lc or _linkage_constants(geom)
    psi = beam_angle_psi(theta, geom, lc)
    denom = lc.psi_b - lc.psi_t
    if abs(denom) < 1e-12:
        return np.zeros_like(np.asarray(theta, dtype=float))
    pr = (lc.psi_b - psi) / denom
    return np.clip(pr, 0.0, 1.0)


def stroke_displacement(
    theta: np.ndarray,
    A: float,
    C: float,
    P: float,
    R: float,
    *,
    I: float | None = None,
    H: float | None = None,
    G: float | None = None,
    K: float | None = None,
    s0_m: float | None = None,
) -> np.ndarray:
    """
    S = f(θ, A, C, P, R) [м].

    Если заданы только A,C,P,R — используется типовая геометрия C-640 для I,H,G.
    K можно задать явно (иначе K = √((H−G)²+I²)).
    s0_m — масштабирование хода (калибровка скважины).
    """
    if I is None or H is None or G is None:
        if K is not None:
            raise ValueError("stroke_displacement: при явном K нужны I, H, G для углов α, γ")
        geom = ApiUnitGeometry.c640_default()
        if not (np.isclose(A, geom.A) or A > 0):
            geom = ApiUnitGeometry(A=A, C=C, P=P, R=R, I=geom.I, H=geom.H, G=geom.G)
        else:
            geom = ApiUnitGeometry(A=A, C=C, P=P, R=R, I=geom.I, H=geom.H, G=geom.G)
    else:
        geom = ApiUnitGeometry(A=A, C=C, P=P, R=R, I=I, H=H, G=G)

    lc = _linkage_constants(geom)
    s0 = float(s0_m) if s0_m is not None else lc.stroke_geom
    scale = s0 / lc.stroke_geom if lc.stroke_geom > 1e-12 else 1.0
    return stroke_fraction(theta, geom, lc) * lc.stroke_geom * scale


def _theta_at_bottom(geom: ApiUnitGeometry, lc: _LinkageAngles) -> float:
    """Угол кривошипа при НМТ (PR ≈ 0), начало подъёма."""
    grid = np.linspace(0.0, 2.0 * np.pi, 7200, endpoint=False)
    pr = stroke_fraction(grid, geom, lc)
    return float(grid[int(np.argmin(pr))])


def resolve_api_geometry(kinematics_cfg: dict | None) -> ApiUnitGeometry:
    """Геометрия из config.kinematics.api_geometry или типовая C-640."""
    cfg = kinematics_cfg or {}
    api = cfg.get("api_geometry") or {}
    if api.get("use_default", True) and not any(
        api.get(k) is not None for k in ("A_m", "C_m", "P_m", "R_m", "A_in", "C_in", "P_in", "R_in")
    ):
        return ApiUnitGeometry.c640_default()

    def _len(key_m: str, key_in: str, default_in: float) -> float:
        if api.get(key_m) is not None:
            return float(api[key_m])
        if api.get(key_in) is not None:
            return float(api[key_in]) * INCH_TO_M
        return default_in * INCH_TO_M

    base = ApiUnitGeometry.c640_default()
    return ApiUnitGeometry(
        A=_len("A_m", "A_in", C640D_305_120_IN["A_in"]),
        C=_len("C_m", "C_in", C640D_305_120_IN["C_in"]),
        P=_len("P_m", "P_in", C640D_305_120_IN["P_in"]),
        R=_len("R_m", "R_in", C640D_305_120_IN["R_in"]),
        I=_len("I_m", "I_in", C640D_305_120_IN["I_in"]),
        H=_len("H_m", "H_in", C640D_305_120_IN["H_in"]),
        G=_len("G_m", "G_in", C640D_305_120_IN["G_in"]),
        name=str(api.get("unit", base.name)),
    )


def compute_kinematics_api_geometry(
    s0_m: float,
    period_seconds: float,
    n_samples: int,
    kinematics_cfg: dict | None = None,
    geometry: ApiUnitGeometry | None = None,
) -> KinematicsResult:
    """
    S(t), v(t), a(t) по четырёхзвеннику API 11E.

    t=0 — НМТ (S≈0), далее подъём; ω = 2π/T постоянна.
    """
    geom = geometry or resolve_api_geometry(kinematics_cfg)
    lc = _linkage_constants(geom)
    s0 = float(s0_m)
    scale = s0 / lc.stroke_geom if lc.stroke_geom > 1e-12 else 1.0

    omega = 2.0 * np.pi / period_seconds
    t = np.linspace(0.0, period_seconds, n_samples, endpoint=False)
    dt = period_seconds / n_samples
    theta0 = _theta_at_bottom(geom, lc)
    theta = (theta0 + omega * t) % (2.0 * np.pi)

    s = stroke_fraction(theta, geom, lc) * lc.stroke_geom * scale
    v = np.gradient(s, dt)
    a = np.gradient(v, dt)

    logger.info(
        "API geometry [%s]: S₀_geom=%.4f m, S₀_cfg=%.4f m, |v|_max=%.4f m/s, K=%.3f m",
        geom.name, lc.stroke_geom, s0, float(np.max(np.abs(v))), geom.K,
    )

    return KinematicsResult(
        gamma_0=lc.psi_b,
        gamma_1=lc.psi_t,
        delta_max=lc.psi_b - lc.psi_t,
        S0=s0,
        S=s,
        v=v,
        a=a,
        omega=omega,
        t=t,
        period_seconds=period_seconds,
        s0_source="config",
        kinematics_mode="api_geometry",
    )


def kinematics_from_config(
    kinematics_cfg: dict | None,
    s0_m: float,
    period_seconds: float,
    n_samples: int,
) -> KinematicsResult:
    """Роутер kinematics.mode: harmonic | api_geometry."""
    from src.yashin_kinematics import compute_kinematics_simple

    mode = (kinematics_cfg or {}).get("mode", "harmonic")
    if mode == "api_geometry":
        return compute_kinematics_api_geometry(
            s0_m, period_seconds, n_samples, kinematics_cfg=kinematics_cfg,
        )
    if mode == "harmonic":
        return compute_kinematics_simple(s0_m, period_seconds, n_samples)
    raise ValueError(f"kinematics.mode={mode!r}; допустимо 'harmonic' или 'api_geometry'")
