"""Валидация: выравнивание по фазе, RMSE, MAPE, площадь петли (Раздел З ТЗ)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    rmse_f: float
    mape_f: float
    area_calc: float
    area_ref: float
    area_diff_percent: float
    rmse_s: Optional[float] = None
    rmse_f_upstroke: Optional[float] = None
    rmse_f_downstroke: Optional[float] = None
    branch_ratio_calc: Optional[float] = None
    branch_ratio_ref: Optional[float] = None


def _force_to_newtons(f: np.ndarray, unit: str) -> np.ndarray:
    """Приводит силу к Н для сравнения."""
    if unit in ("N", "n"):
        return f
    if unit in ("kgf", "kg", "кг"):
        return f * 9.80665
    if unit in ("daN", "dan"):
        return f * 10.0
    if unit in ("kN", "kn"):
        return f * 1000.0
    return f


def _apply_v_near_zero_mask(f: np.ndarray, v_near_zero: Optional[np.ndarray]) -> np.ndarray:
    """Заменяет F в точках |v|≈0 на NaN перед интерполяцией."""
    f = np.asarray(f, dtype=float).copy()
    if v_near_zero is not None and len(v_near_zero) == len(f):
        f[v_near_zero] = np.nan
    return f


def _sanitize_force(f: np.ndarray) -> np.ndarray:
    """Интерполяция NaN/Inf (мёртвые точки хода) для метрик и графиков."""
    f = np.asarray(f, dtype=float)
    bad = ~np.isfinite(f)
    if not np.any(bad):
        return f
    if np.all(bad):
        return f
    good_idx = np.where(~bad)[0]
    bad_idx = np.where(bad)[0]
    f_out = f.copy()
    f_out[bad_idx] = np.interp(bad_idx, good_idx, f[good_idx])
    return f_out


def _phase_parametrize(s: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Параметризация по фазе τ ∈ [0, 1] через накопленную дугу в (S, F)."""
    ds = np.diff(s, prepend=s[0])
    df = np.diff(f, prepend=f[0])
    arc = np.cumsum(np.sqrt(ds**2 + df**2))
    if arc[-1] < 1e-12:
        tau = np.linspace(0, 1, len(s))
    else:
        tau = arc / arc[-1]
    return tau, s, f


def _resample_by_phase(tau: np.ndarray, s: np.ndarray, f: np.ndarray, n_points: int):
    grid = np.linspace(0, 1, n_points)
    s_r = np.interp(grid, tau, s)
    f_r = np.interp(grid, tau, f)
    return grid, s_r, f_r


def _loop_area(s: np.ndarray, f: np.ndarray) -> float:
    """Площадь замкнутой петли ∮F dS (работа за цикл)."""
    return float(abs(np.trapezoid(f, s)))


def _stroke_branch_masks(
    s: np.ndarray,
    v: np.ndarray | None = None,
    s_ref: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Маски подъёма/спуска (для метрик).

    Для расчёта: v>0 — подъём, v<0 — спуск.
    Для эталона (без v): dS>0 — подъём, dS<0 — спуск.
    """
    if v is not None and len(v) == len(s):
        return v > 0, v < 0
    if s_ref is not None:
        ds = np.diff(s_ref, prepend=s_ref[0])
        return ds > 0, ds < 0
    ds = np.diff(s, prepend=s[0])
    return ds > 0, ds < 0


def split_reference_branches(
    s: np.ndarray,
    f: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Эталон: две ветви с разрывом у ВМТ (мёртвая зона dS=0 не рисуется).

    Подъём — от НМТ до последней точки с dS>0; спуск — с первой dS<0.
    """
    s = np.asarray(s, dtype=float)
    f = np.asarray(f, dtype=float)
    ds = np.diff(s, prepend=s[0])
    dn_start = int(np.flatnonzero(ds < 0)[0])
    up_pos = np.flatnonzero(ds > 0)
    if len(up_pos) == 0:
        return s[:1], f[:1], s[1:], f[1:]
    up_end = int(up_pos[-1]) + 1
    return s[:up_end], f[:up_end], s[dn_start:], f[dn_start:]


def _valid_plot_mask(
    f: np.ndarray,
    dead_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Корректные точки для графика: не |v|≈0 и конечная F."""
    good = np.isfinite(f)
    if dead_mask is not None and len(dead_mask) == len(f):
        good &= ~np.asarray(dead_mask, dtype=bool)
    return good


def strip_dead_points(
    s: np.ndarray,
    f: np.ndarray,
    dead_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Убирает мёртвые точки; matplotlib соединяет соседние корректные хордой.

    Интерполяция F не выполняется — только прямой отрезок между
    последней и следующей корректной точками.
    """
    s = np.asarray(s, dtype=float)
    f = np.asarray(f, dtype=float)
    good = _valid_plot_mask(f, dead_mask)
    return s[good], f[good]


def _stroke_dn0(v: np.ndarray) -> int:
    neg = np.flatnonzero(v < 0)
    return int(neg[0]) if len(neg) else len(v) // 2


def split_calc_strokes_for_plot(
    s: np.ndarray,
    f: np.ndarray,
    v: np.ndarray,
    dead_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Расчёт для графика: ↑/↓ без мёртвых точек (|v|≈0).

    Разрывы заполняются хордой между последней и следующей корректной точкой
    (при отрисовке через ax.plot).
    """
    s = np.asarray(s, dtype=float)
    f = np.asarray(f, dtype=float)
    v = np.asarray(v, dtype=float)
    good = _valid_plot_mask(f, dead_mask)
    dn0 = _stroke_dn0(v)

    up = good.copy()
    up[dn0:] = False
    dn = good.copy()
    dn[:dn0] = False
    return s[up], f[up], s[dn], f[dn]


def split_calc_branches(
    s: np.ndarray,
    f: np.ndarray,
    v: np.ndarray,
    v_gap_fraction: float = 0.06,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Расчёт: подъём (v>0) и спуск (v<0) с разрывом у |v|≈0 (ВМТ/НМТ).

    Точки с |v| < v_gap не входят ни в одну ветвь — на графике разрыв, как у эталона.
    """
    s = np.asarray(s, dtype=float)
    f = np.asarray(f, dtype=float)
    v = np.asarray(v, dtype=float)
    v_lim = float(v_gap_fraction * np.max(np.abs(v)))
    up = v > v_lim
    dn = v < -v_lim
    return s[up], f[up], s[dn], f[dn]


def branch_force_metrics(
    s_calc: np.ndarray,
    f_calc: np.ndarray,
    s_ref: np.ndarray,
    f_ref: np.ndarray,
    v_calc: np.ndarray | None = None,
    calc_force_unit: str = "N",
    ref_force_unit: str = "kgf",
    s_tol: float = 0.08,
    s_grid: np.ndarray | None = None,
) -> dict[str, float]:
    """
    RMSE по ветвям S-F (подъём/спуск) — физически значимая метрика петли.

    В отличие от MAPE F(τ), не скрывает ошибку формы при циклическом сдвиге.
    """
    f_calc = _force_to_newtons(f_calc, calc_force_unit)
    f_ref = _force_to_newtons(f_ref, ref_force_unit)
    grid = s_grid if s_grid is not None else np.linspace(
        max(float(np.min(s_calc)), float(np.min(s_ref)), 0.05),
        min(float(np.max(s_calc)), float(np.max(s_ref)), 3.45),
        50,
    )

    up_c, dn_c = _stroke_branch_masks(s_calc, v_calc)
    up_r, dn_r = _stroke_branch_masks(s_ref, s_ref=s_ref)

    def _branch_rmse(branch_c: np.ndarray, branch_r: np.ndarray) -> float:
        errs: list[float] = []
        for sg in grid:
            fc = f_calc[(np.abs(s_calc - sg) < s_tol) & branch_c]
            fr = f_ref[(np.abs(s_ref - sg) < s_tol) & branch_r]
            if len(fc) and len(fr):
                errs.append(float(fc.mean() - fr.mean()))
        if not errs:
            return float("nan")
        return float(np.sqrt(np.mean(np.array(errs) ** 2)))

    rmse_up = _branch_rmse(up_c, up_r)
    rmse_dn = _branch_rmse(dn_c, dn_r)

    def _branch_ratio(s_arr, f_arr, branch):
        vals = f_arr[branch]
        return float(np.mean(vals) / max(np.median(vals), 1e-6)) if np.any(branch) else float("nan")

    ratio_c = float("nan")
    ratio_r = float("nan")
    if np.any(up_c) and np.any(dn_c):
        ratio_c = float(np.mean(f_calc[dn_c]) / max(np.mean(f_calc[up_c]), 1e-6))
    if np.any(up_r) and np.any(dn_r):
        ratio_r = float(np.mean(f_ref[dn_r]) / max(np.mean(f_ref[up_r]), 1e-6))

    return {
        "rmse_f_upstroke": rmse_up,
        "rmse_f_downstroke": rmse_dn,
        "branch_ratio_calc": ratio_c,
        "branch_ratio_ref": ratio_r,
    }


def _align_phase(f_calc: np.ndarray, f_ref: np.ndarray) -> float:
    """
    Циклический сдвиг f_calc для минимизации RMSE по F(τ).
    Возвращает дробный shift (в индексах сетки); применяется через _roll_frac.
    """
    n = len(f_calc)
    best_shift = 0.0
    best_rmse = np.inf
    # целые сдвиги + субдискретное уточнение ±0.5 шага
    for shift_i in range(n):
        for delta in (0.0, -0.25, 0.25, -0.5, 0.5):
            shift = (shift_i + delta) % n
            rolled = _roll_frac(f_calc, shift)
            rmse = float(np.sqrt(np.mean((rolled - f_ref) ** 2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_shift = shift
    return best_shift


def _roll_frac(x: np.ndarray, shift: float) -> np.ndarray:
    """Циклический сдвиг массива на дробное число индексов (линейная интерполяция)."""
    n = len(x)
    if n == 0:
        return x
    shift = shift % n
    idx = np.arange(n, dtype=float)
    src = (idx - shift) % n
    i0 = np.floor(src).astype(int) % n
    i1 = (i0 + 1) % n
    w = src - np.floor(src)
    return (1.0 - w) * x[i0] + w * x[i1]


def validate_dynamograms(
    s_calc: np.ndarray,
    f_calc: np.ndarray,
    s_ref: np.ndarray,
    f_ref: np.ndarray,
    n_resample_points: int = 200,
    calc_force_unit: str = "N",
    ref_force_unit: str = "kgf",
    v_near_zero: Optional[np.ndarray] = None,
    scale_stroke_to_ref: bool = False,
    v_calc: np.ndarray | None = None,
) -> ValidationMetrics:
    """Выравнивание по фазе τ, RMSE/MAPE по F(τ), разница площадей."""
    s_calc = np.asarray(s_calc, dtype=float).copy()
    if scale_stroke_to_ref:
        s_ref_max = float(np.nanmax(s_ref))
        s_calc_max = float(np.nanmax(s_calc))
        if s_calc_max > 1e-9 and s_ref_max > 0:
            s_calc = s_calc * (s_ref_max / s_calc_max)

    f_calc = _apply_v_near_zero_mask(f_calc, v_near_zero)
    # Дополнительно: большие |F| там, где |v| мало (выбросы у мёртвых точек)
    if v_near_zero is not None and len(v_near_zero) == len(f_calc):
        f_calc = np.asarray(f_calc, dtype=float)
        f_calc[v_near_zero] = np.nan
    f_calc = _sanitize_force(_force_to_newtons(f_calc, calc_force_unit))
    f_ref = _sanitize_force(_force_to_newtons(f_ref, ref_force_unit))
    tau_c, _, _ = _phase_parametrize(s_calc, f_calc)
    tau_r, _, _ = _phase_parametrize(s_ref, f_ref)

    _, s_c, f_c = _resample_by_phase(tau_c, s_calc, f_calc, n_resample_points)
    _, s_r, f_r = _resample_by_phase(tau_r, s_ref, f_ref, n_resample_points)

    shift = _align_phase(f_c, f_r)
    f_c = _roll_frac(f_c, shift)
    s_c = _roll_frac(s_c, shift)

    rmse_f = float(np.sqrt(np.mean((f_c - f_r) ** 2)))
    mask = np.abs(f_r) > 1e-6
    mape_f = float(np.mean(np.abs((f_c[mask] - f_r[mask]) / f_r[mask])) * 100) if np.any(mask) else float("nan")
    rmse_s = float(np.sqrt(np.mean((s_c - s_r) ** 2)))

    area_calc = _loop_area(s_calc, f_calc)
    area_ref = _loop_area(s_ref, f_ref)
    area_diff = abs(area_calc - area_ref) / max(abs(area_ref), 1e-9) * 100

    branch = branch_force_metrics(
        s_calc, f_calc, s_ref, f_ref,
        v_calc=v_calc,
        calc_force_unit="N",
        ref_force_unit="N",
    )

    return ValidationMetrics(
        rmse_f=rmse_f,
        mape_f=mape_f,
        rmse_s=rmse_s,
        area_calc=area_calc,
        area_ref=area_ref,
        area_diff_percent=area_diff,
        rmse_f_upstroke=branch["rmse_f_upstroke"],
        rmse_f_downstroke=branch["rmse_f_downstroke"],
        branch_ratio_calc=branch["branch_ratio_calc"],
        branch_ratio_ref=branch["branch_ratio_ref"],
    )


def save_metrics(metrics: ValidationMetrics, path: str | Path, extra: Optional[dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(metrics)
    if extra:
        data.update(extra)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_dynamogram_overlay(
    s_calc: np.ndarray,
    f_calc: np.ndarray,
    s_ref: np.ndarray,
    f_ref: np.ndarray,
    v_near_zero: np.ndarray,
    output_path: str | Path,
    v_calc: np.ndarray | None = None,
    calc_force_unit: str = "N",
    ref_force_unit: str = "kgf",
) -> None:
    """dynamogram_overlay.png — ↑/↓; |v|≈0 пропущены, разрывы соединены хордой."""
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(8, 6))

    f_ref_kgf = _force_to_newtons(f_ref, ref_force_unit) / 9.80665
    s_r_up, f_r_up, s_r_dn, f_r_dn = split_reference_branches(s_ref, f_ref_kgf)
    ax.plot(s_r_up, f_r_up, "b-", linewidth=1.5, label="Эталон ↑")
    ax.plot(s_r_dn, f_r_dn, "b--", linewidth=1.2, alpha=0.9, label="Эталон ↓")

    f_calc_n = _force_to_newtons(f_calc, calc_force_unit)
    dead = v_near_zero if len(v_near_zero) == len(s_calc) else None
    if v_calc is not None and len(v_calc) == len(s_calc):
        s_c_up, f_c_up, s_c_dn, f_c_dn = split_calc_strokes_for_plot(
            s_calc, f_calc_n, v_calc, dead,
        )
        f_c_up /= 9.80665
        f_c_dn /= 9.80665
        ax.plot(s_c_up, f_c_up, "r-", linewidth=1.5, label="Расчёт ↑")
        if len(s_c_up) and len(s_c_dn):
            ax.plot(
                np.r_[s_c_up[-1], s_c_dn],
                np.r_[f_c_up[-1], f_c_dn],
                "r--",
                linewidth=1.2,
                alpha=0.9,
                label="Расчёт ↓",
            )
        elif len(s_c_dn):
            ax.plot(s_c_dn, f_c_dn, "r--", linewidth=1.2, alpha=0.9, label="Расчёт ↓")
        if len(s_c_dn) and len(s_c_up):
            ax.plot(
                [s_c_dn[-1], s_c_up[0]],
                [f_c_dn[-1], f_c_up[0]],
                "r-",
                linewidth=1.2,
                alpha=0.9,
            )
    else:
        s_plot, f_plot = strip_dead_points(s_calc, f_calc_n, dead)
        ax.plot(s_plot, f_plot / 9.80665, "r--", linewidth=1.5, label="Расчётная")

    ax.set_xlabel("S, м")
    ax.set_ylabel("F, kgf")
    ax.set_title("Динамограмма: ↑ подъём, ↓ спуск (|v|≈0 — хорда)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_force_vs_phase(
    s_calc: np.ndarray,
    f_calc: np.ndarray,
    s_ref: np.ndarray,
    f_ref: np.ndarray,
    n_points: int,
    output_path: str | Path,
) -> None:
    """force_vs_phase.png — Раздел И, п.3."""
    tau_c, _, _ = _phase_parametrize(s_calc, f_calc)
    tau_r, _, _ = _phase_parametrize(s_ref, f_ref)
    grid_c, _, f_c = _resample_by_phase(tau_c, s_calc, f_calc, n_points)
    grid_r, _, f_r = _resample_by_phase(tau_r, s_ref, f_ref, n_points)
    shift = _align_phase(f_c, f_r)
    f_c = _roll_frac(f_c, shift)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(grid_r, f_r, "b-", label="Эталонная F(τ)")
    ax.plot(grid_c, f_c, "r--", label="Расчётная F(τ)")
    ax.set_xlabel("Фаза цикла τ")
    ax.set_ylabel("F")
    ax.set_title("F(τ) после выравнивания по фазе")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_metrics_summary(metrics: ValidationMetrics, extra: dict, output_path: str | Path) -> None:
    """metrics_summary.png — Раздел И, п.4."""
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    rows = [
        ["RMSE F(τ)", f"{metrics.rmse_f:.2f}"],
        ["MAPE F(τ), %", f"{metrics.mape_f:.2f}"],
        ["RMSE F↑ (S-F)", f"{metrics.rmse_f_upstroke:.0f}" if metrics.rmse_f_upstroke is not None else "—"],
        ["RMSE F↓ (S-F)", f"{metrics.rmse_f_downstroke:.0f}" if metrics.rmse_f_downstroke is not None else "—"],
        ["F↓/F↑ расч.", f"{metrics.branch_ratio_calc:.2f}" if metrics.branch_ratio_calc is not None else "—"],
        ["F↓/F↑ эталон", f"{metrics.branch_ratio_ref:.2f}" if metrics.branch_ratio_ref is not None else "—"],
        ["RMSE S(τ), м", f"{metrics.rmse_s:.2f}" if metrics.rmse_s is not None else "—"],
        ["Площадь расч., Дж", f"{metrics.area_calc:.1f}"],
        ["Площадь эталон, Дж", f"{metrics.area_ref:.1f}"],
        ["Δ площади, %", f"{metrics.area_diff_percent:.2f}"],
        ["Метод ток→момент", extra.get("method", "—")],
        ["dt, с", str(extra.get("dt_seconds", "—"))],
        ["SPM", f"{extra.get('spm', '—')}"],
    ]
    table = ax.table(cellText=rows, colLabels=["Метрика", "Значение"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title("Метрики валидации")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
