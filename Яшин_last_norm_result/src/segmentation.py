"""Сегментация токограммы на циклы качания (Раздел В.3 ТЗ)."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, find_peaks, savgol_filter

logger = logging.getLogger(__name__)


@dataclass
class CycleSlice:
    """Один полный цикл качания."""

    index: int
    start: int
    end: int
    current: np.ndarray
    length_samples: float
    n_current_peaks: int


@dataclass
class SegmentationResult:
    """Результат сегментации."""

    boundaries: np.ndarray
    period_samples: float
    period_seconds: Optional[float]
    smoothed: np.ndarray
    cycle_index: int
    cycle_start: int
    cycle_end: int
    cycle_current: np.ndarray
    full_cycle_indices: list[int] = field(default_factory=list)


def _parabolic_refine(signal: np.ndarray, peak_idx: int) -> float:
    if peak_idx <= 0 or peak_idx >= len(signal) - 1:
        return float(peak_idx)
    y0, y1, y2 = signal[peak_idx - 1], signal[peak_idx], signal[peak_idx + 1]
    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-12:
        return float(peak_idx)
    delta = 0.5 * (y0 - y2) / denom
    return peak_idx + float(np.clip(delta, -0.5, 0.5))


def _estimate_period_acf(
    signal: np.ndarray,
    dt_seconds: float,
    spm_min: float,
    spm_max: float,
) -> float:
    n = len(signal)
    sig = signal - np.mean(signal)
    acf = correlate(sig, sig, mode="full")[n - 1 :]
    acf = acf / (acf[0] + 1e-12)

    samples_per_min = 60.0 / dt_seconds
    lag_min = max(int(np.floor(samples_per_min / spm_max)), 2)
    lag_max = min(int(np.ceil(samples_per_min / spm_min)), n - 1)
    if lag_min >= lag_max:
        lag_max = min(n - 1, lag_min + 1)

    search = acf[lag_min:lag_max]
    if len(search) == 0:
        return float(n // 8)
    return float(lag_min + int(np.argmax(search)))


def _count_current_peaks(segment: np.ndarray, global_range: float, prominence_fraction: float) -> int:
    """
    Число основных токовых пиков в сегменте (как в ТЗ В.1: height≈30 А на скв.16427).
    Плато-пульсации не считаются — порог height = I_min + prominence_fraction · range.
    """
    if len(segment) < 3:
        return 0
    height_min = float(np.min(segment) + prominence_fraction * global_range)
    distance = max(5, len(segment) // 5)
    peaks, _ = find_peaks(segment, height=height_min, distance=distance)
    return len(peaks)


def _identify_full_cycles(
    boundaries: np.ndarray,
    period_samples: float,
    current: np.ndarray,
    global_range: float,
    prominence_fraction: float,
    tolerance: float = 0.2,
    max_length_factor: float = 1.4,
) -> list[int]:
    """
    Полные циклы: длина ≈ период (до max_length_factor·T) и ровно 1 основной токовый пик.
    """
    lo = (1.0 - tolerance) * period_samples
    hi = max_length_factor * period_samples
    full: list[int] = []
    for i in range(len(boundaries) - 1):
        start = int(np.floor(boundaries[i]))
        end = int(np.floor(boundaries[i + 1]))
        length = boundaries[i + 1] - boundaries[i]
        if not (lo <= length <= hi):
            continue
        seg = current[start:end]
        n_peaks = _count_current_peaks(seg, global_range, prominence_fraction)
        if n_peaks == 1:
            full.append(i)
        else:
            logger.debug(
                "Цикл #%d отброшен: длина=%.1f, основных пиков=%d (ожидается 1)",
                i, length, n_peaks,
            )
    return full


def _select_cycle_index(
    cycle_index: Union[int, str],
    n_cycles: int,
    full_cycle_indices: list[int],
) -> int:
    """
    Выбор рабочего цикла (Раздел В.3, п.7).

    cycle_index=-1 → последний ПОЛНЫЙ цикл (не обрезанный по краю записи).
    cycle_index="all" → не используется здесь; см. segment_all_cycles().
    """
    if n_cycles == 0:
        return 0

    if isinstance(cycle_index, str) and cycle_index == "all":
        # fallback для одиночного режима — последний полный
        return full_cycle_indices[-1] if full_cycle_indices else max(0, n_cycles - 1)

    if full_cycle_indices:
        if int(cycle_index) == -1:
            return full_cycle_indices[-1]
        idx = int(cycle_index)
        if idx < 0:
            idx = len(full_cycle_indices) + idx
        idx = int(np.clip(idx, 0, len(full_cycle_indices) - 1))
        return full_cycle_indices[idx]

    # нет полных циклов — legacy fallback
    idx = int(cycle_index)
    if idx < 0:
        idx = n_cycles + idx
    return int(np.clip(idx, 0, max(0, n_cycles - 1)))


def _refine_boundaries_xcorr(
    signal: np.ndarray,
    boundaries: np.ndarray,
    period_samples: float,
) -> np.ndarray:
    if len(boundaries) < 2:
        return boundaries
    start0 = int(np.floor(boundaries[0]))
    end0 = int(np.floor(boundaries[1]))
    template = signal[start0:end0]
    if len(template) < 4:
        return boundaries

    refined = [boundaries[0]]
    half_win = int(max(3, 0.1 * period_samples))
    for i in range(1, len(boundaries) - 1):
        expected = int(round(refined[-1] + period_samples))
        lo = max(0, expected - half_win)
        hi = min(len(signal), expected + half_win)
        segment = signal[lo:hi]
        if len(segment) <= len(template):
            refined.append(boundaries[i])
            continue
        corr = correlate(segment, template, mode="valid")
        refined.append(float(lo + int(np.argmax(corr))))
    refined.append(boundaries[-1])
    return np.array(refined, dtype=float)


def segment_tokogram(
    current: np.ndarray,
    dt_seconds: float,
    smoothing_window: int = 7,
    smoothing_polyorder: int = 2,
    spm_min: float = 3.0,
    spm_max: float = 12.0,
    prominence_fraction: float = 0.3,
    period_search_margin: float = 0.3,
    refine_by_xcorr: bool = False,
    cycle_index: Union[int, str] = -1,
) -> SegmentationResult:
    """Алгоритм Раздела В.3: SG → ACF → впадины → парабола → (опц.) xcorr."""
    current = np.asarray(current, dtype=float)
    n = len(current)

    window = smoothing_window if smoothing_window % 2 == 1 else smoothing_window + 1
    window = min(window, n if n % 2 == 1 else n - 1)
    if window < smoothing_polyorder + 2:
        window = smoothing_polyorder + 3
        if window % 2 == 0:
            window += 1
    smoothed = savgol_filter(current, window, smoothing_polyorder)

    period_samples = _estimate_period_acf(smoothed, dt_seconds, spm_min, spm_max)
    period_seconds = period_samples * dt_seconds
    spm_est = 60.0 / period_seconds if period_seconds > 0 else 0.0

    if spm_est < spm_min * 0.8 or spm_est > spm_max * 1.2:
        warnings.warn(
            f"Оцененный период ({spm_est:.2f} SPM) выходит за [{spm_min}, {spm_max}] SPM >20%",
            stacklevel=2,
        )

    i_range = current.max() - current.min()
    prominence = prominence_fraction * i_range
    expected_valleys = max(2, int(round(n / period_samples)))

    # distance = (1 − margin) · T по ТЗ; если впадин слишком мало — ослабить (эмпирика ТЗ: ~0.35·T)
    distance = int(max(5, round((1.0 - period_search_margin) * period_samples)))
    valleys, _ = find_peaks(-smoothed, distance=distance, prominence=prominence)
    if len(valleys) < expected_valleys - 1:
        distance = int(max(5, round(0.35 * period_samples)))
        valleys, _ = find_peaks(-smoothed, distance=distance, prominence=prominence)
        logger.info(
            "Повторный поиск впадин: distance=%d (0.35·T), найдено %d (ожидалось ~%d)",
            distance, len(valleys), expected_valleys,
        )
    if len(valleys) == 0:
        warnings.warn("Не найдено впадин; равномерная сетка по ACF", stacklevel=2)
        valleys = np.arange(0, n, int(round(period_samples)), dtype=int)

    boundaries = np.array([_parabolic_refine(smoothed, int(v)) for v in valleys], dtype=float)

    # Уточняем период по медиане расстояний между впадинами
    if len(boundaries) >= 2:
        spacings = np.diff(boundaries)
        period_samples = float(np.median(spacings))
        period_seconds = period_samples * dt_seconds

    if refine_by_xcorr and len(boundaries) >= 3:
        boundaries = _refine_boundaries_xcorr(smoothed, boundaries, period_samples)

    # Границы только по впадинам — без искусственных 0/end (обрезанные сегменты отфильтруются)
    n_cycles = max(0, len(boundaries) - 1)
    full_cycle_indices = _identify_full_cycles(
        boundaries, period_samples, current, i_range, prominence_fraction,
    )

    idx = _select_cycle_index(cycle_index, n_cycles, full_cycle_indices)
    start = int(np.floor(boundaries[idx]))
    end = int(np.floor(boundaries[idx + 1])) if idx + 1 < len(boundaries) else n
    if end <= start:
        end = min(start + int(round(period_samples)), n)

    logger.info(
        "Сегментация: %d сегментов, %d полных (1 пик/сегмент), период=%.1f отсч. (%.2f SPM), "
        "рабочий цикл #%d [%d:%d]",
        n_cycles,
        len(full_cycle_indices),
        period_samples,
        60.0 / period_seconds if period_seconds else 0,
        idx,
        start,
        end,
    )

    return SegmentationResult(
        boundaries=boundaries,
        period_samples=period_samples,
        period_seconds=period_seconds,
        smoothed=smoothed,
        cycle_index=idx,
        cycle_start=start,
        cycle_end=end,
        cycle_current=current[start:end],
        full_cycle_indices=full_cycle_indices,
    )


def segment_all_cycles(
    current: np.ndarray,
    seg: SegmentationResult,
    prominence_fraction: float = 0.3,
) -> list[CycleSlice]:
    """
    Режим cycle_index="all" (Раздел В.3, п.7): все полные циклы.
    Возвращает список CycleSlice только для циклов с ровно одним токовым пиком.
    """
    current = np.asarray(current, dtype=float)
    i_range = current.max() - current.min()
    prominence = prominence_fraction * i_range

    indices = seg.full_cycle_indices
    if not indices:
        indices = list(range(max(0, len(seg.boundaries) - 1)))

    cycles: list[CycleSlice] = []
    for seq, i in enumerate(indices):
        start = int(np.floor(seg.boundaries[i]))
        end = int(np.floor(seg.boundaries[i + 1]))
        seg_data = current[start:end]
        cycles.append(
            CycleSlice(
                index=seq,
                start=start,
                end=end,
                current=seg_data,
                length_samples=float(end - start),
                n_current_peaks=_count_current_peaks(seg_data, i_range, prominence_fraction),
            )
        )
    return cycles


def plot_segmentation(
    current: np.ndarray,
    result: SegmentationResult,
    output_path: str | Path,
    dt_seconds: Optional[float] = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(current)
    x = np.arange(n) * dt_seconds if dt_seconds else np.arange(n)
    xlabel = "Время, с" if dt_seconds else "Отсчёт"

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, current, color="black", linewidth=0.8, label="I(t)")

    n_cycles = max(0, len(result.boundaries) - 1)
    colors = ["#aec7e8", "#ffbb78"]
    labeled_full = False
    for i in range(n_cycles):
        b0 = int(np.floor(result.boundaries[i]))
        b1 = int(np.floor(result.boundaries[i + 1])) if i + 1 < len(result.boundaries) else n - 1
        is_full = i in result.full_cycle_indices
        alpha = 0.35 if is_full else 0.12
        lbl = "Полный цикл (1 пик)" if is_full and not labeled_full else None
        if lbl:
            labeled_full = True
        ax.axvspan(x[b0], x[min(b1, n - 1)], alpha=alpha, color=colors[i % 2], label=lbl)

    for b in result.boundaries:
        ax.axvline(x[int(np.floor(b))], color="red", linestyle="--", linewidth=0.9, alpha=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Ток, А")
    ax.set_title(
        f"Сегментация: {len(result.full_cycle_indices)} полных / {n_cycles} сегментов, "
        f"период ≈ {result.period_samples:.1f} отсч."
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
