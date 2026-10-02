"""Чтение токограммы из .xlsx (Раздел Д.1 ТЗ)."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TokogramData:
    """Результат чтения токограммы."""

    current_a: np.ndarray
    timestamp: Optional[datetime]
    dt_seconds: Optional[float]
    dt_source: str  # "file" | "config" | "missing"


def _parse_timestamp(value) -> Optional[datetime]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    for fmt in ("%d.%m.%Y %H:%M", "%d.%m.%Y %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def read_tokogram(
    path: str | Path,
    dt_seconds: Optional[float] = None,
) -> TokogramData:
    """
    Читает .xlsx строго по структуре Раздела Д.1 ТЗ:
    3 колонки без построчного заголовка; timestamp только в одной ячейке;
    дискретность — из колонки C, иначе из конфига.
    """
    path = Path(path)
    df = pd.read_excel(path, header=None, sheet_name=0)

    if df.shape[1] < 2:
        raise ValueError(f"Ожидается минимум 2 колонки в {path}, получено {df.shape[1]}")

    # Колонка A: timestamp в первой непустой ячейке данных (строка 1, индекс 1)
    timestamp = None
    for val in df.iloc[:, 0]:
        ts = _parse_timestamp(val)
        if ts is not None:
            timestamp = ts
            break

    # Колонка B: ток до первого NaN
    current_raw = df.iloc[1:, 1]
    current_numeric = pd.to_numeric(current_raw, errors="coerce")
    valid_mask = current_numeric.notna()
    current_a = current_numeric[valid_mask].to_numpy(dtype=float)

    # Отбрасываем хвостовые служебные нули (в файле скв. 16427 последний отсчёт = 0)
    while len(current_a) > 0 and current_a[-1] == 0.0:
        current_a = current_a[:-1]

    if len(current_a) == 0:
        raise ValueError(f"Не найдено числовых значений тока в колонке B файла {path}")

    # Колонка C: дискретность (если заполнена)
    dt_from_file: Optional[float] = None
    if df.shape[1] >= 3:
        disc_raw = df.iloc[1:, 2]
        disc_numeric = pd.to_numeric(disc_raw, errors="coerce")
        disc_valid = disc_numeric.dropna()
        if len(disc_valid) > 0:
            unique_vals = disc_valid.unique()
            dt_from_file = float(unique_vals[0])
            if len(unique_vals) > 1:
                warnings.warn(
                    f"Колонка 'Дискретность' содержит несколько значений {unique_vals}; "
                    f"используется первое: {dt_from_file} с",
                    stacklevel=2,
                )

    if dt_from_file is not None:
        dt = dt_from_file
        dt_source = "file"
    elif dt_seconds is not None:
        dt = float(dt_seconds)
        dt_source = "config"
        warnings.warn(
            f"Колонка 'Дискретность' в {path.name} пуста; "
            f"используется dt_seconds={dt} из конфига (ДОПУЩЕНИЕ, см. Раздел Ж ТЗ, п.1)",
            stacklevel=2,
        )
    else:
        dt = None
        dt_source = "missing"
        warnings.warn(
            f"Дискретность токограммы не найдена ни в файле, ни в конфиге. "
            f"Задайте tokogram.dt_seconds (Раздел Ж ТЗ, п.1).",
            stacklevel=2,
        )

    logger.info(
        "Токограмма: %d точек, I=[%.1f, %.1f] А, timestamp=%s, dt=%s (%s)",
        len(current_a),
        current_a.min(),
        current_a.max(),
        timestamp,
        dt,
        dt_source,
    )

    return TokogramData(
        current_a=current_a,
        timestamp=timestamp,
        dt_seconds=dt,
        dt_source=dt_source,
    )
