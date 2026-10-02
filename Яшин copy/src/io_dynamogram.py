"""Чтение эталонной динамограммы из .txt (Раздел Д.2 ТЗ)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DynamogramData:
    """Эталонная динамограмма."""

    s_m: np.ndarray
    f_force: np.ndarray
    timestamp: Optional[datetime]


def _parse_timestamp(line: str) -> Optional[datetime]:
    text = line.strip()
    for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def read_reference_dynamogram(path: str | Path) -> DynamogramData:
    """
    Читает .txt: первая строка — метка времени, далее S\\tF.
    Хвостовая строка NaN\\tNaN отбрасывается.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()

    if not lines:
        raise ValueError(f"Пустой файл динамограммы: {path}")

    timestamp = _parse_timestamp(lines[0])
    s_values: list[float] = []
    f_values: list[float] = []

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            s = float(parts[0])
            f = float(parts[1])
        except ValueError:
            continue
        if np.isnan(s) and np.isnan(f):
            continue
        s_values.append(s)
        f_values.append(f)

    s_m = np.array(s_values, dtype=float)
    f_force = np.array(f_values, dtype=float)

    if len(s_m) == 0:
        raise ValueError(f"Не найдено пар S/F в {path}")

    logger.info(
        "Эталонная динамограмма: %d точек, S=[%.2f, %.2f] м, F=[%.1f, %.1f], timestamp=%s",
        len(s_m),
        s_m.min(),
        s_m.max(),
        f_force.min(),
        f_force.max(),
        timestamp,
    )

    return DynamogramData(s_m=s_m, f_force=f_force, timestamp=timestamp)
