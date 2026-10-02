"""Unit-тесты чтения эталонной динамограммы."""

from pathlib import Path

import numpy as np

from src.io_dynamogram import read_reference_dynamogram

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DYNAMOGRAM_PATH = DATA_DIR / "12.07.2026 1137 скв 16427 динамограмма.txt"


def test_read_real_dynamogram():
    data = read_reference_dynamogram(DYNAMOGRAM_PATH)
    # Фактически 200 точек данных + хвост NaN\tNaN (ТЗ указывает ~201)
    assert len(data.s_m) == 200
    assert len(data.f_force) == 200
    assert data.s_m.dtype == np.float64
    assert data.f_force.dtype == np.float64
    assert data.s_m.max() <= 3.51
    assert data.f_force.min() > 1000
    assert data.f_force.max() < 4000
    assert data.timestamp is not None
    assert not (np.isnan(data.s_m).any() or np.isnan(data.f_force).any())
