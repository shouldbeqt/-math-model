"""Unit-тесты чтения токограммы."""

from pathlib import Path

import numpy as np
import pytest

from src.io_tokogram import read_tokogram

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TOKOGRAM_PATH = DATA_DIR / "скв_16427_токограмма_за_период_с_12_07_2026_по_12_07_2026.xlsx"


def test_read_real_tokogram():
    data = read_tokogram(TOKOGRAM_PATH, dt_seconds=0.2)
    # 512 точек после отбрасывания хвостового служебного 0 (в xlsx 513 ячеек, ТЗ ~513)
    assert len(data.current_a) == 512
    assert data.current_a.dtype == np.float64
    assert data.current_a.min() > 20
    assert data.current_a.max() < 40
    assert data.timestamp is not None
    assert data.dt_source == "config"


def test_read_tokogram_without_dt_raises_warning():
    with pytest.warns(UserWarning, match="Дискретность"):
        data = read_tokogram(TOKOGRAM_PATH)
    assert data.dt_seconds is None
    assert data.dt_source == "missing"
