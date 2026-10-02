"""Тесты разбиения ветвей динамограммы для визуализации."""

from pathlib import Path

import numpy as np

from src.io_dynamogram import read_reference_dynamogram
from src.validation import (
    split_calc_strokes_for_plot,
    split_reference_branches,
    strip_dead_points,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DYNAMOGRAM_PATH = DATA_DIR / "12.07.2026 1137 скв 16427 динамограмма.txt"


def test_reference_branches_gap_at_vmt():
    """Эталон: подъём и спуск не пересекаются по индексам (разрыв у ВМТ)."""
    ref = read_reference_dynamogram(DYNAMOGRAM_PATH)
    s, f = ref.s_m, ref.f_force
    s_up, f_up, s_dn, f_dn = split_reference_branches(s, f)

    ds = np.diff(s, prepend=s[0])
    dn_start = int(np.flatnonzero(ds < 0)[0])
    up_pos = np.flatnonzero(ds > 0)
    up_end = int(up_pos[-1]) + 1

    assert len(s_up) == up_end
    assert len(s_dn) == len(s) - dn_start
    assert s_up[-1] == s[up_end - 1]
    assert s_dn[0] == s[dn_start]
    assert dn_start > up_end - 1
    assert s_up.max() >= 3.4


def test_calc_strokes_skip_dead_chord():
    """Расчёт: мёртвые точки исключены, без интерполяции F."""
    n = 100
    t = np.linspace(0, 1, n, endpoint=False)
    s = 1.75 * (1 - np.cos(2 * np.pi * t))
    v = np.gradient(s, t[1] - t[0])
    f = 2000 + 500 * np.sin(2 * np.pi * t)
    dead = np.abs(v) < 0.06 * np.max(np.abs(v))
    f = f.copy()
    f[dead] = 1e6

    s_up, f_up, s_dn, f_dn = split_calc_strokes_for_plot(s, f, v, dead)
    s_all, f_all = strip_dead_points(s, f, dead)

    assert len(s_up) + len(s_dn) == len(s_all)
    assert np.all(f_up < 5000)
    assert np.all(f_dn < 5000)
    assert dead.sum() + len(s_all) == n
    assert len(s_up) > 0
    assert len(s_dn) > 0
