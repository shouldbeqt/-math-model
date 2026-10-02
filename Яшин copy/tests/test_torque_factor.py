"""Тесты TF-метода (Torque Factor) для инверсии F = M_кр / TF."""

import numpy as np

from src.yashin_inverse import compute_rod_load, compute_rod_load_tf


def test_tf_formula_m_crank_over_tf():
    omega = 2.0 * np.pi / 10.0
    v = np.array([0.5, 1.0, 0.0])
    m = np.array([1000.0, 2000.0, 500.0])
    res = compute_rod_load_tf(m, v, omega, tf_min=1e-6, interpolate=False)
    tf = v / omega
    expected = m / np.where(np.abs(tf) >= 1e-6, tf, np.nan)
    np.testing.assert_allclose(res.p_tpsh[:2], expected[:2], rtol=1e-9)


def test_tf_masks_and_interpolates_dead_points():
    omega = 1.0
    v = np.array([0.0, 1e-8, 0.5, 1.0])
    m = np.array([1000.0, 1000.0, 2000.0, 3000.0])
    res = compute_rod_load_tf(
        m, v, omega, v_zero_threshold_fraction=0.01, tf_min=1e-6, interpolate=True,
    )
    assert res.v_near_zero[0]
    assert np.isfinite(res.p_tpsh[0])  # интерполяция
    assert not res.v_near_zero[-1]


def test_tf_matches_power_balance_net_moment():
    """net_power + tf_scale=coeff эквивалентен (2.32) вне мёртвых точек."""
    omega = 2.0 * np.pi / 10.0
    v = np.linspace(0.05, 2.0, 50)
    p = 12000 + 3000 * np.sin(np.linspace(0, 2 * np.pi, 50))
    coeff = 1.5 / (2.5 * 0.88 * 0.95)
    p_delta = np.zeros_like(v)
    p_r = np.zeros_like(v)
    m_net = (p - 1000 + p_delta + p_r) / omega

    tf_res = compute_rod_load_tf(
        m_net, v, omega, tf_scale=coeff, tf_min=1e-9, interpolate=False,
    )
    pb_res = compute_rod_load(
        p, p_delta, p_r, v,
        {"l1": 1.5, "l2": 2.5},
        {"eta_ed": 0.88, "eta_per": 0.95, "P_xx_w": 1000},
    )
    valid = ~tf_res.v_near_zero & ~pb_res.v_near_zero
    np.testing.assert_allclose(tf_res.p_tpsh[valid], pb_res.p_tpsh[valid], rtol=1e-9)
