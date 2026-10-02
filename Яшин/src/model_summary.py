"""Сводка расчётной модели и констант (для output/model_summary.txt)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_model_summary(config: dict, metrics: Any | None = None) -> str:
    """Текстовое описание формул и параметров скважины."""
    kin = config.get("kinematics", {})
    motor = config.get("motor", {})
    inv = config.get("inverse", {})
    par = config.get("parasitic", {})
    pump = config.get("pump_load", {})
    dyn = config.get("dynamics", {})
    rw = config.get("rod_wave", {})
    fr = config.get("friction", {})
    tok = config.get("tokogram", {})
    mode = config.get("model", {}).get("mode", "mvp")
    wave_on = bool(config.get("wave_model", False))

    lines = [
        "=== Модель расчёта динамограммы (скв. 16427) ===",
        "",
        f"Режим: {mode}",
        "",
        "1. Сегментация токограммы → период T, цикл I(t)  [без изменений]",
        "",
        "2. Кинематика:",
        f"   mode = {kin.get('mode', 'harmonic')}",
    ]
    if kin.get("mode") == "api_geometry":
        lines += [
            "   S(θ) = PR(θ)·S₀, PR = (ψ_B−ψ(θ))/(ψ_B−ψ_T)  [API 11E / Svinos]",
            "   v = dS/dt,  a = d²S/dt²,  ω = 2π/T",
            f"   Станок: {kin.get('api_geometry', {}).get('unit', 'C-640 (типовая)')}",
            f"   S₀ = {kin.get('S0_m', '?')} м (масштаб / калибровка)",
        ]
    else:
        lines += [
            f"   S₀ = {kin.get('S0_m', '?')} м",
            "   S(t) = S₀/2 · (1 − cos(2πt/T))",
            "   v(t) = dS/dt",
        ]
    lines += [
        "3. Мощность двигателя (полином):",
        f"   P(I) = P₀ + a·I + b·I² + c·I³",
        f"   P₀ = {motor.get('P0_w', '?')} Вт",
        f"   a  = {motor.get('P_a_w_per_a', '?')} Вт/А",
        f"   b  = {motor.get('P_b_w_per_a2', '?')} Вт/А²",
        f"   c  = {motor.get('P_c_w_per_a3', 0)} Вт/А³",
        f"   Сдвиг I(t): {tok.get('i_phase_shift_samples', 0)} отсч.",
        "",
        "4. Паразитные мощности P_par(t):",
        f"   mode = {par.get('mode', 'constant')}",
    ]
    if par.get("mode") in ("harmonic", "harmonic2"):
        lines += [
            f"   P₀={par.get('P0', 0)} Вт, P₁={par.get('P1', 0)} Вт, φ={par.get('phi', 0)} рад",
        ]
        if par.get("mode") == "harmonic2":
            lines.append(
                f"   P₂={par.get('P2', 0)} Вт, φ₂={par.get('phi2', 0)} рад"
            )
    lines += [
        "",
        "5. Баланс мощности (2.32) → динамическая сила:",
        "   P_net = P(I) − P₀ + P_par",
        f"   k_tf = {inv.get('k_tf', inv.get('k_v', '?'))}  (Torque Factor)",
        "   M_net = P_net / ω_кр,  F_dyn = M_net / (k_tf · v / ω_кр)",
        "",
        "6. Гидравлическая добавка (насос, спуск):",
        f"   mode = {pump.get('mode', 'none')}",
    ]
    if pump.get("mode") == "force_profile":
        lines += [
            f"   F_add = (F₀ + F₁·S/S₀) при v<0  [N]",
            f"   F₀ = {pump.get('F0_down_n', 0)} N",
            f"   F₁ = {pump.get('F1_down_n', 0)} N",
            f"   F_up = {pump.get('F_up_n', 0)} N (вычитается при v>0)",
        ]
    lines += [
        "",
        "7. Инерция штока:",
        f"   m_eq = {dyn.get('m_eq', config.get('m_eq', 0))} кг",
        "   F_inertia = m_eq · a",
        "",
        "8. Волна по колонне (Gibbs):",
        f"   wave_model = {wave_on}",
    ]
    if wave_on:
        lines += [
            "   u_tt + ν u_t = c² u_xx + g",
            f"   L={rw.get('L', 1500)} м, E={rw.get('E', 2.1e11)} Па, ρ={rw.get('rho', 7850)}",
            f"   D={rw.get('diameter', 0.022)} м, ν={rw.get('nu', 0.5)}, N={rw.get('n_nodes', 51)}",
            "   BC: u(0)=S(t),  EA u_x(L)=F_head(t) → F_surface = EA ε(0)",
        ]
    el = config.get("elastic", {})
    lines += [
        "",
        "9. Упругость (Гук):",
        f"   elastic.enabled = {el.get('enabled', False)}",
        "   F_elastic = E·A·(ΔL/L),  A=πD²/4,  ΔL=S−min(S)",
        f"   E={el.get('E', 2.1e11)}, L={el.get('rod_length', '?')} м, D={el.get('rod_diameter', '?')} м",
        "",
        f"10. Трение: mode={fr.get('mode', 'none')}",
        "",
        "11. Итог: F = F_surface + F_elastic + F_fluid + F_inertia + F_friction",
        "",
        "12. Визуализация dynamogram_overlay.png:",
        "   • Эталон: подъём/спуск с разрывом у ВМТ (dS=0 в данных)",
        "   • Расчёт: |v|≈0 пропущены; разрывы — хорда между корректными точками",
        "",
    ]
    if metrics is not None:
        lines += [
            "=== Метрики ===",
            f"MAPE F(τ) = {getattr(metrics, 'mape_f', '?')}%",
            f"RMSE F↑ = {getattr(metrics, 'rmse_f_upstroke', '?')} N",
            f"RMSE F↓ = {getattr(metrics, 'rmse_f_downstroke', '?')} N",
            f"F↓/F↑ = {getattr(metrics, 'branch_ratio_calc', '?')} (эталон {getattr(metrics, 'branch_ratio_ref', '?')})",
            f"Δ площади = {getattr(metrics, 'area_diff_percent', '?')}%",
        ]
    return "\n".join(lines)


def write_model_summary(
    config: dict,
    output_dir: str | Path,
    metrics: Any | None = None,
) -> Path:
    path = Path(output_dir) / "model_summary.txt"
    path.write_text(build_model_summary(config, metrics), encoding="utf-8")
    return path
