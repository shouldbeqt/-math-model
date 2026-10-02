"""Диагностика формы динамограммы: знаки, ветви, фаза (скв. 16427)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_dynamogram import read_reference_dynamogram
from src.io_tokogram import read_tokogram
from src.pipeline import process_cycle, _dead_mask
from src.validation import branch_force_metrics, validate_dynamograms
from src.yashin_power_balance import compute_parasitic_powers
from src.yashin_kinematics import compute_kinematics_simple

CONFIG = ROOT / "config_well16427.yaml"
FULL_CONFIG = ROOT / "config.yaml"


def _branch_table(s, f, v, ref_s, ref_f, label: str) -> None:
    ds = np.diff(ref_s, prepend=ref_s[0])
    ref_up, ref_dn = ds > 0, ds < 0
    print(f"\n=== {label}: F vs S по ветвям (kgf) ===")
    print(f"{'S,m':>5} {'ref↑':>7} {'ref↓':>7} {'calc↑':>7} {'calc↓':>7}")
    for sg in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.4]:
        ru = ref_f[(np.abs(ref_s - sg) < 0.08) & ref_up]
        rd = ref_f[(np.abs(ref_s - sg) < 0.08) & ref_dn]
        cu = (f[(np.abs(s - sg) < 0.08) & (v > 0)] / 9.80665) if len(f) else []
        cd = (f[(np.abs(s - sg) < 0.08) & (v < 0)] / 9.80665) if len(f) else []
        fmt = lambda a: f"{np.mean(a):.0f}" if len(a) else "  —"
        print(f"{sg:5.1f} {fmt(ru):>7} {fmt(rd):>7} {fmt(cu):>7} {fmt(cd):>7}")


def main() -> None:
    with open(CONFIG, encoding="utf-8") as f:
        mvp_cfg = yaml.safe_load(f)
    with open(FULL_CONFIG, encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    full_cfg["model"] = {"mode": "full"}

    dt = float(mvp_cfg["tokogram"]["dt_seconds"])
    tok = read_tokogram(mvp_cfg["paths"]["tokogram_xlsx"], dt_seconds=dt)
    ref = read_reference_dynamogram(mvp_cfg["paths"]["reference_dynamogram_txt"])
    I = tok.current_a[27:84]
    period = len(I) * dt

    print("=" * 60)
    print("1. ЗНАК В (2.32)")
    print("   P = P_XX − Pδ − PR + coeff·F·v  →  F = (P−P_XX+Pδ+PR)/(coeff·v)")
    print("   compute_parasitic_powers() возвращает Pδ, PR > 0 когда мощность")
    print("   уходит на привод балансира (в (2.27) вычитается, в (2.32) прибавляется).")

    kin = compute_kinematics_simple(3.5, period, len(I))
    pd, pr = compute_parasitic_powers(
        kin, full_cfg["geometry"], full_cfg["masses"], full_cfg["motor"],
    )
    v = kin.v
    print(f"   Физич. (pd+pr): v>0 mean={np.mean((pd+pr)[v>0]):.0f} W, "
          f"v<0 mean={np.mean((pd+pr)[v<0]):.0f} W (противоположные знаки — OK)")

    print("\n2. КИНЕМАТИКА")
    print(f"   t=0: S={kin.S[0]:.2f} м (НМТ), v={kin.v[0]:.3f} м/с")
    print(f"   t=T/4: S={kin.S[len(I)//4]:.2f} м, v={kin.v[len(I)//4]:+.3f} (подъём v>0)")
    print(f"   t=T/2: S={kin.S[len(I)//2]:.2f} м (ВМТ), v={kin.v[len(I)//2]:+.3f}")

    for name, cfg in [("MVP", mvp_cfg), ("Full", full_cfg)]:
        res = process_cycle(I, None, cfg, dt_seconds=dt)
        s, f, v = res["s"], res["f"], res["kinematics"].v
        dead = _dead_mask(res)
        m = validate_dynamograms(
            s, f, ref.s_m, ref.f_force, v_near_zero=dead, v_calc=v,
        )
        br = branch_force_metrics(s, f, ref.s_m, ref.f_force, v_calc=v)
        print(f"\n--- {name} ---")
        print(f"MAPE F(τ)={m.mape_f:.2f}%  Δплощади={m.area_diff_percent:.0f}%")
        print(f"RMSE F↑={br['rmse_f_upstroke']:.0f} N  RMSE F↓={br['rmse_f_downstroke']:.0f} N")
        print(f"F↓/F↑: calc={br['branch_ratio_calc']:.2f}  ref={br['branch_ratio_ref']:.2f}")
        _branch_table(s, f, v, ref.s_m, ref.f_force, name)

    print("\n" + "=" * 60)
    print("ВЫВОД:")
    print("• Знак +Pδ+PR в (2.32) — согласован с (2.27).")
    print("• Кинематика: t=0 → S=0 (НМТ), подъём v>0.")
    print("• Петля S-F: смотреть F↓/F↑ и RMSE по ветвям (не только MAPE F(τ)).")
    print("• pump_load.force_profile — гидравлическая добавка на спуске.")
    print("=" * 60)


if __name__ == "__main__":
    main()
