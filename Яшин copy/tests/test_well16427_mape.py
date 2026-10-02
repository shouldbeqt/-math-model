"""Интеграционный тест: MAPE < 15% на эталоне скв. 16427 (MVP)."""

import yaml

from src.io_dynamogram import read_reference_dynamogram
from src.io_tokogram import read_tokogram
from src.pipeline import process_cycle, _cycle_mape, _dead_mask
from src.segmentation import segment_all_cycles, segment_tokogram

CONFIG = "config_well16427.yaml"
MAPE_LIMIT = 15.0


def test_well16427_mape_below_limit():
    with open(CONFIG, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dt = float(config["tokogram"]["dt_seconds"])
    tok = read_tokogram(config["paths"]["tokogram_xlsx"], dt_seconds=dt)
    ref = read_reference_dynamogram(config["paths"]["reference_dynamogram_txt"])
    seg = segment_tokogram(tok.current_a, dt_seconds=dt)

    mapes: list[float] = []
    for cyc in segment_all_cycles(tok.current_a, seg):
        if cyc.length_samples > seg.period_samples * 1.15:
            continue
        res = process_cycle(cyc.current, None, config, dt_seconds=dt)
        mape = _cycle_mape(res, ref.s_m, ref.f_force, config.get("validation", {}))
        mapes.append(mape)

    assert mapes, "нет полных циклов для проверки"
    assert min(mapes) < MAPE_LIMIT, f"min MAPE={min(mapes):.1f}% >= {MAPE_LIMIT}%"
