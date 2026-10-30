"""Rescore saved BCAD c=6 alarm runs as c=72 under the revised evaluation rule."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from main_detect_residual_baseline import build_target_event_summary
from scripts.analysis.prepare_thesis_seven_settings_split6h import evaluate


RUN = "all6_multi3_seq12_h64_l1_bs64_lr1e-03_wd0_do0_e60_p8_seed42"
SOURCE = ROOT / "results" / "kelmarsh" / "experiments" / RUN / "no_warning_bcad"
OUTPUT = ROOT / "results" / "kelmarsh" / "thesis_revision_split6h" / "bcad_window_scan"
WINDOWS = range(36, 1081, 36)
PATTERNS = {
    "folded": "foldedmle_samecenterabs_w{w}_q990_vm5_c6",
    "gaussian": "abslog_w{w}_q990_vm5_c6",
    "maximum": "weibullmle_foldedcalib2021_2022_w{w}_q990_c6",
    "joint": "weibulljointmle_foldedcalib2021_2022_w{w}_q990_c6",
}


def rescore_one(mode: str, window: int, events: pd.DataFrame, evaluation_cache: dict) -> dict:
    setting = PATTERNS[mode].format(w=window)
    source = SOURCE / f"bcad_alarm_episodes_{setting}.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    episodes = pd.read_csv(source, parse_dates=["start_time", "end_time"])
    episodes = episodes.loc[episodes["alarm_points"].ge(67)].copy()
    episodes["start_time"] += pd.Timedelta(minutes=660)
    episodes["alarm_points"] -= 66
    scored, metrics = evaluate(episodes, events, 72, evaluation_cache=evaluation_cache)
    return {"mode": mode, "window_size": window, "setting": setting.replace("_c6", "_c72"),
            "source_c6_episodes": str(source.relative_to(ROOT)), **metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=[*PATTERNS, "all"], default="all")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    events = build_target_event_summary(ROOT, horizon_days=7, include_long_warnings=False)
    if len(events) != 459:
        raise ValueError(f"Expected 459 target events, found {len(events)}")
    for mode in PATTERNS if args.mode == "all" else [args.mode]:
        output = OUTPUT / f"{mode}_w36-1080_c72.csv"
        rows = (pd.read_csv(output, encoding="utf-8-sig").to_dict("records")
                if output.exists() else [])
        completed = {int(row["window_size"]) for row in rows}
        evaluation_cache = {"target_events": events}
        for index, window in enumerate(WINDOWS, start=1):
            if window in completed:
                print(f"{mode} [{index}/30] Reusing saved W={window}", flush=True)
                continue
            result = rescore_one(mode, window, events, evaluation_cache)
            rows.append(result)
            pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
            print(f"{mode} [{index}/30] W={window}: "
                  f"{result['detected_events']}/459 events, "
                  f"FA={result['operational_false_alarm_episodes']}, "
                  f"F1={result['f1']:.4f}", flush=True)
        best = max(rows, key=lambda row: row["f1"])
        print(f"{mode} best: W={best['window_size']}, F1={best['f1']:.4f}; saved {output}", flush=True)


if __name__ == "__main__":
    main()
