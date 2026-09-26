from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from main_detect_residual_baseline import find_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MLE-calibrated Weibull BCAD over a window-size grid."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--window-start", type=int, default=72)
    parser.add_argument("--window-stop", type=int, default=720)
    parser.add_argument("--window-step", type=int, default=36)
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--horizon-days", type=int, default=7)
    return parser.parse_args()


def format_setting(window_size: int, quantile: float) -> str:
    return f"weibullmle_calib2021_2022_w{window_size}_q{int(quantile * 1000):03d}_c6"


def result_row(window_size: int, performance: dict) -> dict:
    event = performance["event_level"]
    episode = performance["alarm_episode_level"]
    recall = float(event["detection_rate"])
    precision = (
        float(episode["true_alarm_episodes"] / episode["operational_alarm_opportunities"])
        if episode["operational_alarm_opportunities"]
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "window_size": window_size,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "miss_rate": float(event["miss_rate"]),
        "operational_false_alarm_rate": float(episode["operational_false_alarm_rate"]),
        "detected_events": int(event["detected_events"]),
        "total_events": int(event["total_events"]),
        "true_alarm_episodes": int(episode["true_alarm_episodes"]),
        "operational_false_alarm_episodes": int(episode["operational_false_alarm_episodes"]),
    }


def plot_grid(grid: pd.DataFrame, quantile: float, output_path: Path) -> None:
    best = grid.loc[grid["f1"].idxmax()]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
        constrained_layout=True,
    )
    axes[0].plot(grid["window_size"], grid["f1"], marker="o", label="F1 score")
    axes[0].plot(grid["window_size"], grid["precision"], marker="o", label="Precision")
    axes[0].plot(grid["window_size"], grid["recall"], marker="o", label="Recall")
    axes[0].scatter([best["window_size"]], [best["f1"]], s=110, color="#D62728", zorder=5, label=f"Best F1: W={int(best['window_size'])}, F1={best['f1']:.3f}")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("score")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="lower left")

    axes[1].plot(grid["window_size"], grid["operational_false_alarm_rate"], marker="o", color="#D62728", label="Operational FA rate")
    axes[1].plot(grid["window_size"], grid["miss_rate"], marker="o", color="#9467BD", label="Miss rate")
    axes[1].set_xlabel("window size W (10-minute points)")
    axes[1].set_ylabel("rate")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")
    fig.suptitle(f"BCAD Weibull MLE window-size trend | q={quantile:.2f}, c=6")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.window_step < 1 or args.window_start > args.window_stop:
        raise ValueError("Invalid window grid")
    windows = list(range(args.window_start, args.window_stop + 1, args.window_step))
    project_root = PROJECT_ROOT
    result_dir, _, _ = find_experiment(project_root, args.run_id)
    rows = []

    for index, window_size in enumerate(windows, start=1):
        setting = format_setting(window_size, args.quantile)
        performance_path = result_dir / f"bcad_performance_{setting}.json"
        if performance_path.exists():
            print(f"[{index}/{len(windows)}] Existing result reused: W={window_size}", flush=True)
        else:
            print(f"[{index}/{len(windows)}] Running W={window_size}", flush=True)
            command = [
                sys.executable,
                str(project_root / "main_detect_bcad.py"),
                "--run-id",
                args.run_id,
                "--distribution",
                "weibull_h_weibull_a_mle_calibration",
                "--residual-mode",
                "absolute",
                "--window-size",
                str(window_size),
                "--quantile",
                str(args.quantile),
                "--horizon-days",
                str(args.horizon_days),
                "--skip-detections-file",
            ]
            subprocess.run(command, cwd=project_root, check=True)
        performance = json.loads(performance_path.read_text(encoding="utf-8"))
        rows.append(result_row(window_size, performance))

    grid = pd.DataFrame(rows).sort_values("window_size").reset_index(drop=True)
    label = f"w{args.window_start}to{args.window_stop}_step{args.window_step}_q{int(args.quantile * 1000):03d}_c6"
    grid_path = result_dir / f"bcad_grid_weibullmle_calib2021_2022_{label}.csv"
    figure_path = project_root / "results" / "kelmarsh" / "figures" / "bcad_window_trends" / f"bcad_window_trend_weibullmle_calib2021_2022_{label}.png"
    grid.to_csv(grid_path, index=False, encoding="utf-8-sig")
    plot_grid(grid, args.quantile, figure_path)

    best = grid.loc[grid["f1"].idxmax()]
    best_window = int(best["window_size"])
    best_setting = format_setting(best_window, args.quantile)
    best_detection_path = result_dir / f"bcad_detections_{best_setting}.csv"
    if not best_detection_path.exists():
        print(f"Saving point-level detections for best F1 setting W={best_window}", flush=True)
        command = [
            sys.executable,
            str(project_root / "main_detect_bcad.py"),
            "--run-id",
            args.run_id,
            "--distribution",
            "weibull_h_weibull_a_mle_calibration",
            "--residual-mode",
            "absolute",
            "--window-size",
            str(best_window),
            "--quantile",
            str(args.quantile),
            "--horizon-days",
            str(args.horizon_days),
        ]
        subprocess.run(command, cwd=project_root, check=True)

    print(grid.to_string(index=False))
    print("Grid saved to:", grid_path)
    print("Trend figure saved to:", figure_path)
    print("Best F1 setting:", best_setting, f"F1={best['f1']:.6f}")


if __name__ == "__main__":
    main()
