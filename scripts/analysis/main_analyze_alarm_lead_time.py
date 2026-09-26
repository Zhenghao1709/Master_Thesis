from __future__ import annotations

import argparse
import json
import re
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

from main_detect_residual_baseline import build_target_event_summary, find_experiment


DEFAULT_RUN_ID = "all6_multi3_seq12_h64_l1_bs64_lr1e-03_wd0_do0_e60_p8_seed42"
METHODS = ("scc", "cusum", "bcad")
FREQ_MINUTES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze how early SCC, CUSUM, or BCAD alarms occur before target events."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--setting", required=True, help="Setting suffix between *_detections_ and .csv.")
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--freq-minutes", type=int, default=FREQ_MINUTES)
    return parser.parse_args()


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def load_events(project_root: Path, horizon_days: int) -> pd.DataFrame:
    events = build_target_event_summary(project_root, horizon_days=horizon_days).copy()
    for col in ["horizon_start", "event_start", "event_end"]:
        events[col] = pd.to_datetime(events[col], errors="coerce")
    events = events.dropna(subset=["horizon_start", "event_start"]).reset_index(drop=True)
    events["event_uid"] = events.index + 1
    return events


def analyze_episode_trigger_times(
    episodes: pd.DataFrame,
    events: pd.DataFrame,
    freq_minutes: int,
    trigger_time_col: str = "start_time",
) -> pd.DataFrame:
    episodes = episodes.copy()
    episodes["start_time"] = pd.to_datetime(episodes["start_time"], errors="coerce")
    episodes["end_time"] = pd.to_datetime(episodes["end_time"], errors="coerce")
    if trigger_time_col not in episodes.columns:
        raise ValueError(f"Missing episode trigger time column: {trigger_time_col}")
    episodes[trigger_time_col] = pd.to_datetime(episodes[trigger_time_col], errors="coerce")
    episodes = episodes.dropna(subset=["start_time", "end_time"])
    end_exclusive = episodes["end_time"] + pd.Timedelta(minutes=freq_minutes)
    episodes = episodes.assign(end_exclusive=end_exclusive)

    rows: list[dict] = []
    for _, event in events.iterrows():
        candidates = episodes.loc[
            episodes["turbine_id"].eq(event["turbine_id"])
            & (episodes["start_time"] < event["event_start"])
            & (episodes["end_exclusive"] >= event["horizon_start"])
            & (episodes[trigger_time_col] < event["event_start"])
        ].copy()
        if candidates.empty:
            continue

        candidates = candidates.sort_values(trigger_time_col)
        alarm = candidates.iloc[0]
        lead = event["event_start"] - alarm[trigger_time_col]
        rows.append(
            {
                "event_uid": event["event_uid"],
                "turbine_id": event["turbine_id"],
                "event_type": event.get("event_type"),
                "event_start": event["event_start"],
                "event_end": event.get("event_end"),
                "horizon_start": event["horizon_start"],
                "alarm_time": alarm[trigger_time_col],
                "alarm_target": alarm.get("target"),
                "episode_id": alarm.get("episode_id"),
                "episode_end_time": alarm["end_time"],
                "alarm_points": alarm.get("alarm_points"),
                "lead_hours": lead.total_seconds() / 3600,
                "lead_days": lead.total_seconds() / 86400,
                "trigger_definition": trigger_time_col,
            }
        )
    return pd.DataFrame(rows)


def analyze_alarm_point_times(detections: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    detections = detections.copy()
    detections["Date and time"] = pd.to_datetime(detections["Date and time"], errors="coerce")
    detections = detections.dropna(subset=["Date and time"])
    detections = detections.loc[bool_series(detections["is_alarm"])].copy()

    rows: list[dict] = []
    for _, event in events.iterrows():
        candidates = detections.loc[
            detections["turbine_id"].eq(event["turbine_id"])
            & (detections["Date and time"] >= event["horizon_start"])
            & (detections["Date and time"] < event["event_start"])
        ].copy()
        if candidates.empty:
            continue

        candidates = candidates.sort_values("Date and time")
        alarm = candidates.iloc[0]
        lead = event["event_start"] - alarm["Date and time"]
        rows.append(
            {
                "event_uid": event["event_uid"],
                "turbine_id": event["turbine_id"],
                "event_type": event.get("event_type"),
                "event_start": event["event_start"],
                "event_end": event.get("event_end"),
                "horizon_start": event["horizon_start"],
                "alarm_time": alarm["Date and time"],
                "alarm_target": alarm.get("target"),
                "episode_id": pd.NA,
                "episode_end_time": pd.NaT,
                "alarm_points": pd.NA,
                "lead_hours": lead.total_seconds() / 3600,
                "lead_days": lead.total_seconds() / 86400,
                "trigger_definition": "alarm_point",
            }
        )
    return pd.DataFrame(rows)


def make_summary(method: str, setting: str, lead_times: pd.DataFrame, total_events: int) -> dict:
    detected_events = int(len(lead_times))
    summary = {
        "method": method,
        "setting": setting,
        "total_events": int(total_events),
        "events_with_valid_ta_alarm": detected_events,
        "events_without_valid_ta_alarm": int(total_events - detected_events),
        "detection_rate_among_events": float(detected_events / total_events) if total_events else 0.0,
    }
    if not lead_times.empty:
        desc = lead_times["lead_days"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        for key, value in desc.items():
            summary[f"lead_days_{key}"] = float(value)
    return summary


def format_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def parse_signed_scc_k(setting: str) -> float | None:
    match = re.fullmatch(r"signed_meanstd_abslimit_k([0-9]+(?:p[0-9]+)?)_c6", setting)
    if not match:
        return None
    return float(match.group(1).replace("p", "."))


def parse_cusum_std_k_h(setting: str) -> tuple[float, float] | None:
    match = re.fullmatch(
        r"twosided_std_k([0-9]+(?:p[0-9]+)?)_h([0-9]+(?:p[0-9]+)?)_reset",
        setting,
    )
    if not match:
        return None
    k = float(match.group(1).replace("p", "."))
    h = float(match.group(2).replace("p", "."))
    return k, h


def ensure_scc_episode_file(
    project_root: Path,
    run_id: str,
    result_dir: Path,
    setting: str,
    horizon_days: int,
) -> Path:
    path = result_dir / f"scc_alarm_episodes_{setting}.csv"
    if path.exists():
        return path

    signed_k = parse_signed_scc_k(setting)
    if signed_k is None:
        raise FileNotFoundError(path)

    cmd = [
        sys.executable,
        str(project_root / "main_detect_scc.py"),
        "--run-id",
        run_id,
        "--k",
        str(signed_k),
        "--horizon-days",
        str(horizon_days),
    ]
    print("SCC signed episode file not found. Generating it first:")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=project_root, text=True, capture_output=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Failed to generate SCC signed setting: {setting}")
    if not path.exists():
        expected = result_dir / f"scc_alarm_episodes_signed_meanstd_abslimit_k{format_number(signed_k)}_c6.csv"
        raise FileNotFoundError(expected)
    return path


def ensure_cusum_detection_file(
    project_root: Path,
    run_id: str,
    result_dir: Path,
    setting: str,
    horizon_days: int,
) -> Path:
    path = result_dir / f"cusum_detections_{setting}.csv"
    if path.exists():
        return path

    parsed = parse_cusum_std_k_h(setting)
    if parsed is None:
        raise FileNotFoundError(path)
    k, h = parsed

    cmd = [
        sys.executable,
        str(project_root / "main_detect_cusum.py"),
        "--run-id",
        run_id,
        "--scale-type",
        "std",
        "--reference-value",
        str(k),
        "--decision-threshold",
        str(h),
        "--horizon-days",
        str(horizon_days),
    ]
    print("CUSUM buffer-6h detection file not found. Generating it first:")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=project_root, text=True, capture_output=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Failed to generate CUSUM setting: {setting}")
    if not path.exists():
        expected = result_dir / (
            f"cusum_detections_twosided_std_k{format_number(k)}_h{format_number(h)}_reset.csv"
        )
        raise FileNotFoundError(expected)
    return path


def ensure_cusum_episode_file(
    project_root: Path,
    run_id: str,
    result_dir: Path,
    setting: str,
    horizon_days: int,
) -> Path:
    path = result_dir / f"cusum_alarm_episodes_{setting}.csv"
    if path.exists():
        return path
    parsed = parse_cusum_std_k_h(setting)
    if parsed is None:
        raise FileNotFoundError(path)
    k, h = parsed
    subprocess.run(
        [
            sys.executable, str(project_root / "main_detect_cusum.py"),
            "--run-id", run_id,
            "--scale-type", "std",
            "--reference-value", str(k),
            "--decision-threshold", str(h),
            "--horizon-days", str(horizon_days),
        ],
        cwd=project_root, check=True,
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def plot_lead_time_distribution(
    lead_times: pd.DataFrame,
    summary: dict,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("white")

    ax = axes[0]
    if lead_times.empty:
        ax.text(0.5, 0.5, "No detected events for this setting", ha="center", va="center", transform=ax.transAxes)
    else:
        bins = min(24, max(8, int(lead_times["lead_days"].nunique())))
        ax.hist(lead_times["lead_days"], bins=bins, color="#4c78a8", alpha=0.82, edgecolor="white")
        median = lead_times["lead_days"].median()
        ax.axvline(median, color="#d62728", linestyle="--", linewidth=1.4, label=f"median={median:.2f} days")
        ax.legend(loc="upper right")
    ax.set_title("Lead time distribution for detected target events")
    ax.set_xlabel("lead time before event start (days)")
    ax.set_ylabel("event count")
    ax.grid(True, alpha=0.25)

    ax2 = axes[1]
    if not lead_times.empty:
        ax2.boxplot(lead_times["lead_days"], vert=False, widths=0.55, patch_artist=True)
        ax2.set_yticks([])
    ax2.set_xlabel("lead time before event start (days)")
    ax2.grid(True, axis="x", alpha=0.25)

    fig.suptitle(
        (
            f"{summary['method'].upper()} lead time | {summary['setting']}\n"
            f"detected events={summary['events_with_valid_ta_alarm']}/{summary['total_events']}"
        ),
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_root = PROJECT_ROOT
    result_dir, _, _ = find_experiment(project_root, args.run_id)
    events = load_events(project_root, horizon_days=args.horizon_days)

    if args.method == "scc":
        path = ensure_scc_episode_file(
            project_root,
            args.run_id,
            result_dir,
            args.setting,
            horizon_days=args.horizon_days,
        )
        episodes = pd.read_csv(path)
        lead_times = analyze_episode_trigger_times(episodes, events, freq_minutes=args.freq_minutes)
    elif args.method == "bcad":
        path = result_dir / f"bcad_alarm_episodes_{args.setting}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        episodes = pd.read_csv(path)
        trigger_col = "first_trigger_time" if "first_trigger_time" in episodes.columns else "start_time"
        lead_times = analyze_episode_trigger_times(
            episodes,
            events,
            freq_minutes=args.freq_minutes,
            trigger_time_col=trigger_col,
        )
    else:
        path = ensure_cusum_episode_file(
            project_root,
            args.run_id,
            result_dir,
            args.setting,
            horizon_days=args.horizon_days,
        )
        episodes = pd.read_csv(path)
        lead_times = analyze_episode_trigger_times(episodes, events, freq_minutes=args.freq_minutes)

    summary = make_summary(args.method, args.setting, lead_times, total_events=len(events))
    output_method = "cusum_episode" if args.method == "cusum" else args.method
    out_csv = result_dir / f"{output_method}_lead_times_{args.setting}.csv"
    out_json = result_dir / f"{output_method}_lead_time_summary_{args.setting}.json"
    out_fig = project_root / "results" / "kelmarsh" / "figures" / "alarm_lead_times" / (
        f"{output_method}_lead_time_{args.setting}.png"
    )

    lead_times.to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_lead_time_distribution(lead_times, summary, out_fig)

    print("Method:", args.method)
    print("Setting:", args.setting)
    print("Lead-time CSV saved to:", out_csv)
    print("Summary JSON saved to:", out_json)
    print("Figure saved to:", out_fig)
    print("Detected events:", f"{summary['events_with_valid_ta_alarm']}/{summary['total_events']}")
    if not lead_times.empty:
        print("Median lead time days:", f"{lead_times['lead_days'].median():.3f}")
        print("Mean lead time days:", f"{lead_times['lead_days'].mean():.3f}")
        print("Max lead time days:", f"{lead_times['lead_days'].max():.3f}")


if __name__ == "__main__":
    main()
