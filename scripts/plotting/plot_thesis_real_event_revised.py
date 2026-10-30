"""Plot an illustrative real event using the revised selected detector episodes."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.analysis.main_analyze_alarm_lead_time import load_events


DATA = ROOT / "results" / "kelmarsh" / "thesis_revision_split6h" / "chapter7_selected_revised"
THESIS = ROOT.parent / "Thesis_LaTeX" / "figures" / "chapter7"
EVENT_UID = 230
STEP = pd.Timedelta(minutes=10)
TRACKS = (
    ("scc", "SCC", "#3976A8", False),
    ("cusum_one_sided", "Absolute CUSUM", "#C77824", True),
    ("bcad_folded", "Folded-normal BCAD", "#8054A3", False),
)


def merge_intervals(intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    merged: list[list[pd.Timestamp]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def main() -> None:
    events = pd.read_csv(DATA / "scc_lead_time_events.csv",
                         parse_dates=["event_start", "event_end", "horizon_start"])
    event = events.loc[events["event_uid"].eq(EVENT_UID)]
    if len(event) != 1:
        raise ValueError(f"Expected one target event with event_uid={EVENT_UID}")
    event = event.iloc[0]
    selected = pd.read_csv(DATA / "chapter7_comparison.csv").set_index("variant")
    rendered = []
    for slug, name, color, extend_cusum in TRACKS:
        episodes = pd.read_csv(DATA / f"{slug}_evaluated_episodes.csv",
                               parse_dates=["start_time", "end_time"])
        matching = episodes.loc[
            episodes["turbine_id"].eq(event.turbine_id)
            & episodes["start_time"].lt(event.event_start)
            & (episodes["end_time"] + STEP).ge(event.horizon_start)
        ].copy()
        if matching.empty:
            raise ValueError(f"{slug} does not detect the selected example event")
        lead = (event.event_start - matching["start_time"].min()).total_seconds() / 86400
        extension = pd.Timedelta(hours=12) if extend_cusum else STEP
        covered = merge_intervals([
            (max(row.start_time, event.horizon_start), min(row.end_time + extension, event.event_start))
            for row in matching.itertuples()
        ])
        coverage = sum((end - start).total_seconds() for start, end in covered) / 86400
        lead_row = pd.read_csv(DATA / f"{slug}_lead_time_events.csv")
        coverage_row = pd.read_csv(DATA / f"{slug}_horizon_coverage_events.csv")
        expected_lead = lead_row.loc[lead_row["event_uid"].eq(event.event_uid), "lead_days"].iloc[0]
        expected_coverage = coverage_row.loc[coverage_row["event_uid"].eq(event.event_uid), "covered_days"].iloc[0]
        if abs(lead - expected_lead) > 1e-6 or abs(coverage - expected_coverage) > 1e-6:
            raise ValueError(f"{slug} event example disagrees with event-level summary")
        setting = str(selected.loc[slug, "setting"])
        rendered.append((slug, name, color, setting, covered if extend_cusum else
                         merge_intervals(list(zip(matching.start_time, matching.end_time + STEP))),
                         lead, coverage))

    start = min(event.horizon_start - pd.Timedelta(days=1),
                min(event.event_start - pd.Timedelta(days=row[5]) for row in rendered) - pd.Timedelta(days=1))
    end = event.event_end + pd.Timedelta(hours=18)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    fig.subplots_adjust(left=0.27, right=0.81, bottom=0.21, top=0.84)
    # Enlarge only the short event interval, keeping all actual timestamps intact.
    event_left = mdates.date2num(event.event_start)
    event_right = mdates.date2num(event.event_end)
    actual_width = event_right-event_left
    displayed_width = max(actual_width, 1.0)
    expansion = displayed_width/actual_width
    def forward(values):
        values = np.asarray(values)
        return np.where(values <= event_left, values,
                        np.where(values <= event_right,
                                 event_left+(values-event_left)*expansion,
                                 values+displayed_width-actual_width))
    def inverse(values):
        values = np.asarray(values)
        return np.where(values <= event_left, values,
                        np.where(values <= event_left+displayed_width,
                                 event_left+(values-event_left)/expansion,
                                 values-displayed_width+actual_width))
    ax.set_xscale("function", functions=(forward, inverse))
    ax.axvspan(event.horizon_start, event.event_start, color="#E7F1F8", zorder=0)
    ax.axvspan(event.event_start, event.event_end, color="#F5D9D6", zorder=0)
    ax.axvline(event.horizon_start, color="#7098B4", linestyle="--", linewidth=0.8)
    ax.axvline(event.event_start, color="#B74C48", linestyle="--", linewidth=0.9)
    ax.axvline(event.event_end, color="#B74C48", linestyle=":", linewidth=0.8)
    labels = []
    for y, (slug, name, color, setting, intervals, lead, coverage) in zip((2, 1, 0), rendered):
        if slug == "scc":
            label = "SCC\n$k=5$, $c=6$"
        elif slug == "bcad_folded":
            label = "Folded-normal BCAD\n$W=72$, $c=72$"
        else:
            parameters = selected.loc[slug, "setting"]
            import re
            match = re.search(r"_k([0-9p]+)_h([0-9p]+)_", parameters)
            if match is None:
                raise ValueError(f"Cannot parse CUSUM setting: {parameters}")
            k, h = (item.replace("p", ".") for item in match.groups())
            label = f"Absolute CUSUM\n$k={k}$, $h={h}$"
        labels.append(label)
        bars = [(mdates.date2num(first), (last - first).total_seconds() / 86400)
                for first, last in intervals]
        ax.broken_barh(bars, (y - 0.11, 0.22), facecolors=color, zorder=3)
        trigger = event.event_start - pd.Timedelta(days=lead)
        arrow_y = y + 0.25
        ax.annotate("", xy=(event.event_start, arrow_y), xytext=(trigger, arrow_y),
                    arrowprops={"arrowstyle": "<->", "color": "black",
                                "linewidth": 1.25, "shrinkA": 0, "shrinkB": 0},
                    zorder=5)
        ax.plot([trigger, trigger], [y+0.12, arrow_y+0.035],
                color="black", linestyle=":", linewidth=0.8, zorder=4)
        ax.text(trigger + (event.event_start-trigger)/2, arrow_y+0.035,
                f"$L={lead:.2f}$ d", ha="center", va="bottom", fontsize=11,
                color="black", zorder=6)
        ax.text(1.03, y, f"$E={coverage:.2f}$ d",
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=12)
        print(f"{name}: L={lead:.2f} d, E={coverage:.2f} d", flush=True)
    ax.set_yticks([2, 1, 0], labels)
    ax.set_ylim(-0.45, 2.65)
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlabel(str(event.event_start.year))
    ax.grid(axis="x", color="#E2E6E9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.text(event.horizon_start + (event.event_start - event.horizon_start) / 2,
            1.05, "Early-warning\nhorizon", color="#456C86", ha="center", va="bottom",
            transform=ax.get_xaxis_transform(), fontsize=12)
    ax.text(event.event_start + (event.event_end-event.event_start)/2,
            1.05, "Target\nevent", color="#9E403D", ha="center", va="bottom",
            transform=ax.get_xaxis_transform(), fontsize=12)
    THESIS.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(THESIS / f"real_event_method_comparison.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
