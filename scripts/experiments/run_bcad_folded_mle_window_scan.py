"""Complete the final folded-normal BCAD window scan without overwriting saved runs."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
RUN = "all6_multi3_seq12_h64_l1_bs64_lr1e-03_wd0_do0_e60_p8_seed42"
RESULTS = ROOT / "results" / "kelmarsh" / "experiments" / RUN
WINDOWS = range(36, 1081, 36)


def performance_path(window: int) -> Path:
    return RESULTS / f"bcad_performance_foldedmle_samecenterabs_w{window}_q990_vm5_c6.json"


def run_window(window: int) -> tuple[int, str]:
    path = performance_path(window)
    if not path.exists():
        command = [
            sys.executable,
            str(ROOT / "main_detect_bcad.py"),
            "--run-id", RUN,
            "--window-size", str(window),
            "--quantile", "0.99",
            "--variance-multiplier", "5",
            "--residual-mode", "absolute",
            "--distribution", "folded_normal_mle_shared_center",
            "--skip-detections-file",
        ]
        completed = subprocess.run(
            command, cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        )
        if completed.returncode != 0 or not path.exists():
            raise RuntimeError(f"W={window} failed:\n{completed.stdout[-3000:]}")
    result = json.loads(path.read_text(encoding="utf-8"))
    detected = result["event_level"]["detected_events"]
    false_rate = result["alarm_episode_level"]["operational_false_alarm_rate"]
    return window, f"W={window}: {detected}/433 events, operational FA={false_rate:.2%}"


def main() -> None:
    # main_detect_bcad.py updates one shared metadata.json non-atomically.
    # Run sequentially so concurrent writers cannot corrupt it.
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = [pool.submit(run_window, window) for window in WINDOWS]
        for future in as_completed(futures):
            _, message = future.result()
            print(message, flush=True)
    assert all(performance_path(window).exists() for window in WINDOWS)
    print("Folded-normal scan complete: 36 to 1080 in steps of 36.", flush=True)


if __name__ == "__main__":
    main()
