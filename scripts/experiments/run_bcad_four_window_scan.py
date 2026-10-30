"""Run BCAD window scans under the current c72, split6h evaluation rule."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
RUN = "all6_multi3_seq12_h64_l1_bs64_lr1e-03_wd0_do0_e60_p8_seed42"
RESULTS = ROOT / "results" / "kelmarsh" / "experiments" / RUN
WINDOWS = range(36, 1081, 36)
MODES = {
    "folded": ("folded_normal_mle_shared_center", "foldedmle_samecenterabs_w{w}_q990_vm5_c72_split6h_no_warning_turbine"),
    "gaussian": ("gaussian_abs_logscore", "abslog_w{w}_q990_vm5_c72_split6h_no_warning_turbine"),
    "maximum": ("weibull_h_weibull_a_mle_folded_calibration",
                "weibullmle_foldedcalib2021_2022_w{w}_q990_c72_split6h_no_warning_turbine"),
    "joint": ("weibull_h_weibull_a_joint_mle_folded_calibration",
              "weibulljointmle_foldedcalib2021_2022_w{w}_q990_c72_split6h_no_warning_turbine"),
}


def run_mode(name: str) -> None:
    distribution, suffix_pattern = MODES[name]
    for index, window in enumerate(WINDOWS, start=1):
        suffix = suffix_pattern.format(w=window)
        output = RESULTS / f"bcad_performance_{suffix}.json"
        if not output.exists():
            command = [
                sys.executable, str(ROOT / "main_detect_bcad.py"),
                "--run-id", RUN, "--distribution", distribution,
                "--residual-mode", "absolute", "--window-size", str(window),
                "--quantile", "0.99", "--variance-multiplier", "5",
                "--skip-detections-file",
            ]
            completed = subprocess.run(
                command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            if completed.returncode != 0 or not output.exists():
                raise RuntimeError(f"{name}, W={window} failed:\n{completed.stdout[-4000:]}")
        performance = json.loads(output.read_text(encoding="utf-8"))
        detected = performance["event_level"]["detected_events"]
        false_rate = performance["alarm_episode_level"]["operational_false_alarm_rate"]
        print(f"{name} [{index}/30] W={window}: {detected}/459 events, "
              f"operational FA={false_rate:.2%}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=[*MODES, "all"], default="all")
    args = parser.parse_args()
    names = MODES if args.mode == "all" else [args.mode]
    for name in names:
        run_mode(name)
    print("Requested BCAD scans complete.", flush=True)


if __name__ == "__main__":
    main()
