import unittest

import pandas as pd

from scripts.analysis.main_analyze_alarm_horizon_coverage import (
    PROFILES,
    analyze_method,
    summarize,
    union_coverage,
)


class AlarmHorizonCoverageTest(unittest.TestCase):
    def test_thesis_profile_uses_selected_episode_settings(self):
        self.assertEqual(PROFILES["thesis_best"], {
            "SCC": "abs_meanstd_k6_c6",
            "BCAD": "weibullmle_foldedcalib2021_2022_w936_q990_c6",
            "CUSUM": "twosided_std_k0_h100_reset",
        })

    def test_union_clips_and_does_not_count_overlaps_twice(self):
        start = pd.Timestamp("2024-01-01")
        end = start + pd.Timedelta(days=7)
        intervals = [
            (start - pd.Timedelta(days=1), start + pd.Timedelta(days=2)),
            (start + pd.Timedelta(days=1), start + pd.Timedelta(days=3)),
            (start + pd.Timedelta(days=6), end + pd.Timedelta(days=1)),
        ]
        hours = union_coverage(intervals, start, end)
        self.assertEqual(hours, 4 * 24)

    def test_missed_event_has_zero_coverage(self):
        start = pd.Timestamp("2024-01-08")
        events = pd.DataFrame([{
            "event_uid": 1, "turbine_id": "Kelmarsh_1",
            "horizon_start": start - pd.Timedelta(days=7), "event_start": start,
        }])
        episodes = pd.DataFrame(columns=["turbine_id", "start_time", "end_time"])
        result = analyze_method("SCC", episodes, events)
        self.assertEqual(result.loc[0, "covered_days"], 0)
        self.assertEqual(result.loc[0, "coverage_fraction"], 0)

    def test_cusum_12_hour_carry_can_cross_horizon_boundary(self):
        start = pd.Timestamp("2024-01-08")
        events = pd.DataFrame([{
            "event_uid": 1, "turbine_id": "Kelmarsh_1",
            "horizon_start": start - pd.Timedelta(days=7), "event_start": start,
        }])
        alarm_time = start - pd.Timedelta(days=7, hours=6)
        episodes = pd.DataFrame([{
            "turbine_id": "Kelmarsh_1", "start_time": alarm_time,
            "end_time": alarm_time,
        }])
        result = analyze_method("CUSUM", episodes, events)
        self.assertEqual(result.loc[0, "covered_hours"], 6)
        self.assertAlmostEqual(result.loc[0, "coverage_fraction"], 6 / (7 * 24))

    def test_coverage_is_valid_without_reaching_event_start(self):
        start = pd.Timestamp("2024-01-08")
        events = pd.DataFrame([{
            "event_uid": 1, "turbine_id": "Kelmarsh_1",
            "horizon_start": start - pd.Timedelta(days=7), "event_start": start,
        }])
        alarm_time = start - pd.Timedelta(days=3)
        episodes = pd.DataFrame([{
            "turbine_id": "Kelmarsh_1", "start_time": alarm_time,
            "end_time": alarm_time,
        }])
        result = analyze_method("CUSUM", episodes, events)
        self.assertEqual(result.loc[0, "covered_hours"], 12)
        self.assertAlmostEqual(result.loc[0, "coverage_fraction"], 12 / (7 * 24))

    def test_summary_excludes_undetected_events_from_distribution(self):
        coverage = pd.DataFrame({
            "covered_days": [0.0, 1.0, 3.0],
            "coverage_fraction": [0.0, 1 / 7, 3 / 7],
        })
        result = summarize(coverage)
        self.assertEqual(result["total_events"], 3)
        self.assertEqual(result["events_with_coverage"], 2)
        self.assertEqual(result["mean_covered_days_detected_events"], 2.0)
        self.assertEqual(result["median_covered_days_detected_events"], 2.0)
        self.assertAlmostEqual(
            result["mean_horizon_coverage_percent_detected_events"], 200 / 7,
        )


if __name__ == "__main__":
    unittest.main()
