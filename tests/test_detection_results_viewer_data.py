import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from main_detect_residual_baseline import (
    annotate_alarm_episode_operating_context,
    contained_many_prepared_intervals,
    fuse_alarm_episodes,
    longest_operational_alarm_run,
    merge_operating_intervals,
    operating_context_intervals,
    prepare_continuous_interval_bounds,
    update_event_summary_from_alarm_episodes,
)
from scripts.analysis.detection_results_viewer_data import (
    merge_alarm_intervals, min_consecutive_for_setting, plot_timeline, visible_alarm_intervals,
)


class DetectionResultsViewerDataTest(unittest.TestCase):
    def test_remaining_run_threshold_comes_from_setting(self):
        self.assertEqual(min_consecutive_for_setting("scc", "abs_meanstd_k6_c6"), 6)
        self.assertEqual(min_consecutive_for_setting("bcad", "foldedmle_w792_q990_c6"), 6)
        self.assertEqual(min_consecutive_for_setting("cusum_two_sided", "twosided_std_k0_h140_reset"), 1)

    def test_split_threshold_is_read_from_bcad_setting(self):
        self.assertEqual(min_consecutive_for_setting(
            "bcad", "foldedmle_samecenterabs_w360_q990_vm5_c144"), 144)
        self.assertEqual(min_consecutive_for_setting(
            "bcad", "foldedmle_samecenterabs_w180_q990_vm5_c72"), 72)

    def test_alarm_intervals_merge_across_targets_and_clip_to_period(self):
        episodes = pd.DataFrame(
            {
                "target": ["A", "B", "C"],
                "start_time": pd.to_datetime(
                    ["2022-12-31 23:50", "2023-01-01 00:10", "2023-01-01 01:00"]
                ),
                "end_time": pd.to_datetime(
                    ["2023-01-01 00:10", "2023-01-01 00:20", "2023-01-01 01:00"]
                ),
            }
        )
        intervals = merge_alarm_intervals(
            episodes, pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")
        )
        self.assertEqual(
            intervals,
            [
                (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-01 00:30")),
                (pd.Timestamp("2023-01-01 01:00"), pd.Timestamp("2023-01-01 01:10")),
            ],
        )

    def test_timeline_has_target_ta_fa_lanes_and_optional_non_operational_shading(self):
        events = pd.DataFrame(
            {"event_start": pd.to_datetime(["2023-01-02"]),
             "event_end": pd.to_datetime(["2023-01-02 01:00"])}
        )
        episodes = pd.DataFrame({
            "target": ["A", "B", "C"],
            "start_time": pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-04"]),
            "end_time": pd.to_datetime(["2023-01-01 00:10", "2023-01-03 00:10", "2023-01-04 00:10"]),
            "in_fault_horizon": [True, False, False],
            "in_non_operational_interval": [False, False, True],
        })
        non_operational = [(pd.Timestamp("2023-01-04"), pd.Timestamp("2023-01-04 00:20"))]
        figure = plot_timeline(events, episodes, "Kelmarsh_1", 2023, "scc", "test", non_operational)
        try:
            self.assertEqual(
                [label.get_text() for label in figure.axes[0].get_yticklabels()],
                ["Target event", "TA", "FA"],
            )
            self.assertEqual(len(figure.axes[0].collections), 4)
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)

    def test_non_operational_episode_is_excluded_entirely(self):
        episodes = pd.DataFrame({
            "start_time": pd.to_datetime(["2023-01-01 00:00", "2023-01-02 00:00"]),
            "end_time": pd.to_datetime(["2023-01-01 01:00", "2023-01-02 00:10"]),
            "in_fault_horizon": [True, False],
            "in_non_operational_interval": [True, False],
        })
        ta, fa = visible_alarm_intervals(episodes, pd.Timestamp("2023-01-01"),
                                         pd.Timestamp("2024-01-01"))
        self.assertEqual(ta, [])
        self.assertEqual(fa, [(pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-02 00:20"))])

    def test_target_episodes_fuse_by_turbine_and_time(self):
        source = pd.DataFrame({
            "turbine_id": ["Kelmarsh_1"] * 3 + ["Kelmarsh_2"],
            "target": ["A", "B", "C", "A"],
            "start_time": pd.to_datetime(["2023-01-01 00:00", "2023-01-01 00:10",
                                          "2023-01-01 01:00", "2023-01-01 00:00"]),
            "end_time": pd.to_datetime(["2023-01-01 00:10", "2023-01-01 00:20",
                                        "2023-01-01 01:00", "2023-01-01 00:10"]),
        })
        fused = fuse_alarm_episodes(source)
        self.assertEqual(len(fused), 3)
        self.assertNotIn("target", fused)
        self.assertEqual(fused.loc[0, "alarm_points"], 3)
        self.assertEqual(fused.loc[0, "end_time"], pd.Timestamp("2023-01-01 00:20"))

    def test_non_operational_episode_cannot_detect_event(self):
        events = pd.DataFrame({
            "turbine_id": ["Kelmarsh_1"],
            "event_start": pd.to_datetime(["2023-01-02"]),
            "horizon_start": pd.to_datetime(["2022-12-26"]),
        })
        episodes = pd.DataFrame({
            "turbine_id": ["Kelmarsh_1"],
            "start_time": pd.to_datetime(["2023-01-01"]),
            "end_time": pd.to_datetime(["2023-01-01 00:10"]),
            "alarm_points": [2],
            "in_non_operational_interval": [True],
        })
        scored = update_event_summary_from_alarm_episodes(events, episodes)
        self.assertFalse(bool(scored.loc[0, "detected_in_horizon"]))

    def test_non_operational_requires_containment_in_continuous_interval(self):
        day = pd.Timestamp("2023-01-01")
        bounds = prepare_continuous_interval_bounds([
            (day, day + pd.Timedelta(hours=1)),
            (day + pd.Timedelta(hours=2), day + pd.Timedelta(hours=3)),
            (day + pd.Timedelta(hours=4), day + pd.Timedelta(hours=5)),
            (day + pd.Timedelta(hours=4, minutes=40), day + pd.Timedelta(hours=6)),
        ])
        starts = pd.Series([day + pd.Timedelta(minutes=20),
                            day + pd.Timedelta(minutes=20),
                            day + pd.Timedelta(hours=4, minutes=30)])
        ends = pd.Series([day + pd.Timedelta(minutes=50),
                          day + pd.Timedelta(hours=2, minutes=20),
                          day + pd.Timedelta(hours=5, minutes=30)])
        self.assertEqual(contained_many_prepared_intervals(starts, ends, bounds).tolist(),
                         [True, False, True])

    def test_true_alarm_takes_priority_over_episode_level_non_operational_rules(self):
        day = pd.Timestamp("2023-01-01")
        episodes = pd.DataFrame({
            "turbine_id": ["Kelmarsh_1"] * 4,
            "start_time": [day, day + pd.Timedelta(hours=2),
                           day + pd.Timedelta(hours=3), day + pd.Timedelta(hours=3)],
            "end_time": [day + pd.Timedelta(minutes=10),
                         day + pd.Timedelta(minutes=10, hours=2),
                         day + pd.Timedelta(minutes=10, hours=4),
                         day + pd.Timedelta(minutes=10, hours=4)],
            "alarm_points": [2, 2, 8, 8],
            "in_fault_horizon": [False, False, False, True],
        })
        sources = {
            "in_maintenance": [], "in_communication": [], "in_curtailment": [],
            "auxiliary_maintenance": [], "non_full_performance": [], "target_event": [],
            "environmental_buffer": [
                (day + pd.Timedelta(hours=1), day + pd.Timedelta(hours=4))
            ],
        }
        event = pd.DataFrame({
            "turbine_id": ["Kelmarsh_1"],
            "event_start": [day + pd.Timedelta(hours=5)],
            "horizon_start": [day - pd.Timedelta(days=7)],
        })
        for rule, expected in (("overlap", [False, True, True, False]),
                               ("contained", [False, True, False, False])):
            with self.subTest(rule=rule), \
                 patch("main_detect_residual_baseline.build_target_event_summary", return_value=pd.DataFrame()), \
                 patch("main_detect_residual_baseline.operating_context_intervals", return_value=sources) as build:
                result = annotate_alarm_episode_operating_context(
                    episodes, Path("unused"), buffer_hours=3, non_operational_rule=rule,
                )
            self.assertEqual(result["in_non_operational_interval"].tolist(), expected)
            self.assertEqual(result["in_fault_horizon"].tolist(), [False, False, False, True])
            self.assertEqual(result["overlaps_non_operational_interval"].tolist(),
                             [False, True, True, True])
            self.assertTrue(bool(update_event_summary_from_alarm_episodes(
                event, result.iloc[[3]],
            ).loc[0, "detected_in_horizon"]))
            self.assertEqual(build.call_args.kwargs["buffer_hours"], 3)

    def test_remaining_consecutive_splits_non_true_alarm_runs(self):
        day = pd.Timestamp("2023-01-01")
        bounds = prepare_continuous_interval_bounds([
            (day + pd.Timedelta(minutes=40), day + pd.Timedelta(hours=1)),
        ])
        self.assertEqual(longest_operational_alarm_run(
            day, day + pd.Timedelta(hours=2), bounds,
        ), 7)
        sources = {
            "in_maintenance": [], "in_communication": [], "in_curtailment": [],
            "auxiliary_maintenance": [], "non_full_performance": [], "target_event": [],
            "environmental_buffer": [(day + pd.Timedelta(minutes=40),
                                      day + pd.Timedelta(hours=1))],
        }
        episodes = pd.DataFrame({
            "turbine_id": ["Kelmarsh_1"],
            "start_time": [day],
            "end_time": [day + pd.Timedelta(hours=2)],
            "in_fault_horizon": [False],
        })
        with patch("main_detect_residual_baseline.build_target_event_summary", return_value=pd.DataFrame()), \
             patch("main_detect_residual_baseline.operating_context_intervals", return_value=sources):
            c6 = annotate_alarm_episode_operating_context(
                episodes, Path("unused"), non_operational_rule="remaining_consecutive",
                min_operational_consecutive=6,
            )
            c8 = annotate_alarm_episode_operating_context(
                episodes, Path("unused"), non_operational_rule="remaining_consecutive",
                min_operational_consecutive=8,
            )
        self.assertEqual(len(c6), 1)
        self.assertEqual(c6.iloc[0]["start_time"], day + pd.Timedelta(hours=1))
        self.assertEqual(c6.iloc[0]["end_time"], day + pd.Timedelta(hours=2))
        self.assertEqual(c6.iloc[0]["alarm_points"], 7)
        self.assertFalse(bool(c6.iloc[0]["in_fault_horizon"]))
        self.assertTrue(bool(c6.iloc[0]["is_operational_false_alarm"]))
        self.assertFalse(bool(c6.iloc[0]["in_non_operational_interval"]))
        self.assertTrue(c8.empty)
        self.assertEqual(c8.attrs["suppressed_source_episodes"], 1)

    def test_remaining_consecutive_preserves_true_alarm_before_splitting(self):
        day = pd.Timestamp("2023-01-01")
        event = pd.DataFrame({
            "turbine_id": ["Kelmarsh_1"],
            "horizon_start": [day + pd.Timedelta(minutes=30)],
            "event_start": [day + pd.Timedelta(hours=3)],
        })
        episodes = pd.DataFrame({
            "turbine_id": ["Kelmarsh_1"],
            "start_time": [day],
            "end_time": [day + pd.Timedelta(hours=2)],
            "alarm_points": [13],
            "in_fault_horizon": [False],
        })
        sources = {
            "in_maintenance": [], "in_communication": [], "in_curtailment": [],
            "auxiliary_maintenance": [], "non_full_performance": [], "target_event": [],
            "environmental_buffer": [(day + pd.Timedelta(minutes=40),
                                      day + pd.Timedelta(hours=1))],
        }
        with patch("main_detect_residual_baseline.build_target_event_summary", return_value=event), \
             patch("main_detect_residual_baseline.operating_context_intervals", return_value=sources):
            result = annotate_alarm_episode_operating_context(
                episodes, Path("unused"), non_operational_rule="remaining_consecutive",
                min_operational_consecutive=72,
            )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["start_time"], day)
        self.assertEqual(result.iloc[0]["end_time"], day + pd.Timedelta(hours=2))
        self.assertTrue(bool(result.iloc[0]["in_fault_horizon"]))
        self.assertFalse(bool(result.iloc[0]["in_non_operational_interval"]))

    def test_remaining_consecutive_counts_fragments_as_independent_false_alarms(self):
        day = pd.Timestamp("2023-01-01")
        episodes = pd.DataFrame({
            "turbine_id": ["Kelmarsh_1"],
            "start_time": [day],
            "end_time": [day + pd.Timedelta(hours=2)],
            "in_fault_horizon": [False],
        })
        sources = {
            "in_maintenance": [], "in_communication": [], "in_curtailment": [],
            "auxiliary_maintenance": [], "non_full_performance": [], "target_event": [],
            "environmental_buffer": [(day + pd.Timedelta(minutes=50),
                                      day + pd.Timedelta(hours=1)),
                                     (day + pd.Timedelta(hours=1, minutes=50),
                                      day + pd.Timedelta(hours=2))],
        }
        with patch("main_detect_residual_baseline.build_target_event_summary", return_value=pd.DataFrame()), \
             patch("main_detect_residual_baseline.operating_context_intervals", return_value=sources):
            result = annotate_alarm_episode_operating_context(
                episodes, Path("unused"), non_operational_rule="remaining_consecutive",
                min_operational_consecutive=5,
            )
        self.assertEqual(result["alarm_points"].tolist(), [5, 5])
        self.assertEqual(result["is_operational_false_alarm"].tolist(), [True, True])
        self.assertEqual(result["start_time"].tolist(),
                         [day, day + pd.Timedelta(hours=1)])

    def test_operating_interval_union_does_not_fill_gaps(self):
        day = pd.Timestamp("2023-01-01")
        merged = merge_operating_intervals({
            "target_event": [(day, day + pd.Timedelta(hours=1))],
            "non_full_performance": [
                (day + pd.Timedelta(minutes=30), day + pd.Timedelta(hours=2))
            ],
            "environmental_buffer": [
                (day + pd.Timedelta(hours=3), day + pd.Timedelta(hours=4))
            ],
        })
        self.assertEqual(merged, [
            (day, day + pd.Timedelta(hours=2)),
            (day + pd.Timedelta(hours=3), day + pd.Timedelta(hours=4)),
        ])

    def test_operating_sources_exclude_preprocessing_event_flag(self):
        targets = pd.DataFrame({
            "turbine_id": ["Kelmarsh_1"],
            "event_start": [pd.Timestamp("2023-01-02")],
            "event_end": [pd.Timestamp("2023-01-03")],
        })
        with patch("main_detect_residual_baseline.intervals_from_flag_file", return_value={
            "in_maintenance": [], "in_communication": [], "in_curtailment": [],
        }) as read_flags, \
             patch("main_detect_residual_baseline.auxiliary_maintenance_intervals", return_value=[]), \
             patch("main_detect_residual_baseline.non_full_performance_intervals_from_status", return_value=[]), \
             patch("main_detect_residual_baseline.informational_environmental_spec_intervals_from_status") as read_buffer:
            sources = operating_context_intervals(
                Path("unused"), "Kelmarsh_1", buffer_hours=0, target_events=targets,
            )
        self.assertEqual(set(read_flags.call_args.args[1]),
                         {"in_maintenance", "in_communication", "in_curtailment"})
        self.assertEqual(sources["target_event"], [
            (pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-03"))
        ])
        read_buffer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
