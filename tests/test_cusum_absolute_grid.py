import unittest

import numpy as np

from scripts.experiments.main_grid_cusum_fixed_meanstd import build_cusum_alarm_rows


class AbsoluteCusumGridTest(unittest.TestCase):
    def test_opposite_signed_errors_accumulate_in_absolute_mode(self):
        groups = [{
            "times": np.array(["2024-01-01T00:00", "2024-01-01T00:10",
                               "2024-01-01T00:20", "2024-01-01T00:30"], dtype="datetime64[m]"),
            "z": np.array([1.0, -1.0, 1.0, -1.0]),
            "turbine_id": "Kelmarsh_1",
            "segment_id": 1,
            "target": "temperature",
        }]
        absolute = build_cusum_alarm_rows(groups, 0.5, 1.5, score_mode="absolute")
        two_sided = build_cusum_alarm_rows(groups, 0.5, 1.5, score_mode="two-sided")
        self.assertEqual(len(absolute), 1)
        self.assertEqual(str(absolute.iloc[0]["Date and time"]), "2024-01-01 00:30:00")
        self.assertTrue(two_sided.empty)

    def test_absolute_mode_resets_after_each_alarm(self):
        groups = [{
            "times": np.array(["2024-01-01T00:00", "2024-01-01T00:10"], dtype="datetime64[m]"),
            "z": np.array([2.0, 2.0]),
            "turbine_id": "Kelmarsh_1", "segment_id": 1, "target": "temperature",
        }]
        alarms = build_cusum_alarm_rows(groups, 0.0, 1.0, score_mode="absolute")
        self.assertEqual(len(alarms), 2)


if __name__ == "__main__":
    unittest.main()
