import unittest

import pandas as pd

from scripts.analysis.main_analyze_alarm_lead_time import analyze_episode_trigger_times


class EpisodeLeadTimeTest(unittest.TestCase):
    def test_episode_start_before_horizon_counts_if_episode_reaches_it(self):
        events = pd.DataFrame([{
            "event_uid": 1,
            "turbine_id": "Kelmarsh_1",
            "event_type": "fault",
            "event_start": pd.Timestamp("2024-01-10 00:00:00"),
            "event_end": pd.Timestamp("2024-01-10 01:00:00"),
            "horizon_start": pd.Timestamp("2024-01-03 00:00:00"),
        }])
        episodes = pd.DataFrame([{
            "episode_id": 12,
            "turbine_id": "Kelmarsh_1",
            "target": "Rear bearing temperature",
            "start_time": "2024-01-02 23:50:00",
            "end_time": "2024-01-03 00:10:00",
            "alarm_points": 3,
        }])
        result = analyze_episode_trigger_times(episodes, events, freq_minutes=10)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["episode_id"], 12)
        self.assertGreater(result.iloc[0]["lead_days"], 7)

    def test_episode_ending_before_horizon_does_not_count(self):
        events = pd.DataFrame([{
            "event_uid": 1, "turbine_id": "Kelmarsh_1",
            "event_start": pd.Timestamp("2024-01-10"),
            "horizon_start": pd.Timestamp("2024-01-03"),
        }])
        episodes = pd.DataFrame([{
            "turbine_id": "Kelmarsh_1", "start_time": "2024-01-02 23:30:00",
            "end_time": "2024-01-02 23:40:00",
        }])
        result = analyze_episode_trigger_times(episodes, events, freq_minutes=10)
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
