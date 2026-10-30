import unittest

import numpy as np
import pandas as pd

from src.detection.bcad import _bcad_scores_for_series, _weibull_logpdf


class WeibullJointScoreTest(unittest.TestCase):
    def test_folded_calibration_scores_window_maximum_not_joint_likelihood(self):
        values = pd.Series([0.5, 1.0, 2.0, 3.0])
        kwargs = dict(center=0.0, scale=1.0, variance_multiplier=1.0,
                      window_size=3, healthy_weibull_shape=1.5,
                      healthy_weibull_scale=1.0, weibull_shape=2.0,
                      weibull_scale=2.0)
        folded = _bcad_scores_for_series(
            values, distribution="weibull_h_weibull_a_mle_folded_calibration", **kwargs
        )
        previous = _bcad_scores_for_series(
            values, distribution="weibull_h_weibull_a_mle_calibration", **kwargs
        )
        joint = _bcad_scores_for_series(
            values, distribution="weibull_h_weibull_a_joint_mle_calibration", **kwargs
        )
        np.testing.assert_allclose(folded.to_numpy(), previous.to_numpy(), equal_nan=True)
        self.assertNotAlmostEqual(folded.iloc[-1], joint.iloc[-1])

    def test_score_sums_point_log_likelihood_ratios(self):
        values = pd.Series([0.5, 1.0, 2.0, 3.0])
        scores = _bcad_scores_for_series(
            values,
            center=0.0,
            scale=1.0,
            variance_multiplier=1.0,
            window_size=3,
            distribution="weibull_h_weibull_a_joint_mle_calibration",
            healthy_weibull_shape=1.5,
            healthy_weibull_scale=1.0,
            weibull_shape=2.0,
            weibull_scale=2.0,
        )
        point_ratio = _weibull_logpdf(values, 2.0, 2.0) - _weibull_logpdf(values, 1.5, 1.0)
        self.assertTrue(np.isnan(scores.iloc[0]))
        self.assertTrue(np.isnan(scores.iloc[1]))
        self.assertAlmostEqual(scores.iloc[2], point_ratio.iloc[:3].sum())
        self.assertAlmostEqual(scores.iloc[3], point_ratio.iloc[1:].sum())

    def test_folded_calibrated_joint_uses_point_log_likelihood_ratios(self):
        values = pd.Series([0.5, 1.0, 2.0, 3.0])
        kwargs = dict(
            center=0.0, scale=1.0, variance_multiplier=1.0,
            window_size=3, healthy_weibull_shape=1.5,
            healthy_weibull_scale=1.0, weibull_shape=2.0,
            weibull_scale=2.0,
        )
        previous = _bcad_scores_for_series(
            values, distribution="weibull_h_weibull_a_joint_mle_calibration", **kwargs
        )
        folded = _bcad_scores_for_series(
            values, distribution="weibull_h_weibull_a_joint_mle_folded_calibration", **kwargs
        )
        np.testing.assert_allclose(folded, previous, equal_nan=True)

    def test_window_max_log_score_uses_same_weibull_densities(self):
        values = pd.Series([0.5, 1.0, 2.0, 3.0])
        kwargs = dict(
            center=0.0, scale=1.0, variance_multiplier=1.0,
            window_size=3, healthy_weibull_shape=1.5,
            healthy_weibull_scale=1.0, weibull_shape=2.0,
            weibull_scale=2.0,
        )
        old = _bcad_scores_for_series(
            values, distribution="weibull_h_weibull_a_mle_calibration", **kwargs
        )
        log_score = _bcad_scores_for_series(
            values, distribution="weibull_h_weibull_a_mle_calibration_logscore", **kwargs
        )
        np.testing.assert_allclose(np.log(old / (1 - old)), log_score, equal_nan=True)
        maximum = values.rolling(3, min_periods=3).max()
        expected = _weibull_logpdf(maximum, 2.0, 2.0) - _weibull_logpdf(maximum, 1.5, 1.0)
        np.testing.assert_allclose(log_score, expected, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
