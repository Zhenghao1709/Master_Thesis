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


if __name__ == "__main__":
    unittest.main()
