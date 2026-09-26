import unittest

import numpy as np
import pandas as pd

from src.detection.bcad import _bcad_scores_for_series, fit_folded_normal_mle


class FoldedNormalMleTest(unittest.TestCase):
    def test_fit_recovers_near_half_normal_scale(self):
        values = pd.Series(np.abs(np.random.default_rng(42).normal(0, 2, 5000)))
        center, scale = fit_folded_normal_mle(values)
        self.assertGreaterEqual(center, 0)
        self.assertLess(center, 0.5)
        self.assertAlmostEqual(scale, 2.0, delta=0.15)

    def test_mle_setting_preserves_log_odds_without_probability_saturation(self):
        values = pd.Series([0.1, 0.3, 0.7, 1.2])
        kwargs = dict(
            center=0.0, scale=0.5, variance_multiplier=5.0, window_size=3,
            abnormal_center=0.0, abnormal_scale=0.5,
        )
        old = _bcad_scores_for_series(values, distribution="folded_normal", **kwargs)
        mle = _bcad_scores_for_series(values, distribution="folded_normal_mle_shared_center", **kwargs)
        np.testing.assert_allclose(np.log(old / (1 - old)), mle, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
