# BCAD H/A Distribution Experiments Summary

Date: 2026-09-07

This note summarizes the attempts made so far to find a more suitable healthy distribution **H** and abnormal distribution **A** for BCAD detection. The focus here is on the distribution assumptions, not on the later window-size tuning.

## Current BCAD Setting Used for Comparison

Most comparisons below use:

```text
BCAD setting: W=72, q=0.99, vm=5, c=6
Residual input: GRU prediction residuals on 2023-2024 test data
H source: healthy validation residuals
Evaluation: same prediction target events, non-operation intervals, and 7-day horizon as the current detection strategy
```

The key question is:

```text
How should H and A be represented so that the detector separates healthy residual windows from abnormal residual windows more reliably?
```

## 1. Old Abs Gaussian Version

### What Was Changed

This was the first practical BCAD version:

```text
residual = abs(y_true - y_pred)
H = Gaussian fitted from healthy absolute validation residuals
A = Gaussian with the same mean as H but larger variance
```

So the difference between H and A was mainly:

```text
A has broader variance than H
```

### Distribution Shape

Because `abs(error)` is always non-negative, the real residual distribution is not a normal Gaussian. It is closer to a folded-normal or half-normal-like shape. However, this version still used a Gaussian approximation.

![Old abs Gaussian assumption](D:\硕士论文\Project_vision1\results\kelmarsh\figures\bcad_old_abs_gaussian_distribution_all6_multi3_seq12_h64_l1_bs64_lr1e-03_wd0_do0_e60_p8_seed42_all_targets_vm5.png)

### Result

```text
Setting: abs_w72_q990_vm5_c6
Detected events: 425 / 433
Recall: 98.15%
Operational false alarm rate: 20.42%
Precision: 79.58%
F1: 87.89%
```

This was one of the strongest distribution-based versions. Although the Gaussian assumption is theoretically imperfect for absolute residuals, the practical result was good.

## 2. Signed Gaussian Version

### What Was Changed

To make the residual distribution closer to a standard Gaussian assumption, the absolute value was removed:

```text
residual = y_true - y_pred
H = Gaussian fitted from signed healthy validation error
A = Gaussian with broader variance
```

This is more theoretically consistent because signed prediction error can be approximately centered around zero.

### Distribution Shape

The signed validation error is more symmetric and closer to a Gaussian-like shape than the absolute residual.

![Signed and absolute validation error distributions](D:\硕士论文\Project_vision1\results\kelmarsh\figures\bcad_validation_error_distribution_all6_multi3_seq12_h64_l1_bs64_lr1e-03_wd0_do0_e60_p8_seed42_all_targets_vm5.png)

### Result

```text
Setting: signed_w72_q990_vm5_c6
Detected events: 426 / 433
Recall: 98.38%
Operational false alarm rate: 24.89%
Precision: 75.11%
F1: 85.19%
```

The signed version is more theoretically clean, but in the current data it produced more operational false alarms than the old abs Gaussian version.

## 3. Folded-Normal Abs Version

### What Was Changed

Since `abs(error)` does not follow a standard normal distribution, the next attempt was to model the absolute residual using a folded-normal distribution:

```text
residual = abs(y_true - y_pred)
H = folded-normal fitted from healthy absolute validation residuals
A = folded-normal with broader spread
```

### Distribution Shape

This is more consistent with the non-negative shape of absolute residuals:

```text
abs Gaussian error -> folded-normal-like residual
```

### Result

```text
Setting: foldedabs_w72_q990_vm5_c6
Detected events: 426 / 433
Recall: 98.38%
Operational false alarm rate: 24.43%
Precision: 75.57%
F1: 85.48%
```

This was theoretically more suitable than old abs Gaussian, but the performance was worse than the old abs Gaussian approximation.

## 4. Hybrid Version: H Folded-Normal, A Gaussian

### What Was Changed

The next idea was:

```text
H = folded-normal
A = Gaussian
```

The motivation was that healthy absolute residuals are naturally folded, while abnormal residuals may shift to larger values and become less similar to a folded healthy distribution.

### Result

```text
Setting: hybridabs_w72_q990_vm5_c6
Detected events: 427 / 433
Recall: 98.61%
Operational false alarm rate: 26.15%
Precision: 73.85%
F1: 84.46%
```

This increased recall slightly, but it also increased false alarms. With `vm=1`, the method performed much worse:

```text
Setting: hybridabs_w72_q990_vm1_c6
Detected events: 216 / 433
Recall: 49.88%
Operational false alarm rate: 53.07%
F1: 48.36%
```

So this hybrid assumption does not seem stable enough in the current setup.

## 5. A Mean Multiplier Test

### Why This Was Tested

From the actual true-alarm residual windows, A seemed not only broader than H, but also shifted to larger residual values. Therefore, we tested:

```text
H = Gaussian(mean_H, var_H)
A = Gaussian(mean_H * abnormal_mean_multiplier, var_H * vm)
```

with:

```text
abnormal_mean_multiplier = 1.0 to 5.0
W=72, q=0.99, vm=5, c=6
```

### Actual H vs A Window-Max Residual Shape

The true-alarm episode window-max residuals show that the three target signals have very different A-like shapes.

![Generator bearing front temperature true alarm residuals](D:\硕士论文\Project_vision1\results\kelmarsh\figures\bcad_healthy_vs_true_alarm_episode_window_max_residual_all6_multi3_seq12_h64_l1_bs64_lr1e-03_wd0_do0_e60_p8_seed42_abs_w72_q990_vm5_c6_Generator_bearing_front_temperature.png)

![Stator temperature 1 true alarm residuals](D:\硕士论文\Project_vision1\results\kelmarsh\figures\bcad_healthy_vs_true_alarm_episode_window_max_residual_all6_multi3_seq12_h64_l1_bs64_lr1e-03_wd0_do0_e60_p8_seed42_abs_w72_q990_vm5_c6_Stator_temperature_1.png)

![Rear bearing temperature true alarm residuals](D:\硕士论文\Project_vision1\results\kelmarsh\figures\bcad_healthy_vs_true_alarm_episode_window_max_residual_all6_multi3_seq12_h64_l1_bs64_lr1e-03_wd0_do0_e60_p8_seed42_abs_w72_q990_vm5_c6_Rear_bearing_temperature.png)

Key statistics:

| Target | H median | True-alarm A median | H q99 | True-alarm A q99 |
|---|---:|---:|---:|---:|
| Generator bearing front temperature | 0.87 | 2.10 | 2.72 | 21.23 |
| Stator temperature 1 | 2.04 | 3.23 | 4.03 | 34.66 |
| Rear bearing temperature | 3.07 | 10.11 | 7.88 | 57.78 |

This suggests that a single shared A assumption may be too simple. Rear bearing temperature has the strongest A shift, while Stator temperature 1 has a smaller median shift but a strong long tail.

### Result of A Mean Multiplier Test

| A mean multiplier | Detected events | Recall | Operational FA rate | Precision | F1 |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 425 / 433 | 98.15% | 20.42% | 79.58% | 87.89% |
| 1.5 | 425 / 433 | 98.15% | 21.23% | 78.77% | 87.40% |
| 2.5 | 426 / 433 | 98.38% | 22.57% | 77.43% | 86.66% |
| 5.0 | 427 / 433 | 98.61% | 24.78% | 75.22% | 85.35% |

Increasing the A mean did not improve the final detection performance. It slightly increased recall, but the false alarm rate also increased, so F1 decreased.

## Overall Comparison

| Version | Main H/A assumption | Detected events | Recall | Operational FA rate | Precision | F1 |
|---|---|---:|---:|---:|---:|---:|
| Old abs Gaussian | H and A both Gaussian on abs residual; same mean, A larger variance | 425 / 433 | 98.15% | 20.42% | 79.58% | 87.89% |
| Signed Gaussian | H and A Gaussian on signed error | 426 / 433 | 98.38% | 24.89% | 75.11% | 85.19% |
| Folded-normal abs | H and A folded-normal on abs residual | 426 / 433 | 98.38% | 24.43% | 75.57% | 85.48% |
| Hybrid | H folded-normal, A Gaussian | 427 / 433 | 98.61% | 26.15% | 73.85% | 84.46% |
| A mean multiplier | A mean shifted from 1.5x to 5x H mean | up to 427 / 433 | up to 98.61% | increased | decreased | decreased |

## Current Interpretation

The signed Gaussian version is more consistent with the theoretical assumption of Gaussian residuals, because signed errors can be centered around zero. However, the best practical result so far still comes from the old absolute-residual Gaussian approximation.

The true-alarm window-max distributions show that the three target signals do not share the same abnormal residual behavior. This is important because BCAD currently uses a relatively simple H/A assumption for each target. The next more theoretically justified direction would be:

```text
Use signed error
Model H with a Bayesian Normal-Inverse-Gamma prior estimated from healthy validation error
Model A with a broader Normal-Inverse-Gamma prior
Allow A to depart in both mean and variance
Keep H/A target-specific
```

This would be closer to the reference paper's idea:

```text
H: informative prior from healthy data
A: broader prior allowing changes in mean and variance
```

## Practical Conclusion

For the current project stage:

```text
Best empirical BCAD distribution choice:
old abs Gaussian, W=72, q=0.99, vm=5, c=6
```

But for the thesis methodology, it is worth explaining that this is a practical approximation. A more principled follow-up is to implement Bayesian BCAD using signed residuals and Normal-Inverse-Gamma priors, because that would align better with the reference method and avoid the theoretical inconsistency caused by applying a Gaussian model directly to absolute residuals.
