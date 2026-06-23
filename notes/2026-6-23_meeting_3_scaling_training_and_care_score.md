# Meeting Notes 3 - Signal Scaling, Epoch Selection, and CARE-like Scoring

**Date:** 23 June 2026

## 1. Signal-specific scaling strategy

Based on the discussion in the previous meeting and the distribution analysis in `signal_viewer.ipynb`, I replaced the previous single-scaler approach with a signal-specific scaling strategy. Each scaler is fitted only on the healthy training set and then applied unchanged to the validation, test, and future fault-event data.

### RobustScaler

RobustScaler is used for signals with relatively strong skewness or outliers:

- `Wind speed (m/s)`
- `Stator temperature 1 (deg C)`
- `Rear bearing temperature (deg C)`

It centers the data using the median and scales it using the interquartile range, making it less sensitive to extreme values.

### StandardScaler

StandardScaler is used for signals with comparatively regular distributions:

- `Power (kW)`
- `Nacelle ambient temperature (deg C)`
- `Nacelle temperature (deg C)`
- `Ambient temperature (converter) (deg C)`
- `Generator bearing front temperature (deg C)`

It centers each signal using its mean and scales it using its standard deviation.

### MinMaxScaler

MinMaxScaler is used for the three highly correlated rotational-speed signals:

- `Generator RPM (RPM)`
- `Rotor speed (RPM)`
- `Gearbox speed (RPM)`

These signals are transformed using the minimum and maximum values learned from the training set. In the current baseline input set, only Generator RPM is retained because the three speed signals contain highly redundant information. The other two remain available for later ablation experiments.


---

## 2. GRU epoch comparison for Kelmarsh_1

To investigate a reasonable number of training epochs, I trained the same Kelmarsh_1 GRU model using maximum epoch values of 20, 40, and 60. All three experiments used the same data split, input signals, model architecture, scaling strategy, and random seed.

| Maximum epochs | Best epoch | Best validation loss | Validation MAE | Validation RMSE |
|---:|---:|---:|---:|---:|
| 20 | 20 | 0.003371 | 0.159 deg C | 0.323 deg C |
| 40 | 38 | 0.003237 | 0.157 deg C | 0.317 deg C |
| 60 | 50 | 0.003039 | 0.153 deg C | 0.307 deg C |

The 60-epoch experiment achieved its lowest validation loss at epoch 50. After approximately epoch 50, the training loss continued to decrease, while the validation loss fluctuated and did not improve further. This suggests that the current model has largely converged around 50 epochs and that additional training provides little benefit.

### Question

- Would it be reasonable to use a maximum of 60 epochs with early stopping for the following experiments?
- Alternatively, should 50 epochs be used as a fixed training length to reduce computation time?
- My current preference is to set the maximum to 60 and use early stopping with a patience of 8, so the model automatically retains the epoch with the lowest validation loss.

---

## 3. Anomaly Scoring Strategy for Early Fault Detection

After reading the paper shared in the previous meeting, I do not think the original CARE SCORE approach can be applied directly to my dataset. My SCADA data do not contain complete sample-level fault labels or fault classes. I only have healthy training periods and a limited manually prepared event dataset containing event timestamps and descriptions.

Therefore, a supervised score that depends on complete labeled fault data may not be appropriate for the current setting.

For the initial baseline, I would like to start with SCC because it is the simplest method to implement and interpret. After establishing the SCC baseline, I would then implement CUSUM and compare the two approaches. Since the main objective is early fault detection, CUSUM may be more suitable for detecting small but persistent residual shifts caused by gradual component degradation. The comparison should therefore consider warning lead time and event detection performance, as well as the false-alarm rate.
