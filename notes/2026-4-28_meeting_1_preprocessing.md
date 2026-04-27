# Meeting Notes 1 – Preprocessing Progress

## 1. Current feature selection
For the first baseline, I selected the following features for `Kelmarsh_1`:

- `Wind speed (m/s)`
- `Power (kW)`
- `Nacelle ambient temperature (°C)`
- `Nacelle temperature (°C)`
- `Generator RPM (RPM)`
- `Rotor speed (RPM)`
- `Stator temperature 1 (°C)`
- `Generator bearing front temperature (°C)` as both target history and prediction target

### Question
- Are these 8 features sufficient for the first baseline?



- Should more features be added at this stage, or is it better to keep the model simple first?



---

## 2. Key preprocessing parameter choices
Current baseline parameter settings:

- **Sequence length**: `12`  
  This means each input sequence contains the past 12 SCADA points, i.e. 2 hours of history at 10-minute resolution.

- **Minimum healthy segment length**: `24`  
  This means only continuous healthy segments of at least 24 points (4 hours) are kept.

- **Maximum missing ratio per row**: `0.2`  
  A row is kept only if no more than 20% of the selected features are missing.

- **Physical limits**:  
  A coarse plausibility filter is applied, for example:
  - Wind speed: `0 ~ 40`
  - Power: `-100 ~ 3000`
  - Generator RPM: `0 ~ 2500`
  - Rotor speed: `0 ~ 30`
  - Generator bearing front temperature: `-20 ~ 150`

### Question
- Are these parameter values reasonable for the first baseline?
- In particular, is `SEQ_LEN = 12` a good starting point, considering the final objective is early detection?



---

## 3. Heuristic preprocessing workflow
The current heuristic pipeline is:

1. Read and merge SCADA data and status logs  
2. Align status intervals to SCADA timestamps  
3. Apply row-wise missing-value filtering  
4. Apply physical plausibility checks  
5. Apply normal operating condition checks  
6. Mark dirty data, event-like periods, and healthy candidates  
7. Split healthy candidate data into continuous healthy segments

In summary, a data point is kept as a healthy candidate only if it is:
- not dirty,
- not event-like,
- and under a normal operating condition.

### Question
- Does this heuristic logic look appropriate as a first preprocessing strategy for Kelmarsh?

---

## 4. First preprocessing results (Kelmarsh_1, 2016 only)
Using only `Kelmarsh_1` data from 2016, the first preprocessing run produced the following results:

### Healthy candidate filtering ratios
- `is_good_quality ≈ 65.5%`
- `is_physically_valid ≈ 99.98%`
- `is_normal_operating_condition ≈ 54.1%`
- `is_dirty ≈ 34.6%`
- `is_event_like ≈ 17.8%`
- `is_healthy_candidate ≈ 46.3%`

### Healthy segment results
- Number of healthy segments: **172**
- Mean segment length: **124.4 points**
- Median segment length: **63.5 points**
- Maximum segment length: **818 points**
- Total number of trainable sequences: **19,337**


### Question
- Does this amount of healthy data look sufficient and reasonable for training the first GRU-based NBM?