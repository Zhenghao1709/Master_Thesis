# Meeting 4 - Prediction and Residual Baseline Detection

Date: 2026-08-14

## 1. Current Prediction and Detection Logic



Residual baseline steps:

```text
1. Use validation residual quantile to define the threshold.
2. Compare test prediction with the measured test value to get residual.
3. Mark points with residual above the threshold as abnormal points.
4. Raise an alarm when there are n consecutive abnormal points.
```

Because SCADA resolution is 10 minutes:

```text
min_consecutive = 6  -> about 1 hour
min_consecutive = 12 -> about 2 hours
min_consecutive = 36 -> about 6 hours
```

## 2. Tested Residual Baseline Settings

| Setting | Quantile | Min consecutive | Detected events | Missed events | Detection rate | Miss rate | False alarm episode rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| q950_c6 | 0.950 | 6 | 134 / 138 | 4 | 97.10% | 2.90% | 78.67% |
| q990_c6 | 0.990 | 6 | 132 / 138 | 6 | 95.65% | 4.35% | 79.55% |
| q990_c12 | 0.990 | 12 | 120 / 138 | 18 | 86.96% | 13.04% | 80.43% |
| q990_c36 | 0.990 | 36 | 77 / 138 | 61 | 55.80% | 44.20% | 82.12% |
| q995_c12 | 0.995 | 12 | 118 / 138 | 20 | 85.51% | 14.49% | 80.82% |

Main observation:

```text
Increasing quantile / min_consecutive reduces alarms,
but false alarm episode rate remains high and missed events increase.
```

This suggests that many false alarms are probably not short isolated spikes. They may come from longer periods of high residual.

## 3. Possible Reason: Alarm Classification May Be Too Simple

Current evaluation:

```text
alarm in 7 days before event start -> true early alarm
no alarm in 7 days before event start -> missed event
alarm outside all 7-day windows -> false alarm
```

Problem:

```text
Alarms during maintenance / downtime / communication unavailable periods
may currently be counted as false alarms,
even though these periods are not normal operation.
```

## 4. Proposed Alarm Classification




Explicit timeline idea:

```text
time  -------------------------------------------------------------------->

      normal operation        event start - 7d          event start        event end
      |----------------------|=========================|##################|---------|
      false alarm zone        true early alarm zone      event-period zone  normal/post-event
                              missed event if no alarm   should be separate

      maintenance / downtime / communication interval:
      |***************|
      alarm here should probably be reported separately,
      not simply counted as operational false alarm.
```

Suggested categories:

| Alarm location | Suggested label |
|---|---|
| Normal operation, outside any event horizon | Operational false alarm |
| Within 7 days before event start | True early alarm |
| No alarm within 7 days before event start | Missed event |
| During event interval | Event-period alarm |
| During maintenance / downtime / communication unavailable | Non-normal-operation alarm |


