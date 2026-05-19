# Meeting Notes 2 – Data Size Issue and Target Signal Selection

## 1. Large SCADA files in 2023 and 2024
During the raw SCADA extraction step, I found that the 2023 and 2024 files are much larger than the files from earlier years. While most yearly files are around 1 GB, the 2023 and 2024 files are around 16 GB, which caused serious memory issues during extraction and merging.

### Current observation
- The extraction and merge process works for earlier years, but 2023–2024 are too large to process in the same way on the current machine.
- Because of this, I temporarily limited the exploratory signal-viewing workflow to **2016–2022**.

### Questions
- Do we know why the 2023 and 2024 SCADA files are much larger than the previous years?
- Is the difference caused by higher sampling frequency, more recorded variables or export format changes?
- For the next stage, would it be reasonable to first complete the workflow on 2016–2022 and then integrate 2023–2024 separately with a chunk-based or staged processing strategy?

### Current idea for integration
At the moment, my plan is:
1. use **2016–2022** for signal exploration and initial baseline work,
2. keep 2023–2024 separate for now,
3. and later integrate them into the full dataset using a lighter pipeline, for example:
   - reading only selected columns,
   - extracting per file first,
   - and merging in smaller steps instead of loading everything at once.

---

## 2. Event windows and visual signal inspection
I used the manually prepared event dataset together with the signal viewer to inspect the SCADA signals around labeled event periods.

### Current observation
For many manually labeled event windows, the key signals do **not** show an obvious anomaly by visual inspection, or at least not in a way that is clearly distinguishable by eye.

This means that:
- manual visual inspection alone is not sufficient to decide which variables should be used as target signals,
- and target selection should not rely only on whether a signal looks abnormal in raw plots around events.

### Input signal selection
    "Wind speed (m/s)",
    "Power (kW)",
    "Nacelle ambient temperature (°C)",
    "Nacelle temperature (°C)",
    "Generator RPM (RPM)",
    "Rotor speed (RPM)",
    "Gearbox speed (RPM)",
    "Ambient temperature (converter) (°C)",
    "Stator temperature 1 (°C)", #target
    "Generator bearing front temperature (°C)"
    "Rear bearing temperature (°C)"

The correlation between "Generator RPM", "Rotor speed", and "Gearbox speed" is nearly identical, I think it's not necessarily required to include all three.

The correlation between "Nacelle ambient temperature" and "Ambient temperature (converter)" is approximately 0.978668; the two are also highly redundant. 

### Target signal selection
Based on the current signal exploration and correlation inspection, I selected the following three target signals for the next stage:

    "Stator temperature 1 (°C)", #target
    "Generator bearing front temperature (°C)"
    "Rear bearing temperature (°C)"

### Rationale
The target selection was made based on visual observation of the signal behavior and their relationships with the other selected SCADA variables.



### Current conclusion
At this stage, these three signals seem to be the most suitable target candidates among the currently selected variables, because they are both physically meaningful and show patterns that may support normal behavior modeling.We can appropriately increase the number of input features, as they may provide more information on operating conditions and thermal states; however, at this stage, I hope to keep the number of target signals within 2-3 to maintain a clear modeling and evaluation framework. Too many targets do not necessarily reduce accuracy, but they will significantly increase model complexity, training and comparison costs, and may also weaken the analytical depth of each target.