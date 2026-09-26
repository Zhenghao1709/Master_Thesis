# Utility scripts

The project root contains only the main data, training, prediction, and
detection entry points. Supporting scripts are grouped here:

- `plotting/`: thesis figures and detection-result visualisations.
- `experiments/`: parameter grids and BCAD calibration utilities.
- `analysis/`: post-detection analyses such as alarm lead time.

Run a script with the project Python environment from the project root, for
example:

```powershell
C:\Users\10980\anaconda3\envs\pytorch\python.exe scripts\plotting\main_plot_thesis_preprocessing_pipeline.py
```
