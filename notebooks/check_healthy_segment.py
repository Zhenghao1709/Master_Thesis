from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
healthy_segments_path = PROJECT_ROOT / "data" / "processed" / "kelmarsh" / "healthy_segments" / "kelmarsh_1_healthy_segments.parquet"

healthy_segments = pd.read_parquet(healthy_segments_path)

print("shape:", healthy_segments.shape)
print("columns:", healthy_segments.columns.tolist())

segment_summary = healthy_segments[["turbine_id", "segment_id", "segment_len"]].drop_duplicates().copy()

print("\n连续健康片段数量:", len(segment_summary))
print("\nsegment_len 描述统计（按片段）:")
print(segment_summary["segment_len"].describe())

segment_summary["hours"] = segment_summary["segment_len"] * 10 / 60
segment_summary["days"] = segment_summary["hours"] / 24

print("\n长度换算成小时后的统计:")
print(segment_summary["hours"].describe())

SEQ_LEN = 12
segment_summary["n_sequences"] = (segment_summary["segment_len"] - SEQ_LEN).clip(lower=0)

print("\n每段可切出的序列数统计:")
print(segment_summary["n_sequences"].describe())
print("总可切序列数:", segment_summary["n_sequences"].sum())

plt.figure(figsize=(10, 5))
plt.hist(segment_summary["segment_len"], bins=50)
plt.xlabel("Segment length (number of 10-min points)")
plt.ylabel("Count")
plt.title("Distribution of Healthy Segment Lengths")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(segment_summary["hours"], bins=50)
plt.xlabel("Segment length (hours)")
plt.ylabel("Count")
plt.title("Distribution of Healthy Segment Lengths in Hours")
plt.tight_layout()
plt.show()