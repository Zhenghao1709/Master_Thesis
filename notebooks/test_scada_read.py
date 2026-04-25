from pathlib import Path
from src.data.read_status import read_status_folder
from src.preprocessing.clean_status import clean_status_df

print("Running test_scada_read.py ...")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
print("PROJECT_ROOT:", PROJECT_ROOT)

status_folder = PROJECT_ROOT / "data" / "raw" / "kelmarsh" / "status" / "Kelmarsh_1"
print("status_folder:", status_folder)
print("exists:", status_folder.exists())

if status_folder.exists():
    print("files in folder:")
    for p in status_folder.iterdir():
        print(" -", p.name)

status_df = read_status_folder(
    status_folder,
    turbine_id="Kelmarsh_1"
)

status_df = clean_status_df(status_df)

print(status_df.head())
print(status_df[["Timestamp start", "Timestamp end", "Status", "Message", "IEC category", "status_bucket"]].head(10))

########
print("最早开始时间:", status_df["Timestamp start"].min())
print("最晚开始时间:", status_df["Timestamp start"].max())
print("最晚结束时间:", status_df["Timestamp end"].max())
print("总行数:", len(status_df))

print("\n最后10行：")
print(status_df.tail(10)[["Timestamp start", "Timestamp end", "Status", "Message", "IEC category", "status_bucket"]].to_string())

print("\n按月份统计：")
monthly_counts = status_df["Timestamp start"].dt.to_period("M").value_counts().sort_index()
print(monthly_counts.to_string())