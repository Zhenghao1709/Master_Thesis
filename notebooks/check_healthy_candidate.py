from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

flags_path = (
    PROJECT_ROOT
    / "data" / "interim" / "kelmarsh" / "flags"
    / "kelmarsh_1_with_flags.parquet"
)

flagged_df = pd.read_parquet(flags_path)

summary_cols = [
    "is_good_quality",
    "is_physically_valid",
    "is_dirty",
    "is_event_like",
    "is_healthy_candidate"
]
summary = flagged_df[[c for c in summary_cols if c in flagged_df.columns]].mean()

print(summary)

print("\nAs percentages:")
for k, v in summary.items():
    print(f"{k}: {v:.2%}")
