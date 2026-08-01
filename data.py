"""One-time 80/20 time-based split into frozen train/test files.

Usage: python data.py <creditcard_csv> <output_dir>
Writes train.csv and test.csv to <output_dir>. Run once; all models
train on train.csv and are evaluated on test.csv thereafter.
"""
import sys
from pathlib import Path

import pandas as pd

csv_path, out_dir = sys.argv[1], Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path).sort_values("Time")
cut = int(len(df) * 0.8)
train, test = df.iloc[:cut], df.iloc[cut:]

print(f"train: {len(train):,} rows, {int(train['Class'].sum())} frauds")
print(f"test:  {len(test):,} rows, {int(test['Class'].sum())} frauds")

train.to_csv(out_dir / "train.csv", index=False)
test.to_csv(out_dir / "test.csv", index=False)
