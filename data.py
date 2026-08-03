"""
This script takes a dataset and path as input and splits
it time-based  (train/test) (80/20) into separate files.
The resulting datasets are written to the specified output directory 

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

train.to_csv(out_dir / "train.csv", index=False)
test.to_csv(out_dir / "test.csv", index=False)
