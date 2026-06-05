"""
Ultra-conservative candidate after v11 decreased the leaderboard score.

v11 moved away from the known-good v10 with a profile calibration and sample-row
override. This version does neither. It stays near v10 while slightly reducing
the v9 intraday component, which earlier comments flagged as leaderboard-risky.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

test = pd.read_csv(ROOT / "dataset" / "test.csv")
v5 = pd.read_csv(OUT / "submission_v5_improved.csv")
v6 = pd.read_csv(OUT / "submission_v6_regenerated.csv")
v9 = pd.read_csv(OUT / "submission_v9_intraday.csv")

for name, df in [("v5", v5), ("v6", v6), ("v9", v9)]:
    if not df["Index"].equals(test["Index"]):
        raise ValueError(f"{name} index order does not match test.csv")

# v10 was 0.42*v6 + 0.38*v5 + 0.20*v9.
# Move only a little toward the stable v5/v6 pair and away from v9.
pred = (
    0.45 * v6["demand"].to_numpy()
    + 0.41 * v5["demand"].to_numpy()
    + 0.14 * v9["demand"].to_numpy()
)
pred = np.clip(pred, 0.0, 1.0)

submission = pd.DataFrame({"Index": test["Index"], "demand": pred})
if submission.shape != (41778, 2):
    raise ValueError(f"Bad shape: {submission.shape}")
if submission.columns.tolist() != ["Index", "demand"]:
    raise ValueError(f"Bad columns: {submission.columns.tolist()}")
if not submission["Index"].equals(test["Index"]):
    raise ValueError("Index order changed")
if not np.isfinite(submission["demand"]).all():
    raise ValueError("Non-finite predictions")

submission.to_csv(OUT / "submission_v12_ultra_conservative.csv", index=False)
print("Saved outputs/submission_v12_ultra_conservative.csv")
print(submission["demand"].describe())
print(submission.head(10))
