"""
Stable final ensemble after leaderboard feedback.

Avoids the two changes that hurt submissions:
- no aggressive v7 meta-only correction stack
- no heavy HGB/sample-row override from v8

Uses the two strongest stable pipelines (v5 fixed + v6) and a small intraday
horizon correction from v9. This keeps predictions close to v6 while adding
variance reduction and a test-horizon-specific signal.
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

# v5/v6 are almost identical but use different category treatment and imputation.
# v9 is intentionally kept small because its Day48 pseudo-validation is leaky,
# but it was trained on the same time horizon as the hidden test.
pred = (
    0.42 * v6["demand"].to_numpy()
    + 0.38 * v5["demand"].to_numpy()
    + 0.20 * v9["demand"].to_numpy()
)
pred = np.clip(pred, 0.0, 1.0)

submission = pd.DataFrame({"Index": test["Index"], "demand": pred})

if submission.shape != (41778, 2):
    raise ValueError(f"Bad shape: {submission.shape}")
if submission.columns.tolist() != ["Index", "demand"]:
    raise ValueError(f"Bad columns: {submission.columns.tolist()}")
if not np.isfinite(submission["demand"]).all():
    raise ValueError("Non-finite demand value")
if not submission["Index"].equals(test["Index"]):
    raise ValueError("Index order changed")

submission.to_csv(ROOT / "submission.csv", index=False)
submission.to_csv(OUT / "submission_v10_stable.csv", index=False)

print("Saved submission.csv and outputs/submission_v10_stable.csv")
print(submission["demand"].describe())
print(submission.head(10))
