"""Feature engineering, shared by training and (eventually) serving.

Every feature is computable from the transaction data alone — no labels
(the fraud-rate-per-hour idea is deferred: it's target encoding, which
leaks unless built out-of-fold and causally), and no statistics fit on
data (those belong in the model pipeline).
"""
import numpy as np

# EDA shortlist: the strongest univariate separators (fraud vs legit).
SHORTLIST = ["V17", "V14", "V12", "V10", "V16", "V3"]

FAMILIES = ("time", "interactions", "log_amount", "activity")


def add(df, families):
    """Return a copy of df with the requested feature families appended."""
    out = df.copy()

    if "time" in families:
        # Time = seconds since the dataset's first transaction; the EDA's
        # volume curve suggests that's roughly midnight, so day-phase is
        # relative, not verified wall-clock. sin/cos make 23:00 and 01:00
        # neighbors; buckets one-hot the day's phases (afternoon/evening =
        # both dummies zero, so it's the reference class).
        hour = (out["Time"] / 3600) % 24
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        out["is_night"] = ((hour >= 0) & (hour < 6)).astype("float32")
        out["is_morning"] = ((hour >= 6) & (hour < 14)).astype("float32")

    if "interactions" in families:
        # All 15 pairwise products of the shortlist: lets a linear model
        # score frauds that are only jointly (not individually) unusual.
        for i, a in enumerate(SHORTLIST):
            for b in SHORTLIST[i + 1:]:
                out[f"{a}x{b}"] = out[a] * out[b]

    if "log_amount" in families:
        out["log_amount"] = np.log1p(out["Amount"])

    if "activity" in families:
        # Transactions in this row's (absolute) hour — "how busy is the
        # system right now". At serving time this comes from the live
        # traffic stream; here, from the batch itself. Uses no labels.
        hour_abs = (out["Time"] // 3600).astype(int)
        out["txns_per_hour"] = hour_abs.map(hour_abs.value_counts()).astype("float32")

    return out
