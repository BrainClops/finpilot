"""
parser.py
Handles reading uploaded bank statement / transaction files (CSV)
and normalizing them into a standard DataFrame format:
    date | description | amount | type (debit/credit)
"""

import pandas as pd


def parse_csv(uploaded_file) -> pd.DataFrame:
    """
    Reads an uploaded CSV file and tries to normalize common column names
    into a standard schema. Adjust `column_map` if your bank's CSV uses
    different header names.
    """
    df = pd.read_csv(uploaded_file)

    # Normalize column names to lowercase for matching
    df.columns = [c.strip().lower() for c in df.columns]

    # Common variations you might encounter in real bank statements
    column_map = {
        "date": ["date", "transaction date", "txn date"],
        "description": ["description", "narration", "details", "particulars"],
        "amount": ["amount", "amt", "transaction amount"],
        "type": ["type", "dr/cr", "debit/credit"],
    }

    standardized = {}
    for target, options in column_map.items():
        for opt in options:
            if opt in df.columns:
                standardized[target] = df[opt]
                break

    result = pd.DataFrame(standardized)

    # Basic cleanup
    if "date" in result:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if "amount" in result:
        result["amount"] = pd.to_numeric(result["amount"], errors="coerce")

    result = result.dropna(subset=["date", "amount"])
    result = result.sort_values("date").reset_index(drop=True)

    return result


def load_sample_data() -> pd.DataFrame:
    """
    Returns mock transaction data for demo purposes,
    in case a user doesn't want to upload a real file live.
    Great fallback for hackathon demo day.
    """
    import numpy as np

    dates = pd.date_range("2026-08-01", periods=45, freq="D")
    merchants = [
        ("Swiggy", -350, "Food"),
        ("Zomato", -420, "Food"),
        ("Netflix", -199, "Subscription"),
        ("Spotify", -119, "Subscription"),
        ("Rent Payment", -12000, "Rent"),
        ("Electricity Bill", -1400, "Utilities"),
        ("Uber", -220, "Transport"),
        ("Amazon", -1500, "Shopping"),
        ("Salary Credit", 35000, "Income"),
        ("Gym Membership", -1200, "Health"),
    ]

    rows = []
    rng = np.random.default_rng(42)
    for d in dates:
        n_txns = rng.integers(0, 3)
        for _ in range(n_txns):
            merchant, base_amt, _ = merchants[rng.integers(0, len(merchants))]
            amt = base_amt + rng.integers(-20, 20)
            rows.append({"date": d, "description": merchant, "amount": amt})

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df
