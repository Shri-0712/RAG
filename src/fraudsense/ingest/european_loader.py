"""
European Credit Card Fraud dataset loader and normalizer.
Normalizes the tabular PCA dataset (Time, V1..V28, Amount, Class) to canonical schema.
Designated for risk head and DQN sanity checks (no graph).
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from ..config import get_project_root, load_config
from .schema import UNIFIED_COLUMNS, TransactionEvent

BASE_TIME = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)


def load_european(
    file_path: (str or Path) | None = None,
    n_rows: int | None = None,
    as_dataframe: bool = False,
) -> pd.DataFrame or list[TransactionEvent]:
    """
    Loads and normalizes the European Credit Card Fraud dataset.

    Args:
        file_path: Path to creditcard.csv.
        n_rows: Number of rows to load. If None, checks config or loads full file.
        as_dataframe: If True, returns a pandas DataFrame; otherwise returns List[TransactionEvent].

    Returns:
        DataFrame or List of TransactionEvent objects matching canonical schema.
    """
    if file_path is None:
        file_path = get_project_root() / "data" / "raw" / "creditcard.csv"
    else:
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"European credit card file not found at: {file_path}")

    if n_rows is None:
        try:
            cfg = load_config()
            n_rows = cfg.get("data", {}).get("sampling", {}).get("european_rows")
        except (FileNotFoundError, KeyError, ValueError):
            n_rows = None

    df = pd.read_csv(file_path, nrows=n_rows)

    # Normalize fields
    df["event_id"] = [f"EU_{i}" for i in df.index]
    df["ts"] = df["Time"].apply(lambda s: BASE_TIME + timedelta(seconds=int(s)))
    df["amount"] = df["Amount"].astype(float)
    # Synthetic entity IDs for tabular sanity checks
    df["payer_id"] = [f"EU_CARD_{i % 5000:04d}" for i in df.index]
    df["payee_id"] = [f"EU_MERCH_{i % 500:03d}" for i in df.index]
    df["channel"] = "ONLINE_POS"
    df["device_id"] = [f"DEV_EU_{i % 2000:04d}" for i in df.index]
    # Class column may be quoted or int
    df["label"] = df["Class"].astype(int)

    # Preserve PCA components V1..V28 and original Time
    v_cols = [c for c in df.columns if c.startswith("V")] + ["Time"]
    df["meta"] = df[v_cols].to_dict(orient="records")

    normalized_df = df[UNIFIED_COLUMNS + ["meta"]]

    if as_dataframe:
        return normalized_df

    events = [
        TransactionEvent(
            event_id=row["event_id"],
            ts=row["ts"],
            amount=row["amount"],
            payer_id=row["payer_id"],
            payee_id=row["payee_id"],
            channel=row["channel"],
            device_id=row["device_id"],
            label=row["label"],
            meta=row["meta"],
        )
        for _, row in normalized_df.iterrows()
    ]
    return events
