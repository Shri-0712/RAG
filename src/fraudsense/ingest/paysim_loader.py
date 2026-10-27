"""
PaySim dataset loader and normalizer.
Normalizes PaySim transaction logs to the canonical TransactionEvent schema.
"""

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from ..config import get_project_root, load_config
from .schema import UNIFIED_COLUMNS, TransactionEvent

BASE_TIME = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)


def _generate_synthetic_device(payer_id: str) -> str:
    """Generates a deterministic surrogate device ID for PaySim accounts."""
    h = hashlib.md5(payer_id.encode("utf-8")).hexdigest()[:8]
    return f"DEV_{h}"


def load_paysim(
    file_path: (str or Path) | None = None,
    n_rows: int | None = None,
    as_dataframe: bool = False,
) -> pd.DataFrame or list[TransactionEvent]:
    """
    Loads and normalizes the PaySim financial transaction dataset.

    Args:
        file_path: Path to PS_20174392719_1491204439457_log.csv.
        n_rows: Number of rows to load. If None, checks config or loads full file.
        as_dataframe: If True, returns a pandas DataFrame; otherwise returns List[TransactionEvent].

    Returns:
        DataFrame or List of TransactionEvent objects matching canonical schema.
    """
    if file_path is None:
        file_path = get_project_root() / "data" / "raw" / "PS_20174392719_1491204439457_log.csv"
    else:
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PaySim file not found at: {file_path}")

    if n_rows is None:
        try:
            cfg = load_config()
            n_rows = cfg.get("data", {}).get("sampling", {}).get("paysim_rows")
        except (FileNotFoundError, KeyError, ValueError):
            n_rows = None

    # Load with selected columns
    usecols = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
    ]
    df = pd.read_csv(file_path, nrows=n_rows, usecols=usecols)

    # Normalize fields
    df["event_id"] = [f"PAYSIM_{i}" for i in df.index]
    df["ts"] = df["step"].apply(lambda s: BASE_TIME + timedelta(hours=int(s)))
    df["amount"] = df["amount"].astype(float)
    df["payer_id"] = df["nameOrig"].astype(str)
    df["payee_id"] = df["nameDest"].astype(str)
    df["channel"] = df["type"].astype(str)
    df["device_id"] = df["payer_id"].apply(_generate_synthetic_device)
    df["label"] = df["isFraud"].astype(int)

    # Preserve domain metadata
    meta_cols = ["step", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    df["meta"] = df[meta_cols].to_dict(orient="records")

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
