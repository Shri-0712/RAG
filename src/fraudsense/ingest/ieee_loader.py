"""
IEEE-CIS Fraud Detection dataset loader and normalizer.
Joins transaction and identity tables and normalizes into canonical TransactionEvent schema.
"""

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from ..config import get_project_root, load_config
from .schema import UNIFIED_COLUMNS, TransactionEvent

BASE_TIME = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)


def _build_payer_id(row: pd.Series) -> str:
    """Creates a deterministic entity identifier from card and address fields."""
    parts = [
        str(row.get("card1", "")),
        str(row.get("card2", "")),
        str(row.get("card3", "")),
        str(row.get("card4", "")),
        str(row.get("card5", "")),
        str(row.get("card6", "")),
        str(row.get("addr1", "")),
    ]
    raw_str = "_".join(p for p in parts if p and p != "nan")
    if not raw_str:
        return f"CARD_ANON_{row.get('TransactionID')}"
    h = hashlib.md5(raw_str.encode("utf-8")).hexdigest()[:12]
    return f"CARD_{h}"


def _build_payee_id(row: pd.Series) -> str:
    """Creates a deterministic payee/merchant identifier from email domain and addr2."""
    email = str(row.get("P_emaildomain", "")).strip()
    addr2 = str(row.get("addr2", "")).strip()
    prod = str(row.get("ProductCD", "")).strip()
    if email and email != "nan":
        return f"MERCH_{email}_{addr2 if addr2 != 'nan' else '0'}"
    return f"MERCH_{prod}_{addr2 if addr2 != 'nan' else '0'}"


def _build_device_id(row: pd.Series, payer_id: str) -> str:
    """Extracts or synthesizes device identifier."""
    info = str(row.get("DeviceInfo", "")).strip()
    dtype = str(row.get("DeviceType", "")).strip()
    if info and info != "nan":
        h = hashlib.md5(f"{info}_{dtype}".encode()).hexdigest()[:10]
        return f"DEV_{h}"
    if dtype and dtype != "nan":
        return f"DEV_{dtype}_{payer_id[-6:]}"
    return f"DEV_IEEE_{payer_id[-6:]}"


def load_ieee(
    trans_path: (str or Path) | None = None,
    identity_path: (str or Path) | None = None,
    n_rows: int | None = None,
    as_dataframe: bool = False,
) -> pd.DataFrame or list[TransactionEvent]:
    """
    Loads and normalizes the IEEE-CIS Fraud Detection dataset.

    Args:
        trans_path: Path to train_transaction.csv.
        identity_path: Path to train_identity.csv.
        n_rows: Number of rows to load from transaction table.
        as_dataframe: If True, returns a pandas DataFrame; otherwise returns List[TransactionEvent].

    Returns:
        DataFrame or List of TransactionEvent objects matching canonical schema.
    """
    root = get_project_root()
    if trans_path is None:
        trans_path = root / "data" / "raw" / "train_transaction.csv"
    else:
        trans_path = Path(trans_path)

    if identity_path is None:
        identity_path = root / "data" / "raw" / "train_identity.csv"
    else:
        identity_path = Path(identity_path)

    if not trans_path.exists():
        raise FileNotFoundError(f"IEEE transaction file not found at: {trans_path}")

    if n_rows is None:
        try:
            cfg = load_config()
            n_rows = cfg.get("data", {}).get("sampling", {}).get("ieee_rows")
        except (FileNotFoundError, KeyError, ValueError):
            n_rows = None

    # Load core transaction columns
    trans_cols = [
        "TransactionID",
        "isFraud",
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
        "P_emaildomain",
    ]
    df_trans = pd.read_csv(trans_path, nrows=n_rows, usecols=trans_cols)

    # Load identity table if available
    if identity_path.exists():
        id_cols = ["TransactionID", "DeviceType", "DeviceInfo"]
        # If sampling, only read enough rows from identity or filter by TransactionID
        df_id = pd.read_csv(identity_path, usecols=id_cols, nrows=n_rows * 4 if n_rows else None)
        df = df_trans.merge(df_id, on="TransactionID", how="left")
    else:
        df = df_trans
        df["DeviceType"] = None
        df["DeviceInfo"] = None

    # Normalization
    df["event_id"] = df["TransactionID"].apply(lambda tid: f"IEEE_{tid}")
    df["ts"] = df["TransactionDT"].apply(lambda s: BASE_TIME + timedelta(seconds=int(s)))
    df["amount"] = df["TransactionAmt"].astype(float)
    df["payer_id"] = df.apply(_build_payer_id, axis=1)
    df["payee_id"] = df.apply(_build_payee_id, axis=1)
    df["channel"] = df["ProductCD"].fillna("UNKNOWN").astype(str)
    df["device_id"] = [
        _build_device_id(row, pid) for row, pid in zip(df.to_dict(orient="records"), df["payer_id"])
    ]
    df["label"] = df["isFraud"].astype(int)

    # Domain metadata for downstream Text2SQL & graph join
    meta_cols = [
        "TransactionID",
        "TransactionDT",
        "ProductCD",
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
        "P_emaildomain",
        "DeviceType",
        "DeviceInfo",
    ]
    df["meta"] = df[[c for c in meta_cols if c in df.columns]].to_dict(orient="records")

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
