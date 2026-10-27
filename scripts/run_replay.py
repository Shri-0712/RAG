"""Replay a public PaySim sample through the simulated FraudSense scoring API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi.testclient import TestClient

from fraudsense.audit.log import DatabaseManager
from fraudsense.ingest.paysim_loader import load_paysim
from fraudsense.ingest.replay import EventReplayEngine
from fraudsense.scoring.api import create_app


def main() -> None:
    """Scores a bounded, timestamp-ordered public-data replay in local SQLite."""
    parser = argparse.ArgumentParser(description="SIMULATED FraudSense replay")
    parser.add_argument("--samples", type=int, default=100, help="Public events to replay")
    parser.add_argument("--db", type=Path, default=None, help="Local SQLite path")
    args = parser.parse_args()
    db = DatabaseManager(args.db) if args.db else DatabaseManager()
    client = TestClient(create_app(db_manager=db))
    engine = EventReplayEngine(load_paysim(n_rows=args.samples))
    count = 0
    for event, _ in engine.stream(mask_label=True):
        response = client.post(
            "/score",
            json={
                "event_id": event.event_id,
                "amount": event.amount,
                "payer_id": event.payer_id,
                "payee_id": event.payee_id,
                "channel": event.channel,
                "device_id": event.device_id,
                "ts": event.ts.isoformat(),
                "meta": event.meta,
            },
        )
        response.raise_for_status()
        count += 1
    print(f"SIMULATED replay complete: {count} public-dataset events scored locally.")


if __name__ == "__main__":
    main()
