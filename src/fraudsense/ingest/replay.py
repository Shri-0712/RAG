"""
Event replay engine for FraudSense.
Streams normalized transaction events strictly in timestamp order,
supporting virtual clock progression and simulated label maturity delay.
"""

from collections.abc import Iterator
from datetime import datetime, timedelta

import pandas as pd

from ..config import load_config
from .european_loader import load_european
from .ieee_loader import load_ieee
from .paysim_loader import load_paysim
from .schema import TransactionEvent, events_to_dataframe


class EventReplayEngine:
    """
    Chronological event stream replayer.

    Guarantees:
    1. Events are yielded strictly in timestamp order (ts ascending).
    2. Ground-truth labels remain hidden from training/feedback until the
       simulated label maturity delay has elapsed (label_maturity_sim_days).
    """

    def __init__(
        self,
        events: list[TransactionEvent] | None = None,
        config: dict | None = None,
    ):
        self.config = config or load_config()
        self.label_maturity_days: int = self.config.get("labels", {}).get(
            "label_maturity_sim_days", 7
        )
        self._events: list[TransactionEvent] = []
        # Buffer for tracking label maturity: (maturity_timestamp, event_id, label)
        self._pending_labels: list[tuple[datetime, str, int]] = []
        self._matured_labels: dict[str, int] = {}
        self.current_sim_time: datetime | None = None

        if events is not None:
            self.set_events(events)

    def set_events(self, events: list[TransactionEvent]) -> None:
        """Sets and sorts events strictly in chronological order."""
        # Sort by ts, then by event_id for deterministic ordering
        self._events = sorted(events, key=lambda e: (e.ts, e.event_id))
        self._pending_labels.clear()
        self._matured_labels.clear()
        self.current_sim_time = self._events[0].ts if self._events else None

    @classmethod
    def from_dataset(
        cls,
        dataset_name: str,
        n_rows: int | None = None,
        config: dict | None = None,
    ) -> "EventReplayEngine":
        """
        Factory method to initialize replay engine directly from a named dataset.

        Args:
            dataset_name: One of 'paysim', 'ieee', or 'european'.
            n_rows: Optional limit on rows to load.
            config: Optional config dict.
        """
        name = dataset_name.lower()
        if name == "paysim":
            events = load_paysim(n_rows=n_rows)
        elif name in ("ieee", "ieee-cis"):
            events = load_ieee(n_rows=n_rows)
        elif name in ("european", "creditcard"):
            events = load_european(n_rows=n_rows)
        else:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Must be 'paysim', 'ieee', or 'european'."
            )

        engine = cls(events=events, config=config)
        return engine

    def __len__(self) -> int:
        return len(self._events)

    def stream(
        self,
        mask_label: bool = True,
        start_idx: int = 0,
        end_idx: int | None = None,
    ) -> Iterator[tuple[TransactionEvent, int | None]]:
        """
        Yields events sequentially in chronological order.

        Args:
            mask_label: If True, yields (event_with_hidden_label, ground_truth_label)
                        so scoring models cannot accidentally access ground truth.
            start_idx: Starting event index.
            end_idx: Optional slice upper bound.

        Yields:
            Tuple of (event_for_scoring, true_label)
        """
        slice_events = self._events[start_idx:end_idx]
        for event in slice_events:
            self.current_sim_time = event.ts
            # Register event in label maturity queue
            maturity_ts = event.ts + timedelta(days=self.label_maturity_days)
            self._pending_labels.append((maturity_ts, event.event_id, event.label))

            # Mature any labels whose delay has passed relative to current_sim_time
            self._refresh_matured_labels()

            if mask_label:
                # Provide an event with masked label (-1) to guarantee zero leakage
                scoring_event = TransactionEvent(
                    event_id=event.event_id,
                    ts=event.ts,
                    amount=event.amount,
                    payer_id=event.payer_id,
                    payee_id=event.payee_id,
                    channel=event.channel,
                    device_id=event.device_id,
                    label=-1,  # Masked
                    meta=event.meta,
                )
                yield scoring_event, event.label
            else:
                yield event, event.label

    def _refresh_matured_labels(self) -> None:
        """Checks and transfers matured labels into matured registry."""
        if not self.current_sim_time:
            return

        still_pending = []
        for maturity_ts, eid, lbl in self._pending_labels:
            if maturity_ts <= self.current_sim_time:
                self._matured_labels[eid] = lbl
            else:
                still_pending.append((maturity_ts, eid, lbl))
        self._pending_labels = still_pending

    def get_matured_label(self, event_id: str) -> int | None:
        """Returns the matured ground truth label if the maturity window has passed; else None."""
        return self._matured_labels.get(event_id, None)

    def to_dataframe(self) -> pd.DataFrame:
        """Returns the sorted events as a normalized DataFrame."""
        return events_to_dataframe(self._events, include_meta=True)
