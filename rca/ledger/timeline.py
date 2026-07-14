from __future__ import annotations

from rca.causal.types import CausalResult
from rca.contract import TimelineEvent
from rca.ingest.types import IncidentData
from rca.ledger.faults import sig_ref
from rca.schema import AnomalySignal


def build_timeline(
    result: CausalResult, signals: list[AnomalySignal], data: IncidentData
) -> list[TimelineEvent]:
    events: list[TimelineEvent] = []
    for cc in data.config_changes:
        if cc.ts is not None:
            events.append(
                TimelineEvent(
                    t=cc.ts,
                    event=f"{cc.change_type} on {cc.entity_id}",
                    entity_id=cc.entity_id,
                    refs=[f"cfg:{cc.entity_id}"],
                )
            )
    for s in signals:
        if s.window_start is not None:
            events.append(
                TimelineEvent(
                    t=s.window_start,
                    event=f"{s.entity_id}: {s.indicator}",
                    entity_id=s.entity_id,
                    refs=[sig_ref(s)],
                )
            )
    events.sort(key=lambda e: e.t)
    return events
