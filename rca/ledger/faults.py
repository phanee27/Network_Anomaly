from __future__ import annotations

from typing import Optional

from rca.causal.graph import TopologyGraph
from rca.schema import AnomalySignal

_FAULT_KEYWORDS: list[tuple[str, tuple[str, str]]] = [
    ("mem", ("memory leak / pressure", "Inspect heap usage and restart or scale the affected pod; check for unbounded caches or leaks.")),
    ("cpu", ("CPU saturation", "Profile hot code paths and scale replicas or raise CPU limits.")),
    ("disk", ("disk I/O or capacity pressure", "Check disk usage and I/O wait; free space or move to faster storage.")),
    ("socket", ("socket / connection exhaustion", "Check open sockets and ulimits; tune connection pooling.")),
    ("latency", ("elevated latency", "Check downstream dependency latency, timeouts and connection pools.")),
    ("delay", ("network delay on the path", "Inspect network path and QoS between the affected services.")),
    ("loss", ("network packet loss", "Inspect NIC/switch health and retransmissions on the path.")),
    ("error", ("elevated error rate", "Inspect recent errors and logs on the service and its dependencies.")),
    ("workload", ("workload / traffic surge", "Check upstream traffic and autoscaling behaviour.")),
]


def fault_from_text(text: Optional[str]) -> tuple[str, str]:
    t = (text or "").lower()
    for keyword, (label, action) in _FAULT_KEYWORDS:
        if keyword in t:
            return label, action
    return "anomalous behaviour", "Inspect the service metrics, logs and recent changes."


def cause_signal(entity: str, signals: list[AnomalySignal]) -> Optional[AnomalySignal]:
    best: Optional[AnomalySignal] = None
    for s in signals:
        if s.entity_id == entity and (best is None or s.score > best.score):
            best = s
    return best


def fault_hint_for(entity: str, signals: list[AnomalySignal]) -> str:
    s = cause_signal(entity, signals)
    label, _ = fault_from_text(s.indicator if s else None)
    return label


def sig_ref(signal: AnomalySignal) -> str:
    return f"sig:{signal.detector.value}:{signal.entity_id}"


def on_impact_path(graph: TopologyGraph, entity: str, anomalous: list[str]) -> bool:
    for sym in anomalous:
        if sym == entity or graph.reaches(sym, entity):
            return True
    return False
