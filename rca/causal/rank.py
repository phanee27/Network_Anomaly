from __future__ import annotations

from datetime import datetime
from typing import Optional

from rca.causal.graph import TopologyGraph
from rca.causal.types import CausalResult, RootCauseCandidate
from rca.config import Settings, get_settings
from rca.schema import AnomalySignal, ConfigChange


def severity_by_entity(signals: list[AnomalySignal]) -> dict[str, float]:
    out: dict[str, float] = {}
    for s in signals:
        if not s.entity_id:
            continue
        out[s.entity_id] = max(out.get(s.entity_id, 0.0), abs(s.score))
    return out


def onset_by_entity(signals: list[AnomalySignal]) -> dict[str, datetime]:
    out: dict[str, datetime] = {}
    for s in signals:
        if not s.entity_id or s.window_start is None:
            continue
        cur = out.get(s.entity_id)
        if cur is None or s.window_start < cur:
            out[s.entity_id] = s.window_start
    return out


def rank_root_causes(
    signals: list[AnomalySignal],
    graph: TopologyGraph,
    inject_time: Optional[datetime] = None,
    config_changes: Optional[list[ConfigChange]] = None,
    settings: Optional[Settings] = None,
    extra_bias: Optional[dict[str, float]] = None,
) -> CausalResult:
    settings = settings or get_settings()
    severity = severity_by_entity(signals)
    onset = onset_by_entity(signals)
    anomalous = [e for e in severity if graph.has_node(e)]
    if not anomalous:
        return CausalResult(candidates=[])

    frz = graph.fault_relevance_zone(anomalous, hops=1)
    personalization = dict(severity)
    if extra_bias:
        for node, bonus in extra_bias.items():
            if graph.has_node(node):
                personalization[node] = personalization.get(node, 0.0) + float(bonus) * settings.gnn_bias_weight
    scores = graph.personalized_pagerank(
        personalization, nodes=frz, damping=settings.ppr_damping, max_iter=settings.ppr_max_iter
    )
    if not scores:
        scores = {e: severity[e] for e in anomalous}

    ranked = sorted(scores.items(), key=lambda kv: (kv[1], severity.get(kv[0], 0.0)), reverse=True)
    cause = ranked[0][0]
    cause_onset = onset.get(cause)

    symptoms = [e for e in anomalous if e != cause]
    cause_path: list[str] = [cause]
    for sym in sorted(symptoms, key=lambda e: severity.get(e, 0.0), reverse=True):
        ip = graph.impact_path(cause, sym)
        if ip and len(ip) > len(cause_path):
            cause_path = ip

    candidates: list[RootCauseCandidate] = []
    for ent, sc in ranked:
        if ent != cause and severity.get(ent, 0.0) <= 0:
            continue
        cand = RootCauseCandidate(
            entity_id=ent,
            score=float(sc),
            onset=onset.get(ent),
            severity=severity.get(ent, 0.0),
        )
        if ent == cause:
            cand.impact_path = cause_path
            cand.reasons.append("highest topology-weighted anomaly mass (personalized PageRank)")
            if cause_onset is not None:
                cand.reasons.append("earliest anomaly onset among causally-linked candidates")
        else:
            ip = graph.impact_path(cause, ent)
            later = cause_onset is not None and onset.get(ent) is not None and onset[ent] >= cause_onset
            if ip is not None:
                cand.impact_path = ip
                if later:
                    cand.demoted = True
                    cand.note = f"symptom downstream of {cause} on the impact path with later onset"
                    cand.reasons.append(f"reachable from {cause} via dependency path; onset not earlier than cause")
        candidates.append(cand)

    return CausalResult(candidates=candidates, frz=frz, pc_applied=False)
