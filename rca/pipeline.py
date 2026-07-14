from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from rca.causal.engine import analyze
from rca.causal.graph import TopologyGraph
from rca.config import Settings, get_settings
from rca.contract import (
    GroundingDecision,
    Incident,
    IncidentStatus,
    NarrationSource,
)
from rca.detect.registry import build_detectors, run_detectors
from rca.ingest.types import IncidentData
from rca.ledger.builder import build_hypotheses
from rca.ledger.recommend import recommend
from rca.ledger.timeline import build_timeline
from rca.narrate.templated import render_narration, render_summary
from rca.schema import AnomalySignal


class _Audit:
    def __init__(self, repo, incident_id: str) -> None:
        self.repo = repo
        self.incident_id = incident_id
        self.entries: list[dict] = []

    def record(self, actor: str, action: str, input_ref=None, output_ref=None, detail=None) -> None:
        self.entries.append({"actor": actor, "action": action, "detail": detail or {}})
        if self.repo is not None:
            self.repo.append_audit(self.incident_id, actor, action, input_ref, output_ref, detail)


def _detected_at(signals: list[AnomalySignal], data: IncidentData) -> Optional[datetime]:
    starts = [s.window_start for s in signals if s.window_start is not None]
    if starts:
        return min(starts)
    return data.inject_time


def analyze_incident(
    data: IncidentData,
    settings: Optional[Settings] = None,
    persist: bool = True,
    narrate: bool = True,
) -> Incident:
    settings = settings or get_settings()
    repo = None
    if persist:
        from rca.db.postgres import get_repo

        repo = get_repo()
    audit = _Audit(repo, data.incident_id)

    graph = TopologyGraph(data.edges)
    detectors = build_detectors(settings)
    signals, det_errors = run_detectors(data, detectors)
    audit.record("detector", "detect", detail={"signals": len(signals), "errors": det_errors})

    result = analyze(data, signals, graph=graph, settings=settings)
    audit.record("causal", "rank", detail={"candidates": len(result.candidates), "pc_applied": result.pc_applied})

    hypotheses = build_hypotheses(result, signals, data, graph, settings=settings)
    timeline = build_timeline(result, signals, data)
    steps = recommend(result, signals, data, graph)
    audit.record("ledger", "build_evidence", detail={"hypotheses": len(hypotheses)})

    status = IncidentStatus.resolved if hypotheses else IncidentStatus.abstained
    incident = Incident(
        incident_id=data.incident_id,
        status=status,
        detected_at=_detected_at(signals, data),
        ranked_hypotheses=hypotheses,
        timeline=timeline,
        recommended_next_steps=steps,
        grounding_decision=GroundingDecision.resolved if hypotheses else GroundingDecision.abstain,
        audit_trail_id=f"aud:{data.incident_id}",
    )
    incident.summary = render_summary(incident)

    if narrate:
        from rca.narrate.narrator import narrate_incident

        narrate_incident(incident, settings=settings, audit=audit)
    else:
        incident.narration = render_narration(incident)
        incident.narration_source = NarrationSource.template

    if persist and repo is not None:
        repo.upsert_entities(data.entities)
        repo.upsert_topology_edges(data.edges)
        repo.insert_signals(signals)
        repo.insert_config_changes(data.config_changes)
        repo.replace_hypotheses(incident.incident_id, hypotheses)
        repo.upsert_incident(incident)
        audit.record("api", "persist", detail={"status": incident.status.value})

    return incident
