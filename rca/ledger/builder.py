from __future__ import annotations

from typing import Optional

from rca.causal.graph import TopologyGraph
from rca.causal.types import CausalResult, RootCauseCandidate
from rca.config import Settings, get_settings
from rca.contract import EvidenceBucket, EvidenceItem, EvidenceLedger, RankedHypothesis
from rca.ingest.types import IncidentData
from rca.ledger.faults import cause_signal, fault_from_text, fault_hint_for, on_impact_path, sig_ref
from rca.schema import AnomalySignal


def _group_signals(signals: list[AnomalySignal]) -> dict[str, list[AnomalySignal]]:
    out: dict[str, list[AnomalySignal]] = {}
    for s in signals:
        if s.entity_id:
            out.setdefault(s.entity_id, []).append(s)
    return out


def _cause_text(entity: str, own: list[AnomalySignal]) -> str:
    s = cause_signal(entity, own)
    label, _ = fault_from_text(s.indicator if s else None)
    if s:
        return f"{label} ({s.indicator})"
    return label


def _has_frame(frame: object) -> bool:
    return frame is not None and len(getattr(frame, "columns", [])) > 0


def _missing_evidence(data: IncidentData, entity: str, signals: list[AnomalySignal]) -> list[EvidenceItem]:
    out: list[EvidenceItem] = []
    if not _has_frame(data.log_templates):
        out.append(EvidenceItem(bucket=EvidenceBucket.missing, ref_type="modality", ref_id="logs",
                                 detail="no log-template evidence ingested for this incident"))
    if not _has_frame(data.trace_latency):
        out.append(EvidenceItem(bucket=EvidenceBucket.missing, ref_type="modality", ref_id="traces",
                                 detail="no trace/latency evidence ingested for this incident"))
    label = fault_hint_for(entity, signals)
    if "memory" in label:
        out.append(EvidenceItem(bucket=EvidenceBucket.missing, ref_type="metric", ref_id="node_memory",
                                 detail="no node-level memory-pressure metric to confirm the physical cause"))
    return out


def build_hypotheses(
    result: CausalResult,
    signals: list[AnomalySignal],
    data: IncidentData,
    graph: TopologyGraph,
    settings: Optional[Settings] = None,
) -> list[RankedHypothesis]:
    settings = settings or get_settings()
    if not result.candidates:
        return []
    by_entity = _group_signals(signals)
    cause = result.candidates[0].entity_id
    anomalous = [c.entity_id for c in result.candidates]
    hyps: list[RankedHypothesis] = []

    for rank_i, cand in enumerate(result.candidates, start=1):
        cand.fault_hint = fault_hint_for(cand.entity_id, signals)
        ledger = EvidenceLedger()
        own = by_entity.get(cand.entity_id, [])
        is_primary = cand.entity_id == cause and not cand.demoted

        for s in own:
            item = EvidenceItem(
                bucket=EvidenceBucket.confirmed if is_primary else EvidenceBucket.correlated,
                ref_type="signal",
                ref_id=sig_ref(s),
                detail=f"{s.entity_id}: {s.indicator}",
                entity_id=s.entity_id,
                score=s.score,
            )
            (ledger.confirmed if is_primary else ledger.correlated).append(item)

        if is_primary:
            for sym in result.candidates[1:]:
                for s in by_entity.get(sym.entity_id, []):
                    ledger.correlated.append(
                        EvidenceItem(
                            bucket=EvidenceBucket.correlated,
                            ref_type="signal",
                            ref_id=sig_ref(s),
                            detail=f"{sym.entity_id}: {s.indicator} (downstream symptom)",
                            entity_id=sym.entity_id,
                            score=s.score,
                        )
                    )
            for cc in data.config_changes:
                on_path = on_impact_path(graph, cc.entity_id, anomalous)
                suffix = "" if on_path else " — off the impact path, excluded as cause"
                ledger.correlated.append(
                    EvidenceItem(
                        bucket=EvidenceBucket.correlated,
                        ref_type="config_change",
                        ref_id=f"cfg:{cc.entity_id}",
                        detail=f"{cc.change_type} on {cc.entity_id}{suffix}",
                        entity_id=cc.entity_id,
                    )
                )
            for item in _missing_evidence(data, cand.entity_id, signals):
                ledger.missing.append(item)

        hyps.append(
            RankedHypothesis(
                rank=rank_i,
                component=cand.entity_id,
                cause=_cause_text(cand.entity_id, own),
                score=round(float(cand.score), 4),
                impact_path=cand.impact_path,
                evidence=ledger,
                demoted=cand.demoted,
                note=cand.note,
            )
        )
    return hyps
