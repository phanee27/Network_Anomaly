from __future__ import annotations

from typing import Optional

from rca.causal.graph import TopologyGraph
from rca.causal.types import CausalResult
from rca.contract import RecommendedStep
from rca.ingest.types import IncidentData
from rca.ledger.faults import cause_signal, fault_from_text, on_impact_path


def recommend(
    result: CausalResult, signals: list, data: IncidentData, graph: TopologyGraph
) -> list[RecommendedStep]:
    if not result.candidates:
        return []
    steps: list[RecommendedStep] = []
    cause = result.candidates[0]
    top_sig = cause_signal(cause.entity_id, signals)
    label, action = fault_from_text(top_sig.indicator if top_sig else cause.fault_hint)
    steps.append(
        RecommendedStep(
            action=f"Inspect {cause.entity_id}: {action}",
            rationale=f"Ranked #1 root cause ({label}).",
            refs=[f"hyp:1:{cause.entity_id}"],
        )
    )

    anomalous = [c.entity_id for c in result.candidates]
    on_path = [cc for cc in data.config_changes if on_impact_path(graph, cc.entity_id, anomalous)]
    off_path = [cc for cc in data.config_changes if not on_impact_path(graph, cc.entity_id, anomalous)]

    for cc in on_path:
        steps.append(
            RecommendedStep(
                action=f"Review the recent {cc.change_type} on {cc.entity_id}",
                rationale="This change is on the dependency path to the affected flow.",
                refs=[f"cfg:{cc.entity_id}"],
            )
        )
    if off_path:
        names = sorted({cc.entity_id for cc in off_path})
        steps.append(
            RecommendedStep(
                action=f"No action needed on {', '.join(names)}",
                rationale="Recent change(s) here are off the impact path to the symptom and are not causally linked.",
                refs=[f"cfg:{n}" for n in names],
            )
        )
    return steps
