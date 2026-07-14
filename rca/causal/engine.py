from __future__ import annotations

from typing import Optional

from rca.causal.discover import build_entity_matrix, topology_constrained_pc
from rca.causal.graph import TopologyGraph
from rca.causal.rank import rank_root_causes
from rca.causal.types import CausalResult
from rca.config import Settings, get_settings
from rca.ingest.types import IncidentData
from rca.schema import AnomalySignal


def analyze(
    data: IncidentData,
    signals: list[AnomalySignal],
    graph: Optional[TopologyGraph] = None,
    settings: Optional[Settings] = None,
) -> CausalResult:
    settings = settings or get_settings()
    graph = graph or TopologyGraph(data.edges)
    bias = _gnn_bias(data, signals, graph, settings)
    result = rank_root_causes(
        signals,
        graph,
        inject_time=data.inject_time,
        config_changes=data.config_changes,
        settings=settings,
        extra_bias=bias,
    )
    if not result.candidates:
        return result
    if bias:
        result.gnn_applied = True
        result.candidates[0].reasons.append("GNN root-cause scorer applied as an additive re-weighting")
    try:
        _apply_pc(data, graph, result, settings)
    except Exception:
        result.pc_applied = False
    return result


def _gnn_bias(data: IncidentData, signals, graph, settings):
    try:
        from rca.causal.gnn import gnn_scores

        return gnn_scores(signals, data.entity_ids(), graph, settings)
    except Exception:
        return None


def _apply_pc(data: IncidentData, graph: TopologyGraph, result: CausalResult, settings: Settings) -> None:
    matrix, names = build_entity_matrix(data.metrics, set(result.frz))
    if matrix is None or len(names) < 3:
        return
    edges = topology_constrained_pc(matrix, names, graph, settings.pc_alpha)
    result.pc_applied = True
    result.edges_used = edges
    cause = result.candidates[0].entity_id
    confirmed = sorted({b for (a, b) in edges if a == cause})
    if confirmed:
        result.candidates[0].reasons.append(
            "PC conditional-independence test confirms a directed causal edge to " + ", ".join(confirmed)
        )
