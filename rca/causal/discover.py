from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from rca.causal.graph import TopologyGraph
from rca.detect.common import parse_metric_col


def build_entity_matrix(metrics: Optional[pd.DataFrame], entities: set[str]) -> tuple[Optional[np.ndarray], list[str]]:
    if metrics is None or len(metrics) == 0:
        return None, []
    cols_by_entity: dict[str, list[str]] = {}
    for col in metrics.columns:
        entity, _ = parse_metric_col(str(col))
        if entity in entities:
            cols_by_entity.setdefault(entity, []).append(col)
    series: dict[str, pd.Series] = {}
    for entity, cols in cols_by_entity.items():
        sub = metrics[cols].apply(pd.to_numeric, errors="coerce")
        stds = sub.std()
        if not np.isfinite(stds.max()) or stds.max() == 0:
            continue
        series[entity] = sub[stds.idxmax()]
    if len(series) < 3:
        return None, []
    frame = pd.DataFrame(series).dropna()
    if len(frame) < 10:
        return None, []
    return frame.to_numpy(dtype=float), list(frame.columns)


def topology_constrained_pc(
    matrix: np.ndarray, names: list[str], graph: TopologyGraph, alpha: float
) -> list[tuple[str, str]]:
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

    nodes = [GraphNode(n) for n in names]
    index = {n: i for i, n in enumerate(names)}
    bk = BackgroundKnowledge()
    for a in names:
        for b in names:
            if a == b:
                continue
            if not graph.reaches(b, a):
                bk.add_forbidden_by_node(nodes[index[a]], nodes[index[b]])

    cg = pc(matrix, alpha, "fisherz", background_knowledge=bk, node_names=names, show_progress=False)
    adj = cg.G.graph
    edges: list[tuple[str, str]] = []
    n = len(names)
    for i in range(n):
        for j in range(n):
            if adj[i, j] == -1 and adj[j, i] == 1:
                edges.append((names[i], names[j]))
    return edges
