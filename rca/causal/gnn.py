from __future__ import annotations

import json
from typing import Optional

from rca.causal.graph import TopologyGraph
from rca.config import Settings, get_settings
from rca.schema import AnomalySignal


def _edge_index(order: list[str], graph: TopologyGraph):
    import torch

    idx = {e: i for i, e in enumerate(order)}
    src, dst = [], []
    for u, v in graph.g.edges():
        if u in idx and v in idx:
            src.append(idx[u])
            dst.append(idx[v])
    if not src:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def gnn_scores(
    signals: list[AnomalySignal], entities: list[str], graph: TopologyGraph, settings: Optional[Settings] = None
) -> Optional[dict[str, float]]:
    settings = settings or get_settings()
    if not settings.use_gnn:
        return None
    root = settings.artifacts_root
    if not (root / "gnn.pt").exists():
        return None
    try:
        import torch

        from rca.ml.features import NODE_FEATURE_DIM, node_feature_matrix
        from rca.ml.models import RCAScorer

        meta = json.loads((root / "gnn.json").read_text()) if (root / "gnn.json").exists() else {}
        X, order = node_feature_matrix(signals, entities, graph)
        if len(order) == 0:
            return None
        model = RCAScorer(meta.get("in_dim", NODE_FEATURE_DIM), meta.get("hidden", 64))
        model.load_state_dict(torch.load(root / "gnn.pt", map_location="cpu"))
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float), _edge_index(order, graph))
            probs = torch.softmax(logits, dim=0).cpu().numpy()
        return {order[i]: float(probs[i]) for i in range(len(order))}
    except Exception:
        return None
