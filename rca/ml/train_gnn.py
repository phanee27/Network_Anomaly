from __future__ import annotations

import argparse
import json

import numpy as np

from rca.causal.graph import TopologyGraph
from rca.config import get_settings
from rca.detect.registry import run_detectors
from rca.eval import find_cases
from rca.ingest.rcaeval import load_rcaeval_case
from rca.ml.features import NODE_FEATURE_DIM, node_feature_matrix


def _base_detectors(settings):
    from rca.detect.eps_diagnosis import EpsDiagnosisDetector
    from rca.detect.iforest import IsolationForestDetector
    from rca.detect.log_drain import LogTemplateDetector
    from rca.detect.trace_latency import TraceLatencyDetector

    return [
        EpsDiagnosisDetector(settings),
        IsolationForestDetector(settings),
        LogTemplateDetector(settings),
        TraceLatencyDetector(settings),
    ]


def train_gnn(root: str, epochs: int = 50, hidden: int = 64, limit: int = 0) -> None:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.utils import softmax

    from rca.ml.models import RCAScorer
    from rca.ml.utils import get_device

    settings = get_settings()
    detectors = _base_detectors(settings)
    cases = find_cases(root)
    if limit:
        cases = cases[:limit]

    graphs = []
    for case in cases:
        data = load_rcaeval_case(case, settings=settings)
        truth = (data.ground_truth or {}).get("service")
        if not truth:
            continue
        graph = TopologyGraph(data.edges)
        signals, _ = run_detectors(data, detectors)
        entities = data.entity_ids()
        if not entities:
            continue
        X, order = node_feature_matrix(signals, entities, graph)
        y = np.array([1.0 if e == truth else 0.0 for e in order], dtype="float32")
        if y.sum() == 0:
            continue
        idx = {e: i for i, e in enumerate(order)}
        src, dst = [], []
        for u, v in graph.g.edges():
            if u in idx and v in idx:
                src.append(idx[u])
                dst.append(idx[v])
        edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)
        graphs.append(Data(x=torch.tensor(X, dtype=torch.float), edge_index=edge_index, y=torch.tensor(y, dtype=torch.float)))

    if not graphs:
        raise SystemExit("no labelled graphs could be built (need ground-truth service per case)")

    device = get_device()
    model = RCAScorer(NODE_FEATURE_DIM, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    loader = DataLoader(graphs, batch_size=16, shuffle=True)
    model.train()
    for ep in range(epochs):
        total = 0.0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index)
            probs = softmax(logits, batch.batch)
            loss = -(batch.y * torch.log(probs + 1e-12)).sum() / batch.num_graphs
            loss.backward()
            opt.step()
            total += float(loss)
        print(f"[gnn] epoch {ep + 1}/{epochs} loss {total / len(loader):.5f}")

    out = settings.artifacts_root
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "gnn.pt")
    (out / "gnn.json").write_text(json.dumps({"in_dim": NODE_FEATURE_DIM, "hidden": hidden}))
    print(f"[gnn] saved to {out} ({len(graphs)} training graphs)")


def main() -> None:
    p = argparse.ArgumentParser(description="Train the GNN root-cause scorer on RCAEval cases.")
    p.add_argument("--root", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    train_gnn(args.root, args.epochs, args.hidden, args.limit)


if __name__ == "__main__":
    main()
