from __future__ import annotations

import argparse
import json

import numpy as np

from rca.config import get_settings
from rca.eval import find_cases
from rca.ingest.rcaeval import load_rcaeval_case
from rca.ml.features import CANONICAL_METRICS, autoencoder_normal_matrix


def train_autoencoder(root: str, epochs: int = 50, hidden: int = 16, bottleneck: int = 4, limit: int = 0) -> float:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from rca.ml.models import MetricAutoencoder
    from rca.ml.utils import get_device

    settings = get_settings()
    cases = find_cases(root)
    if limit:
        cases = cases[:limit]
    mats = []
    for case in cases:
        data = load_rcaeval_case(case, settings=settings)
        m = autoencoder_normal_matrix(data.metrics, data.inject_time, settings.normal_window_frac)
        if len(m):
            mats.append(m)
    if not mats:
        raise SystemExit("no normal metric data found under root")
    X = np.vstack(mats).astype("float32")

    device = get_device()
    model = MetricAutoencoder(X.shape[1], hidden, bottleneck).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.tensor(X)), batch_size=256, shuffle=True)
    model.train()
    for ep in range(epochs):
        total = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), xb)
            loss.backward()
            opt.step()
            total += float(loss)
        print(f"[autoencoder] epoch {ep + 1}/{epochs} loss {total / len(loader):.5f}")

    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X).to(device)
        err = torch.mean((model(xt) - xt) ** 2, dim=1).cpu().numpy()
    threshold = float(err.mean() + 3 * err.std())

    out = settings.artifacts_root
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "autoencoder.pt")
    (out / "autoencoder.json").write_text(
        json.dumps(
            {"in_dim": X.shape[1], "hidden": hidden, "bottleneck": bottleneck, "threshold": threshold, "metrics": CANONICAL_METRICS}
        )
    )
    print(f"[autoencoder] saved to {out} (threshold={threshold:.4f})")
    return threshold


def main() -> None:
    p = argparse.ArgumentParser(description="Train the metric autoencoder on RCAEval normal windows.")
    p.add_argument("--root", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--bottleneck", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    train_autoencoder(args.root, args.epochs, args.hidden, args.bottleneck, args.limit)


if __name__ == "__main__":
    main()
