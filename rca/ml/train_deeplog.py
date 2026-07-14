from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rca.config import get_settings
from rca.eval import find_cases

SERVICE_COL = "container_name"
MESSAGE_COL = "message"
TIME_COL = "timestamp"


def _read_inject_ns(case: Path) -> float:
    path = case / "inject_time.txt"
    if not path.exists():
        return float("inf")
    lines = path.read_text().strip().splitlines()
    try:
        return float(int(float(lines[0].strip()))) * 1e9
    except Exception:
        return float("inf")


def _build_sequences(cases, miner, max_lines):
    sequences = []
    want = {SERVICE_COL, MESSAGE_COL, TIME_COL}
    for case in cases:
        logs = Path(case) / "logs.csv"
        if not logs.exists():
            continue
        try:
            df = pd.read_csv(logs, usecols=lambda c: c in want, nrows=max_lines)
        except Exception:
            continue
        if MESSAGE_COL not in df.columns or SERVICE_COL not in df.columns:
            continue
        df = df.dropna(subset=[MESSAGE_COL])
        if TIME_COL in df.columns:
            inject_ns = _read_inject_ns(Path(case))
            df = df[pd.to_numeric(df[TIME_COL], errors="coerce") < inject_ns]
            df = df.sort_values(TIME_COL)
        for _, group in df.groupby(SERVICE_COL):
            seq = [miner.add_log_message(str(m))["cluster_id"] - 1 for m in group[MESSAGE_COL]]
            if len(seq) > 1:
                sequences.append(seq)
    return sequences


def train_deeplog(
    root: str, epochs: int = 50, window: int = 10, topg: int = 9, hidden: int = 64, layers: int = 2,
    max_lines: int = 100000, limit: int = 0
) -> None:
    import torch
    from drain3.file_persistence import FilePersistence
    from drain3.template_miner import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
    from torch.utils.data import DataLoader, TensorDataset

    from rca.ingest.logs import default_masking
    from rca.ml.models import DeepLog
    from rca.ml.utils import get_device

    settings = get_settings()
    out = settings.artifacts_root
    out.mkdir(parents=True, exist_ok=True)

    config = TemplateMinerConfig()
    config.drain_sim_th = settings.drain_sim_th
    config.drain_depth = settings.drain_depth
    config.drain_max_children = settings.drain_max_children
    config.masking_instructions = default_masking()
    miner = TemplateMiner(persistence_handler=FilePersistence(str(out / "drain3_state.bin")), config=config)

    cases = find_cases(root)
    if limit:
        cases = cases[:limit]
    sequences = _build_sequences(cases, miner, max_lines)
    num_keys = len(list(miner.drain.clusters))
    miner.save_state("deeplog training complete")
    if num_keys == 0:
        raise SystemExit("no log templates learned")

    xs, ys = [], []
    for seq in sequences:
        for i in range(len(seq) - window):
            xs.append(seq[i : i + window])
            ys.append(seq[i + window])
    if not xs:
        raise SystemExit("no training windows built (need sequences longer than the window)")

    device = get_device()
    x = torch.tensor(xs, dtype=torch.float).unsqueeze(-1)
    y = torch.tensor(ys, dtype=torch.long)
    model = DeepLog(num_keys, hidden, layers).to(device)
    opt = torch.optim.Adam(model.parameters())
    crit = torch.nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(x, y), batch_size=2048, shuffle=True)
    model.train()
    for ep in range(epochs):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss)
        print(f"[deeplog] epoch {ep + 1}/{epochs} loss {total / len(loader):.5f}")

    torch.save(model.state_dict(), out / "deeplog.pt")
    (out / "deeplog.json").write_text(
        json.dumps(
            {
                "num_keys": num_keys, "window": window, "topg": topg, "hidden": hidden, "layers": layers,
                "service_col": SERVICE_COL, "message_col": MESSAGE_COL, "time_col": TIME_COL,
                "max_lines": max_lines, "anom_frac": 0.1,
            }
        )
    )
    print(f"[deeplog] saved to {out} (num_keys={num_keys})")


def main() -> None:
    p = argparse.ArgumentParser(description="Train DeepLog on RCAEval normal-window logs (global Drain3 vocabulary).")
    p.add_argument("--root", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--topg", type=int, default=9)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--max-lines", type=int, default=100000)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    train_deeplog(args.root, args.epochs, args.window, args.topg, args.hidden, args.layers, args.max_lines, args.limit)


if __name__ == "__main__":
    main()
