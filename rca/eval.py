from __future__ import annotations

import argparse
import json
from pathlib import Path

from rca.config import Settings, get_settings
from rca.ingest.rcaeval import load_rcaeval_case
from rca.pipeline import analyze_incident


def find_cases(root: str | Path) -> list[Path]:
    return sorted(p.parent for p in Path(root).rglob("inject_time.txt"))


def evaluate(cases: list[Path], settings: Settings, ks: tuple[int, ...] = (1, 2, 3, 4, 5)) -> dict:
    hits = {k: 0 for k in ks}
    reciprocal = 0.0
    total = 0
    for case in cases:
        data = load_rcaeval_case(case, settings=settings)
        truth = (data.ground_truth or {}).get("service")
        if not truth:
            continue
        incident = analyze_incident(data, settings=settings, persist=False, narrate=False)
        ranked = [h.component for h in incident.ranked_hypotheses]
        total += 1
        if truth in ranked:
            reciprocal += 1.0 / (ranked.index(truth) + 1)
        for k in ks:
            if truth in ranked[:k]:
                hits[k] += 1
    if total == 0:
        return {"cases": 0}
    ac = {f"AC@{k}": round(hits[k] / total, 4) for k in ks}
    return {
        "cases": total,
        **ac,
        "Avg@5": round(sum(hits[k] for k in ks) / (len(ks) * total), 4),
        "MRR": round(reciprocal / total, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RCA accuracy (AC@k / Avg@5 / MRR) over RCAEval cases.")
    parser.add_argument("--root", required=True, help="dataset root, e.g. data/rcaeval/RE2/RE2-OB")
    parser.add_argument("--no-dl", action="store_true", help="disable deep-learning signals for an ablation baseline")
    parser.add_argument("--limit", type=int, default=0, help="cap the number of cases (0 = all)")
    args = parser.parse_args()

    settings = get_settings()
    if args.no_dl:
        settings = settings.model_copy(update={"use_autoencoder": False, "use_deeplog": False, "use_gnn": False})

    cases = find_cases(args.root)
    if args.limit:
        cases = cases[: args.limit]
    print(json.dumps(evaluate(cases, settings), indent=2))


if __name__ == "__main__":
    main()
