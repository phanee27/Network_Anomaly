from __future__ import annotations

import argparse

from rca.config import get_settings
from rca.ingest.rcaeval import load_rcaeval_case
from rca.pipeline import analyze_incident


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RCA pipeline on one RCAEval case, offline.")
    parser.add_argument("case", help="path to an RCAEval case dir (.../<service>_<fault>/<instance>)")
    parser.add_argument("--narrate", action="store_true", help="use the local LLM narrator (needs Ollama)")
    parser.add_argument("--persist", action="store_true", help="persist to Postgres (needs the DB)")
    parser.add_argument("--drain3", action="store_true", help="parse raw logs with Drain3 instead of logts.csv")
    args = parser.parse_args()

    settings = get_settings()
    data = load_rcaeval_case(args.case, use_drain3=args.drain3, settings=settings)
    incident = analyze_incident(data, settings=settings, persist=args.persist, narrate=args.narrate)
    print(incident.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
