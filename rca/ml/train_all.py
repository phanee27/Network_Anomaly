from __future__ import annotations

import argparse

from rca.config import get_settings


def main() -> None:
    settings = get_settings()
    p = argparse.ArgumentParser(description="Train the deep-learning suite (run inside the L40s window).")
    p.add_argument("--rcaeval-root", required=True, help="e.g. data/rcaeval/RE2/RE2-OB")
    p.add_argument("--unsw-csv", default=None, help="path to UNSW_NB15_training-set.csv (for the flow classifier)")
    p.add_argument("--models", default="autoencoder,gnn,deeplog,lightgbm")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    wanted = {m.strip() for m in args.models.split(",") if m.strip()}
    root = settings.artifacts_root

    if "autoencoder" in wanted and (args.force or not (root / "autoencoder.pt").exists()):
        from rca.ml.train_autoencoder import train_autoencoder

        train_autoencoder(args.rcaeval_root, epochs=args.epochs, limit=args.limit)

    if "gnn" in wanted and (args.force or not (root / "gnn.pt").exists()):
        from rca.ml.train_gnn import train_gnn

        train_gnn(args.rcaeval_root, epochs=args.epochs, limit=args.limit)

    if "deeplog" in wanted and (args.force or not (root / "deeplog.pt").exists()):
        from rca.ml.train_deeplog import train_deeplog

        train_deeplog(args.rcaeval_root, epochs=args.epochs, limit=args.limit)

    if "lightgbm" in wanted and args.unsw_csv and (args.force or not (root / "lightgbm_unsw.joblib").exists()):
        from rca.ml.train_lightgbm import train_lightgbm

        train_lightgbm(args.unsw_csv)

    print(f"training complete; artifacts in {root}")


if __name__ == "__main__":
    main()
