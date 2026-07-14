from __future__ import annotations

import argparse
import json

import pandas as pd

from rca.config import get_settings

UNSW_FEATURES = [
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb",
    "dwin", "tcprtt", "synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
]
UNSW_CATEGORICAL = ["proto", "service", "state"]


def train_lightgbm(train_csv: str, n_estimators: int = 400) -> None:
    import joblib
    from lightgbm import LGBMClassifier
    from sklearn.preprocessing import LabelEncoder

    settings = get_settings()
    df = pd.read_csv(train_csv)
    features = [c for c in UNSW_FEATURES if c in df.columns]
    X = df[features].copy()
    for c in UNSW_CATEGORICAL:
        if c in X.columns:
            X[c] = X[c].astype("category")
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["attack_cat"].astype(str))

    clf = LGBMClassifier(
        objective="multiclass", class_weight="balanced", n_estimators=n_estimators, learning_rate=0.05, num_leaves=63
    )
    clf.fit(X, y)

    out = settings.artifacts_root
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out / "lightgbm_unsw.joblib")
    (out / "lightgbm_unsw.json").write_text(
        json.dumps(
            {"features": features, "categorical": UNSW_CATEGORICAL, "classes": list(encoder.classes_), "normal_label": "Normal"}
        )
    )
    print(f"[lightgbm] saved to {out} ({len(encoder.classes_)} classes)")


def main() -> None:
    p = argparse.ArgumentParser(description="Train the LightGBM UNSW-NB15 network-flow classifier.")
    p.add_argument("--train-csv", required=True, help="path to UNSW_NB15_training-set.csv")
    p.add_argument("--n-estimators", type=int, default=400)
    args = p.parse_args()
    train_lightgbm(args.train_csv, args.n_estimators)


if __name__ == "__main__":
    main()
