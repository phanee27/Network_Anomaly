from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from rca.detect.common import metric_col, parse_metric_col, split_windows
from rca.schema import AnomalySignal, Detector

CANONICAL_METRICS = ["cpu", "mem", "diskio", "socket", "workload", "error", "latency"]
DETECTOR_ORDER = [Detector.eps_diag, Detector.iforest, Detector.drain3, Detector.trace]
NODE_FEATURE_DIM = len(DETECTOR_ORDER) + 2


def entities_of(metrics: pd.DataFrame) -> list[str]:
    seen: list[str] = []
    for c in metrics.columns:
        e, _ = parse_metric_col(str(c))
        if e not in seen:
            seen.append(e)
    return seen


def entity_metric_frame(metrics: pd.DataFrame, entity: str, metrics_list: list[str]) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    for m in metrics_list:
        col = metric_col(entity, m)
        if col in metrics.columns:
            data[m] = pd.to_numeric(metrics[col], errors="coerce")
        else:
            data[m] = pd.Series(0.0, index=metrics.index)
    return pd.DataFrame(data, index=metrics.index).fillna(0.0)


def zscore_by_normal(
    frame: pd.DataFrame, inject_time, normal_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    normal, abnormal = split_windows(frame, inject_time, normal_frac)
    mu = normal.mean()
    sd = normal.std(ddof=0).replace(0.0, 1.0)
    z = (frame - mu) / sd
    return z.loc[normal.index], z.loc[abnormal.index]


def autoencoder_normal_matrix(
    metrics: Optional[pd.DataFrame], inject_time, normal_frac: float, metrics_list: list[str] = CANONICAL_METRICS
) -> np.ndarray:
    if metrics is None or len(metrics) == 0:
        return np.empty((0, len(metrics_list)))
    rows: list[np.ndarray] = []
    for e in entities_of(metrics):
        z_normal, _ = zscore_by_normal(entity_metric_frame(metrics, e, metrics_list), inject_time, normal_frac)
        if len(z_normal):
            rows.append(z_normal.to_numpy(dtype=float))
    if not rows:
        return np.empty((0, len(metrics_list)))
    return np.nan_to_num(np.vstack(rows))


def node_feature_matrix(
    signals: list[AnomalySignal], entities: list[str], graph
) -> tuple[np.ndarray, list[str]]:
    order = list(entities)
    idx = {e: i for i, e in enumerate(order)}
    X = np.zeros((len(order), NODE_FEATURE_DIM), dtype=float)
    for s in signals:
        if s.entity_id in idx and s.detector in DETECTOR_ORDER:
            j = DETECTOR_ORDER.index(s.detector)
            X[idx[s.entity_id], j] = max(X[idx[s.entity_id], j], abs(s.score))
    base = len(DETECTOR_ORDER)
    for e in order:
        if graph.has_node(e):
            X[idx[e], base] = float(graph.g.in_degree(e))
            X[idx[e], base + 1] = float(graph.g.out_degree(e))
    return X, order
