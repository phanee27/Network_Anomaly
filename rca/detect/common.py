from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from rca.config import Settings
from rca.schema import AnomalySignal, Detector

METRIC_SEP = "::"


def metric_col(entity: str, metric: str) -> str:
    return f"{entity}{METRIC_SEP}{metric}"


def parse_metric_col(col: str) -> tuple[str, str]:
    if METRIC_SEP in col:
        entity, metric = col.split(METRIC_SEP, 1)
        return entity, metric
    return col, col


def entities_in(df: pd.DataFrame) -> list[str]:
    seen: list[str] = []
    for col in df.columns:
        ent, _ = parse_metric_col(str(col))
        if ent not in seen:
            seen.append(ent)
    return seen


def to_datetime(x: object) -> datetime:
    return x.to_pydatetime() if hasattr(x, "to_pydatetime") else x


def split_windows(
    df: pd.DataFrame, inject_time: Optional[datetime], normal_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or len(df) == 0:
        return df, df
    if inject_time is not None:
        ts = pd.Timestamp(inject_time)
        idx_tz = getattr(df.index, "tz", None)
        if ts.tzinfo is None and idx_tz is not None:
            ts = ts.tz_localize(idx_tz)
        elif ts.tzinfo is not None and idx_tz is None:
            ts = ts.tz_localize(None)
        normal = df[df.index < ts]
        abnormal = df[df.index >= ts]
        if len(normal) >= 3 and len(abnormal) >= 3:
            return normal, abnormal
    cut = max(1, int(len(df) * normal_frac))
    return df.iloc[:cut], df.iloc[cut:]


def first_deviation(normal_s: pd.Series, abnormal_s: pd.Series, z: float) -> Optional[datetime]:
    mu = float(normal_s.mean())
    sd = float(normal_s.std(ddof=0))
    if not np.isfinite(mu):
        return None
    threshold = abs(z) * (sd if sd > 0 else 1.0)
    dev = (abnormal_s - mu).abs()
    hits = abnormal_s.index[dev > threshold]
    if len(hits) > 0:
        return to_datetime(hits[0])
    return None


def two_sample_shift_signals(
    frame: pd.DataFrame,
    incident_id: str,
    inject_time: Optional[datetime],
    detector: Detector,
    settings: Settings,
    indicator: str,
) -> list[AnomalySignal]:
    if frame is None or len(frame) == 0:
        return []
    normal, abnormal = split_windows(frame, inject_time, settings.normal_window_frac)
    if len(normal) < 3 or len(abnormal) < 3:
        return []
    alpha = settings.eps_diag_alpha
    z = settings.anomaly_zscore
    best: dict[str, tuple[float, str, Optional[datetime]]] = {}
    for col in frame.columns:
        n_series = pd.to_numeric(normal[col], errors="coerce").dropna()
        a_series = pd.to_numeric(abnormal[col], errors="coerce").dropna()
        n = n_series.to_numpy()
        a = a_series.to_numpy()
        if len(n) < 3 or len(a) < 3:
            continue
        if np.std(n) == 0 and np.std(a) == 0:
            continue
        try:
            _, p = stats.ks_2samp(n, a)
        except ValueError:
            continue
        if p >= alpha:
            continue
        mu, sd = float(np.mean(n)), float(np.std(n))
        shift = abs(float(np.mean(a)) - mu) / (sd if sd > 0 else 1.0)
        if shift < z:
            continue
        entity, metric = parse_metric_col(str(col))
        onset = first_deviation(n_series, a_series, z)
        cur = best.get(entity)
        if cur is None or shift > cur[0]:
            best[entity] = (shift, metric, onset)
    signals: list[AnomalySignal] = []
    w_end = to_datetime(abnormal.index[-1])
    for entity, (shift, metric, onset) in best.items():
        w_start = onset or to_datetime(abnormal.index[0])
        signals.append(
            AnomalySignal(
                incident_id=incident_id,
                entity_id=entity,
                detector=detector,
                window_start=w_start,
                window_end=w_end,
                score=float(shift),
                indicator=indicator.format(metric=metric, shift=shift),
            )
        )
    return signals
