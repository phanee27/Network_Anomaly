from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest

from rca.config import Settings, get_settings
from rca.detect.common import entities_in, parse_metric_col, split_windows, to_datetime
from rca.ingest.types import IncidentData
from rca.schema import AnomalySignal, Detector


class IsolationForestDetector:
    name = "isolation_forest"
    modality = "metric"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def detect(self, data: IncidentData) -> list[AnomalySignal]:
        df = data.metrics
        if df is None or len(df) == 0:
            return []
        normal, abnormal = split_windows(df, data.inject_time, self.settings.normal_window_frac)
        if len(normal) < 5 or len(abnormal) < 3:
            return []
        signals: list[AnomalySignal] = []
        w_start = to_datetime(abnormal.index[0])
        w_end = to_datetime(abnormal.index[-1])
        for entity in entities_in(df):
            cols = [c for c in df.columns if parse_metric_col(str(c))[0] == entity]
            if not cols:
                continue
            xn = np.nan_to_num(normal[cols].apply(_numeric).to_numpy(dtype=float))
            xa = np.nan_to_num(abnormal[cols].apply(_numeric).to_numpy(dtype=float))
            if xn.shape[0] < 5 or np.allclose(xn.std(axis=0), 0):
                continue
            model = IsolationForest(
                n_estimators=self.settings.iforest_n_estimators,
                contamination=self.settings.iforest_contamination,
                random_state=0,
            )
            model.fit(xn)
            frac = float(np.mean(model.predict(xa) == -1))
            if frac <= self.settings.iforest_contamination:
                continue
            shift = float(np.mean(-model.score_samples(xa)) - np.mean(-model.score_samples(xn)))
            signals.append(
                AnomalySignal(
                    incident_id=data.incident_id,
                    entity_id=entity,
                    detector=Detector.iforest,
                    window_start=w_start,
                    window_end=w_end,
                    score=max(shift, frac),
                    indicator=f"multivariate outlier fraction {frac:.2f}",
                )
            )
        return signals


def _numeric(series):
    import pandas as pd

    return pd.to_numeric(series, errors="coerce")
