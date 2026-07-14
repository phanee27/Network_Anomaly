from __future__ import annotations

import json
from typing import Optional

import numpy as np

from rca.config import Settings, get_settings
from rca.detect.common import split_windows, to_datetime
from rca.ingest.types import IncidentData
from rca.ml.features import CANONICAL_METRICS, entities_of, entity_metric_frame, zscore_by_normal
from rca.schema import AnomalySignal, Detector


class AutoencoderDetector:
    name = "autoencoder"
    modality = "metric"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._model = None
        self._meta = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch

        from rca.ml.models import MetricAutoencoder

        root = self.settings.artifacts_root
        self._meta = json.loads((root / "autoencoder.json").read_text())
        model = MetricAutoencoder(self._meta["in_dim"], self._meta.get("hidden", 16), self._meta.get("bottleneck", 4))
        model.load_state_dict(torch.load(root / "autoencoder.pt", map_location="cpu"))
        model.eval()
        self._model = model

    def detect(self, data: IncidentData) -> list[AnomalySignal]:
        if data.metrics is None or len(data.metrics) == 0:
            return []
        self._ensure_loaded()
        import torch

        metrics_list = self._meta.get("metrics", CANONICAL_METRICS)
        threshold = float(self._meta["threshold"])
        normal, abnormal = split_windows(data.metrics, data.inject_time, self.settings.normal_window_frac)
        if len(abnormal) == 0:
            return []
        w_start, w_end = to_datetime(abnormal.index[0]), to_datetime(abnormal.index[-1])
        signals: list[AnomalySignal] = []
        for entity in entities_of(data.metrics):
            _, z_ab = zscore_by_normal(
                entity_metric_frame(data.metrics, entity, metrics_list), data.inject_time, self.settings.normal_window_frac
            )
            if len(z_ab) == 0:
                continue
            x = torch.tensor(np.nan_to_num(z_ab.to_numpy(dtype=float)), dtype=torch.float)
            with torch.no_grad():
                recon = self._model(x)
                err = torch.mean((recon - x) ** 2, dim=1).cpu().numpy()
            score = float(np.mean(err))
            if score > threshold:
                signals.append(
                    AnomalySignal(
                        incident_id=data.incident_id,
                        entity_id=entity,
                        detector=Detector.autoencoder,
                        window_start=w_start,
                        window_end=w_end,
                        score=float(score / threshold),
                        indicator=f"autoencoder reconstruction error {score:.2f} above baseline {threshold:.2f}",
                    )
                )
        return signals
