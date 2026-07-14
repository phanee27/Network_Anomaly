from __future__ import annotations

import json
from typing import Optional

from rca.config import Settings, get_settings
from rca.ingest.types import IncidentData
from rca.schema import AnomalySignal, Detector


class LightGBMFlowDetector:
    name = "lightgbm_flow"
    modality = "flow"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._model = None
        self._meta = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import joblib

        root = self.settings.artifacts_root
        self._model = joblib.load(root / "lightgbm_unsw.joblib")
        self._meta = json.loads((root / "lightgbm_unsw.json").read_text())

    def detect(self, data: IncidentData) -> list[AnomalySignal]:
        flows = data.flows
        if flows is None or len(flows) == 0 or "entity_id" not in getattr(flows, "columns", []):
            return []
        self._ensure_loaded()

        features = [c for c in self._meta["features"] if c in flows.columns]
        normal_label = self._meta.get("normal_label", "Normal")
        classes = self._meta.get("classes")
        X = flows[features].copy()
        for c in self._meta.get("categorical", []):
            if c in X.columns:
                X[c] = X[c].astype("category")
        preds = self._model.predict(X)
        labels = [classes[int(p)] if classes and int(p) < len(classes) else str(p) for p in preds]
        tagged = flows.assign(_pred=labels)

        signals: list[AnomalySignal] = []
        for entity, group in tagged.groupby("entity_id"):
            frac = float((group["_pred"] != normal_label).mean())
            if frac > self.settings.iforest_contamination:
                signals.append(
                    AnomalySignal(
                        incident_id=data.incident_id,
                        entity_id=str(entity),
                        detector=Detector.lightgbm,
                        window_start=data.inject_time,
                        window_end=data.inject_time,
                        score=frac,
                        indicator=f"network-flow classifier flagged {frac:.0%} of flows as malicious",
                    )
                )
        return signals
