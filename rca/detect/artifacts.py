from __future__ import annotations

from typing import Optional

from rca.config import Settings, get_settings
from rca.detect.base import Detector as DetectorProtocol


def load_artifact_detectors(settings: Optional[Settings] = None) -> list[DetectorProtocol]:
    settings = settings or get_settings()
    root = settings.artifacts_root
    out: list[DetectorProtocol] = []

    if (root / "lightgbm_unsw.joblib").exists():
        try:
            from rca.detect.flow_lightgbm import LightGBMFlowDetector

            out.append(LightGBMFlowDetector(settings))
        except Exception:
            pass

    if settings.use_autoencoder and (root / "autoencoder.pt").exists():
        try:
            from rca.detect.dl_autoencoder import AutoencoderDetector

            out.append(AutoencoderDetector(settings))
        except Exception:
            pass

    if settings.use_deeplog and (root / "deeplog.pt").exists():
        try:
            from rca.detect.dl_deeplog import DeepLogDetector

            out.append(DeepLogDetector(settings))
        except Exception:
            pass

    return out
