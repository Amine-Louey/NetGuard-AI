"""NetGuard AI — inference, analytics and evaluation for the NSL-KDD NIDS.

The package deliberately separates:
- artifact loading (:mod:`nids.artifacts`),
- input validation (:mod:`nids.validation`),
- preprocessing and prediction (:mod:`nids.inference`),
- prediction statistics (:mod:`nids.analytics`),
- saved experimental evaluation (:mod:`nids.evaluation`).

The Streamlit application only orchestrates these functions; it contains no
machine-learning logic of its own.
"""

from nids.analytics import AttackStatistics, build_attack_statistics
from nids.artifacts import Artifacts, load_artifacts
from nids.errors import (
    ArtifactError,
    InputValidationError,
    NIDSError,
    PredictionError,
)
from nids.inference import PredictionResult, predict_flows, predict_single_flow

__all__ = [
    "Artifacts",
    "AttackStatistics",
    "ArtifactError",
    "InputValidationError",
    "NIDSError",
    "PredictionError",
    "PredictionResult",
    "build_attack_statistics",
    "load_artifacts",
    "predict_flows",
    "predict_single_flow",
]
