"""Project-wide paths and constants.

Paths can be overridden with the ``NIDS_MODELS_DIR`` and ``NIDS_REPORTS_DIR``
environment variables, which is useful for deployment platforms.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("NIDS_MODELS_DIR", PROJECT_ROOT / "models")).resolve()
REPORTS_DIR = Path(os.environ.get("NIDS_REPORTS_DIR", PROJECT_ROOT / "reports")).resolve()
DATA_DIR = PROJECT_ROOT / "data"

FEATURE_CONFIG_PATH = "feature_columns.json"
PREPROCESSOR_PATH = "preprocessor.joblib"
LABEL_ENCODER_PATH = "label_encoder.joblib"
RF_MODEL_PATH = "rf_model.joblib"
METRICS_SUMMARY_PATH = "metrics_summary.json"
MODEL_EVALUATION_PATH = "model_evaluation.json"

# Presentation order used by the interface.  It is not the model's internal
# encoding order: predictions are always mapped through the saved LabelEncoder.
CLASS_ORDER = ("Normal", "DoS", "Probe", "R2L", "U2R")
ATTACK_CLASSES = ("DoS", "Probe", "R2L", "U2R")

# Metadata columns that a labelled NSL-KDD export may legitimately contain.
# They are not model inputs and are ignored (with an explicit warning).
IGNORED_LABEL_COLUMNS = (
    "attack_type",
    "attack_category",
    "difficulty_level",
    "label",
    "class",
    "target",
)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_ROWS = 100_000
