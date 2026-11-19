"""Generate the demonstration CSV and fixed inference fixtures.

The sample is drawn from the independent test split (never from training).  It
contains only the 41 raw features: no label is exposed to the model.  The
fixture records expected predictions so the test suite can detect any future
change in artifact loading or feature ordering.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nids import config  # noqa: E402
from nids.artifacts import load_artifacts  # noqa: E402
from nids.dataset import reproduce_split  # noqa: E402
from nids.inference import predict_flows  # noqa: E402

DEMO_PER_CLASS = {"Normal": 120, "DoS": 85, "Probe": 25, "R2L": 15, "U2R": 5}
FIXTURE_PER_CLASS = {"Normal": 20, "DoS": 20, "Probe": 20, "R2L": 20, "U2R": 12}


def _sample_by_class(frame: pd.DataFrame, labels: pd.Series, sizes: dict[str, int]) -> pd.DataFrame:
    chunks = []
    for class_name, size in sizes.items():
        subset = frame[labels == class_name]
        take = min(size, len(subset))
        chunks.append(subset.sample(n=take, random_state=42))
    return pd.concat(chunks).sort_index()


def main() -> int:
    split = reproduce_split(PROJECT_ROOT / "data" / "raw")
    artifacts = load_artifacts(config.MODELS_DIR)

    samples_dir = PROJECT_ROOT / "data" / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    demo_raw = _sample_by_class(split.X_test, split.y_test, DEMO_PER_CLASS)
    demo_path = samples_dir / "nsl_kdd_sample.csv"
    demo_raw.to_csv(demo_path, index=False)
    print(f"Échantillon de démonstration : {demo_path} ({len(demo_raw)} lignes)")

    fixture_raw = _sample_by_class(split.X_test, split.y_test, FIXTURE_PER_CLASS)
    prediction = predict_flows(fixture_raw, artifacts)
    fixture = fixture_raw.reset_index(drop=True).copy()
    fixture["expected_class"] = prediction.results["predicted_class"].to_numpy()
    fixture["expected_confidence"] = prediction.results["confidence"].to_numpy()
    for class_name in artifacts.class_names:
        fixture[f"p_{class_name}"] = prediction.probabilities[class_name].to_numpy()
    fixture["true_class"] = split.y_test.loc[fixture_raw.index].to_numpy()

    fixtures_dir = PROJECT_ROOT / "tests" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = fixtures_dir / "expected_predictions.csv"
    fixture.to_csv(fixture_path, index=False)
    print(f"Fixture d'inférence : {fixture_path} ({len(fixture)} lignes)")
    print("Distribution attendue :", prediction.results["predicted_class"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
