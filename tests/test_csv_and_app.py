"""CSV ingestion and Streamlit smoke tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nids.errors import InputValidationError
from nids.inference import predict_flows
from nids.validation import read_csv_bytes

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_bom_csv_is_read(demo_sample):
    content = demo_sample.head(3).to_csv(index=False).encode("utf-8-sig")
    frame = read_csv_bytes(content, "bom.csv")
    assert list(frame.columns) == list(demo_sample.columns)


def test_csv_with_index_column_is_handled(demo_sample, artifacts):
    content = demo_sample.head(3).to_csv(index=True).encode("utf-8")
    frame = read_csv_bytes(content, "with_index.csv")
    prediction = predict_flows(frame, artifacts)
    assert len(prediction) == 3
    assert any("supplémentaires" in warning for warning in prediction.warnings)


def test_sample_csv_is_valid_and_predictable(demo_sample, artifacts):
    prediction = predict_flows(demo_sample, artifacts)
    assert len(prediction) == 250
    assert prediction.results["predicted_class"].isin(artifacts.class_names).all()


def test_semicolon_csv_is_not_silently_accepted(artifacts):
    content = b"duration;protocol_type;service\n0;tcp;http\n"
    frame = read_csv_bytes(content, "semicolon.csv")
    with pytest.raises(InputValidationError):
        predict_flows(frame, artifacts)


def test_streamlit_app_starts_and_pages_run():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PROJECT_ROOT / "app" / "app.py"), default_timeout=60)
    app.run()
    assert not app.exception
    assert app.radio[0].options == ["Accueil", "Analyse", "Attack Analytics", "Model Evaluation"]

    for page in ["Analyse", "Attack Analytics", "Model Evaluation"]:
        app.radio[0].set_value(page).run()
        assert not app.exception, f"La page {page} a levé une exception"
