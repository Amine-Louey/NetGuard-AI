"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nids.artifacts import load_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def artifacts():
    return load_artifacts(PROJECT_ROOT / "models")


@pytest.fixture(scope="session")
def expected_frame() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "expected_predictions.csv")


@pytest.fixture(scope="session")
def raw_flows(expected_frame, artifacts) -> pd.DataFrame:
    return expected_frame[artifacts.feature_columns].copy()


@pytest.fixture(scope="session")
def demo_sample() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "data" / "samples" / "nsl_kdd_sample.csv")


@pytest.fixture()
def valid_one_row(raw_flows) -> pd.DataFrame:
    return raw_flows.iloc[[0]].copy()


@pytest.fixture()
def valid_many_rows(raw_flows) -> pd.DataFrame:
    return raw_flows.iloc[:20].copy()
