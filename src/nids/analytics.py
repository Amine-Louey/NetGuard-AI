"""Statistics computed exclusively from the current prediction batch."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nids import config
from nids.errors import InputValidationError

_REQUIRED_COLUMNS = ("flow_id", "predicted_class", "confidence", "status")


@dataclass(frozen=True)
class AttackStatistics:
    total: int
    normal: int
    malicious: int
    malicious_rate: float
    distribution: pd.DataFrame
    alerts: pd.DataFrame
    mean_confidence: float
    mean_alert_confidence: float | None


def build_attack_statistics(results: pd.DataFrame) -> AttackStatistics:
    """Derive every displayed statistic from ``results``.

    No value is hard-coded: totals, shares and confidence summaries are all
    computed from the prediction table produced by the model.
    """

    missing = [column for column in _REQUIRED_COLUMNS if column not in results.columns]
    if missing:
        raise InputValidationError(
            "Tableau de résultats incomplet : " + ", ".join(missing)
        )
    if results.empty:
        raise InputValidationError("Aucune prédiction disponible pour l'analyse.")

    total = int(len(results))
    normal = int((results["predicted_class"] == "Normal").sum())
    malicious = total - normal

    counts = results["predicted_class"].value_counts()
    distribution = pd.DataFrame(
        {
            "category": list(config.CLASS_ORDER),
            "count": [int(counts.get(name, 0)) for name in config.CLASS_ORDER],
        }
    )
    distribution["share"] = distribution["count"] / total

    alerts = results[results["predicted_class"] != "Normal"].copy()
    alerts = alerts.sort_values(
        by=["confidence", "flow_id"], ascending=[False, True]
    ).reset_index(drop=True)

    mean_confidence = float(results["confidence"].mean())
    mean_alert_confidence = (
        float(alerts["confidence"].mean()) if not alerts.empty else None
    )

    return AttackStatistics(
        total=total,
        normal=normal,
        malicious=malicious,
        malicious_rate=malicious / total if total else 0.0,
        distribution=distribution,
        alerts=alerts,
        mean_confidence=mean_confidence,
        mean_alert_confidence=mean_alert_confidence,
    )
