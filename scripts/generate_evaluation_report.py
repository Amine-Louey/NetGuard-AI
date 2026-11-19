"""Rebuild, verify and save the experimental evaluation report.

Usage:
    python scripts/generate_evaluation_report.py [--no-neural]

Requires the raw NSL-KDD files in ``data/raw`` and the frozen artifacts in
``models/``.  The script fails if the recomputed Random Forest test metrics do
not match the Week 3 export.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nids import config  # noqa: E402
from nids.reproducibility import (  # noqa: E402
    build_evaluation_report,
    save_evaluation_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-neural",
        action="store_true",
        help="Ne pas recharger les modèles Keras (plus rapide, pas de détail MLP/CNN).",
    )
    args = parser.parse_args()

    report = build_evaluation_report(
        raw_dir=PROJECT_ROOT / "data" / "raw",
        models_dir=config.MODELS_DIR,
        include_neural=not args.no_neural,
    )
    path = save_evaluation_report(report, config.REPORTS_DIR)
    reproduction = report["reproduction"]
    print(f"Rapport écrit : {path}")
    print(
        "Random Forest — écart max test : "
        f"{reproduction['random_forest_max_abs_difference_test']:.2e} ; "
        "validation : "
        f"{reproduction['random_forest_max_abs_difference_validation']:.2e}"
    )
    for note in reproduction["neural_notes"]:
        print(f" - {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
