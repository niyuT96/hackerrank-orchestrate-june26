"""Sprint 4 evaluation and submission validation tests."""

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

SPEC = importlib.util.spec_from_file_location(
    "claim_evaluation", CODE_DIR / "evaluation" / "main.py"
)
assert SPEC and SPEC.loader
evaluation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluation
SPEC.loader.exec_module(evaluation)


class MetricTests(unittest.TestCase):
    def test_set_metrics_reports_micro_scores(self) -> None:
        metrics = evaluation.set_metrics(
            [{"a", "b"}, {"c"}],
            [{"a"}, {"c", "d"}],
        )
        self.assertAlmostEqual(metrics.precision, 2 / 3)
        self.assertAlmostEqual(metrics.recall, 2 / 3)
        self.assertEqual(metrics.exact_match, 0.0)

    def test_split_is_stable_and_order_independent(self) -> None:
        values = {
            user_id: evaluation.deterministic_split(user_id)
            for user_id in ("alpha", "beta", "gamma")
        }
        reversed_values = {
            user_id: evaluation.deterministic_split(user_id)
            for user_id in ("gamma", "beta", "alpha")
        }
        self.assertEqual(values, reversed_values)


class OutputValidationTests(unittest.TestCase):
    def test_sample_output_passes_schema_validation(self) -> None:
        result = evaluation.validate_output(
            REPO_ROOT / "dataset" / "sample_claims.csv",
            REPO_ROOT / "sample_output.csv",
            expected_rows=20,
        )
        self.assertTrue(result["valid"], result["errors"])

    def test_changed_input_column_is_rejected(self) -> None:
        rows = evaluation._read_csv(
            REPO_ROOT / "dataset" / "sample_claims.csv"
        )
        rows[0]["user_claim"] = "changed"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=evaluation.OUTPUT_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            result = evaluation.validate_output(
                REPO_ROOT / "dataset" / "sample_claims.csv",
                path,
                expected_rows=20,
            )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("input field changed" in item for item in result["errors"])
        )


if __name__ == "__main__":
    unittest.main()
