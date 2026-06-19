from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from claim_agent import (  # noqa: E402
    CompositeClaimParser,
    OUTPUT_COLUMNS,
    DataValidationError,
    LLMClaimParser,
    RuleBasedClaimParser,
    load_bundle,
    parse_claim_text,
    static_llm_client,
    write_output,
    write_prepared_claims,
)


class ClaimParserTests(unittest.TestCase):
    def test_extracts_multiple_car_parts_and_issues(self) -> None:
        parsed = parse_claim_text(
            "Customer: The door is dented and rear bumper is scratched.",
            "car",
        )
        self.assertEqual(parsed.claimed_parts, ("door", "rear_bumper"))
        self.assertEqual(parsed.claimed_issue_types, ("dent", "scratch"))

    def test_ignores_negated_laptop_parts(self) -> None:
        parsed = parse_claim_text(
            "Customer: Not the keyboard or hinge, the screen is cracked.",
            "laptop",
        )
        self.assertEqual(parsed.claimed_parts, ("screen",))
        self.assertEqual(parsed.excluded_parts, ("keyboard", "hinge"))
        self.assertEqual(parsed.claimed_issue_types, ("crack",))

    def test_extracts_package_contents_rule_intent(self) -> None:
        parsed = parse_claim_text(
            "Customer: The actual issue is that the product inside is missing.",
            "package",
        )
        self.assertIn("contents", parsed.claimed_parts)
        self.assertIn("missing_part", parsed.claimed_issue_types)

    def test_extracts_spanish_car_damage(self) -> None:
        parsed = parse_claim_text(
            "Cliente: El parachoques trasero esta danado.",
            "car",
        )
        self.assertEqual(parsed.claimed_parts, ("rear_bumper",))
        self.assertEqual(parsed.claimed_issue_types, ("broken_part",))

    def test_uses_unknown_instead_of_guessing(self) -> None:
        parsed = parse_claim_text(
            "Customer: Something seems wrong, please review it.",
            "car",
        )
        self.assertEqual(parsed.claimed_parts, ("unknown",))
        self.assertEqual(parsed.claimed_issue_types, ("unknown",))
        self.assertEqual(parsed.claimed_severity, "unknown")
        self.assertIn(
            "parser_uncertainty:claimed_parts",
            parsed.parser_diagnostics,
        )


class CompositeParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset_dir = REPO_ROOT / "dataset"
        cls.claim = load_bundle(cls.dataset_dir).claims[0]

    def test_selects_validated_llm_on_disagreement(self) -> None:
        response = {
            self.claim.user_id: {
                "claimed_parts": ["headlight"],
                "claimed_issue_types": ["broken_part"],
                "claimed_severity": "unknown",
                "included_parts": ["headlight"],
                "excluded_parts": [],
                "evidence_quotes": ["headlight"],
                "parser_confidence": 0.9,
                "parser_diagnostics": [],
            }
        }
        parser = CompositeClaimParser(
            rule_parser=RuleBasedClaimParser(),
            llm_parser=LLMClaimParser(static_llm_client(response)),
        )
        decision = parser.parse(self.claim)
        self.assertEqual(decision.selected.parser_name, "llm")
        self.assertIn("parser_disagreement", decision.diagnostics)

    def test_falls_back_when_llm_quote_is_not_in_source(self) -> None:
        response = {
            self.claim.user_id: {
                "claimed_parts": ["hood"],
                "claimed_issue_types": ["dent"],
                "claimed_severity": "high",
                "included_parts": ["hood"],
                "excluded_parts": [],
                "evidence_quotes": ["invented hood quote"],
                "parser_confidence": 0.99,
                "parser_diagnostics": [],
            }
        }
        parser = CompositeClaimParser(
            llm_parser=LLMClaimParser(
                static_llm_client(response),
                max_attempts=1,
            )
        )
        decision = parser.parse(self.claim)
        self.assertEqual(decision.selected.parser_name, "rule")
        self.assertIn("llm_failed_fallback_to_rule", decision.diagnostics)


class DatasetPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset_dir = REPO_ROOT / "dataset"

    def test_loads_and_prepares_all_test_claims(self) -> None:
        bundle = load_bundle(self.dataset_dir)
        self.assertEqual(len(bundle.claims), 44)
        self.assertEqual(len(bundle.prepared_claims), 44)
        first_ids = {item.requirement_id for item in bundle.prepared_claims[0].requirements}
        self.assertIn("REQ_GENERAL_OBJECT_PART", first_ids)
        self.assertIn("REQ_GENERAL_MULTI_IMAGE", first_ids)
        self.assertIn("REQ_CAR_BODY_PANEL", first_ids)
        self.assertIn("REQ_CAR_GLASS_LIGHT_MIRROR", first_ids)
        first = bundle.prepared_claims[0]
        self.assertEqual(
            first.claim.images[0].path,
            "images/test/case_001/img_1.jpg",
        )
        self.assertTrue(first.requirements[0].minimum_image_evidence)

    def test_writes_schema_valid_sprint1_output(self) -> None:
        bundle = load_bundle(self.dataset_dir)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.csv"
            count = write_output(bundle.prepared_claims, output_path)
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
                self.assertEqual(tuple(rows[0].keys()), OUTPUT_COLUMNS)
            self.assertEqual(count, 44)
            self.assertEqual(len(rows), 44)
            self.assertTrue(all(row["claim_status"] == "not_enough_information" for row in rows))
            self.assertTrue(all(row["supporting_image_ids"] == "none" for row in rows))
            self.assertTrue(all(row["object_part"] == "unknown" for row in rows))
            self.assertTrue(all(row["valid_image"] == "false" for row in rows))

    def test_writes_lossless_prepared_claim_json(self) -> None:
        bundle = load_bundle(self.dataset_dir)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "prepared.json"
            count = write_prepared_claims(bundle.prepared_claims, output_path)
            payload = __import__("json").loads(
                output_path.read_text(encoding="utf-8")
            )
            self.assertEqual(count, 44)
            self.assertEqual(payload[0]["source_file"], "claims.csv")
            self.assertEqual(
                payload[0]["images"][0],
                {
                    "image_id": "img_1",
                    "path": "images/test/case_001/img_1.jpg",
                },
            )
            self.assertIn(
                "minimum_image_evidence",
                payload[0]["requirements"][0],
            )
            self.assertIn("rule_result", payload[0]["claim_intent"])
            self.assertIn("history_summary", payload[0]["history"])

    def test_missing_claim_file_has_clear_error(self) -> None:
        with self.assertRaisesRegex(DataValidationError, "does not exist"):
            load_bundle(self.dataset_dir, "missing.csv")

    def test_duplicate_claim_rows_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            with (self.dataset_dir / "claims.csv").open(
                "r", encoding="utf-8", newline=""
            ) as source_handle:
                source_rows = list(csv.DictReader(source_handle))
            with (dataset_dir / "claims.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=source_rows[0].keys())
                writer.writeheader()
                writer.writerow(source_rows[0])
                writer.writerow(source_rows[0])
            for filename in ("user_history.csv", "evidence_requirements.csv"):
                (dataset_dir / filename).write_bytes(
                    (self.dataset_dir / filename).read_bytes()
                )
            with self.assertRaisesRegex(DataValidationError, "duplicate claim row"):
                load_bundle(dataset_dir, require_images=False)


if __name__ == "__main__":
    unittest.main()
