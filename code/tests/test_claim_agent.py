from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from claim_agent import (  # noqa: E402
    CompositeClaimParser,
    OUTPUT_COLUMNS,
    DataValidationError,
    LLMClaimParser,
    OpenAIResponsesClaimClient,
    RuleBasedClaimParser,
    claim_parser_schema,
    load_claim_lexicon,
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

    def test_keeps_generic_spanish_damage_unknown(self) -> None:
        parsed = parse_claim_text(
            "Cliente: El parachoques trasero esta danado.",
            "car",
        )
        self.assertEqual(parsed.claimed_parts, ("rear_bumper",))
        self.assertEqual(parsed.claimed_issue_types, ("unknown",))

    def test_does_not_map_generic_mark_to_stain(self) -> None:
        parsed = parse_claim_text(
            "Customer: There is a mark on the hood. "
            "Customer: The hood has a scratch.",
            "car",
        )
        self.assertEqual(parsed.claimed_parts, ("hood",))
        self.assertEqual(parsed.claimed_issue_types, ("scratch",))

    def test_rule_parser_defers_unmapped_code_switched_damage_to_unknown(self) -> None:
        parsed = parse_claim_text(
            "Customer: Seal wali side phati hui thi.",
            "package",
        )
        self.assertEqual(parsed.claimed_parts, ("seal",))
        self.assertEqual(parsed.claimed_issue_types, ("unknown",))

    def test_shattered_is_damage_type_not_claimed_severity(self) -> None:
        parsed = parse_claim_text(
            "Customer: The screen looks shattered.",
            "laptop",
        )
        self.assertEqual(parsed.claimed_issue_types, ("glass_shatter",))
        self.assertEqual(parsed.claimed_severity, "unknown")

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
                "claimed_issue_types": ["unknown"],
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
            llm_routing="always",
        )
        decision = parser.parse(self.claim)
        self.assertEqual(decision.selected.parser_name, "llm")
        self.assertIn("parser_disagreement", decision.diagnostics)

    def test_default_routing_runs_rule_and_llm_together(self) -> None:
        response = {
            self.claim.user_id: {
                "claimed_parts": ["front_bumper", "headlight"],
                "claimed_issue_types": ["unknown"],
                "claimed_severity": "unknown",
                "included_parts": ["front_bumper", "headlight"],
                "excluded_parts": [],
                "evidence_quotes": ["front bumper", "left headlight"],
                "parser_confidence": 0.9,
                "parser_diagnostics": [],
            }
        }
        parser = CompositeClaimParser(
            llm_parser=LLMClaimParser(static_llm_client(response)),
        )
        decision = parser.parse(self.claim)
        self.assertIsNotNone(decision.llm_result)
        self.assertIn("llm_routing:always", decision.diagnostics)
        self.assertIn("parser_disagreement", decision.diagnostics)
        self.assertIn("parsers_agree_on_claim_intent", decision.diagnostics)
        self.assertIn("difference:evidence_quotes", decision.diagnostics)
        self.assertIn("difference:parser_confidence", decision.diagnostics)
        self.assertIn("difference:parser_diagnostics", decision.diagnostics)

    def test_records_excluded_part_difference_even_when_core_intent_agrees(self) -> None:
        claim = type(self.claim)(
            user_id="scope-difference",
            image_paths=self.claim.image_paths,
            user_claim="Customer: Not the keyboard; the screen is cracked.",
            claim_object="laptop",
            source_file=self.claim.source_file,
            images=self.claim.images,
        )
        response = {
            claim.user_id: {
                "claimed_parts": ["screen"],
                "claimed_issue_types": ["crack"],
                "claimed_severity": "unknown",
                "included_parts": ["screen"],
                "excluded_parts": [],
                "evidence_quotes": ["screen", "cracked"],
                "parser_confidence": 0.9,
                "parser_diagnostics": [],
            }
        }
        parser = CompositeClaimParser(
            llm_parser=LLMClaimParser(static_llm_client(response)),
        )
        decision = parser.parse(claim)
        self.assertIn("parsers_agree_on_claim_intent", decision.diagnostics)
        self.assertIn("difference:excluded_parts", decision.diagnostics)

    def test_rejects_llm_when_each_specific_field_is_not_independently_sourced(
        self,
    ) -> None:
        claim = type(self.claim)(
            user_id="missing-field-evidence",
            image_paths=self.claim.image_paths,
            user_claim="Customer: There is a deep dent on the door.",
            claim_object="car",
            source_file=self.claim.source_file,
            images=self.claim.images,
        )
        response = {
            claim.user_id: {
                "claimed_parts": ["door"],
                "claimed_issue_types": ["dent"],
                "claimed_severity": "high",
                "included_parts": ["door"],
                "excluded_parts": [],
                "evidence_quotes": ["door"],
                "parser_confidence": 0.99,
                "parser_diagnostics": [],
            }
        }
        parser = CompositeClaimParser(
            llm_parser=LLMClaimParser(
                static_llm_client(response),
                max_attempts=1,
            ),
        )
        decision = parser.parse(claim)
        self.assertEqual(decision.selected.parser_name, "rule")
        self.assertIn("llm_failed_fallback_to_rule", decision.diagnostics)
        self.assertIn(
            "Missing explicit source evidence for claimed_issue_types value: dent",
            " | ".join(decision.diagnostics),
        )

    def test_rejects_excluded_part_without_negated_source_quote(self) -> None:
        claim = type(self.claim)(
            user_id="missing-negation-evidence",
            image_paths=self.claim.image_paths,
            user_claim="Customer: Not the keyboard; the screen is cracked.",
            claim_object="laptop",
            source_file=self.claim.source_file,
            images=self.claim.images,
        )
        response = {
            claim.user_id: {
                "claimed_parts": ["screen"],
                "claimed_issue_types": ["crack"],
                "claimed_severity": "unknown",
                "included_parts": ["screen"],
                "excluded_parts": ["keyboard"],
                "evidence_quotes": ["keyboard", "screen", "cracked"],
                "parser_confidence": 0.99,
                "parser_diagnostics": [],
            }
        }
        parser = CompositeClaimParser(
            llm_parser=LLMClaimParser(
                static_llm_client(response),
                max_attempts=1,
            ),
        )
        decision = parser.parse(claim)
        self.assertEqual(decision.selected.parser_name, "rule")
        self.assertIn(
            "Missing negated source evidence for excluded_parts value: keyboard",
            " | ".join(decision.diagnostics),
        )

    def test_rejects_specific_issue_inferred_only_from_generic_damage_word(
        self,
    ) -> None:
        claim = type(self.claim)(
            user_id="generic-damage",
            image_paths=self.claim.image_paths,
            user_claim="Customer: The hood looks damaged.",
            claim_object="car",
            source_file=self.claim.source_file,
            images=self.claim.images,
        )
        response = {
            claim.user_id: {
                "claimed_parts": ["hood"],
                "claimed_issue_types": ["dent"],
                "claimed_severity": "unknown",
                "included_parts": ["hood"],
                "excluded_parts": [],
                "evidence_quotes": ["hood", "damaged"],
                "parser_confidence": 0.99,
                "parser_diagnostics": [],
            }
        }
        parser = CompositeClaimParser(
            llm_parser=LLMClaimParser(
                static_llm_client(response),
                max_attempts=1,
            ),
        )
        decision = parser.parse(claim)
        self.assertEqual(decision.selected.parser_name, "rule")
        self.assertEqual(
            decision.selected.claimed_issue_types,
            ("unknown",),
        )
        self.assertIn("generic damage wording is insufficient", " | ".join(
            decision.diagnostics
        ))

    def test_accepts_complete_per_field_source_evidence(self) -> None:
        claim = type(self.claim)(
            user_id="complete-provenance",
            image_paths=self.claim.image_paths,
            user_claim=(
                "Customer: Not the hood. There is a severe dent on the door."
            ),
            claim_object="car",
            source_file=self.claim.source_file,
            images=self.claim.images,
        )
        response = {
            claim.user_id: {
                "claimed_parts": ["door"],
                "claimed_issue_types": ["dent"],
                "claimed_severity": "high",
                "included_parts": ["door"],
                "excluded_parts": ["hood"],
                "evidence_quotes": ["Not the hood", "door", "dent", "severe"],
                "parser_confidence": 0.99,
                "parser_diagnostics": ["source_evidence_complete"],
            }
        }
        parser = CompositeClaimParser(
            llm_parser=LLMClaimParser(static_llm_client(response)),
        )
        decision = parser.parse(claim)
        self.assertIsNotNone(decision.llm_result)
        self.assertEqual(decision.selected.parser_name, "llm")

    def test_rejects_unknown_mixed_with_specific_issue(self) -> None:
        claim = type(self.claim)(
            user_id="mixed-issue-sentinel",
            image_paths=self.claim.image_paths,
            user_claim="Customer: The hinge is broken and the screen wobbles.",
            claim_object="laptop",
            source_file=self.claim.source_file,
            images=self.claim.images,
        )
        response = {
            claim.user_id: {
                "claimed_parts": ["hinge", "screen"],
                "claimed_issue_types": ["broken_part", "unknown"],
                "claimed_severity": "unknown",
                "included_parts": ["hinge", "screen"],
                "excluded_parts": [],
                "evidence_quotes": ["hinge", "broken", "screen"],
                "parser_confidence": 0.99,
                "parser_diagnostics": [],
            }
        }
        parser = CompositeClaimParser(
            llm_parser=LLMClaimParser(
                static_llm_client(response),
                max_attempts=1,
            ),
        )
        decision = parser.parse(claim)
        self.assertEqual(decision.selected.parser_name, "rule")
        self.assertIn(
            "claimed_issue_types cannot combine sentinel values",
            " | ".join(decision.diagnostics),
        )

    def test_normalizes_only_paired_wrapper_quotes_before_exact_source_check(
        self,
    ) -> None:
        claim = type(self.claim)(
            user_id="wrapped-quotes",
            image_paths=self.claim.image_paths,
            user_claim=(
                "Customer: The back of the car has a dent now. "
                "Mostly the rear bumper area."
            ),
            claim_object="car",
            source_file=self.claim.source_file,
            images=self.claim.images,
        )
        response = {
            claim.user_id: {
                "claimed_parts": ["rear_bumper"],
                "claimed_issue_types": ["dent"],
                "claimed_severity": "unknown",
                "included_parts": ["rear_bumper"],
                "excluded_parts": [],
                "evidence_quotes": [
                    '"The back of the car has a dent now."',
                    "“Mostly the rear bumper area.”",
                ],
                "parser_confidence": 0.99,
                "parser_diagnostics": [],
            }
        }
        parser = CompositeClaimParser(
            llm_parser=LLMClaimParser(static_llm_client(response)),
        )
        decision = parser.parse(claim)
        self.assertIsNotNone(decision.llm_result)
        self.assertEqual(
            decision.llm_result.evidence_quotes,
            (
                "The back of the car has a dent now.",
                "Mostly the rear bumper area.",
            ),
        )
        self.assertEqual(decision.selected.parser_name, "llm")

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
            ),
            llm_routing="always",
        )
        decision = parser.parse(self.claim)
        self.assertEqual(decision.selected.parser_name, "rule")
        self.assertIn("llm_failed_fallback_to_rule", decision.diagnostics)

    def test_auto_routing_skips_llm_when_rule_result_is_sufficient(self) -> None:
        clear_claim = type(self.claim)(
            user_id="clear",
            image_paths=self.claim.image_paths,
            user_claim="Customer: There is a deep dent on the door.",
            claim_object="car",
            source_file=self.claim.source_file,
            images=self.claim.images,
        )
        client = mock.Mock(
            return_value={
                "claimed_parts": ["door"],
                "claimed_issue_types": ["dent"],
                "claimed_severity": "unknown",
                "included_parts": ["door"],
                "excluded_parts": [],
                "evidence_quotes": ["door", "dent"],
                "parser_confidence": 0.9,
                "parser_diagnostics": [],
            }
        )
        parser = CompositeClaimParser(
            llm_parser=LLMClaimParser(client),
            llm_routing="auto",
        )
        decision = parser.parse(clear_claim)
        client.assert_not_called()
        self.assertEqual(decision.selected.parser_name, "rule")
        self.assertIn("llm_skipped_rule_sufficient", decision.diagnostics)

    def test_auto_routing_invokes_llm_for_unknown_rule_issue(self) -> None:
        claim = type(self.claim)(
            user_id="spanish",
            image_paths=self.claim.image_paths,
            user_claim="Cliente: El parachoques trasero esta danado.",
            claim_object="car",
            source_file=self.claim.source_file,
            images=self.claim.images,
        )
        response = {
            "spanish": {
                "claimed_parts": ["rear_bumper"],
                "claimed_issue_types": ["unknown"],
                "claimed_severity": "unknown",
                "included_parts": ["rear_bumper"],
                "excluded_parts": [],
                "evidence_quotes": ["parachoques trasero"],
                "parser_confidence": 0.8,
                "parser_diagnostics": ["generic_damage_term"],
            }
        }
        parser = CompositeClaimParser(
            llm_parser=LLMClaimParser(static_llm_client(response)),
            llm_routing="auto",
        )
        decision = parser.parse(claim)
        self.assertIsNotNone(decision.llm_result)
        self.assertIn("llm_routing:rule_unknown_issue", decision.diagnostics)


class OpenAIClaimClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.claim = load_bundle(REPO_ROOT / "dataset").claims[0]

    def test_claim_schema_is_scoped_to_current_object(self) -> None:
        schema = claim_parser_schema(self.claim)
        part_values = schema["properties"]["claimed_parts"]["items"]["enum"]
        self.assertIn("headlight", part_values)
        self.assertNotIn("screen", part_values)
        self.assertFalse(schema["additionalProperties"])
        self.assertNotIn("uniqueItems", json.dumps(schema))

    def test_responses_client_requests_strict_structured_output(self) -> None:
        payload = {
            "claimed_parts": ["front_bumper", "headlight"],
            "claimed_issue_types": ["unknown"],
            "claimed_severity": "unknown",
            "included_parts": ["front_bumper", "headlight"],
            "excluded_parts": [],
            "evidence_quotes": ["front bumper", "headlight"],
            "parser_confidence": 0.9,
            "parser_diagnostics": [],
        }
        api_response = {
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(payload),
                        }
                    ]
                }
            ]
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return json.dumps(api_response).encode("utf-8")

        with mock.patch(
            "claim_agent.urllib.request.urlopen",
            return_value=FakeResponse(),
        ) as urlopen:
            client = OpenAIResponsesClaimClient(
                api_key="test-key",
                model="gpt-5.4-mini",
            )
            result = client(self.claim, "test prompt")

        request = urlopen.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "gpt-5.4-mini")
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertEqual(
            body["text"]["format"]["schema"]["additionalProperties"],
            False,
        )
        self.assertEqual(json.loads(result), payload)


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

    def test_rule_lexicon_is_external_and_not_keyed_by_dataset_identity(self) -> None:
        lexicon = load_claim_lexicon()
        serialized = json.dumps(lexicon, ensure_ascii=False).casefold()
        self.assertNotIn("user_", serialized)
        self.assertNotIn("case_", serialized)
        self.assertNotIn("sample_claims", serialized)

    def test_rule_baseline_processes_both_datasets_without_illegal_combinations(
        self,
    ) -> None:
        for filename, expected_count in (
            ("sample_claims.csv", 20),
            ("claims.csv", 44),
        ):
            bundle = load_bundle(self.dataset_dir, filename)
            self.assertEqual(len(bundle.prepared_claims), expected_count)
            for prepared in bundle.prepared_claims:
                parsed = prepared.parsed_claim
                self.assertFalse(
                    "unknown" in parsed.claimed_parts
                    and len(parsed.claimed_parts) > 1
                )
                self.assertFalse(
                    {"unknown", "none"} & set(parsed.claimed_issue_types)
                    and len(parsed.claimed_issue_types) > 1
                )

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
