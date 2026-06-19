from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from claim_agent import DataValidationError, load_bundle  # noqa: E402
from main import (  # noqa: E402
    DEFAULT_REVIEW_MODEL,
    DEFAULT_VISION_MODEL,
    load_environment,
)
from visual_agent import (  # noqa: E402
    ESCALATION_REASONS,
    ReplayVisionClient,
    VisionReviewer,
    encode_image,
    image_observation_schema,
    local_escalation_reasons,
    parse_image_observation,
    write_visual_reviews,
    write_visual_traces,
)


class EnvironmentConfigurationTests(unittest.TestCase):
    def test_loads_repository_style_dotenv_without_overriding_process_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "OPENAI_API_KEY=dotenv-key\n"
                "OPENAI_VISION_MODEL=gpt-5.5\n"
                "OPENAI_REVIEW_MODEL=gpt-5.5-pro\n",
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "process-key"},
                clear=True,
            ):
                self.assertTrue(load_environment(env_path))
                self.assertEqual(os.environ["OPENAI_API_KEY"], "process-key")
                self.assertEqual(os.environ["OPENAI_VISION_MODEL"], "gpt-5.5")
                self.assertEqual(
                    os.environ["OPENAI_REVIEW_MODEL"], "gpt-5.5-pro"
                )

    def test_default_vision_model_is_current_cost_quality_choice(self) -> None:
        self.assertEqual(DEFAULT_VISION_MODEL, "gpt-5.4-mini")
        self.assertEqual(DEFAULT_REVIEW_MODEL, "gpt-5.5")


def valid_payload(prepared, image):
    part = next(
        (
            value
            for value in prepared.parsed_claim.claimed_parts
            if value != "unknown"
        ),
        "unknown",
    )
    issue = next(
        (
            value
            for value in prepared.parsed_claim.claimed_issue_types
            if value not in {"unknown", "none"}
        ),
        "none",
    )
    return {
        "image_id": image.image_id,
        "path": image.path,
        "actual_object": prepared.claim.claim_object,
        "visible_parts": [part],
        "visible_issue_types": [issue],
        "severity": "medium" if issue != "none" else "none",
        "target_part_visibility": (
            "visible" if part != "unknown" else "unknown"
        ),
        "requirement_results": [
            {
                "requirement_id": requirement.requirement_id,
                "status": "met",
                "reason": "The required target and issue are visible.",
            }
            for requirement in prepared.requirements
        ],
        "fact_summary": "The claimed object and target area are visible.",
        "risk_flags": (
            ["none"] if issue != "none" else ["damage_not_visible"]
        ),
        "reviewable": True,
        "claim_target_clear": part != "unknown",
        "diagnostics": [],
    }


class ImageEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prepared = load_bundle(
            REPO_ROOT / "dataset", "sample_claims.csv"
        ).prepared_claims[0]

    def test_encodes_and_bounds_image_dimensions(self) -> None:
        encoded = encode_image(
            self.prepared.claim.images[0].resolved_path,
            max_dimension=512,
        )
        self.assertTrue(encoded.data_url.startswith("data:image/jpeg;base64,"))
        self.assertLessEqual(encoded.processed_width, 512)
        self.assertLessEqual(encoded.processed_height, 512)
        self.assertEqual(len(encoded.sha256), 64)

    def test_decodes_avif_content_even_when_file_uses_jpg_extension(self) -> None:
        encoded = encode_image(
            REPO_ROOT / "dataset/images/test/case_001/img_1.jpg",
            max_dimension=512,
        )
        self.assertTrue(encoded.data_url.startswith("data:image/jpeg;base64,"))
        self.assertGreater(encoded.original_width, 0)
        self.assertGreater(encoded.original_height, 0)


class ObservationValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prepared = load_bundle(
            REPO_ROOT / "dataset", "sample_claims.csv"
        ).prepared_claims[0]
        cls.image = cls.prepared.claim.images[0]

    def test_accepts_strict_per_image_observation(self) -> None:
        observation = parse_image_observation(
            valid_payload(self.prepared, self.image),
            self.prepared,
            self.image,
            model_name="test",
            attempts=1,
        )
        self.assertEqual(observation.image_id, self.image.image_id)
        self.assertNotIn("claim_status", observation.to_dict())

    def test_openai_schema_avoids_unsupported_unique_items_keyword(self) -> None:
        schema = image_observation_schema(self.prepared, self.image)
        self.assertNotIn("uniqueItems", json.dumps(schema))

    def test_rejects_invented_requirement_id(self) -> None:
        payload = valid_payload(self.prepared, self.image)
        payload["requirement_results"][0]["requirement_id"] = "REQ_INVENTED"
        with self.assertRaisesRegex(DataValidationError, "every prepared"):
            parse_image_observation(
                payload,
                self.prepared,
                self.image,
                model_name="test",
                attempts=1,
            )

    def test_rejects_final_claim_status_field(self) -> None:
        payload = valid_payload(self.prepared, self.image)
        payload["claim_status"] = "supported"
        with self.assertRaisesRegex(DataValidationError, "fields mismatch"):
            parse_image_observation(
                payload,
                self.prepared,
                self.image,
                model_name="test",
                attempts=1,
            )

    def test_wrong_object_requires_risk_flag(self) -> None:
        payload = valid_payload(self.prepared, self.image)
        payload["actual_object"] = "laptop"
        payload["visible_parts"] = ["screen"]
        payload["target_part_visibility"] = "not_visible"
        with self.assertRaisesRegex(DataValidationError, "wrong_object"):
            parse_image_observation(
                payload,
                self.prepared,
                self.image,
                model_name="test",
                attempts=1,
            )

    def test_wrong_object_part_requires_risk_flag(self) -> None:
        payload = valid_payload(self.prepared, self.image)
        payload["visible_parts"] = ["front_bumper"]
        payload["target_part_visibility"] = "not_visible"
        with self.assertRaisesRegex(DataValidationError, "wrong_object_part"):
            parse_image_observation(
                payload,
                self.prepared,
                self.image,
                model_name="test",
                attempts=1,
            )

    def test_no_visible_damage_requires_damage_not_visible_risk(self) -> None:
        payload = valid_payload(self.prepared, self.image)
        payload["visible_issue_types"] = ["none"]
        payload["severity"] = "none"
        payload["risk_flags"] = ["none"]
        with self.assertRaisesRegex(DataValidationError, "damage_not_visible"):
            parse_image_observation(
                payload,
                self.prepared,
                self.image,
                model_name="test",
                attempts=1,
            )

    def test_conflicting_visible_issue_requires_claim_mismatch_risk(self) -> None:
        payload = valid_payload(self.prepared, self.image)
        payload["visible_issue_types"] = ["scratch"]
        with self.assertRaisesRegex(DataValidationError, "claim_mismatch"):
            parse_image_observation(
                payload,
                self.prepared,
                self.image,
                model_name="test",
                attempts=1,
            )

    def test_high_risk_flags_have_fixed_escalation_reasons(self) -> None:
        expected = {
            "possible_manipulation": "possible_manipulation",
            "non_original_image": "non_original_image",
            "text_instruction_present": "text_instruction_present",
        }
        for risk, reason in expected.items():
            with self.subTest(risk=risk):
                payload = valid_payload(self.prepared, self.image)
                payload["risk_flags"] = [risk]
                observation = parse_image_observation(
                    payload,
                    self.prepared,
                    self.image,
                    model_name="test",
                    attempts=1,
                )
                reasons = local_escalation_reasons(
                    observation, self.prepared
                )
                self.assertIn(reason, reasons)
                self.assertTrue(set(reasons) <= ESCALATION_REASONS)


class VisionPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prepared = load_bundle(
            REPO_ROOT / "dataset", "sample_claims.csv"
        ).prepared_claims

    def test_reviews_every_image_independently_and_writes_artifacts(self) -> None:
        responses = {
            image.path: valid_payload(prepared, image)
            for prepared in self.prepared
            for image in prepared.claim.images
        }
        reviewer = VisionReviewer(
            ReplayVisionClient(responses),
            max_attempts=1,
            max_dimension=512,
        )
        cases, traces = reviewer.review(self.prepared)
        expected_images = sum(
            len(prepared.claim.images) for prepared in self.prepared
        )
        self.assertEqual(len(cases), len(self.prepared))
        self.assertEqual(len(traces), expected_images)
        self.assertEqual(
            sum(len(case.observations) for case in cases),
            expected_images,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            reviews_path = Path(temp_dir) / "observations.json"
            traces_path = Path(temp_dir) / "trace.jsonl"
            self.assertEqual(
                write_visual_reviews(cases, reviews_path),
                len(self.prepared),
            )
            self.assertEqual(
                write_visual_traces(traces, traces_path),
                expected_images,
            )
            payload = json.loads(reviews_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload[0]["observations"][0]["path"],
                self.prepared[0].claim.images[0].path,
            )

    def test_invalid_single_image_falls_back_without_stopping_batch(self) -> None:
        responses = {}
        images = [
            (prepared, image)
            for prepared in self.prepared[:2]
            for image in prepared.claim.images
        ]
        for prepared, image in images[1:]:
            responses[image.path] = valid_payload(prepared, image)
        reviewer = VisionReviewer(
            ReplayVisionClient(responses),
            max_attempts=1,
            max_dimension=512,
        )
        cases, traces = reviewer.review(self.prepared[:2])
        observations = [
            observation
            for case in cases
            for observation in case.observations
        ]
        self.assertEqual(len(observations), len(images))
        self.assertFalse(observations[0].reviewable)
        self.assertEqual(observations[0].model_name, "fallback")
        self.assertTrue(any(item["status"] == "completed" for item in traces))

    def test_forced_escalation_calls_review_client_and_records_audit(self) -> None:
        prepared = self.prepared[0]
        image = prepared.claim.images[0]
        primary_payload = valid_payload(prepared, image)
        primary_payload["risk_flags"] = ["possible_manipulation"]
        review_payload = dict(primary_payload)
        reviewer = VisionReviewer(
            ReplayVisionClient({image.path: primary_payload}),
            review_client=ReplayVisionClient({image.path: review_payload}),
            max_attempts=1,
            review_max_attempts=1,
            max_dimension=512,
        )
        observation, trace = reviewer.review_image(prepared, image)
        self.assertEqual(observation.escalation.status, "resolved")
        self.assertEqual(
            observation.escalation.reasons, ("possible_manipulation",)
        )
        self.assertEqual(trace["status"], "completed_after_escalation")

    def test_review_conflict_preserves_primary_and_routes_to_human(self) -> None:
        prepared = self.prepared[0]
        image = prepared.claim.images[0]
        primary_payload = valid_payload(prepared, image)
        primary_payload["risk_flags"] = ["possible_manipulation"]
        review_payload = valid_payload(prepared, image)
        review_payload["visible_issue_types"] = ["scratch"]
        review_payload["risk_flags"] = [
            "possible_manipulation",
            "claim_mismatch",
        ]
        reviewer = VisionReviewer(
            ReplayVisionClient({image.path: primary_payload}),
            review_client=ReplayVisionClient({image.path: review_payload}),
            max_attempts=1,
            review_max_attempts=1,
            max_dimension=512,
        )
        observation, trace = reviewer.review_image(prepared, image)
        self.assertEqual(observation.visible_issue_types, ("dent",))
        self.assertIn("manual_review_required", observation.risk_flags)
        self.assertIn(
            "visible_issue_types", observation.escalation.conflicts
        )
        self.assertEqual(trace["status"], "manual_review_required")

    def test_missing_review_client_routes_forced_escalation_to_human(self) -> None:
        prepared = self.prepared[0]
        image = prepared.claim.images[0]
        payload = valid_payload(prepared, image)
        payload["risk_flags"] = ["non_original_image"]
        reviewer = VisionReviewer(
            ReplayVisionClient({image.path: payload}),
            max_attempts=1,
            max_dimension=512,
        )
        observation, trace = reviewer.review_image(prepared, image)
        self.assertIn("manual_review_required", observation.risk_flags)
        self.assertEqual(
            observation.escalation.diagnostics,
            ("review_client_not_configured",),
        )
        self.assertEqual(trace["status"], "manual_review_required")

    def test_review_can_fill_unknown_primary_fields_without_overwriting_facts(
        self,
    ) -> None:
        prepared = self.prepared[0]
        image = prepared.claim.images[0]
        primary_payload = valid_payload(prepared, image)
        primary_payload.update(
            {
                "actual_object": "unknown",
                "visible_parts": ["unknown"],
                "visible_issue_types": ["unknown"],
                "severity": "unknown",
                "target_part_visibility": "unknown",
                "requirement_results": [
                    {
                        "requirement_id": requirement.requirement_id,
                        "status": "unknown",
                        "reason": "The primary model could not determine this.",
                    }
                    for requirement in prepared.requirements
                ],
                "fact_summary": "The primary result is uncertain.",
                "reviewable": False,
                "claim_target_clear": False,
            }
        )
        review_payload = valid_payload(prepared, image)
        reviewer = VisionReviewer(
            ReplayVisionClient({image.path: primary_payload}),
            review_client=ReplayVisionClient({image.path: review_payload}),
            max_attempts=1,
            review_max_attempts=1,
            max_dimension=512,
        )
        observation, trace = reviewer.review_image(prepared, image)
        self.assertEqual(observation.actual_object, "car")
        self.assertEqual(observation.visible_issue_types, ("dent",))
        self.assertTrue(observation.reviewable)
        self.assertEqual(observation.escalation.status, "resolved")
        self.assertEqual(trace["status"], "completed_after_escalation")

    def test_successful_review_clears_fallback_manual_marker(self) -> None:
        prepared = self.prepared[0]
        image = prepared.claim.images[0]
        review_payload = valid_payload(prepared, image)
        reviewer = VisionReviewer(
            ReplayVisionClient({}),
            review_client=ReplayVisionClient({image.path: review_payload}),
            max_attempts=1,
            review_max_attempts=1,
            max_dimension=512,
        )
        observation, trace = reviewer.review_image(prepared, image)
        self.assertNotIn("manual_review_required", observation.risk_flags)
        self.assertEqual(observation.model_name, "replay")
        self.assertTrue(observation.reviewable)
        self.assertEqual(trace["status"], "completed_after_escalation")

    def test_multi_image_identity_signal_escalates_each_related_image(self) -> None:
        prepared = self.prepared[1]
        primary_responses = {}
        review_responses = {}
        for image in prepared.claim.images:
            payload = valid_payload(prepared, image)
            if image.image_id == "img_2":
                payload["risk_flags"] = ["claim_mismatch"]
            primary_responses[image.path] = payload
            review_responses[image.path] = payload
        reviewer = VisionReviewer(
            ReplayVisionClient(primary_responses),
            review_client=ReplayVisionClient(review_responses),
            max_attempts=1,
            review_max_attempts=1,
            max_dimension=512,
        )
        cases, traces = reviewer.review([prepared])
        observations = cases[0].observations
        self.assertTrue(
            all(
                "multi_image_identity_conflict"
                in item.escalation.reasons
                for item in observations
            )
        )
        self.assertTrue(
            all("escalation" in item for item in traces)
        )


if __name__ == "__main__":
    unittest.main()
