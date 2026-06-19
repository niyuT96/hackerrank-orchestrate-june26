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
from main import DEFAULT_VISION_MODEL, load_environment  # noqa: E402
from visual_agent import (  # noqa: E402
    ReplayVisionClient,
    VisionReviewer,
    encode_image,
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
                "OPENAI_VISION_MODEL=gpt-5.5\n",
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

    def test_default_vision_model_is_current_cost_quality_choice(self) -> None:
        self.assertEqual(DEFAULT_VISION_MODEL, "gpt-5.4-mini")


def valid_payload(prepared, image):
    part = next(
        (
            value
            for value in prepared.parsed_claim.claimed_parts
            if value != "unknown"
        ),
        "unknown",
    )
    return {
        "image_id": image.image_id,
        "path": image.path,
        "actual_object": prepared.claim.claim_object,
        "visible_parts": [part],
        "visible_issue_types": ["unknown"],
        "severity": "unknown",
        "target_part_visibility": (
            "visible" if part != "unknown" else "unknown"
        ),
        "requirement_results": [
            {
                "requirement_id": requirement.requirement_id,
                "status": "unknown",
                "reason": "The image requires visual review.",
            }
            for requirement in prepared.requirements
        ],
        "fact_summary": "The claimed object and target area are visible.",
        "risk_flags": ["none"],
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


if __name__ == "__main__":
    unittest.main()
