"""Sprint 3 aggregation, evidence judgment, and three-state decision tests."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from claim_agent import (  # noqa: E402
    DataValidationError,
    ReviewResult,
    load_bundle,
)
from visual_agent import (  # noqa: E402
    ImageObservation,
    RequirementObservation,
    VisualReviewCase,
)
from decision_agent import (  # noqa: E402
    aggregate_visual_case,
    write_final_output,
    _history_risk_flags,
    _evaluate_evidence_standard,
    _decide_claim_status,
    _select_supporting_images,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_observation(
    prepared,
    image,
    *,
    actual_object: str | None = None,
    visible_parts: list[str] | None = None,
    visible_issue_types: list[str] | None = None,
    severity: str = "medium",
    target_part_visibility: str = "visible",
    req_status: str = "met",
    risk_flags: list[str] | None = None,
    reviewable: bool = True,
    claim_target_clear: bool = True,
) -> ImageObservation:
    """Build a minimal ImageObservation for a given PreparedClaim image."""
    part = next(
        (p for p in prepared.parsed_claim.claimed_parts if p != "unknown"),
        "unknown",
    )
    issue = next(
        (i for i in prepared.parsed_claim.claimed_issue_types
         if i not in {"unknown", "none"}),
        "none",
    )
    if actual_object is None:
        actual_object = prepared.claim.claim_object
    if visible_parts is None:
        visible_parts = [part]
    if visible_issue_types is None:
        visible_issue_types = [issue] if issue != "none" else ["none"]
    if risk_flags is None:
        risk_flags = (
            ["none"]
            if issue != "none"
            else ["damage_not_visible"]
            if target_part_visibility == "visible"
            else ["none"]
        )
    if target_part_visibility == "visible" and visible_issue_types == ["none"]:
        if "damage_not_visible" not in risk_flags:
            risk_flags = ["damage_not_visible"]

    if severity == "medium" and visible_issue_types == ["none"]:
        severity = "none"

    return ImageObservation(
        image_id=image.image_id,
        path=image.path,
        actual_object=actual_object,
        visible_parts=tuple(visible_parts),
        visible_issue_types=tuple(visible_issue_types),
        severity=severity,
        target_part_visibility=target_part_visibility,
        requirement_results=tuple(
            RequirementObservation(
                requirement_id=req.requirement_id,
                status=req_status,
                reason="test",
            )
            for req in prepared.requirements
        ),
        fact_summary="Test observation.",
        risk_flags=tuple(risk_flags),
        reviewable=reviewable,
        claim_target_clear=claim_target_clear,
        model_name="test",
        attempts=1,
    )


def _make_case(prepared, observations: list[ImageObservation]) -> VisualReviewCase:
    return VisualReviewCase(prepared, tuple(observations))


# ---------------------------------------------------------------------------
# Load test fixtures
# ---------------------------------------------------------------------------

SAMPLE_BUNDLE = load_bundle(REPO_ROOT / "dataset", "sample_claims.csv")
SAMPLE_PREPARED = SAMPLE_BUNDLE.prepared_claims


def _prepared_by_user(user_id: str):
    return next(p for p in SAMPLE_PREPARED if p.claim.user_id == user_id)


# ---------------------------------------------------------------------------
# History risk tests
# ---------------------------------------------------------------------------

class HistoryRiskTests(unittest.TestCase):
    def test_clean_history_produces_no_flags(self) -> None:
        prepared = _prepared_by_user("user_001")
        flags = _history_risk_flags(prepared.history)
        self.assertEqual(flags, frozenset())

    def test_high_rejection_rate_triggers_user_history_risk(self) -> None:
        prepared = _prepared_by_user("user_005")
        flags = _history_risk_flags(prepared.history)
        self.assertIn("user_history_risk", flags)

    def test_history_flags_do_not_include_illegal_values(self) -> None:
        for prepared in SAMPLE_PREPARED:
            flags = _history_risk_flags(prepared.history)
            illegal = flags - {"user_history_risk", "manual_review_required"}
            self.assertEqual(illegal, frozenset(), msg=prepared.claim.user_id)


# ---------------------------------------------------------------------------
# Evidence standard tests
# ---------------------------------------------------------------------------

class EvidenceStandardTests(unittest.TestCase):
    def test_all_requirements_met_returns_true(self) -> None:
        prepared = _prepared_by_user("user_001")
        image = prepared.claim.images[0]
        obs = _make_observation(prepared, image)
        met, reason = _evaluate_evidence_standard((obs,), prepared)
        self.assertTrue(met)
        self.assertGreater(len(reason), 0)

    def test_any_unmet_requirement_returns_false(self) -> None:
        prepared = _prepared_by_user("user_001")
        image = prepared.claim.images[0]
        obs = _make_observation(prepared, image, req_status="not_met")
        met, reason = _evaluate_evidence_standard((obs,), prepared)
        self.assertFalse(met)
        self.assertIn("not satisfied", reason)

    def test_unknown_requirement_status_returns_false(self) -> None:
        prepared = _prepared_by_user("user_001")
        image = prepared.claim.images[0]
        obs = _make_observation(prepared, image, req_status="unknown")
        met, reason = _evaluate_evidence_standard((obs,), prepared)
        self.assertFalse(met)

    def test_non_reviewable_image_excluded_from_evidence(self) -> None:
        prepared = _prepared_by_user("user_001")
        image = prepared.claim.images[0]
        obs = _make_observation(prepared, image, req_status="met", reviewable=False)
        # Non-reviewable obs should not count towards meeting requirements
        met, _ = _evaluate_evidence_standard((obs,), prepared)
        self.assertFalse(met)

    def test_multi_image_one_met_one_not_met_returns_true(self) -> None:
        """If at least one image meets a requirement, it is satisfied."""
        prepared = _prepared_by_user("user_003")
        images = prepared.claim.images
        blurry = _make_observation(
            prepared, images[0], req_status="not_met",
            visible_parts=["unknown"], visible_issue_types=["unknown"],
            severity="unknown", target_part_visibility="unknown",
            risk_flags=["blurry_image"],
        )
        clear = _make_observation(prepared, images[1], req_status="met")
        met, _ = _evaluate_evidence_standard((blurry, clear), prepared)
        self.assertTrue(met)


# ---------------------------------------------------------------------------
# Three-state decision tests
# ---------------------------------------------------------------------------

class ClaimStatusDecisionTests(unittest.TestCase):
    def test_supported_when_claimed_part_and_damage_visible(self) -> None:
        prepared = _prepared_by_user("user_001")
        image = prepared.claim.images[0]
        obs = _make_observation(
            prepared, image,
            visible_parts=["rear_bumper"],
            visible_issue_types=["dent"],
            severity="medium",
        )
        status = _decide_claim_status((obs,), prepared, True, obs)
        self.assertEqual(status, "supported")

    def test_contradicted_when_target_visible_but_no_damage(self) -> None:
        prepared = _prepared_by_user("user_001")
        image = prepared.claim.images[0]
        obs = _make_observation(
            prepared, image,
            visible_parts=["rear_bumper"],
            visible_issue_types=["none"],
            severity="none",
            target_part_visibility="visible",
            risk_flags=["damage_not_visible"],
        )
        status = _decide_claim_status((obs,), prepared, True, obs)
        self.assertEqual(status, "contradicted")

    def test_not_enough_information_when_evidence_not_met(self) -> None:
        prepared = _prepared_by_user("user_001")
        image = prepared.claim.images[0]
        obs = _make_observation(
            prepared, image,
            visible_parts=["unknown"],
            visible_issue_types=["unknown"],
            severity="unknown",
            target_part_visibility="not_visible",
            risk_flags=["wrong_angle"],
        )
        status = _decide_claim_status((obs,), prepared, False, obs)
        self.assertEqual(status, "not_enough_information")

    def test_contradicted_when_wrong_object_shown(self) -> None:
        prepared = _prepared_by_user("user_001")
        image = prepared.claim.images[0]
        obs = _make_observation(
            prepared, image,
            actual_object="laptop",
            visible_parts=["screen"],
            visible_issue_types=["crack"],
            severity="medium",
            target_part_visibility="not_visible",
            risk_flags=["wrong_object"],
            claim_target_clear=False,
        )
        status = _decide_claim_status((obs,), prepared, True, obs)
        self.assertEqual(status, "contradicted")

    def test_contradicted_when_visible_issue_mismatches_claim(self) -> None:
        prepared = _prepared_by_user("user_001")  # claims dent
        image = prepared.claim.images[0]
        obs = _make_observation(
            prepared, image,
            visible_parts=["rear_bumper"],
            visible_issue_types=["scratch"],  # different from claimed dent
            severity="low",
            risk_flags=["claim_mismatch"],
        )
        status = _decide_claim_status((obs,), prepared, True, obs)
        self.assertEqual(status, "contradicted")

    def test_unknown_claimed_parts_cannot_produce_supported(self) -> None:
        prepared = _prepared_by_user("user_006")  # parts unknown
        if not prepared.claim.images:
            self.skipTest("no images")
        image = prepared.claim.images[0]
        obs = _make_observation(
            prepared, image,
            visible_parts=["unknown"],
            visible_issue_types=["unknown"],
            severity="unknown",
            target_part_visibility="unknown",
            risk_flags=["wrong_angle"],
        )
        status = _decide_claim_status((obs,), prepared, False, obs)
        self.assertNotEqual(status, "supported")


# ---------------------------------------------------------------------------
# Supporting image selection tests
# ---------------------------------------------------------------------------

class SupportingImageTests(unittest.TestCase):
    def test_supported_returns_images_showing_damage(self) -> None:
        prepared = _prepared_by_user("user_001")
        image = prepared.claim.images[0]
        obs = _make_observation(
            prepared, image,
            visible_parts=["rear_bumper"],
            visible_issue_types=["dent"],
        )
        ids = _select_supporting_images(
            (obs,), "supported",
            frozenset(["rear_bumper"]), frozenset(["dent"])
        )
        self.assertIn("img_1", ids)

    def test_not_enough_information_returns_empty_for_fully_unknown(self) -> None:
        prepared = _prepared_by_user("user_001")
        image = prepared.claim.images[0]
        obs = _make_observation(
            prepared, image,
            visible_parts=["unknown"],
            visible_issue_types=["unknown"],
            severity="unknown",
            target_part_visibility="not_visible",
            req_status="not_met",
            risk_flags=["wrong_angle"],
        )
        ids = _select_supporting_images(
            (obs,), "not_enough_information",
            frozenset(["rear_bumper"]), frozenset(["dent"])
        )
        self.assertEqual(ids, ())

    def test_identity_conflict_includes_both_images(self) -> None:
        """Multi-image identity conflict → both image IDs listed."""
        prepared = _prepared_by_user("user_002")
        images = prepared.claim.images
        obs1 = _make_observation(
            prepared, images[0],
            visible_parts=["front_bumper"],
            visible_issue_types=["broken_part"],
            risk_flags=["claim_mismatch"],
        )
        obs2 = _make_observation(
            prepared, images[1],
            actual_object="car",
            visible_parts=["body"],
            visible_issue_types=["none"],
            severity="none",
            risk_flags=["wrong_object", "claim_mismatch"],
            claim_target_clear=False,
        )
        ids = _select_supporting_images(
            (obs1, obs2), "not_enough_information",
            frozenset(["front_bumper"]), frozenset(["scratch"])
        )
        # Both images referenced in multi-image conflict
        self.assertEqual(len(ids), 2)


# ---------------------------------------------------------------------------
# Full aggregation integration tests
# ---------------------------------------------------------------------------

class AggregationIntegrationTests(unittest.TestCase):
    """End-to-end tests using replay observations against sample_claims cases."""

    def _make_clean_supported_case(self, user_id: str) -> VisualReviewCase:
        prepared = _prepared_by_user(user_id)
        observations = [
            _make_observation(prepared, img)
            for img in prepared.claim.images
        ]
        return _make_case(prepared, observations)

    def test_clean_supported_produces_valid_review_result(self) -> None:
        case = self._make_clean_supported_case("user_001")
        result = aggregate_visual_case(case)
        self.assertEqual(result.claim_status, "supported")
        self.assertTrue(result.evidence_standard_met)
        self.assertIn("rear_bumper", result.object_part)
        self.assertNotEqual(result.severity, "unknown")
        self.assertNotEqual(result.issue_type, "unknown")
        self.assertIn("img_1", result.supporting_image_ids)

    def test_result_passes_review_result_validation(self) -> None:
        """aggregate_visual_case must always return a ReviewResult that passes validate()."""
        for prepared in SAMPLE_PREPARED[:5]:
            observations = [
                _make_observation(prepared, img)
                for img in prepared.claim.images
            ]
            case = _make_case(prepared, observations)
            result = aggregate_visual_case(case)
            # validate() raises DataValidationError on inconsistency
            result.validate(prepared.claim)

    def test_contradicted_when_part_visible_no_damage(self) -> None:
        prepared = _prepared_by_user("user_001")
        image = prepared.claim.images[0]
        obs = _make_observation(
            prepared, image,
            visible_parts=["rear_bumper"],
            visible_issue_types=["none"],
            severity="none",
            target_part_visibility="visible",
            risk_flags=["damage_not_visible"],
        )
        case = _make_case(prepared, [obs])
        result = aggregate_visual_case(case)
        self.assertEqual(result.claim_status, "contradicted")
        self.assertEqual(result.issue_type, "none")
        self.assertEqual(result.severity, "none")

    def test_not_enough_information_when_wrong_angle(self) -> None:
        prepared = _prepared_by_user("user_001")
        image = prepared.claim.images[0]
        obs = _make_observation(
            prepared, image,
            visible_parts=["unknown"],
            visible_issue_types=["unknown"],
            severity="unknown",
            target_part_visibility="not_visible",
            req_status="not_met",
            risk_flags=["wrong_angle"],
            claim_target_clear=False,
        )
        case = _make_case(prepared, [obs])
        result = aggregate_visual_case(case)
        self.assertEqual(result.claim_status, "not_enough_information")
        self.assertFalse(result.evidence_standard_met)

    def test_user_history_risk_added_but_does_not_change_clear_visual(self) -> None:
        """History risk adds a flag but must not flip a clear supported → contradicted."""
        prepared = _prepared_by_user("user_005")  # has rejected claims history
        image = prepared.claim.images[0]
        obs = _make_observation(
            prepared, image,
            visible_parts=["rear_bumper"],
            visible_issue_types=["dent"],
            severity="medium",
        )
        case = _make_case(prepared, [obs])
        result = aggregate_visual_case(case)
        self.assertIn("user_history_risk", result.risk_flags)
        self.assertEqual(result.claim_status, "supported")

    def test_multi_image_blurry_plus_clear_is_supported(self) -> None:
        """case_007 pattern: img_1 blurry, img_2 clear → supported using img_2."""
        prepared = _prepared_by_user("user_003")
        images = prepared.claim.images
        self.assertEqual(len(images), 2)
        blurry = _make_observation(
            prepared, images[0],
            visible_parts=["unknown"],
            visible_issue_types=["unknown"],
            severity="unknown",
            target_part_visibility="unknown",
            req_status="not_met",
            risk_flags=["blurry_image"],
            reviewable=True,
            claim_target_clear=False,
        )
        clear = _make_observation(
            prepared, images[1],
            visible_parts=["door"],
            visible_issue_types=["dent"],
            severity="medium",
        )
        case = _make_case(prepared, [blurry, clear])
        result = aggregate_visual_case(case)
        self.assertEqual(result.claim_status, "supported")
        self.assertIn("blurry_image", result.risk_flags)
        self.assertIn("img_2", result.supporting_image_ids)
        self.assertNotIn("img_1", result.supporting_image_ids)

    def test_wrong_object_in_multi_image_produces_not_enough_information(self) -> None:
        """case_002 pattern: one image wrong car → identity conflict."""
        prepared = _prepared_by_user("user_002")
        images = prepared.claim.images
        obs1 = _make_observation(
            prepared, images[0],
            visible_parts=["front_bumper"],
            visible_issue_types=["scratch"],
            severity="low",
        )
        obs2 = _make_observation(
            prepared, images[1],
            actual_object="car",
            visible_parts=["body"],
            visible_issue_types=["none"],
            severity="none",
            target_part_visibility="not_visible",
            req_status="not_met",
            risk_flags=["wrong_object", "claim_mismatch"],
            claim_target_clear=False,
        )
        case = _make_case(prepared, [obs1, obs2])
        result = aggregate_visual_case(case)
        self.assertFalse(result.evidence_standard_met)
        self.assertEqual(result.claim_status, "not_enough_information")
        self.assertIn("wrong_object", result.risk_flags)

    def test_issue_type_and_severity_consistent_with_none(self) -> None:
        """issue_type=none must always co-occur with severity=none."""
        prepared = _prepared_by_user("user_001")
        image = prepared.claim.images[0]
        obs = _make_observation(
            prepared, image,
            visible_parts=["rear_bumper"],
            visible_issue_types=["none"],
            severity="none",
            target_part_visibility="visible",
            risk_flags=["damage_not_visible"],
        )
        case = _make_case(prepared, [obs])
        result = aggregate_visual_case(case)
        if result.issue_type == "none":
            self.assertEqual(result.severity, "none")

    def test_supporting_image_ids_all_valid_for_claim(self) -> None:
        """supporting_image_ids must be a subset of the claim's image_ids."""
        for prepared in SAMPLE_PREPARED:
            observations = [
                _make_observation(prepared, img)
                for img in prepared.claim.images
            ]
            case = _make_case(prepared, observations)
            result = aggregate_visual_case(case)
            valid_ids = set(prepared.claim.image_ids)
            for img_id in result.supporting_image_ids:
                self.assertIn(img_id, valid_ids, msg=prepared.claim.user_id)


# ---------------------------------------------------------------------------
# write_final_output integration test
# ---------------------------------------------------------------------------

class FinalOutputWriterTests(unittest.TestCase):
    def test_writes_correct_column_count_and_row_count(self) -> None:
        cases = [
            _make_case(
                prepared,
                [_make_observation(prepared, img) for img in prepared.claim.images]
            )
            for prepared in SAMPLE_PREPARED
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "output.csv"
            count = write_final_output(cases, out_path)
            self.assertEqual(count, len(SAMPLE_PREPARED))
            rows = list(csv.DictReader(out_path.open(encoding="utf-8")))
            self.assertEqual(len(rows), len(SAMPLE_PREPARED))
            # Verify 14 columns in correct order
            from claim_agent import OUTPUT_COLUMNS
            self.assertEqual(list(rows[0].keys()), list(OUTPUT_COLUMNS))

    def test_input_columns_preserved_verbatim(self) -> None:
        prepared = SAMPLE_PREPARED[0]
        case = _make_case(
            prepared,
            [_make_observation(prepared, img) for img in prepared.claim.images]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "output.csv"
            write_final_output([case], out_path)
            rows = list(csv.DictReader(out_path.open(encoding="utf-8")))
            self.assertEqual(rows[0]["user_id"], prepared.claim.user_id)
            self.assertEqual(rows[0]["image_paths"], prepared.claim.image_paths)
            self.assertEqual(rows[0]["user_claim"], prepared.claim.user_claim)
            self.assertEqual(rows[0]["claim_object"], prepared.claim.claim_object)

    def test_boolean_fields_use_lowercase_text(self) -> None:
        prepared = SAMPLE_PREPARED[0]
        case = _make_case(
            prepared,
            [_make_observation(prepared, img) for img in prepared.claim.images]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "output.csv"
            write_final_output([case], out_path)
            rows = list(csv.DictReader(out_path.open(encoding="utf-8")))
            self.assertIn(rows[0]["evidence_standard_met"], {"true", "false"})
            self.assertIn(rows[0]["valid_image"], {"true", "false"})

    def test_all_enums_are_legal(self) -> None:
        from claim_agent import (
            CLAIM_STATUSES, ISSUE_TYPES, OBJECT_PARTS, RISK_FLAGS, SEVERITIES
        )
        cases = [
            _make_case(
                prepared,
                [_make_observation(prepared, img) for img in prepared.claim.images]
            )
            for prepared in SAMPLE_PREPARED
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "output.csv"
            write_final_output(cases, out_path)
            for row in csv.DictReader(out_path.open(encoding="utf-8")):
                self.assertIn(row["claim_status"], CLAIM_STATUSES)
                self.assertIn(row["issue_type"], ISSUE_TYPES)
                claim_object = row["claim_object"]
                self.assertIn(row["object_part"], OBJECT_PARTS[claim_object])
                self.assertIn(row["severity"], SEVERITIES)
                for flag in row["risk_flags"].split(";"):
                    self.assertIn(flag.strip(), RISK_FLAGS)

    def test_supporting_image_ids_none_when_no_support(self) -> None:
        prepared = SAMPLE_PREPARED[0]
        image = prepared.claim.images[0]
        obs = _make_observation(
            prepared, image,
            visible_parts=["unknown"],
            visible_issue_types=["unknown"],
            severity="unknown",
            target_part_visibility="not_visible",
            req_status="not_met",
            risk_flags=["wrong_angle"],
            claim_target_clear=False,
        )
        case = _make_case(prepared, [obs])
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "output.csv"
            write_final_output([case], out_path)
            rows = list(csv.DictReader(out_path.open(encoding="utf-8")))
            self.assertEqual(rows[0]["supporting_image_ids"], "none")

    def test_no_not_enough_information_when_all_requirements_met(self) -> None:
        """A fully-met, claim-aligned observation should never produce NEI."""
        prepared = SAMPLE_PREPARED[0]  # user_001 rear bumper dent
        case = self._clean_supported_case(prepared)
        result = aggregate_visual_case(case)
        self.assertNotEqual(result.claim_status, "not_enough_information")

    def _clean_supported_case(self, prepared):
        return _make_case(
            prepared,
            [_make_observation(prepared, img) for img in prepared.claim.images]
        )


if __name__ == "__main__":
    unittest.main()
