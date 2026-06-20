"""Sprint 3 multi-image aggregation, evidence judgment, and three-state decision."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from claim_agent import (
    CLAIM_STATUSES,
    ISSUE_TYPES,
    OBJECT_PARTS,
    RISK_FLAGS,
    SEVERITIES,
    DataValidationError,
    OUTPUT_COLUMNS,
    PreparedClaim,
    ReviewResult,
    UserHistory,
    _bool_text,
)
from visual_agent import (
    EscalationAudit,
    ImageObservation,
    RequirementObservation,
    VisualReviewCase,
    validate_image_observation,
)


@dataclass(frozen=True)
class CaseVisualFacts:
    """Order-independent, traceable visual facts for one claim.

    Sprint 3 decisions use this case-level view.  A single best image is kept
    only for choosing the primary output fact and concise explanation.
    """

    reviewable_observations: tuple[ImageObservation, ...]
    visible_objects: frozenset[str]
    visible_parts: frozenset[str]
    visible_issue_types: frozenset[str]
    part_image_ids: Mapping[str, tuple[str, ...]]
    issue_image_ids: Mapping[str, tuple[str, ...]]
    requirement_image_ids: Mapping[str, tuple[str, ...]]
    identity_conflict: bool
    conflicting_image_ids: tuple[str, ...]


def _stable_observations(
    observations: Iterable[ImageObservation],
) -> tuple[ImageObservation, ...]:
    return tuple(sorted(observations, key=lambda item: (item.path, item.image_id)))


def build_case_visual_facts(
    observations: tuple[ImageObservation, ...],
    prepared: PreparedClaim,
) -> CaseVisualFacts:
    """Aggregate all per-image observations without depending on input order."""
    ordered = _stable_observations(observations)
    reviewable = tuple(item for item in ordered if item.reviewable)
    visible_objects = frozenset(
        item.actual_object
        for item in reviewable
        if item.actual_object != "unknown"
    )
    visible_parts = frozenset(
        part
        for item in reviewable
        for part in item.visible_parts
        if part != "unknown"
    )
    visible_issues = frozenset(
        issue
        for item in reviewable
        for issue in item.visible_issue_types
        if issue not in {"unknown", "none"}
    )

    def image_map(values: str) -> dict[str, tuple[str, ...]]:
        result: dict[str, list[str]] = {}
        for item in reviewable:
            entries = (
                item.visible_parts
                if values == "parts"
                else item.visible_issue_types
            )
            for entry in entries:
                if entry in {"unknown", "none"}:
                    continue
                result.setdefault(entry, []).append(item.image_id)
        return {
            key: tuple(dict.fromkeys(ids))
            for key, ids in sorted(result.items())
        }

    requirement_images: dict[str, list[str]] = {}
    for item in reviewable:
        for result in item.requirement_results:
            if result.status == "met":
                requirement_images.setdefault(
                    result.requirement_id, []
                ).append(item.image_id)

    concrete_objects = {
        item.actual_object
        for item in ordered
        if item.actual_object != "unknown"
    }
    wrong_object_ids = {
        item.image_id
        for item in ordered
        if (
            "wrong_object" in item.risk_flags
            or item.actual_object
            not in {"unknown", prepared.claim.claim_object}
        )
    }
    identity_conflict = len(ordered) > 1 and (
        len(concrete_objects) > 1 or bool(wrong_object_ids)
    )
    conflicting_ids = (
        tuple(item.image_id for item in ordered)
        if identity_conflict
        else ()
    )
    return CaseVisualFacts(
        reviewable_observations=reviewable,
        visible_objects=visible_objects,
        visible_parts=visible_parts,
        visible_issue_types=visible_issues,
        part_image_ids=image_map("parts"),
        issue_image_ids=image_map("issues"),
        requirement_image_ids={
            key: tuple(dict.fromkeys(ids))
            for key, ids in sorted(requirement_images.items())
        },
        identity_conflict=identity_conflict,
        conflicting_image_ids=conflicting_ids,
    )


# ---------------------------------------------------------------------------
# User-history risk heuristics
# ---------------------------------------------------------------------------

def _history_risk_flags(history: UserHistory) -> frozenset[str]:
    """Return the set of risk flags warranted by the user's claim history.

    Only ``user_history_risk`` and ``manual_review_required`` are ever
    generated here; they must never override a clear visual conclusion.
    """
    flags: set[str] = set()
    if not history:
        return frozenset()
    total = history.past_claim_count
    if total == 0:
        return frozenset()

    reject_rate = history.rejected_claim / total if total else 0.0
    manual_rate = history.manual_review_claim / total if total else 0.0

    risky_history_flags = {
        flag.strip().lower()
        for flag in history.history_flags
        if flag.strip().lower() not in {"none", ""}
    }

    if (
        reject_rate >= 0.3
        or history.rejected_claim >= 2
        or risky_history_flags
        or history.last_90_days_claim_count >= 3
    ):
        flags.add("user_history_risk")

    if (
        (reject_rate >= 0.4 or history.rejected_claim >= 3)
        or (manual_rate >= 0.3 and history.manual_review_claim >= 2)
        or ("high_risk" in risky_history_flags)
        or ("fraud_flag" in risky_history_flags)
    ):
        flags.add("manual_review_required")

    return frozenset(flags)


# ---------------------------------------------------------------------------
# Evidence standard evaluation
# ---------------------------------------------------------------------------

def _evaluate_evidence_standard(
    observations: tuple[ImageObservation, ...],
    prepared: PreparedClaim,
    facts: CaseVisualFacts | None = None,
) -> tuple[bool, str]:
    """Return (met, reason) for the case-level evidence standard.

    A requirement is met if at least one non-failed observation reports
    ``status="met"`` for it.  All requirements must be met for
    ``evidence_standard_met=True``.

    Additional override: a multi-image identity conflict (different actual
    objects or wrong_object + claim_mismatch across images) makes the image
    set unreliable even if individual requirements appear met.
    """
    if not observations:
        return False, "No images were submitted for this claim."

    facts = facts or build_case_visual_facts(observations, prepared)
    if facts.identity_conflict:
        return (
            False,
            "The submitted images appear to show different objects or vehicles, "
            "so the image set cannot reliably support the claim."
        )

    # Build per-requirement best status across all images
    req_status: dict[str, str] = {
        req.requirement_id: "not_met" for req in prepared.requirements
    }
    for obs in facts.reviewable_observations:
        for rr in obs.requirement_results:
            if rr.requirement_id not in req_status:
                continue
            current = req_status[rr.requirement_id]
            # met > unknown > not_met
            if rr.status == "met":
                req_status[rr.requirement_id] = "met"
            elif rr.status == "unknown" and current == "not_met":
                req_status[rr.requirement_id] = "unknown"

    not_met = [rid for rid, status in req_status.items() if status == "not_met"]
    unknown = [rid for rid, status in req_status.items() if status == "unknown"]

    if not_met:
        reason = (
            f"The following evidence requirements were not satisfied: "
            f"{'; '.join(sorted(not_met))}."
        )
        return False, reason

    if unknown:
        reason = (
            f"The following evidence requirements could not be confirmed: "
            f"{'; '.join(sorted(unknown))}. "
            "The available images were insufficient to evaluate all required evidence."
        )
        return False, reason

    # All met — build a concise affirmative reason
    req_texts = {req.requirement_id: req.minimum_image_evidence for req in prepared.requirements}
    first_req = next(iter(req_status), "")
    first_text = req_texts.get(first_req, "")
    if len(req_status) == 1:
        reason = f"The submitted image satisfies {first_req}: {first_text}"
    else:
        reason = (
            f"All {len(req_status)} applicable evidence requirements are satisfied "
            f"by the submitted image set."
        )
    return True, reason


# ---------------------------------------------------------------------------
# Visual-fact aggregation helpers
# ---------------------------------------------------------------------------

def _pick_best_observation(
    observations: tuple[ImageObservation, ...],
    claimed_parts: frozenset[str],
    claimed_issues: frozenset[str],
) -> ImageObservation | None:
    """Pick the single most informative observation for fact extraction.

    Priority:
    1. Reviewable + target part visible + claimed issue visible
    2. Reviewable + target part visible
    3. Reviewable + any concrete visible parts
    4. Reviewable (fallback)
    5. Any observation (final fallback)
    """
    def score(obs: ImageObservation) -> tuple[int, int, int, int]:
        if not obs.reviewable:
            return (0, 0, 0, 0)
        visible_parts = set(obs.visible_parts) - {"unknown"}
        visible_issues = set(obs.visible_issue_types) - {"unknown", "none"}
        target_visible = obs.target_part_visibility == "visible"
        parts_match = bool(claimed_parts & visible_parts) if claimed_parts else bool(visible_parts)
        issues_match = bool(claimed_issues & visible_issues) if claimed_issues else False
        return (
            1,
            int(target_visible),
            int(issues_match),
            int(parts_match),
        )

    if not observations:
        return None
    ordered = _stable_observations(observations)
    return max(ordered, key=lambda item: (score(item), item.path, item.image_id))


def _derive_issue_type(
    obs: ImageObservation,
    claimed_issues: frozenset[str] = frozenset(),
) -> str:
    """Extract the single most relevant issue type from an observation."""
    if obs.visible_issue_types == ("unknown",):
        return "unknown"
    if obs.visible_issue_types == ("none",):
        return "none"
    concrete = [t for t in obs.visible_issue_types if t not in {"unknown", "none"}]
    for issue in concrete:
        if issue in claimed_issues:
            return issue
    return concrete[0] if concrete else "unknown"


def _derive_object_part(
    obs: ImageObservation,
    prepared: PreparedClaim,
) -> str:
    """Pick the most relevant object_part from an observation.

    Prefer a claimed part that is actually visible. Fall back to any visible
    part for the correct object, then to 'unknown'.
    """
    allowed_parts = OBJECT_PARTS[prepared.claim.claim_object]
    claimed_parts = frozenset(prepared.parsed_claim.claimed_parts) - {"unknown"}

    visible = set(obs.visible_parts) - {"unknown"}

    # Prefer a part that is both claimed and visible
    overlap = claimed_parts & visible & allowed_parts
    if overlap:
        # Return the first claimed part found in visible order
        for part in obs.visible_parts:
            if part in overlap:
                return part

    # Fall back to any actually visible allowed part.
    for part in obs.visible_parts:
        if part in allowed_parts and part != "unknown":
            return part

    return "unknown"


def _derive_severity(obs: ImageObservation) -> str:
    return obs.severity if obs.severity in SEVERITIES else "unknown"


def _aggregate_risk_flags(
    observations: tuple[ImageObservation, ...],
    history_flags: frozenset[str],
) -> tuple[str, ...]:
    """Merge per-image risk flags with user-history risk flags."""
    combined: set[str] = set()
    for obs in observations:
        for flag in obs.risk_flags:
            if flag != "none":
                combined.add(flag)
    combined |= history_flags
    if not combined:
        return ("none",)
    return tuple(sorted(combined))


def _is_valid_image_set(
    observations: tuple[ImageObservation, ...],
) -> bool:
    """True if the image set is usable for automated review.

    False only when ALL images are non-reviewable, or the entire set has
    authenticity/manipulation issues that undermine any conclusion.
    """
    if not observations:
        return False
    authenticity_issues = {"possible_manipulation", "non_original_image"}
    all_unreviewed = all(not obs.reviewable for obs in observations)
    all_authenticity_broken = all(
        set(obs.risk_flags) & authenticity_issues for obs in observations
    )
    return not (all_unreviewed or all_authenticity_broken)


# ---------------------------------------------------------------------------
# Three-state decision
# ---------------------------------------------------------------------------

def _decide_claim_status(
    observations: tuple[ImageObservation, ...],
    prepared: PreparedClaim,
    evidence_standard_met: bool,
    best: ImageObservation | None,
    facts: CaseVisualFacts | None = None,
) -> str:
    """Determine supported / contradicted / not_enough_information.

    Priority order from requirements §3.4:
    1. Evidence insufficient → not_enough_information
    2. Evidence sufficient + consistent visual facts → supported
    3. Evidence sufficient + inconsistent visual facts → contradicted
    """
    if not evidence_standard_met or best is None:
        return "not_enough_information"

    facts = facts or build_case_visual_facts(observations, prepared)
    claimed_parts = frozenset(prepared.parsed_claim.claimed_parts) - {"unknown"}
    claimed_issues = frozenset(
        prepared.parsed_claim.claimed_issue_types
    ) - {"unknown", "none"}

    # --- contradicted path 1: wrong object ---
    if any(
        "wrong_object" in item.risk_flags
        or item.actual_object
        not in {"unknown", prepared.claim.claim_object}
        for item in observations
    ):
        return "contradicted"

    # Unknown claim scope must not be expanded from visual facts.
    if not claimed_parts or not claimed_issues:
        return "not_enough_information"

    # Multiple confirmed parts use conservative ALL semantics until Sprint 1
    # supplies an explicit scope operator.  Partial coverage is insufficient.
    if not claimed_parts.issubset(facts.visible_parts):
        return "not_enough_information"

    observations_by_part = {
        part: tuple(
            item
            for item in facts.reviewable_observations
            if part in item.visible_parts
        )
        for part in claimed_parts
    }
    part_has_matching_damage = {
        part: any(
            claimed_issues
            & (set(item.visible_issue_types) - {"unknown", "none"})
            for item in part_observations
        )
        for part, part_observations in observations_by_part.items()
    }

    # Every confirmed target must have traceable matching damage evidence.
    if all(part_has_matching_damage.values()) and claimed_issues.issubset(
        facts.visible_issue_types
    ):
        return "supported"

    # An image showing an opened contents area without an explicit
    # ``missing_part`` observation cannot prove that the expected item was
    # present.  Absence claims require positive evidence of absence or remain
    # insufficient; generic "no damage" is not a contradiction.
    if "missing_part" in claimed_issues and "missing_part" not in (
        facts.visible_issue_types
    ):
        return "not_enough_information"

    # A fully visible target with an explicit no-damage observation contradicts
    # the claim only when no matching damage exists for that target.
    no_damage_parts = {
        part
        for part, part_observations in observations_by_part.items()
        if part_observations
        and any(
            item.target_part_visibility == "visible"
            and item.visible_issue_types == ("none",)
            for item in part_observations
        )
        and not part_has_matching_damage[part]
    }
    if no_damage_parts == claimed_parts:
        return "contradicted"

    # Concrete visual damage that has no overlap with the user's asserted
    # damage is a contradiction once all target parts are visible.
    if facts.visible_issue_types and not (
        claimed_issues & facts.visible_issue_types
    ):
        return "contradicted"

    # Partial target/damage coverage remains insufficient rather than silently
    # broadening the user's claim.
    return "not_enough_information"


# ---------------------------------------------------------------------------
# Supporting image selection
# ---------------------------------------------------------------------------

def _select_supporting_images(
    observations: tuple[ImageObservation, ...],
    claim_status: str,
    claimed_parts: frozenset[str],
    claimed_issues: frozenset[str],
) -> tuple[str, ...]:
    """Return the image IDs that materially support the final conclusion.

    Rules from requirements §3.5:
    - not_enough_information with no useful images → ("none",) sentinel
    - Only images that actually contribute to the conclusion
    - Multi-image identity conflict: list the conflicting pair
    - contradicted: list the image showing the contradiction
    - supported: list only the image(s) showing the claimed damage
    """
    ordered = _stable_observations(observations)
    if claim_status == "not_enough_information":
        # Include images that contributed to the determination even if insufficient
        useful = [
            obs for obs in ordered
            if obs.reviewable and obs.visible_parts != ("unknown",)
        ]
        if not useful:
            return ()  # will become "none" in serialisation
        # If there's a multi-image identity conflict, include all conflicting images
        has_identity_conflict = any(
            "wrong_object" in obs.risk_flags or "claim_mismatch" in obs.risk_flags
            for obs in ordered
        )
        if has_identity_conflict:
            return tuple(obs.image_id for obs in ordered)
        return ()

    if claim_status == "contradicted":
        # The image(s) showing the contradiction
        contradiction_ids: list[str] = []
        for obs in ordered:
            visible_parts = frozenset(obs.visible_parts) - {"unknown"}
            visible_issues = frozenset(obs.visible_issue_types) - {"unknown"}
            target_ok = (
                claimed_parts & visible_parts
                or (not claimed_parts and visible_parts)
                or obs.target_part_visibility == "visible"
            )
            if obs.reviewable and target_ok:
                contradiction_ids.append(obs.image_id)
        return tuple(contradiction_ids) if contradiction_ids else ()

    # supported — the image(s) that clearly show the claimed damage
    assert claim_status == "supported"
    supporting: list[str] = []
    covered_parts: set[str] = set()
    covered_issues: set[str] = set()
    for obs in ordered:
        if not obs.reviewable:
            continue
        visible_parts = frozenset(obs.visible_parts) - {"unknown"}
        visible_issues = frozenset(obs.visible_issue_types) - {"unknown", "none"}
        matched_parts = claimed_parts & visible_parts
        matched_issues = claimed_issues & visible_issues
        parts_ok = bool(matched_parts)
        issues_ok = bool(matched_issues)
        target_visible = obs.target_part_visibility == "visible"
        if parts_ok and issues_ok and target_visible:
            supporting.append(obs.image_id)
            covered_parts.update(matched_parts)
            covered_issues.update(matched_issues)
    # If strict match found nothing but we know claim is supported, fall back to
    # any reviewable image that at least shows the right object
    if not supporting:
        for obs in ordered:
            if obs.reviewable and obs.target_part_visibility == "visible":
                supporting.append(obs.image_id)
    elif not claimed_parts.issubset(covered_parts) or not claimed_issues.issubset(
        covered_issues
    ):
        # A caller must not serialize a partially supported multi-target claim
        # as if the selected images covered the complete confirmed scope.
        return ()
    return tuple(supporting)


# ---------------------------------------------------------------------------
# Justification text generation
# ---------------------------------------------------------------------------

def _build_evidence_reason(
    evidence_standard_met: bool,
    reason_from_reqs: str,
    observations: tuple[ImageObservation, ...],
    best: ImageObservation | None,
) -> str:
    if not evidence_standard_met:
        return reason_from_reqs
    if best is None:
        return reason_from_reqs
    # Enrich with the best image's fact summary if it adds context
    summary = best.fact_summary.strip().rstrip(".")
    return f"{summary}." if summary else reason_from_reqs


def _build_claim_justification(
    claim_status: str,
    observations: tuple[ImageObservation, ...],
    prepared: PreparedClaim,
    best: ImageObservation | None,
    supporting_ids: tuple[str, ...],
    evidence_standard_met: bool,
) -> str:
    """Build a concise, image-fact-grounded justification string."""
    claimed_parts = list(prepared.parsed_claim.claimed_parts)
    claimed_issues = list(prepared.parsed_claim.claimed_issue_types)

    if best is None:
        return "No images were available for review."

    fact = best.fact_summary.strip().rstrip(".")
    img_ref = (
        f" ({best.image_id})" if len(observations) > 1 else ""
    )
    supporting_ref = (
        f" The conclusion is based on image(s): {', '.join(supporting_ids)}."
        if supporting_ids
        else ""
    )

    if claim_status == "not_enough_information":
        if not evidence_standard_met:
            return (
                f"The submitted image(s) are insufficient to evaluate the claim. "
                f"{fact}{img_ref}.{supporting_ref}"
            ).strip()
        return (
            f"The available images do not provide enough information to reach "
            f"a reliable conclusion. {fact}{img_ref}.{supporting_ref}"
        ).strip()

    if claim_status == "supported":
        return (
            f"The image{img_ref} supports the claim. {fact}.{supporting_ref}"
        ).strip()

    # contradicted
    visible_issues = [
        t for t in best.visible_issue_types
        if t not in {"unknown", "none"}
    ]
    if "wrong_object" in best.risk_flags:
        return (
            f"The image{img_ref} does not show the claimed {prepared.claim.claim_object}. "
            f"{fact}.{supporting_ref}"
        ).strip()
    if best.visible_issue_types == ("none",):
        return (
            f"The image{img_ref} shows the claimed part but no visible damage, "
            f"contradicting the damage claim. {fact}.{supporting_ref}"
        ).strip()
    if visible_issues and claimed_issues and not (
        set(claimed_issues) - {"unknown", "none"}
    ) & set(visible_issues):
        visible_str = ", ".join(visible_issues)
        claimed_str = ", ".join(
            t for t in claimed_issues if t not in {"unknown", "none"}
        )
        return (
            f"The image{img_ref} shows {visible_str} rather than the claimed "
            f"{claimed_str}. {fact}.{supporting_ref}"
        ).strip()
    return (
        f"The image evidence contradicts the claim. {fact}.{supporting_ref}"
    ).strip()


# ---------------------------------------------------------------------------
# Main aggregation entry point
# ---------------------------------------------------------------------------

def aggregate_visual_case(case: VisualReviewCase) -> ReviewResult:
    """Convert a Sprint 2 VisualReviewCase into a complete ReviewResult.

    This is the Sprint 3 deterministic aggregation layer. It:
    1. Collects all image risk flags and adds user-history risk context.
    2. Determines valid_image from reviewability and authenticity.
    3. Selects the best observation for visual-fact extraction.
    4. Derives issue_type, object_part, severity from visual facts.
    5. Evaluates evidence standard against matched requirements.
    6. Compares user claim intent with visual facts → three-state status.
    7. Selects supporting image IDs traceable to ImageObservation records.
    8. Generates short, image-grounded reason and justification text.
    """
    prepared = case.prepared_claim
    observations = case.observations

    claimed_parts = frozenset(prepared.parsed_claim.claimed_parts) - {"unknown"}
    claimed_issues = frozenset(
        prepared.parsed_claim.claimed_issue_types
    ) - {"unknown", "none"}

    facts = build_case_visual_facts(observations, prepared)

    # 1. Risk flags
    history_flags = _history_risk_flags(prepared.history)
    risk_flags = _aggregate_risk_flags(observations, history_flags)
    if facts.identity_conflict and "manual_review_required" not in risk_flags:
        risk_flags = tuple(
            sorted(
                (set(risk_flags) - {"none"})
                | {"claim_mismatch", "manual_review_required"}
            )
        )

    # 2. Valid image set
    valid_image = _is_valid_image_set(observations)

    # 3. Best observation for fact extraction
    best = _pick_best_observation(observations, claimed_parts, claimed_issues)

    # 4. Visual fact fields
    if best is not None and best.reviewable:
        issue_type = _derive_issue_type(best, claimed_issues)
        object_part = _derive_object_part(best, prepared)
        severity = _derive_severity(best)
    else:
        issue_type = "unknown"
        object_part = "unknown"
        severity = "unknown"

    # Ensure object_part is valid for this claim_object
    allowed_parts = OBJECT_PARTS[prepared.claim.claim_object]
    if object_part not in allowed_parts:
        object_part = "unknown"

    # 5. Evidence standard
    evidence_standard_met, req_reason = _evaluate_evidence_standard(
        observations, prepared, facts
    )
    evidence_reason = _build_evidence_reason(
        evidence_standard_met, req_reason, observations, best
    )

    # 6. Three-state claim status
    claim_status = _decide_claim_status(
        observations, prepared, evidence_standard_met, best, facts
    )

    # Enforce consistency: issue_type/severity must reflect claim_status
    if claim_status == "not_enough_information" and not facts.identity_conflict:
        if issue_type not in {"none", "unknown"}:
            issue_type = "unknown"
        if severity not in {"none", "unknown"}:
            severity = "unknown"
    if claim_status == "contradicted" and best is not None:
        if best.visible_issue_types == ("none",):
            issue_type = "none"
            severity = "none"

    # Ensure issue_type=none → severity=none invariant
    if issue_type == "none" and severity != "none":
        severity = "none"

    # 7. Supporting image IDs
    supporting_ids = _select_supporting_images(
        observations, claim_status, claimed_parts, claimed_issues
    )
    # Validate: IDs must come from this claim's image_paths
    valid_image_ids = frozenset(prepared.claim.image_ids)
    supporting_ids = tuple(
        img_id for img_id in supporting_ids if img_id in valid_image_ids
    )

    # 8. Justification
    claim_justification = _build_claim_justification(
        claim_status, observations, prepared, best, supporting_ids,
        evidence_standard_met,
    )

    result = ReviewResult(
        evidence_standard_met=evidence_standard_met,
        evidence_standard_met_reason=evidence_reason,
        risk_flags=risk_flags,
        issue_type=issue_type,
        object_part=object_part,
        claim_status=claim_status,
        claim_status_justification=claim_justification,
        supporting_image_ids=supporting_ids,
        valid_image=valid_image,
        severity=severity,
    )
    # Run the built-in cross-field validator before returning
    result.validate(prepared.claim)
    return result


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def load_visual_reviews(
    input_path: Path,
    prepared_claims: Sequence[PreparedClaim],
) -> list[VisualReviewCase]:
    """Load Sprint 2 JSON and bind it to freshly validated PreparedClaims.

    The persisted Sprint 2 artifact intentionally does not duplicate resolved
    paths or full user-history records.  Binding by user_id restores that
    trusted local context and rejects missing, duplicate, reordered, or
    invented image references.
    """
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DataValidationError(
            f"Visual review JSON does not exist: {input_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise DataValidationError(
            f"Visual review JSON is invalid: {exc}"
        ) from exc
    if not isinstance(payload, list):
        raise DataValidationError("Visual review JSON must be an array")

    prepared_by_user = {
        item.claim.user_id: item for item in prepared_claims
    }
    if len(prepared_by_user) != len(prepared_claims):
        raise DataValidationError("Prepared claims contain duplicate user_id values")

    stored_by_user: dict[str, Mapping[str, Any]] = {}
    for raw_case in payload:
        if not isinstance(raw_case, Mapping):
            raise DataValidationError("Each visual review case must be an object")
        user_id = raw_case.get("user_id")
        if not isinstance(user_id, str) or not user_id:
            raise DataValidationError("Visual review case has invalid user_id")
        if user_id in stored_by_user:
            raise DataValidationError(
                f"Visual review JSON contains duplicate user_id={user_id}"
            )
        stored_by_user[user_id] = raw_case

    missing = set(prepared_by_user) - set(stored_by_user)
    extra = set(stored_by_user) - set(prepared_by_user)
    if missing or extra:
        raise DataValidationError(
            "Visual review case mismatch; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    cases: list[VisualReviewCase] = []
    for prepared in prepared_claims:
        raw_case = stored_by_user[prepared.claim.user_id]
        if raw_case.get("claim_object") != prepared.claim.claim_object:
            raise DataValidationError(
                f"Visual review changed claim_object for {prepared.claim.user_id}"
            )
        raw_observations = raw_case.get("observations")
        if not isinstance(raw_observations, list):
            raise DataValidationError(
                f"observations must be an array for {prepared.claim.user_id}"
            )
        images_by_identity = {
            (image.image_id, image.path): image
            for image in prepared.claim.images
        }
        observations: list[ImageObservation] = []
        seen_images: set[tuple[str, str]] = set()
        for raw_observation in raw_observations:
            observation = _stored_image_observation(raw_observation)
            identity = (observation.image_id, observation.path)
            image = images_by_identity.get(identity)
            if image is None:
                raise DataValidationError(
                    "Visual review contains an image outside the claim: "
                    f"{prepared.claim.user_id}:{observation.path}"
                )
            if identity in seen_images:
                raise DataValidationError(
                    f"Duplicate visual observation: {observation.path}"
                )
            validate_image_observation(observation, prepared, image)
            observations.append(observation)
            seen_images.add(identity)
        expected_images = set(images_by_identity)
        if seen_images != expected_images:
            missing_images = sorted(expected_images - seen_images)
            raise DataValidationError(
                f"Missing visual observations for {prepared.claim.user_id}: "
                f"{missing_images}"
            )
        cases.append(
            VisualReviewCase(prepared, _stable_observations(observations))
        )
    return cases


def _stored_image_observation(value: Any) -> ImageObservation:
    if not isinstance(value, Mapping):
        raise DataValidationError("Stored ImageObservation must be an object")
    required = {
        "image_id",
        "path",
        "actual_object",
        "visible_parts",
        "visible_issue_types",
        "severity",
        "target_part_visibility",
        "requirement_results",
        "fact_summary",
        "risk_flags",
        "reviewable",
        "claim_target_clear",
        "model_name",
        "attempts",
        "diagnostics",
        "escalation",
    }
    if set(value) != required:
        raise DataValidationError(
            "Stored ImageObservation fields mismatch; "
            f"missing={sorted(required - set(value))}, "
            f"extra={sorted(set(value) - required)}"
        )
    requirement_values = value["requirement_results"]
    if not isinstance(requirement_values, list):
        raise DataValidationError("requirement_results must be an array")
    requirement_results: list[RequirementObservation] = []
    for item in requirement_values:
        if not isinstance(item, Mapping) or set(item) != {
            "requirement_id",
            "status",
            "reason",
        }:
            raise DataValidationError("Stored requirement result is invalid")
        requirement_results.append(
            RequirementObservation(
                requirement_id=_stored_string(
                    item["requirement_id"], "requirement_id"
                ),
                status=_stored_string(item["status"], "requirement status"),
                reason=_stored_string(item["reason"], "requirement reason"),
            )
        )
    return ImageObservation(
        image_id=_stored_string(value["image_id"], "image_id"),
        path=_stored_string(value["path"], "path"),
        actual_object=_stored_string(value["actual_object"], "actual_object"),
        visible_parts=_stored_string_tuple(
            value["visible_parts"], "visible_parts"
        ),
        visible_issue_types=_stored_string_tuple(
            value["visible_issue_types"], "visible_issue_types"
        ),
        severity=_stored_string(value["severity"], "severity"),
        target_part_visibility=_stored_string(
            value["target_part_visibility"], "target_part_visibility"
        ),
        requirement_results=tuple(requirement_results),
        fact_summary=_stored_string(value["fact_summary"], "fact_summary"),
        risk_flags=_stored_string_tuple(value["risk_flags"], "risk_flags"),
        reviewable=_stored_bool(value["reviewable"], "reviewable"),
        claim_target_clear=_stored_bool(
            value["claim_target_clear"], "claim_target_clear"
        ),
        model_name=_stored_string(value["model_name"], "model_name"),
        attempts=_stored_positive_int(value["attempts"], "attempts"),
        diagnostics=_stored_string_tuple(
            value["diagnostics"], "diagnostics", allow_empty=True
        ),
        escalation=_stored_escalation(value["escalation"]),
    )


def _stored_escalation(value: Any) -> EscalationAudit | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise DataValidationError("Stored escalation must be an object or null")
    required = {
        "reasons",
        "status",
        "review_model_name",
        "attempts",
        "conflicts",
        "diagnostics",
        "review_candidate",
    }
    if set(value) != required:
        raise DataValidationError("Stored escalation fields are invalid")
    review_model_name = value["review_model_name"]
    if review_model_name is not None:
        review_model_name = _stored_string(
            review_model_name, "review_model_name"
        )
    review_candidate = value["review_candidate"]
    if review_candidate is not None and not isinstance(review_candidate, Mapping):
        raise DataValidationError("review_candidate must be an object or null")
    return EscalationAudit(
        reasons=_stored_string_tuple(value["reasons"], "escalation reasons"),
        status=_stored_string(value["status"], "escalation status"),
        review_model_name=review_model_name,
        attempts=_stored_non_negative_int(
            value["attempts"], "escalation attempts"
        ),
        conflicts=_stored_string_tuple(
            value["conflicts"], "escalation conflicts", allow_empty=True
        ),
        diagnostics=_stored_string_tuple(
            value["diagnostics"], "escalation diagnostics", allow_empty=True
        ),
        review_candidate=dict(review_candidate)
        if review_candidate is not None
        else None,
    )


def _stored_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _stored_string_tuple(
    value: Any,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise DataValidationError(
            f"{field_name} must be an array of non-empty strings"
        )
    result = tuple(dict.fromkeys(item.strip() for item in value))
    if not allow_empty and not result:
        raise DataValidationError(f"{field_name} cannot be empty")
    return result


def _stored_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise DataValidationError(f"{field_name} must be a boolean")
    return value


def _stored_non_negative_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DataValidationError(f"{field_name} must be a non-negative integer")
    return value


def _stored_positive_int(value: Any, field_name: str) -> int:
    result = _stored_non_negative_int(value, field_name)
    if result < 1:
        raise DataValidationError(f"{field_name} must be at least 1")
    return result


def write_final_output(
    cases: Iterable[VisualReviewCase],
    output_path: Path,
) -> int:
    """Write the final 14-column output.csv from Sprint 3 aggregation results.

    Each VisualReviewCase is aggregated deterministically and written as one
    row, preserving the original input order.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for case in cases:
            result = aggregate_visual_case(case)
            writer.writerow(result.to_output_row(case.prepared_claim.claim))
            count += 1
    return count
