"""Sprint 3 multi-image aggregation, evidence judgment, and three-state decision."""

from __future__ import annotations

import csv
import re
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from claim_agent import (
    CLAIM_STATUSES,
    ISSUE_TYPES,
    OBJECT_PARTS,
    RISK_FLAGS,
    SEVERITIES,
    ClaimRecord,
    DataValidationError,
    OUTPUT_COLUMNS,
    PreparedClaim,
    ReviewResult,
    UserHistory,
    _bool_text,
)
from visual_agent import ImageObservation, VisualReviewCase


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

    # Check for multi-image identity conflict that undermines the whole set
    if len(observations) > 1:
        concrete_objects = {
            obs.actual_object
            for obs in observations
            if obs.actual_object != "unknown"
        }
        has_identity_conflict = (
            len(concrete_objects) > 1
            or any(
                "wrong_object" in obs.risk_flags
                for obs in observations
            )
        )
        if has_identity_conflict:
            return (
                False,
                "The submitted images appear to show different objects or vehicles, "
                "so the image set cannot reliably support the claim."
            )

    # Build per-requirement best status across all images
    req_status: dict[str, str] = {
        req.requirement_id: "not_met" for req in prepared.requirements
    }
    for obs in observations:
        if not obs.reviewable:
            continue
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
    return max(observations, key=score)


def _derive_issue_type(obs: ImageObservation) -> str:
    """Extract the single most relevant issue type from an observation."""
    if obs.visible_issue_types == ("unknown",):
        return "unknown"
    if obs.visible_issue_types == ("none",):
        return "none"
    concrete = [t for t in obs.visible_issue_types if t not in {"unknown", "none"}]
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

    # Fall back to the first claimed part if we at least know what was claimed
    if claimed_parts:
        first_claimed = next(iter(prepared.parsed_claim.claimed_parts))
        if first_claimed != "unknown" and first_claimed in allowed_parts:
            return first_claimed

    # Fall back to any visible allowed part
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
) -> str:
    """Determine supported / contradicted / not_enough_information.

    Priority order from requirements §3.4:
    1. Evidence insufficient → not_enough_information
    2. Evidence sufficient + consistent visual facts → supported
    3. Evidence sufficient + inconsistent visual facts → contradicted
    """
    if not evidence_standard_met or best is None:
        return "not_enough_information"

    claimed_parts = frozenset(prepared.parsed_claim.claimed_parts) - {"unknown"}
    claimed_issues = frozenset(
        prepared.parsed_claim.claimed_issue_types
    ) - {"unknown", "none"}

    visible_parts = frozenset(best.visible_parts) - {"unknown"}
    visible_issues = frozenset(best.visible_issue_types) - {"unknown", "none"}
    issue_is_none = best.visible_issue_types == ("none",)

    # --- contradicted path 1: wrong object ---
    if "wrong_object" in best.risk_flags:
        return "contradicted"

    # --- contradicted path 2: claimed part visible but explicitly no damage ---
    if (
        best.target_part_visibility == "visible"
        and issue_is_none
        and claimed_issues
    ):
        return "contradicted"

    # --- contradicted path 3: visible issue type explicitly mismatches claimed ---
    if (
        claimed_issues
        and visible_issues
        and not claimed_issues & visible_issues
    ):
        return "contradicted"

    # --- not_enough_information: target part not visible or observation unclear ---
    if best.target_part_visibility in {"not_visible", "unknown"}:
        if claimed_parts and not claimed_parts & visible_parts:
            return "not_enough_information"

    # --- not_enough_information: both sides unknown ---
    if not claimed_parts and not visible_parts:
        return "not_enough_information"

    # --- supported: claimed part visible and damage matches (or no damage claimed) ---
    if claimed_parts:
        parts_ok = bool(claimed_parts & visible_parts)
    else:
        parts_ok = bool(visible_parts)  # anything visible is fine if no part specified

    if claimed_issues:
        issues_ok = bool(claimed_issues & visible_issues)
    else:
        # no specific damage claimed — any visible state (including none) is fine
        issues_ok = True

    if parts_ok and issues_ok:
        return "supported"

    # Conservative default when there is partial information
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
    if claim_status == "not_enough_information":
        # Include images that contributed to the determination even if insufficient
        useful = [
            obs for obs in observations
            if obs.reviewable and obs.visible_parts != ("unknown",)
        ]
        if not useful:
            return ()  # will become "none" in serialisation
        # If there's a multi-image identity conflict, include all conflicting images
        has_identity_conflict = any(
            "wrong_object" in obs.risk_flags or "claim_mismatch" in obs.risk_flags
            for obs in observations
        )
        if has_identity_conflict:
            return tuple(obs.image_id for obs in observations)
        return ()

    if claim_status == "contradicted":
        # The image(s) showing the contradiction
        contradiction_ids: list[str] = []
        for obs in observations:
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
    for obs in observations:
        if not obs.reviewable:
            continue
        visible_parts = frozenset(obs.visible_parts) - {"unknown"}
        visible_issues = frozenset(obs.visible_issue_types) - {"unknown", "none"}
        parts_ok = (
            bool(claimed_parts & visible_parts) if claimed_parts else bool(visible_parts)
        )
        issues_ok = (
            bool(claimed_issues & visible_issues) if claimed_issues else True
        )
        target_visible = obs.target_part_visibility == "visible"
        if parts_ok and issues_ok and target_visible:
            supporting.append(obs.image_id)
    # If strict match found nothing but we know claim is supported, fall back to
    # any reviewable image that at least shows the right object
    if not supporting:
        for obs in observations:
            if obs.reviewable and obs.target_part_visibility == "visible":
                supporting.append(obs.image_id)
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

    # 1. Risk flags
    history_flags = _history_risk_flags(prepared.history)
    risk_flags = _aggregate_risk_flags(observations, history_flags)

    # 2. Valid image set
    valid_image = _is_valid_image_set(observations)

    # 3. Best observation for fact extraction
    best = _pick_best_observation(observations, claimed_parts, claimed_issues)

    # 4. Visual fact fields
    if best is not None and best.reviewable:
        issue_type = _derive_issue_type(best)
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
        observations, prepared
    )
    evidence_reason = _build_evidence_reason(
        evidence_standard_met, req_reason, observations, best
    )

    # 6. Three-state claim status
    claim_status = _decide_claim_status(
        observations, prepared, evidence_standard_met, best
    )

    # Enforce consistency: issue_type/severity must reflect claim_status
    if claim_status == "not_enough_information":
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
