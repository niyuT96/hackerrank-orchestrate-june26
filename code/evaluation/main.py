"""Sprint 4 layered evaluation, validation, and submission packaging."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from claim_agent import (  # noqa: E402
    CLAIM_STATUSES,
    ISSUE_TYPES,
    OBJECT_PARTS,
    OUTPUT_COLUMNS,
    RISK_FLAGS,
    SEVERITIES,
    CompositeClaimParser,
    DataValidationError,
    LLMClaimParser,
    OpenAIResponsesClaimClient,
    PreparedClaim,
    load_bundle,
)
from decision_agent import (  # noqa: E402
    aggregate_visual_case,
    load_visual_reviews,
    write_final_output,
)
from visual_agent import (  # noqa: E402
    ImageObservation,
    VisualReviewCase,
    failed_image_observation,
    parse_image_observation,
)


FINAL_SCALAR_FIELDS = (
    "claim_status",
    "evidence_standard_met",
    "valid_image",
    "issue_type",
    "object_part",
    "severity",
)
FINAL_SET_FIELDS = ("risk_flags", "supporting_image_ids")
SENSITIVE_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
IDENTITY_COUPLING_PATTERNS = (
    re.compile(r"\buser_\d{3,}\b", re.IGNORECASE),
    re.compile(r"\bcase_\d{3,}\b", re.IGNORECASE),
    re.compile(r"images/(?:sample|test)/", re.IGNORECASE),
)


@dataclass(frozen=True)
class SetMetrics:
    exact_match: float
    precision: float
    recall: float
    f1: float


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise DataValidationError(f"CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _split_set(value: str | Sequence[str], *, none_is_empty: bool = True) -> set[str]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(";") if item.strip()]
    else:
        items = [str(item).strip() for item in value if str(item).strip()]
    result = set(items)
    if none_is_empty:
        result.discard("none")
    return result


def set_metrics(
    expected_values: Sequence[set[str]],
    actual_values: Sequence[set[str]],
) -> SetMetrics:
    if len(expected_values) != len(actual_values):
        raise ValueError("Metric inputs must have equal length")
    exact = sum(
        expected == actual
        for expected, actual in zip(expected_values, actual_values)
    )
    true_positive = sum(
        len(expected & actual)
        for expected, actual in zip(expected_values, actual_values)
    )
    false_positive = sum(
        len(actual - expected)
        for expected, actual in zip(expected_values, actual_values)
    )
    false_negative = sum(
        len(expected - actual)
        for expected, actual in zip(expected_values, actual_values)
    )
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return SetMetrics(
        exact_match=_ratio(exact, len(expected_values)),
        precision=precision,
        recall=recall,
        f1=f1,
    )


def deterministic_split(user_id: str) -> str:
    """Stable 70/30 case-level split independent of CSV order."""
    bucket = int(hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:8], 16)
    return "development" if bucket % 10 < 7 else "frozen_validation"


def _rows_by_user(rows: Sequence[Mapping[str, str]]) -> dict[str, Mapping[str, str]]:
    result: dict[str, Mapping[str, str]] = {}
    for row in rows:
        user_id = row.get("user_id", "")
        if not user_id or user_id in result:
            raise DataValidationError(f"Missing or duplicate user_id: {user_id!r}")
        result[user_id] = row
    return result


def evaluate_final_rows(
    labels: Sequence[Mapping[str, str]],
    predictions: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    expected = _rows_by_user(labels)
    actual = _rows_by_user(predictions)
    if set(expected) != set(actual):
        raise DataValidationError(
            "Prediction users do not match labels; "
            f"missing={sorted(set(expected) - set(actual))}, "
            f"extra={sorted(set(actual) - set(expected))}"
        )
    ordered_ids = [row["user_id"] for row in labels]
    scalar = {
        field: _mean(
            expected[user_id][field] == actual[user_id][field]
            for user_id in ordered_ids
        )
        for field in FINAL_SCALAR_FIELDS
    }
    set_fields: dict[str, dict[str, float]] = {}
    for field in FINAL_SET_FIELDS:
        metrics = set_metrics(
            [_split_set(expected[user_id][field]) for user_id in ordered_ids],
            [_split_set(actual[user_id][field]) for user_id in ordered_ids],
        )
        set_fields[field] = asdict(metrics)
    exact_rows = _mean(
        all(
            expected[user_id][field] == actual[user_id][field]
            for field in FINAL_SCALAR_FIELDS
        )
        and all(
            _split_set(expected[user_id][field])
            == _split_set(actual[user_id][field])
            for field in FINAL_SET_FIELDS
        )
        for user_id in ordered_ids
    )
    errors = [
        {
            "user_id": user_id,
            "split": deterministic_split(user_id),
            "mismatched_fields": [
                field
                for field in FINAL_SCALAR_FIELDS
                if expected[user_id][field] != actual[user_id][field]
            ]
            + [
                field
                for field in FINAL_SET_FIELDS
                if _split_set(expected[user_id][field])
                != _split_set(actual[user_id][field])
            ],
        }
        for user_id in ordered_ids
        if any(
            expected[user_id][field] != actual[user_id][field]
            for field in FINAL_SCALAR_FIELDS
        )
        or any(
            _split_set(expected[user_id][field])
            != _split_set(actual[user_id][field])
            for field in FINAL_SET_FIELDS
        )
    ]
    split_metrics = {}
    for split in ("development", "frozen_validation"):
        ids = [user_id for user_id in ordered_ids if deterministic_split(user_id) == split]
        split_metrics[split] = {
            field: _mean(
                expected[user_id][field] == actual[user_id][field]
                for user_id in ids
            )
            for field in FINAL_SCALAR_FIELDS
        }
        split_metrics[split]["case_count"] = len(ids)
    return {
        "case_count": len(ordered_ids),
        "scalar_accuracy": scalar,
        "set_metrics": set_fields,
        "exact_row_match": exact_rows,
        "split_metrics": split_metrics,
        "errors": errors,
    }


def parser_diagnostics(
    prepared_claims: Sequence[PreparedClaim],
) -> dict[str, Any]:
    selected = Counter()
    diagnostic_counts: Counter[str] = Counter()
    unknown_counts: Counter[str] = Counter()
    llm_results = 0
    disagreements = 0
    for prepared in prepared_claims:
        decision = prepared.parser_decision
        selected[decision.selected.parser_name] += 1
        llm_results += decision.llm_result is not None
        disagreements += "parser_disagreement" in decision.diagnostics
        for diagnostic in decision.diagnostics:
            diagnostic_counts[diagnostic.split(":", 1)[0]] += 1
        parsed = decision.selected
        unknown_counts["claimed_parts"] += parsed.claimed_parts == ("unknown",)
        unknown_counts["claimed_issue_types"] += (
            parsed.claimed_issue_types == ("unknown",)
        )
        unknown_counts["claimed_severity"] += (
            parsed.claimed_severity == "unknown"
        )
    return {
        "case_count": len(prepared_claims),
        "selected_parser_counts": dict(sorted(selected.items())),
        "validated_llm_result_count": llm_results,
        "parser_disagreement_count": disagreements,
        "unknown_field_counts": dict(sorted(unknown_counts.items())),
        "diagnostic_counts": dict(sorted(diagnostic_counts.items())),
        "note": (
            "These are unlabeled diagnostics. No parser accuracy, precision, "
            "recall, or F1 is computed without independent ground truth."
        ),
    }


def evaluate_requirement_matching(
    prepared_claims: Sequence[PreparedClaim],
) -> dict[str, Any]:
    ids_by_object: dict[str, set[str]] = defaultdict(set)
    invalid = []
    family_counts: Counter[str] = Counter()
    for prepared in prepared_claims:
        for requirement in prepared.requirements:
            ids_by_object[prepared.claim.claim_object].add(
                requirement.requirement_id
            )
            family = requirement.requirement_id.split("_", 2)[:2]
            family_counts["_".join(family)] += 1
            if requirement.claim_object not in {
                "all",
                prepared.claim.claim_object,
            }:
                invalid.append(
                    f"{prepared.claim.user_id}:{requirement.requirement_id}"
                )
    return {
        "invalid_object_rule_matches": invalid,
        "rule_ids_by_object": {
            key: sorted(values) for key, values in sorted(ids_by_object.items())
        },
        "rule_family_counts": dict(sorted(family_counts.items())),
    }


def visual_diagnostics(
    cases: Sequence[VisualReviewCase],
) -> dict[str, Any]:
    observations = [
        observation
        for case in cases
        for observation in case.observations
    ]
    risks: Counter[str] = Counter()
    requirement_statuses: Counter[str] = Counter()
    unknown_parts = 0
    unknown_issues = 0
    escalated = 0
    manual = 0
    for observation in observations:
        risks.update(observation.risk_flags)
        requirement_statuses.update(
            result.status for result in observation.requirement_results
        )
        unknown_parts += observation.visible_parts == ("unknown",)
        unknown_issues += observation.visible_issue_types == ("unknown",)
        escalated += observation.escalation is not None
        manual += "manual_review_required" in observation.risk_flags
    return {
        "image_count": len(observations),
        "reviewable_rate": _mean(item.reviewable for item in observations),
        "unknown_part_rate": _ratio(unknown_parts, len(observations)),
        "unknown_issue_rate": _ratio(unknown_issues, len(observations)),
        "escalation_rate": _ratio(escalated, len(observations)),
        "manual_review_rate": _ratio(manual, len(observations)),
        "risk_flag_counts": dict(sorted(risks.items())),
        "requirement_status_counts": dict(sorted(requirement_statuses.items())),
        "note": (
            "These are unlabeled diagnostics. Visual accuracy is not computed "
            "because sample_claims.csv contains case-level, not per-image, labels."
        ),
    }


def _response_output_text(response: Mapping[str, Any]) -> str:
    texts = []
    for output in response.get("output", []):
        if not isinstance(output, Mapping):
            continue
        for content in output.get("content", []):
            if (
                isinstance(content, Mapping)
                and content.get("type") == "output_text"
                and isinstance(content.get("text"), str)
            ):
                texts.append(content["text"])
    if not texts:
        raise DataValidationError("Trace response contained no output_text")
    return "".join(texts)


def primary_only_cases(
    prepared_claims: Sequence[PreparedClaim],
    trace_path: Path,
) -> tuple[list[VisualReviewCase], int]:
    trace_by_path: dict[str, Mapping[str, Any]] = {}
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            trace_by_path[item["path"]] = item
    failures = 0
    cases = []
    for prepared in prepared_claims:
        observations = []
        for image in prepared.claim.images:
            trace = trace_by_path.get(image.path, {})
            raw = trace.get("primary", {}).get("raw_response")
            try:
                if not isinstance(raw, Mapping):
                    raise DataValidationError("primary raw response unavailable")
                observation = parse_image_observation(
                    _response_output_text(raw),
                    prepared,
                    image,
                    model_name=str(trace.get("primary", {}).get("model_name", "primary")),
                    attempts=int(trace.get("primary", {}).get("attempts", 1)),
                )
            except (DataValidationError, TypeError, ValueError, json.JSONDecodeError):
                failures += 1
                observation = failed_image_observation(
                    prepared,
                    image,
                    "primary_trace_reconstruction_failed",
                    attempts=0,
                )
            observations.append(observation)
        cases.append(VisualReviewCase(prepared, tuple(observations)))
    return cases, failures


def _predictions_from_cases(cases: Sequence[VisualReviewCase]) -> list[dict[str, str]]:
    return [
        aggregate_visual_case(case).to_output_row(case.prepared_claim.claim)
        for case in cases
    ]


def operational_stats(
    trace_path: Path,
    *,
    input_price_per_million: float,
    output_price_per_million: float,
    review_input_price_per_million: float,
    review_output_price_per_million: float,
) -> dict[str, Any]:
    calls = []
    image_count = 0
    statuses: Counter[str] = Counter()
    retry_count = 0
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            trace = json.loads(line)
            image_count += 1
            statuses[str(trace.get("status", "unknown"))] += 1
            primary = trace.get("primary", {})
            retry_count += max(0, int(primary.get("attempts", 0)) - 1)
            if isinstance(primary.get("raw_response"), Mapping):
                calls.append(primary["raw_response"])
            escalation = trace.get("escalation", {})
            retry_count += max(0, int(escalation.get("attempts", 0)) - 1)
            if isinstance(escalation.get("raw_response"), Mapping):
                calls.append(escalation["raw_response"])
    input_tokens = sum(
        int(call.get("usage", {}).get("input_tokens", 0)) for call in calls
    )
    output_tokens = sum(
        int(call.get("usage", {}).get("output_tokens", 0)) for call in calls
    )
    latencies = [
        float(call["completed_at"] - call["created_at"])
        for call in calls
        if isinstance(call.get("created_at"), (int, float))
        and isinstance(call.get("completed_at"), (int, float))
    ]
    models = Counter(str(call.get("model", "unknown")) for call in calls)
    estimated_cost = 0.0
    cost_by_model: dict[str, float] = {}
    for model, model_calls in _group_calls_by_model(calls).items():
        model_input = sum(
            int(call.get("usage", {}).get("input_tokens", 0))
            for call in model_calls
        )
        model_output = sum(
            int(call.get("usage", {}).get("output_tokens", 0))
            for call in model_calls
        )
        is_review = model.startswith("gpt-5.5")
        input_price = (
            review_input_price_per_million
            if is_review
            else input_price_per_million
        )
        output_price = (
            review_output_price_per_million
            if is_review
            else output_price_per_million
        )
        cost = (
            model_input * input_price + model_output * output_price
        ) / 1_000_000
        cost_by_model[model] = cost
        estimated_cost += cost
    return {
        "image_count": image_count,
        "model_call_count": len(calls),
        "model_calls_by_model": dict(sorted(models.items())),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "retry_count": retry_count,
        "status_counts": dict(sorted(statuses.items())),
        "failure_rate": _ratio(
            sum(
                count
                for status, count in statuses.items()
                if status in {"image_read_failed", "vlm_failed"}
            ),
            image_count,
        ),
        "average_latency_seconds": _mean(latencies),
        "p95_latency_seconds": (
            sorted(latencies)[max(0, math.ceil(len(latencies) * 0.95) - 1)]
            if latencies
            else 0.0
        ),
        "estimated_cost_usd": estimated_cost,
        "estimated_cost_by_model_usd": cost_by_model,
        "pricing_assumption": {
            "primary_input_usd_per_million_tokens": input_price_per_million,
            "primary_output_usd_per_million_tokens": output_price_per_million,
            "review_input_usd_per_million_tokens": (
                review_input_price_per_million
            ),
            "review_output_usd_per_million_tokens": (
                review_output_price_per_million
            ),
        },
    }


def _group_calls_by_model(
    calls: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for call in calls:
        grouped[str(call.get("model", "unknown"))].append(call)
    return grouped


def validate_output(
    claims_path: Path,
    output_path: Path,
    *,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    claims = _read_csv(claims_path)
    output = _read_csv(output_path)
    errors: list[str] = []
    if expected_rows is not None and len(output) != expected_rows:
        errors.append(f"Expected {expected_rows} output rows, found {len(output)}")
    if len(claims) != len(output):
        errors.append(
            f"Input/output row count differs: {len(claims)} != {len(output)}"
        )
    if output:
        columns = tuple(output[0])
        if columns != OUTPUT_COLUMNS:
            errors.append(
                f"Output columns differ: expected={OUTPUT_COLUMNS}, actual={columns}"
            )
    for index, (claim, row) in enumerate(zip(claims, output), start=2):
        for field in ("user_id", "image_paths", "user_claim", "claim_object"):
            if claim[field] != row[field]:
                errors.append(f"Row {index}: input field changed: {field}")
        if row["claim_status"] not in CLAIM_STATUSES:
            errors.append(f"Row {index}: invalid claim_status")
        if row["issue_type"] not in ISSUE_TYPES:
            errors.append(f"Row {index}: invalid issue_type")
        if row["object_part"] not in OBJECT_PARTS.get(row["claim_object"], set()):
            errors.append(f"Row {index}: invalid object_part")
        if row["severity"] not in SEVERITIES:
            errors.append(f"Row {index}: invalid severity")
        if row["evidence_standard_met"] not in {"true", "false"}:
            errors.append(f"Row {index}: invalid evidence_standard_met")
        if row["valid_image"] not in {"true", "false"}:
            errors.append(f"Row {index}: invalid valid_image")
        risks = _split_set(row["risk_flags"], none_is_empty=False)
        if not risks or not risks.issubset(RISK_FLAGS):
            errors.append(f"Row {index}: invalid risk_flags")
        if "none" in risks and len(risks) > 1:
            errors.append(f"Row {index}: risk_flags mixes none")
        image_ids = {
            Path(item.strip()).stem
            for item in claim["image_paths"].split(";")
            if item.strip()
        }
        supports = _split_set(row["supporting_image_ids"])
        if not supports.issubset(image_ids):
            errors.append(f"Row {index}: invalid supporting_image_ids")
        if row["issue_type"] == "none" and row["severity"] != "none":
            errors.append(f"Row {index}: issue_type=none requires severity=none")
    return {
        "valid": not errors,
        "row_count": len(output),
        "column_count": len(output[0]) if output else 0,
        "errors": errors,
    }


def scan_submission_sources(paths: Sequence[Path]) -> dict[str, Any]:
    secret_hits = []
    coupling_hits = []
    scanned = 0
    for root in paths:
        candidates = [root] if root.is_file() else list(root.rglob("*"))
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in {
                ".py",
                ".md",
                ".json",
                ".txt",
                ".yml",
                ".yaml",
            }:
                continue
            if path.name == "evaluation_results.json":
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            scanned += 1
            for pattern in SENSITIVE_PATTERNS:
                if pattern.search(text):
                    secret_hits.append(str(path.relative_to(REPO_ROOT)))
                    break
            if path.name not in {"README.md"} and "tests" not in path.parts:
                for pattern in IDENTITY_COUPLING_PATTERNS:
                    if pattern.search(text):
                        coupling_hits.append(str(path.relative_to(REPO_ROOT)))
                        break
    return {
        "files_scanned": scanned,
        "secret_hits": sorted(set(secret_hits)),
        "sample_identity_coupling_hits": sorted(set(coupling_hits)),
    }


def strategy_manifest(paths: Sequence[Path], config: Mapping[str, Any]) -> dict[str, Any]:
    hashes = {}
    for path in paths:
        if path.is_file():
            hashes[str(path.relative_to(REPO_ROOT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return {"file_sha256": hashes, "configuration": dict(config)}


def package_code(output_path: Path) -> int:
    include_roots = [CODE_DIR]
    excluded_parts = {"__pycache__", ".pytest_cache"}
    excluded_suffixes = {".pyc", ".pyo"}
    files = [
        path
        for root in include_roots
        for path in root.rglob("*")
        if path.is_file()
        and not excluded_parts.intersection(path.parts)
        and path.suffix.lower() not in excluded_suffixes
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(REPO_ROOT).as_posix())
    return len(files)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def write_report(path: Path, results: Mapping[str, Any]) -> None:
    final = results["final_evaluation"]
    parser = results["parser_diagnostics"]
    strategies = results["case_strategy_comparison"]
    visual = results["visual_diagnostics"]
    operational = results["operational"]
    validation = results["validation"]
    scan = results["source_scan"]
    lines = [
        "# Evaluation Report",
        "",
        "## Ground-truth contract",
        "",
        (
            "`dataset/sample_claims.csv` is the only source of evaluation "
            "ground truth. No human-, AI-, or model-generated claim-intent or "
            "per-image labels are used to calculate accuracy, precision, "
            "recall, or F1."
        ),
        "",
        (
            "Cases are split deterministically by a SHA-256 hash of `user_id`; "
            "the split is case-level, so images from one claim cannot leak "
            "between development and frozen validation."
        ),
        "",
        _markdown_table(
            ("Split", "Cases"),
            [
                (
                    name,
                    metrics["case_count"],
                )
                for name, metrics in final["split_metrics"].items()
            ],
        ),
        "",
        "## End-to-end metrics",
        "",
        _markdown_table(
            ("Field", "Accuracy"),
            [
                (field, _fmt(value))
                for field, value in final["scalar_accuracy"].items()
            ],
        ),
        "",
        _markdown_table(
            ("Set field", "Exact", "Precision", "Recall", "F1"),
            [
                (
                    field,
                    _fmt(metrics["exact_match"]),
                    _fmt(metrics["precision"]),
                    _fmt(metrics["recall"]),
                    _fmt(metrics["f1"]),
                )
                for field, metrics in final["set_metrics"].items()
            ],
        ),
        "",
        f"Exact full-row match: **{_fmt(final['exact_row_match'])}**.",
        "",
        "## Case-level strategy comparison",
        "",
        _markdown_table(
            (
                "Strategy",
                "Claim status",
                "Evidence",
                "Valid image",
                "Issue",
                "Part",
                "Severity",
            ),
            [
                (
                    name,
                    _fmt(metrics["scalar_accuracy"]["claim_status"]),
                    _fmt(metrics["scalar_accuracy"]["evidence_standard_met"]),
                    _fmt(metrics["scalar_accuracy"]["valid_image"]),
                    _fmt(metrics["scalar_accuracy"]["issue_type"]),
                    _fmt(metrics["scalar_accuracy"]["object_part"]),
                    _fmt(metrics["scalar_accuracy"]["severity"]),
                )
                for name, metrics in strategies.items()
            ],
        ),
        "",
        (
            "Both strategies are scored only after producing complete "
            "case-level output rows. `primary_only` reconstructs the first "
            "vision pass from the trace; `review_routed` uses the persisted "
            "post-escalation observations."
        ),
        "",
        "## Claim-understanding diagnostics (unlabeled)",
        "",
        _markdown_table(
            ("Diagnostic", "Value"),
            [
                ("Cases", parser["case_count"]),
                (
                    "Validated LLM results",
                    parser["validated_llm_result_count"],
                ),
                (
                    "Rule/LLM disagreements",
                    parser["parser_disagreement_count"],
                ),
                (
                    "Selected parser counts",
                    json.dumps(
                        parser["selected_parser_counts"], ensure_ascii=False
                    ),
                ),
                (
                    "Unknown field counts",
                    json.dumps(
                        parser["unknown_field_counts"], ensure_ascii=False
                    ),
                ),
            ],
        ),
        "",
        (
            "These are routing and uncertainty diagnostics, not parser "
            "accuracy metrics. The sample file has no independent claim-intent "
            "labels."
        ),
        "",
        "## Visual diagnostics (unlabeled)",
        "",
        _markdown_table(
            ("Diagnostic", "Primary only", "Review routed"),
            [
                (
                    "Images",
                    visual["primary_only"]["image_count"],
                    visual["review_routed"]["image_count"],
                ),
                (
                    "Reviewable rate",
                    _fmt(visual["primary_only"]["reviewable_rate"]),
                    _fmt(visual["review_routed"]["reviewable_rate"]),
                ),
                (
                    "Unknown part rate",
                    _fmt(visual["primary_only"]["unknown_part_rate"]),
                    _fmt(visual["review_routed"]["unknown_part_rate"]),
                ),
                (
                    "Unknown issue rate",
                    _fmt(visual["primary_only"]["unknown_issue_rate"]),
                    _fmt(visual["review_routed"]["unknown_issue_rate"]),
                ),
                (
                    "Escalation rate",
                    _fmt(visual["primary_only"]["escalation_rate"]),
                    _fmt(visual["review_routed"]["escalation_rate"]),
                ),
                (
                    "Manual-review rate",
                    _fmt(visual["primary_only"]["manual_review_rate"]),
                    _fmt(visual["review_routed"]["manual_review_rate"]),
                ),
            ],
        ),
        "",
        (
            "No per-image accuracy is reported because the sample file "
            "contains case-level final labels only."
        ),
        "",
        "## Requirement matching audit",
        "",
        (
            "Invalid object/rule matches: "
            f"**{len(results['requirement_audit']['invalid_object_rule_matches'])}**."
        ),
        "",
        "```json",
        json.dumps(
            results["requirement_audit"]["rule_ids_by_object"],
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## Operational analysis",
        "",
        _markdown_table(
            ("Metric", "Value"),
            [
                ("Images", operational["image_count"]),
                ("Model calls", operational["model_call_count"]),
                ("Input tokens", operational["input_tokens"]),
                ("Output tokens", operational["output_tokens"]),
                ("Retries", operational["retry_count"]),
                (
                    "Average latency (s)",
                    _fmt(operational["average_latency_seconds"]),
                ),
                ("P95 latency (s)", _fmt(operational["p95_latency_seconds"])),
                (
                    "Estimated cost (USD)",
                    f"{operational['estimated_cost_usd']:.4f}",
                ),
                ("Failure rate", _fmt(operational["failure_rate"])),
            ],
        ),
        "",
        "## Error interpretation",
        "",
        (
            f"End-to-end mismatched cases: **{len(final['errors'])}**. "
            "Because no additional layer-level ground truth is used, the "
            "report does not claim exact attribution to parsing, vision, or "
            "aggregation. Unlabeled diagnostics only indicate likely areas "
            "for investigation."
        ),
        "",
        "## Generalization controls",
        "",
        "- Case-level deterministic development/frozen-validation split.",
        "- Unit tests cover image-order invariance and multi-target coverage.",
        "- Test-set outputs are not used as labels or strategy-selection data.",
        "- No AI-generated proxy labels are used.",
        (
            f"- Source scan secret hits: `{scan['secret_hits']}`; "
            "sample-identity coupling hits: "
            f"`{scan['sample_identity_coupling_hits']}`."
        ),
        "",
        "## Submission validation",
        "",
        f"Output valid: **{validation['valid']}**.",
        "",
    ]
    if validation["errors"]:
        lines.extend(["Validation errors:", ""])
        lines.extend(f"- {error}" for error in validation["errors"])
        lines.append("")
    lines.extend(
        [
            "## Frozen strategy manifest",
            "",
            "```json",
            json.dumps(results["strategy_manifest"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    if results.get("test_operational"):
        test_operational = results["test_operational"]
        lines.extend(
            [
                "## Final test-set operational run",
                "",
                _markdown_table(
                    ("Metric", "Value"),
                    [
                        ("Images", test_operational["image_count"]),
                        ("Model calls", test_operational["model_call_count"]),
                        ("Input tokens", test_operational["input_tokens"]),
                        ("Output tokens", test_operational["output_tokens"]),
                        (
                            "Average latency (s)",
                            _fmt(test_operational["average_latency_seconds"]),
                        ),
                        (
                            "P95 latency (s)",
                            _fmt(test_operational["p95_latency_seconds"]),
                        ),
                        (
                            "Estimated cost (USD)",
                            f"{test_operational['estimated_cost_usd']:.4f}",
                        ),
                    ],
                ),
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=REPO_ROOT / "dataset"
    )
    parser.add_argument("--claims-file", default="sample_claims.csv")
    parser.add_argument(
        "--predictions", type=Path, default=REPO_ROOT / "sample_output.csv"
    )
    parser.add_argument(
        "--visual-json",
        type=Path,
        default=REPO_ROOT / "sprint2_observations.json",
    )
    parser.add_argument(
        "--trace-jsonl",
        type=Path,
        default=REPO_ROOT / "sprint2_trace.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=CODE_DIR / "evaluation" / "evaluation_report.md",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=CODE_DIR / "evaluation" / "evaluation_results.json",
    )
    parser.add_argument(
        "--input-price-per-million", type=float, default=0.75
    )
    parser.add_argument(
        "--output-price-per-million", type=float, default=4.50
    )
    parser.add_argument(
        "--review-input-price-per-million", type=float, default=5.00
    )
    parser.add_argument(
        "--review-output-price-per-million", type=float, default=30.00
    )
    parser.add_argument(
        "--test-trace-jsonl",
        type=Path,
        help="Optional final test-set trace for AC-11 operational analysis.",
    )
    parser.add_argument("--expected-output-rows", type=int)
    parser.add_argument(
        "--package", type=Path, help="Write a submission code.zip after evaluation"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate predictions against input rows without requiring labels.",
    )
    parser.add_argument("--vision-workers", type=int, default=1)
    parser.add_argument("--rpm-limit", type=int, default=60)
    parser.add_argument(
        "--claim-provider",
        choices=("none", "openai"),
        default="none",
        help="Optionally run a real LLM claim parser for strategy comparison.",
    )
    parser.add_argument(
        "--claim-model",
        default="gpt-5.4-mini",
    )
    parser.add_argument("--claim-timeout", type=float, default=30.0)
    parser.add_argument("--claim-max-attempts", type=int, default=2)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    bundle = load_bundle(args.dataset_dir, args.claims_file)
    parser_bundle = bundle
    if args.claim_provider == "openai":
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=False)
        model = os.environ.get("OPENAI_CLAIM_MODEL") or args.claim_model
        llm_parser = LLMClaimParser(
            OpenAIResponsesClaimClient(
                model=model,
                timeout_seconds=args.claim_timeout,
            ),
            max_attempts=args.claim_max_attempts,
        )
        parser_bundle = load_bundle(
            args.dataset_dir,
            args.claims_file,
            parser=CompositeClaimParser(
                llm_parser=llm_parser,
                llm_routing="always",
            ),
        )
    labels = _read_csv(args.dataset_dir / args.claims_file)
    cases = load_visual_reviews(args.visual_json, bundle.prepared_claims)
    if not args.predictions.is_file():
        write_final_output(cases, args.predictions)
    predictions = _read_csv(args.predictions)
    final_metrics = evaluate_final_rows(labels, predictions)
    primary_cases, primary_failures = primary_only_cases(
        bundle.prepared_claims, args.trace_jsonl
    )
    primary_metrics = evaluate_final_rows(
        labels, _predictions_from_cases(primary_cases)
    )
    strategy_metrics = {
        "primary_only": primary_metrics,
        "review_routed": final_metrics,
    }
    visual_stats = {
        "primary_only": {
            **visual_diagnostics(primary_cases),
            "trace_reconstruction_failures": primary_failures,
        },
        "review_routed": {
            **visual_diagnostics(cases),
            "trace_reconstruction_failures": 0,
        },
    }
    output_validation = validate_output(
        args.dataset_dir / args.claims_file,
        args.predictions,
        expected_rows=args.expected_output_rows,
    )
    source_scan = scan_submission_sources([CODE_DIR])
    manifest = strategy_manifest(
        [
            CODE_DIR / "claim_lexicon.json",
            CODE_DIR / "claim_agent.py",
            CODE_DIR / "visual_agent.py",
            CODE_DIR / "decision_agent.py",
        ],
        {
            "claims_file": args.claims_file,
            "vision_workers": args.vision_workers,
            "rpm_limit": args.rpm_limit,
            "claim_provider": args.claim_provider,
            "claim_model": (
                os.environ.get("OPENAI_CLAIM_MODEL") or args.claim_model
            ),
            "input_price_per_million": args.input_price_per_million,
            "output_price_per_million": args.output_price_per_million,
            "review_input_price_per_million": (
                args.review_input_price_per_million
            ),
            "review_output_price_per_million": (
                args.review_output_price_per_million
            ),
        },
    )
    results = {
        "final_evaluation": final_metrics,
        "parser_diagnostics": parser_diagnostics(
            parser_bundle.prepared_claims
        ),
        "requirement_audit": evaluate_requirement_matching(
            bundle.prepared_claims
        ),
        "case_strategy_comparison": strategy_metrics,
        "visual_diagnostics": visual_stats,
        "operational": operational_stats(
            args.trace_jsonl,
            input_price_per_million=args.input_price_per_million,
            output_price_per_million=args.output_price_per_million,
            review_input_price_per_million=(
                args.review_input_price_per_million
            ),
            review_output_price_per_million=(
                args.review_output_price_per_million
            ),
        ),
        "validation": output_validation,
        "source_scan": source_scan,
        "strategy_manifest": manifest,
    }
    if args.test_trace_jsonl:
        results["test_operational"] = operational_stats(
            args.test_trace_jsonl,
            input_price_per_million=args.input_price_per_million,
            output_price_per_million=args.output_price_per_million,
            review_input_price_per_million=(
                args.review_input_price_per_million
            ),
            review_output_price_per_million=(
                args.review_output_price_per_million
            ),
        )
    args.results_json.parent.mkdir(parents=True, exist_ok=True)
    args.results_json.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.report, results)
    if args.package:
        package_code(args.package)
    return results


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.validate_only:
            validation = validate_output(
                args.dataset_dir / args.claims_file,
                args.predictions,
                expected_rows=args.expected_output_rows,
            )
            scan = scan_submission_sources([CODE_DIR])
            if args.package and validation["valid"] and not scan["secret_hits"]:
                package_code(args.package)
            print(
                json.dumps(
                    {"validation": validation, "source_scan": scan},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return (
                0
                if validation["valid"]
                and not scan["secret_hits"]
                and not scan["sample_identity_coupling_hits"]
                else 3
            )
        results = run(args)
    except (DataValidationError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(
        "Evaluation complete: "
        f"claim_status_accuracy="
        f"{results['final_evaluation']['scalar_accuracy']['claim_status']:.3f}; "
        f"report={args.report.resolve()}"
    )
    return 0 if results["validation"]["valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
