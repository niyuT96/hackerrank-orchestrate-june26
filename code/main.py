"""Command-line entry point for Sprint 1 preparation and Sprint 2 vision."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from claim_agent import (
    CompositeClaimParser,
    DataValidationError,
    LLMClaimParser,
    OpenAIResponsesClaimClient,
    load_bundle,
    static_llm_client,
    write_output,
    write_prepared_claims,
)
from visual_agent import (
    OpenAIResponsesVisionClient,
    ReplayVisionClient,
    VisionReviewer,
    load_replay_responses,
    write_visual_reviews,
    write_visual_traces,
)
from decision_agent import write_final_output


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"
DEFAULT_OUTPUT = REPO_ROOT / "sprint1_output.csv"
DEFAULT_PREPARED = REPO_ROOT / "sprint1_summary.json"
DEFAULT_VISUAL_REVIEWS = REPO_ROOT / "sprint2_observations.json"
DEFAULT_VISUAL_TRACES = REPO_ROOT / "sprint2_trace.jsonl"
DEFAULT_FINAL_OUTPUT = REPO_ROOT / "output.csv"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_CLAIM_MODEL = "gpt-5.4-mini"
DEFAULT_VISION_MODEL = "gpt-5.4-mini"
DEFAULT_REVIEW_MODEL = "gpt-5.5"


def load_environment(env_file: Path = DEFAULT_ENV_FILE) -> bool:
    """Load local configuration without overriding exported environment values."""
    return load_dotenv(dotenv_path=env_file, override=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare claims, parse claim intent, match evidence requirements, "
            "and optionally run the Sprint 2 per-image visual review."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--claims-file", default="claims.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--prepared-json",
        "--summary-json",
        dest="prepared_json",
        type=Path,
        default=DEFAULT_PREPARED,
        help="PreparedClaim JSON for the Sprint 2 visual agent.",
    )
    parser.add_argument(
        "--llm-responses-json",
        type=Path,
        help=(
            "Optional offline/replay structured LLM responses keyed by user_id. "
            "Without this option, the deterministic parser is used."
        ),
    )
    parser.add_argument(
        "--claim-provider",
        choices=("none", "replay", "openai"),
        default="none",
        help=(
            "Optional Sprint 1 LLM provider. In auto routing mode it is called "
            "only for uncertain or complex claims."
        ),
    )
    parser.add_argument(
        "--claim-model",
        default=DEFAULT_CLAIM_MODEL,
        help="OpenAI claim parser model; override with OPENAI_CLAIM_MODEL.",
    )
    parser.add_argument(
        "--claim-routing",
        choices=("auto", "always"),
        default="always",
        help=(
            "Run rule and LLM parsers together by default; use auto only as "
            "an explicit cost-optimization experiment."
        ),
    )
    parser.add_argument(
        "--claim-timeout",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--claim-max-attempts",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--skip-image-existence-check",
        action="store_true",
    )
    parser.add_argument(
        "--vision-provider",
        choices=("none", "replay", "openai"),
        default="none",
        help=(
            "Run Sprint 2 with replay responses or the OpenAI Responses API. "
            "The default 'none' runs Sprint 1 only."
        ),
    )
    parser.add_argument(
        "--vision-responses-json",
        type=Path,
        help="Replay VLM responses keyed by image path or user_id:image_id.",
    )
    parser.add_argument(
        "--vision-review-responses-json",
        type=Path,
        help=(
            "Optional replay responses for difficult-case escalation. "
            "If omitted in replay mode, forced escalations go to manual review."
        ),
    )
    parser.add_argument(
        "--vision-output",
        type=Path,
        default=DEFAULT_VISUAL_REVIEWS,
        help="Per-case Sprint 2 ImageObservation JSON.",
    )
    parser.add_argument(
        "--vision-trace",
        type=Path,
        default=DEFAULT_VISUAL_TRACES,
        help="JSONL trace containing image metadata, attempts, and raw responses.",
    )
    parser.add_argument(
        "--vision-model",
        default=DEFAULT_VISION_MODEL,
        help="OpenAI vision-capable model; override with OPENAI_VISION_MODEL.",
    )
    parser.add_argument(
        "--review-model",
        default=DEFAULT_REVIEW_MODEL,
        help=(
            "OpenAI difficult-case review model; override with "
            "OPENAI_REVIEW_MODEL."
        ),
    )
    parser.add_argument(
        "--vision-timeout",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--vision-max-attempts",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--review-max-attempts",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--vision-retry-delay",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--vision-max-dimension",
        type=int,
        default=1600,
    )
    parser.add_argument(
        "--final-output",
        type=Path,
        default=DEFAULT_FINAL_OUTPUT,
        help=(
            "Sprint 3 final output.csv path. Written only when --vision-provider "
            "is not 'none'."
        ),
    )
    return parser


def _claim_parser(args: argparse.Namespace) -> CompositeClaimParser:
    provider = args.claim_provider
    if args.llm_responses_json:
        if provider == "openai":
            raise DataValidationError(
                "--llm-responses-json cannot be combined with "
                "--claim-provider openai"
            )
        provider = "replay"
    if provider == "none":
        return CompositeClaimParser()
    if provider == "replay":
        if not args.llm_responses_json:
            raise DataValidationError(
                "--llm-responses-json is required for replay claim parsing"
            )
        payload = json.loads(
            args.llm_responses_json.read_text(encoding="utf-8")
        )
        if not isinstance(payload, dict):
            raise DataValidationError(
                "LLM responses JSON must be an object keyed by user_id"
            )
        client = static_llm_client(payload)
    else:
        model = os.environ.get("OPENAI_CLAIM_MODEL") or args.claim_model
        client = OpenAIResponsesClaimClient(
            model=model,
            timeout_seconds=args.claim_timeout,
        )
    llm_parser = LLMClaimParser(
        client,
        max_attempts=args.claim_max_attempts,
    )
    return CompositeClaimParser(
        llm_parser=llm_parser,
        llm_routing=args.claim_routing,
    )


def _vision_reviewer(args: argparse.Namespace) -> VisionReviewer | None:
    if args.vision_provider == "none":
        return None
    if args.vision_provider == "replay":
        if not args.vision_responses_json:
            raise DataValidationError(
                "--vision-responses-json is required for replay vision"
            )
        client = ReplayVisionClient(
            load_replay_responses(args.vision_responses_json)
        )
        review_client = (
            ReplayVisionClient(
                load_replay_responses(args.vision_review_responses_json)
            )
            if args.vision_review_responses_json
            else None
        )
    else:
        model = os.environ.get("OPENAI_VISION_MODEL") or args.vision_model
        client = OpenAIResponsesVisionClient(
            model=model,
            timeout_seconds=args.vision_timeout,
        )
        review_model = (
            os.environ.get("OPENAI_REVIEW_MODEL") or args.review_model
        )
        review_client = OpenAIResponsesVisionClient(
            model=review_model,
            timeout_seconds=args.vision_timeout,
        )
    return VisionReviewer(
        client,
        review_client=review_client,
        max_attempts=args.vision_max_attempts,
        review_max_attempts=args.review_max_attempts,
        retry_delay_seconds=args.vision_retry_delay,
        max_dimension=args.vision_max_dimension,
    )


def run(args: argparse.Namespace) -> int:
    bundle = load_bundle(
        args.dataset_dir,
        args.claims_file,
        require_images=not args.skip_image_existence_check,
        parser=_claim_parser(args),
    )
    row_count = write_output(bundle.prepared_claims, args.output.resolve())
    prepared_count = write_prepared_claims(
        bundle.prepared_claims, args.prepared_json.resolve()
    )
    print(
        f"Prepared {prepared_count} claims; wrote placeholder output to "
        f"{args.output.resolve()} and PreparedClaim JSON to "
        f"{args.prepared_json.resolve()}"
    )
    if row_count != prepared_count:
        return 3

    reviewer = _vision_reviewer(args)
    if reviewer is not None:
        cases, traces = reviewer.review(bundle.prepared_claims)
        case_count = write_visual_reviews(cases, args.vision_output.resolve())
        trace_count = write_visual_traces(traces, args.vision_trace.resolve())
        expected_images = sum(
            len(prepared.claim.images) for prepared in bundle.prepared_claims
        )
        print(
            f"Reviewed {trace_count} images across {case_count} claims; wrote "
            f"observations to {args.vision_output.resolve()} and traces to "
            f"{args.vision_trace.resolve()}"
        )
        if case_count != prepared_count or trace_count != expected_images:
            return 4

        # Sprint 3: deterministic aggregation → final output.csv
        final_count = write_final_output(cases, args.final_output.resolve())
        print(
            f"Sprint 3 aggregation complete; wrote {final_count} decisions to "
            f"{args.final_output.resolve()}"
        )
        if final_count != prepared_count:
            return 5

    return 0


def main() -> int:
    load_environment()
    args = build_parser().parse_args()
    try:
        return run(args)
    except (DataValidationError, OSError, json.JSONDecodeError) as exc:
        print(f"Claim pipeline failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
