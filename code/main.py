"""Command-line entry point for the Sprint 1 claim-preparation pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from claim_agent import (
    CompositeClaimParser,
    DataValidationError,
    LLMClaimParser,
    load_bundle,
    static_llm_client,
    write_output,
    write_prepared_claims,
)


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"
DEFAULT_OUTPUT = REPO_ROOT / "sprint1_output.csv"
DEFAULT_PREPARED = REPO_ROOT / "sprint1_summary.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare claims, parse claim intent, match evidence requirements, "
            "and write a schema-valid Sprint 1 placeholder output."
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
        "--skip-image-existence-check",
        action="store_true",
    )
    return parser


def _claim_parser(args: argparse.Namespace) -> CompositeClaimParser:
    if not args.llm_responses_json:
        return CompositeClaimParser()
    payload = json.loads(
        args.llm_responses_json.read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise DataValidationError("LLM responses JSON must be an object keyed by user_id")
    llm_parser = LLMClaimParser(static_llm_client(payload))
    return CompositeClaimParser(llm_parser=llm_parser)


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
    return 0 if row_count == prepared_count else 3


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except (DataValidationError, OSError, json.JSONDecodeError) as exc:
        print(f"Sprint 1 failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
