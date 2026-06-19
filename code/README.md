# Claim Review Agent

This directory contains the runnable solution for the HackerRank Orchestrate
multi-modal evidence review task.

## Sprint 1 scope

Sprint 1 prepares each raw claim for a later visual agent. It:

- validates claims, user history, evidence rules, and local image paths;
- preserves the original `user_claim` and `claim_object`;
- stores every image as a portable relative `path` plus its output-facing
  `image_id`;
- extracts claimed parts, issue types, severity, excluded parts, and source
  quotes;
- keeps a deterministic rule parser as the baseline and fallback;
- supports an optional structured LLM parser through a provider-neutral
  adapter;
- validates parser enums, object-specific parts, confidence, and evidence
  quotes locally;
- records rule/LLM disagreements without merging their outputs;
- matches only valid requirements from `evidence_requirements.csv`;
- stores both requirement IDs and full minimum-evidence text;
- writes a complete `PreparedClaim` JSON handoff for Sprint 2;
- writes a schema-valid placeholder CSV.

Sprint 1 does **not** inspect image contents. Visual output fields therefore
remain `unknown` or `false`, and every status is
`not_enough_information`. Do not submit `sprint1_output.csv` as the final
competition output.

## Requirements

- Python 3.11 or later
- No third-party packages for the deterministic Sprint 1 pipeline

## Run

From the repository root:

```powershell
python -B code/main.py
```

This creates:

- `sprint1_output.csv`: 14-column placeholder output for all claims;
- `sprint1_summary.json`: complete `PreparedClaim` records for Sprint 2.

Run against the labeled sample inputs:

```powershell
python -B code/main.py `
  --claims-file sample_claims.csv `
  --output sample_sprint1_output.csv `
  --prepared-json sample_sprint1_summary.json
```

All paths can be overridden:

```powershell
python -B code/main.py `
  --dataset-dir dataset `
  --claims-file claims.csv `
  --output sprint1_output.csv `
  --prepared-json sprint1_summary.json
```

`--summary-json` remains an alias for `--prepared-json`.

## Parser architecture

`RuleBasedClaimParser` always runs. An optional `LLMClaimParser` can run in
parallel and must return the same structured schema. The local selector:

1. validates both results;
2. accepts agreement directly;
3. records field-level disagreement;
4. selects a validated, sufficiently confident LLM result;
5. falls back to the rule parser when the LLM result is invalid or unavailable.

The repository does not hard-code an LLM provider. `LLMClaimParser` accepts an
injected client callable. For offline testing and reproducible replay, the CLI
accepts a JSON object keyed by `user_id`:

```powershell
python -B code/main.py --llm-responses-json parser_responses.json
```

Example response:

```json
{
  "user_002": {
    "claimed_parts": ["front_bumper", "headlight"],
    "claimed_issue_types": ["broken_part"],
    "claimed_severity": "unknown",
    "included_parts": ["front_bumper", "headlight"],
    "excluded_parts": [],
    "evidence_quotes": ["front bumper", "headlight"],
    "parser_confidence": 0.9,
    "parser_diagnostics": []
  }
}
```

If no response exists for a row or validation fails, the system records the
failure and uses the deterministic parser.

## PreparedClaim JSON

Each record contains:

- source CSV and immutable input fields;
- `images`: unambiguous relative paths and image IDs;
- selected, rule, and optional LLM parsing results;
- parser disagreement and fallback diagnostics;
- matched requirement IDs and full evidence text;
- complete user-history context.

This JSON is the intended Sprint 2 visual-agent input. It contains no machine
absolute paths.

## Test

```powershell
python -B -m unittest discover -s code/tests -v
```

Tests cover:

- multi-part and multilingual extraction;
- negated and excluded parts;
- conservative `unknown` handling;
- LLM validation, disagreement, and fallback;
- requirement matching;
- portable image identity;
- output schema and placeholder semantics;
- complete PreparedClaim JSON;
- duplicate and missing input errors.

## Main modules

- `main.py`: command-line entry point
- `claim_agent.py`: models, parsers, selector, loaders, rule matcher, and writers
- `tests/test_claim_agent.py`: Sprint 1 automated tests

Sprint 2 will consume `PreparedClaim` records and add per-image VLM
observations without changing the original claim intent.
