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
- parses the original multilingual or code-switched conversation directly
  instead of translating it into an intermediate English claim;
- gives the last meaningful customer confirmation priority over earlier
  speculation and keeps explicit scope exclusions separate;
- treats generic words such as `damaged`, `mark`, and `issue` as insufficient
  to infer a specific damage enum;
- keeps a deterministic rule parser as the baseline and fallback;
- loads the versioned high-precision baseline vocabulary from
  `claim_lexicon.json`, which contains no user IDs, case IDs, paths, or
  row-specific answers;
- leaves unsupported code-switched and complex-language expressions as
  `unknown` for the LLM instead of adding sample-specific phrases to regexes;
- parses issue type and claimed severity independently; damage words such as
  `shattered` do not automatically imply `high` severity;
- supports an optional structured LLM parser through a provider-neutral
  adapter;
- validates parser enums, object-specific parts, confidence, and evidence
  quotes locally;
- requires independent exact-source evidence for every included part, excluded
  part, specific issue type, and claimed severity; excluded-part evidence must
  include the negation or exclusion wording;
- rejects LLM-specific damage enums when their only textual basis is a generic
  term such as `damaged`, `mark`, or `issue`;
- rejects arrays that mix `unknown` or `none` with concrete enum values;
- records rule/LLM differences for all parser fields, including scope,
  provenance quotes, confidence, and diagnostics, without merging outputs;
- matches only valid requirements from `evidence_requirements.csv`;
- stores both requirement IDs and full minimum-evidence text;
- writes a complete `PreparedClaim` JSON handoff for Sprint 2;
- writes a schema-valid placeholder CSV.

Sprint 1 does **not** inspect image contents. Visual output fields therefore
remain `unknown` or `false`, and every status is
`not_enough_information`. Do not submit `sprint1_output.csv` as the final
competition output.

## Sprint 2 scope

Sprint 2 consumes the in-memory `PreparedClaim` records and reviews every
submitted image independently. It:

- reads and validates each local image;
- decodes ordinary Pillow formats and AVIF content even when the supplied file
  uses a `.jpg` extension;
- applies EXIF orientation, bounds the longest edge, and JPEG-encodes the
  processed image;
- sends one image, the untrusted claim target, and all matched minimum
  evidence requirements to a provider-neutral vision client;
- supports OpenAI Responses API image input and deterministic replay JSON;
- requires one structured `ImageObservation` per image;
- records actual object, visible parts, visible damage, visual severity,
  target-part visibility, quality/authenticity risks, and requirement results;
- rejects invented image IDs, paths, requirement IDs, enum values, and
  additional fields;
- treats user text and text inside images as untrusted content;
- retries invalid or failed model responses and creates a traceable
  `manual_review_required` fallback observation instead of stopping the batch;
- routes only difficult images to a separately configured review model;
- records fixed local escalation reasons, the review candidate, conflicts,
  attempts, and resolution status in both observation and trace artifacts;
- permits review output to fill primary `unknown` fields, but never silently
  replaces a conflicting primary fact;
- routes unresolved review uncertainty or known-field disagreement to
  `manual_review_required`;
- writes raw-response and retry diagnostics to a separate JSONL trace.

The fixed escalation reason enum is:

- `object_or_part_conflict`
- `multi_image_identity_conflict`
- `possible_manipulation`
- `non_original_image`
- `critical_field_conflict`
- `primary_uncertain_or_unreviewable`
- `text_instruction_present`

Local routing is mandatory when these conditions are present. A model cannot
cancel a forced escalation by claiming that review is unnecessary.

Sprint 2 deliberately does **not** generate `claim_status`,
`evidence_standard_met`, supporting image selection, or a multi-image
decision. Those belong to Sprint 3.

## Sprint 3 scope

Sprint 3 is the deterministic aggregation and decision layer. It consumes the
per-image `VisualReviewCase` records produced by Sprint 2 and generates all
ten output fields. It runs immediately after Sprint 2 in the same pipeline
call. It:

- unions per-image risk flags and adds `user_history_risk` or
  `manual_review_required` from the claim history, but never uses history
  alone to override a clear visual conclusion;
- determines `valid_image`: `true` unless all images are non-reviewable or the
  entire set has authenticity issues;
- selects the single most informative image observation for visual-fact
  extraction based on reviewability, target-part visibility, and damage
  alignment;
- derives `issue_type`, `object_part`, and `severity` from the best
  observation's visible facts;
- evaluates `evidence_standard_met` by checking each matched requirement
  against the best per-image status across all reviewable observations; a
  multi-image identity conflict (different objects across images) overrides
  `evidence_standard_met=false` even if individual requirements appear met;
- applies the three-state decision logic:
  - `not_enough_information` when evidence is insufficient, or the target
    part is not visible, or the claim is too broad to verify;
  - `contradicted` when the target is visible but shows no damage, when the
    wrong object is shown, or when visible damage type mismatches the claim;
  - `supported` when the claimed part and damage type are both visible;
- user claim `unknown` parts or issues are never auto-expanded to a more
  specific supported conclusion;
- `claimed_severity="unknown"` does not affect the damage-existence decision;
  the final `severity` always comes from visual facts;
- selects `supporting_image_ids` traceable to `ImageObservation` records:
  only the images that materially contribute to the conclusion are listed;
  `none` is written when no image supports the conclusion;
- enforces cross-field invariants (`issue_type=none ↔ severity=none`,
  `object_part` in allowed enum for `claim_object`, all output enums legal)
  and calls `ReviewResult.validate()` before writing;
- writes the final `output.csv` with all 14 columns in the required order.

The aggregation entry point is `aggregate_visual_case(case) -> ReviewResult`
in `decision_agent.py`. The writer is `write_final_output(cases, path)`.

## Requirements

- Python 3.11 or later
- Pillow, pillow-avif-plugin, and python-dotenv, declared in `requirements.txt`
- `OPENAI_API_KEY` only when using the OpenAI vision provider

Install:

```powershell
python -m pip install -r code/requirements.txt
```

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

## Configure OpenAI

Create a local `.env` file from the safe template:

```powershell
Copy-Item .env.example .env
```

Then edit `.env`:

```dotenv
OPENAI_API_KEY=replace_with_your_openai_api_key
OPENAI_CLAIM_MODEL=gpt-5.4-mini
OPENAI_VISION_MODEL=gpt-5.4-mini
OPENAI_REVIEW_MODEL=gpt-5.5
```

The CLI automatically loads `.env` from the repository root. Existing process
environment variables take precedence, so production or CI configuration is
never overwritten by the local file. `.env` is excluded by `.gitignore`;
`.env.example` contains no secret and should remain committed.

## Run Sprint 1 with the Claim API

The rule parser always runs. When a Claim provider is configured, the default
`--claim-routing always` mode also runs `LLMClaimParser` for every claim,
retains both results, records field-level differences, and lets the local
selector choose the validated result:

```powershell
python -B code/main.py `
  --claims-file sample_claims.csv `
  --claim-provider openai `
  --prepared-json sample_sprint1_summary.json
```

`OPENAI_CLAIM_MODEL` overrides `--claim-model`; the default is
`gpt-5.4-mini`.

`--claim-routing auto` is an explicit cost-optimization experiment. It calls
the LLM only for unknown or low-confidence rule results, multiple parts,
multilingual/code-switched text, or complex negation/final scope. It is not the
Sprint 1 baseline behavior.

The online client uses Responses API Structured Outputs. Every result is still
checked locally for allowed fields, enums, object-specific parts, exact
per-field source quotes, negated evidence for excluded parts, include/exclude
overlap, generic-term overreach, and confidence. Invalid responses are retried
and then fall back to the deterministic parser. Array uniqueness is enforced
locally because the Structured Outputs JSON Schema subset does not accept the
`uniqueItems` keyword. Paired quotation marks wrapped around an evidence quote
are removed before validation, but the text inside must still occur verbatim
in the original `user_claim`; no fuzzy matching or paraphrase is accepted.

For offline replay:

```powershell
python -B code/main.py `
  --claims-file sample_claims.csv `
  --claim-provider replay `
  --claim-routing always `
  --llm-responses-json parser_responses.json
```

## Run Sprint 3

Sprint 3 runs automatically whenever `--vision-provider` is not `none`. After
Sprint 2 writes `sprint2_observations.json` and `sprint2_trace.jsonl`, Sprint
3 aggregates each `VisualReviewCase` and writes the final decision output:

```powershell
python -B code/main.py `
  --claims-file sample_claims.csv `
  --vision-provider openai `
  --final-output output.csv
```

For offline replay:

```powershell
python -B code/main.py `
  --claims-file sample_claims.csv `
  --vision-provider replay `
  --vision-responses-json vision_replay.json `
  --final-output output.csv
```

The `--final-output` flag defaults to `output.csv` in the repository root.
The Sprint 1 placeholder `sprint1_output.csv` is still written for
diagnostics and must not be submitted as the final prediction.

## Run Sprint 2

Using the OpenAI Responses API:

```powershell
python -B code/main.py `
  --claims-file sample_claims.csv `
  --vision-provider openai `
  --vision-output sample_sprint2_observations.json `
  --vision-trace sample_sprint2_trace.jsonl
```

`OPENAI_VISION_MODEL` is optional and overrides `--vision-model`. The default is
`gpt-5.4-mini`. API keys are never written to output or trace files.
`OPENAI_REVIEW_MODEL` overrides `--review-model` and defaults to `gpt-5.5`.
Normal images are sent only to `OPENAI_VISION_MODEL`. The review model is called
only when a fixed local escalation reason is present.

For deterministic replay testing, difficult-case review responses can be kept
separate from primary responses:

```powershell
python -B code/main.py `
  --claims-file sample_claims.csv `
  --vision-provider replay `
  --vision-responses-json vision_primary_replay.json `
  --vision-review-responses-json vision_review_replay.json
```

If replay mode encounters a forced escalation without
`--vision-review-responses-json`, it emits `manual_review_required`; it does not
reuse the primary replay as if it were an independent review.

### Vision model selection and API cost

Prices below are OpenAI standard API token prices as of June 19, 2026 and
should be rechecked before a production run:

| Model | Recommended use | Input / output price per 1M tokens |
|---|---|---:|
| `gpt-5.4-mini` | Default. Strong cost/quality balance for per-image structured extraction. | $0.75 / $4.50 |
| `gpt-5.5` | Accuracy-focused evaluation or difficult/ambiguous images. | $5.00 / $30.00 |
| `gpt-5.4-nano` | Cheap first pass or simple quality/object classification; benchmark before using for final damage decisions. | $0.20 / $1.25 |

All three accept image input and text output through the Responses API and
support structured outputs. Image inputs are converted to billable input
tokens based on image dimensions and model-specific tokenization. Therefore,
the exact run cost depends on the number and dimensions of images, prompt
length, retries, reasoning tokens, and JSON output length.

Official references:

- Models: https://developers.openai.com/api/docs/models
- Structured Outputs: https://developers.openai.com/api/docs/guides/structured-outputs
- GPT-5.4 mini: https://developers.openai.com/api/docs/models/gpt-5.4-mini
- GPT-5.5: https://developers.openai.com/api/docs/models/gpt-5.5
- GPT-5.4 nano: https://developers.openai.com/api/docs/models/gpt-5.4-nano
- Vision token calculation: https://developers.openai.com/api/docs/guides/images-vision#calculating-costs
- Pricing: https://developers.openai.com/api/docs/pricing

For deterministic offline testing, provide one structured response per image,
keyed by its dataset-relative path:

```powershell
python -B code/main.py `
  --claims-file sample_claims.csv `
  --vision-provider replay `
  --vision-responses-json vision_replay.json `
  --vision-output sample_sprint2_observations.json `
  --vision-trace sample_sprint2_trace.jsonl
```

Example replay key:

```json
{
  "images/sample/case_001/img_1.jpg": {
    "image_id": "img_1",
    "path": "images/sample/case_001/img_1.jpg",
    "actual_object": "car",
    "visible_parts": ["rear_bumper"],
    "visible_issue_types": ["dent"],
    "severity": "medium",
    "target_part_visibility": "visible",
    "requirement_results": [
      {
        "requirement_id": "REQ_GENERAL_OBJECT_PART",
        "status": "met",
        "reason": "The rear bumper is visible."
      },
      {
        "requirement_id": "REQ_CAR_BODY_PANEL",
        "status": "met",
        "reason": "The panel surface can be inspected."
      },
      {
        "requirement_id": "REQ_REVIEW_TRUST",
        "status": "met",
        "reason": "The image is usable and relevant."
      }
    ],
    "fact_summary": "A dent is visible on the rear bumper.",
    "risk_flags": ["none"],
    "reviewable": true,
    "claim_target_clear": true,
    "diagnostics": []
  }
}
```

## Parser architecture

`RuleBasedClaimParser` always runs. `LLMClaimParser` supports the OpenAI
Responses API and injected replay clients, and must return the same structured
schema. In the default `always` mode, both parsers run for every claim when an
LLM provider is configured. The selector:

1. records why the LLM was called or skipped;
2. validates both results when the LLM runs;
3. accepts agreement directly;
4. records field-level disagreement;
5. selects a validated, sufficiently confident LLM result;
6. falls back to the rule parser when the LLM result is invalid or unavailable.

The LLM parser is instructed to resolve multilingual negation and final claim
scope from the original transcript. Exact source quotes remain mandatory.
Translation or an English summary may be retained as diagnostics, but it is
not a source of truth and cannot replace original-text quote validation.

The deterministic fallback intentionally favors precision over recall. It
handles common high-value scope forms such as `not X`, `not claiming X`, and
`X claim nahi`, but returns `unknown` when a generic description or unfamiliar
negation cannot be interpreted safely.

`LLMClaimParser` remains provider-neutral through an injected client callable;
the repository includes an OpenAI Responses API client plus deterministic
replay. Replay JSON is keyed by `user_id`:

```powershell
python -B code/main.py `
  --claim-provider replay `
  --claim-routing always `
  --llm-responses-json parser_responses.json
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
- image decoding, orientation, resizing, and encoding;
- strict per-image observation validation;
- complete requirement-ID reporting;
- rejection of final-decision fields in Sprint 2;
- wrong-object risk consistency;
- wrong-part, no-visible-damage, and claim-mismatch consistency;
- forced escalation for manipulation, non-original images, image text
  instructions, uncertainty, and multi-image identity conflicts;
- independent review-model invocation and fixed audit reasons;
- safe filling of primary `unknown` fields;
- preservation of primary facts and manual routing on review conflicts;
- every sample image being reviewed independently;
- per-image failure fallback without batch termination;
- observation and raw-trace artifact writing.
- history risk flag generation and isolation from visual decisions;
- evidence standard evaluation across single and multi-image cases;
- multi-image identity conflict override of evidence standard;
- three-state decision: supported, contradicted, not_enough_information;
- user history risk added to flags without changing clear visual conclusions;
- multi-image blurry+clear → supported using the clearer image only;
- wrong-object multi-image → not_enough_information with identity conflict;
- issue_type=none ↔ severity=none invariant enforcement;
- supporting_image_ids traceable to ImageObservation records;
- write_final_output: 14-column schema, correct row count, enum legality,
  input column preservation, boolean serialization, and none sentinel.

## Main modules

- `main.py`: command-line entry point
- `claim_agent.py`: models, parsers, selector, loaders, rule matcher, and writers
- `visual_agent.py`: image processing, VLM clients, observation validation,
  retry/fallback handling, and Sprint 2 writers
- `decision_agent.py`: Sprint 3 deterministic aggregation, evidence judgment,
  three-state decision, and final output writer
- `tests/test_claim_agent.py`: Sprint 1 automated tests
- `tests/test_visual_agent.py`: Sprint 2 automated tests
- `tests/test_decision_agent.py`: Sprint 3 automated tests
