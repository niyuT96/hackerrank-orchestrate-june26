# Evaluation Report

## Ground-truth contract

`dataset/sample_claims.csv` is the only source of evaluation ground truth. No human-, AI-, or model-generated claim-intent or per-image labels are used to calculate accuracy, precision, recall, or F1.

Cases are split deterministically by a SHA-256 hash of `user_id`; the split is case-level, so images from one claim cannot leak between development and frozen validation.

| Split | Cases |
| --- | --- |
| development | 15 |
| frozen_validation | 5 |

## End-to-end metrics

| Field | Accuracy |
| --- | --- |
| claim_status | 0.650 |
| evidence_standard_met | 0.750 |
| valid_image | 0.700 |
| issue_type | 0.400 |
| object_part | 0.600 |
| severity | 0.450 |

| Set field | Exact | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| risk_flags | 0.350 | 0.471 | 0.552 | 0.508 |
| supporting_image_ids | 0.550 | 0.909 | 0.500 | 0.645 |

Exact full-row match: **0.100**.

## Case-level strategy comparison

| Strategy | Claim status | Evidence | Valid image | Issue | Part | Severity |
| --- | --- | --- | --- | --- | --- | --- |
| primary_only | 0.600 | 0.800 | 0.800 | 0.350 | 0.750 | 0.450 |
| review_routed | 0.650 | 0.750 | 0.700 | 0.400 | 0.600 | 0.450 |

Both strategies are scored only after producing complete case-level output rows. `primary_only` reconstructs the first vision pass from the trace; `review_routed` uses the persisted post-escalation observations.

## Claim-understanding diagnostics (unlabeled)

| Diagnostic | Value |
| --- | --- |
| Cases | 20 |
| Validated LLM results | 0 |
| Rule/LLM disagreements | 0 |
| Selected parser counts | {"rule": 20} |
| Unknown field counts | {"claimed_issue_types": 3, "claimed_parts": 0, "claimed_severity": 20} |

These are routing and uncertainty diagnostics, not parser accuracy metrics. The sample file has no independent claim-intent labels.

## Visual diagnostics (unlabeled)

| Diagnostic | Primary only | Review routed |
| --- | --- | --- |
| Images | 29 | 29 |
| Reviewable rate | 0.793 | 0.552 |
| Unknown part rate | 0.138 | 0.103 |
| Unknown issue rate | 0.138 | 0.103 |
| Escalation rate | 0.000 | 0.517 |
| Manual-review rate | 0.172 | 0.483 |

No per-image accuracy is reported because the sample file contains case-level final labels only.

## Requirement matching audit

Invalid object/rule matches: **0**.

```json
{
  "car": [
    "REQ_CAR_BODY_PANEL",
    "REQ_CAR_GLASS_LIGHT_MIRROR",
    "REQ_CAR_IDENTITY_OR_SIDE",
    "REQ_GENERAL_MULTI_IMAGE",
    "REQ_GENERAL_OBJECT_PART",
    "REQ_REVIEW_TRUST"
  ],
  "laptop": [
    "REQ_GENERAL_MULTI_IMAGE",
    "REQ_GENERAL_OBJECT_PART",
    "REQ_LAPTOP_BODY_HINGE_PORT",
    "REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD",
    "REQ_REVIEW_TRUST"
  ],
  "package": [
    "REQ_GENERAL_MULTI_IMAGE",
    "REQ_GENERAL_OBJECT_PART",
    "REQ_PACKAGE_CONTENTS",
    "REQ_PACKAGE_EXTERIOR",
    "REQ_PACKAGE_LABEL_OR_STAIN",
    "REQ_REVIEW_TRUST"
  ]
}
```

## Operational analysis

| Metric | Value |
| --- | --- |
| Images | 29 |
| Model calls | 44 |
| Input tokens | 89999 |
| Output tokens | 21980 |
| Retries | 10 |
| Average latency (s) | 7.568 |
| P95 latency (s) | 20.000 |
| Estimated cost (USD) | 0.6709 |
| Failure rate | 0.000 |

## Error interpretation

End-to-end mismatched cases: **18**. Because no additional layer-level ground truth is used, the report does not claim exact attribution to parsing, vision, or aggregation. Unlabeled diagnostics only indicate likely areas for investigation.

## Generalization controls

- Case-level deterministic development/frozen-validation split.
- Unit tests cover image-order invariance and multi-target coverage.
- Test-set outputs are not used as labels or strategy-selection data.
- No AI-generated proxy labels are used.
- Source scan secret hits: `[]`; sample-identity coupling hits: `[]`.

## Submission validation

Output valid: **True**.

## Frozen strategy manifest

```json
{
  "file_sha256": {
    "code\\claim_lexicon.json": "8e5ca46c433993db48290f29d9c878d7a9e0f81667388ce135b7e1057e9b60f6",
    "code\\claim_agent.py": "8b8517d5f38ded8618ba773069852639fd4b1dd0117cb3ca294bbc239d1ae3ed",
    "code\\visual_agent.py": "34026b4f39491149fef2e62f7f49812fceec70e2daebc2a1f1363b3b3a7ba60f",
    "code\\decision_agent.py": "9f8468e3b7efe2a5e9aadaab0290f09a4fac1d032346171c469d12914a391874"
  },
  "configuration": {
    "claims_file": "sample_claims.csv",
    "vision_workers": 1,
    "rpm_limit": 60,
    "claim_provider": "openai",
    "claim_model": "gpt-5.4-mini",
    "input_price_per_million": 0.75,
    "output_price_per_million": 4.5,
    "review_input_price_per_million": 5.0,
    "review_output_price_per_million": 30.0
  }
}
```

## Final test-set operational run

| Metric | Value |
| --- | --- |
| Images | 82 |
| Model calls | 137 |
| Input tokens | 280888 |
| Output tokens | 78164 |
| Average latency (s) | 8.015 |
| P95 latency (s) | 20.000 |
| Estimated cost (USD) | 2.4826 |
