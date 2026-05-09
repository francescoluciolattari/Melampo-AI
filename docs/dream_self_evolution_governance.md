# Melampo Phase 3 Dream Self-Evolution Governance

Phase 3 separates dream generation from learning promotion. Dream traces are never clinical facts. They move through a governed workflow:

```text
low-activity scheduler
  -> dream / counterfactual candidate generation
  -> candidate store
  -> rational-control validation
  -> promotion policy
  -> vector-memory candidate / needs-review / promoted / rejected / retired
  -> reviewed outcome feedback
```

## Learning statuses

- `candidate`: default for dream-generated or synthetic material.
- `needs_review`: validated enough for human/protocol review but not promoted.
- `promoted`: allowed only after rational-control validation and provenance.
- `rejected`: unsafe, unsupported or high-risk candidate.
- `retired`: deprecated memory trace.

## Guardrails

A dream candidate cannot become clinical truth. Promotion is limited to governed vector memory or synthetic curriculum material. Clinical deployment still requires formal validation, regulatory review and human specialist oversight.

## Main modules

- `training/dream_scheduler.py`: synchronous low-activity scheduler.
- `training/dream_candidate_store.py`: append-style candidate registry.
- `training/rational_control_validator.py`: validation rubric for pi score, mismatch, risk, provenance and retrieval coverage.
- `training/promotion_policy.py`: status transition decision policy.
- `training/outcome_feedback.py`: attaches reviewed outcomes and creates memory traces.
- `memory/learning_status.py`: allowed learning statuses and transition validation.
