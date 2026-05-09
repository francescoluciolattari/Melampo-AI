# Phase 4A Specialist Model Cards

These cards describe the safe enterprise adapter layer added in Phase 4A. The
adapters are disabled by default and do not call external systems unless an
infrastructure owner explicitly configures an execution mode, endpoint or local
command.

## Pillar-0 adapter

- Role: primary radiology / volumetric imaging signal provider.
- Area: `visual_diagnostic_area`.
- Supported execution modes: `disabled`, `dry_run`, `mock`, `http_json`, `local_subprocess`.
- Safety boundary: signal provider only, never final diagnostic authority.
- Required validation before clinical use: radiology benchmark, calibration,
  modality/anatomy slice analysis, provenance review and human specialist review.

## Gemma 4 adapter

- Role: grounded clinical text and agentic reasoning provider.
- Areas: `language_listening`, `case_context`, optional workflow reasoning.
- Supported execution modes: `disabled`, `dry_run`, `mock`, `http_json`, `local_subprocess`.
- Safety boundary: must be RAG-grounded and must return uncertainty, missing
  evidence and source references.
- Required validation before clinical use: groundedness, faithfulness, safety rails,
  slice analysis and human review.

## Claude Healthcare/Life Sciences style critic

- Role: optional external critic and scientific/regulatory reviewer.
- Areas: critique, metacognition, regulatory review.
- Supported execution modes: `disabled`, `dry_run`, `mock`, `http_json`, `local_subprocess`.
- Safety boundary: external critic only. It cannot override `MelampoDiagnosticOrchestrator`.
- Required validation before clinical use: unsupported-claim detection benchmark,
  privacy review, audit trace review and human approval.
