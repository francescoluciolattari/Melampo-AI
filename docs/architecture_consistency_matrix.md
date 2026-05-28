# Melampo Architecture Consistency Matrix

This document links the final treatise concepts to canonical documentation and
implementation modules. Its purpose is to prevent architectural drift between
theory, documentation and code while preserving Melampo's research-only clinical
safety boundary.

## Status legend

- `implemented`: executable scaffold or typed contract exists.
- `contract-only`: safe public contract exists; real backend integration is not enabled by default.
- `local-fallback`: local deterministic fallback exists.
- `planned`: documented but not yet implemented.
- `validation-required`: implemented research primitive requires dataset-driven validation.

| Treatise concept | Canonical documentation | Canonical code | Status | Enterprise constraint |
|---|---|---|---|---|
| Melampo is not a single-model assistant | `docs/final_treatise_decision_record.md`, `docs/architecture.md` | `src/melampo/reasoning/diagnostic_orchestrator.py` | implemented | External models cannot be final arbiters |
| Final diagnostic authority | `docs/final_treatise_decision_record.md` | `MelampoDiagnosticOrchestrator`, `DiagnosticResult` | implemented | Audit-first, abstention-capable, escalation-capable |
| Pillar-0 radiology signal provider | `docs/final_treatise_decision_record.md`, `docs/imaging_model_strategy.md` | `Pillar0RadiologyAdapter`, `SpecialistRuntime`, `VisualDiagnosticArea` | contract-only | Disabled by default, no hidden network calls |
| Gemma 4 grounded text reasoning | `docs/final_treatise_decision_record.md` | `Gemma4ClinicalReasoningAdapter`, `SpecialistRuntime`, `LanguageListeningArea` | contract-only | Must be grounded by retrieval context |
| Claude-style external critic | `docs/final_treatise_decision_record.md` | `ClaudeCritiqueAdapter`, `SpecialistRuntime` | contract-only | External critic only, cannot override final result |
| Weaviate semantic object-property memory | `docs/architecture.md` | `MelampoWeaviateSchema`, `WeaviateEnterpriseMemoryAdapter` | local-fallback | Provenance, learning status and license metadata required |
| Visual semantic imprint memory | `docs/architecture.md` | `visual_imprint.py`, `VisualConcept`, `VisualRecognitionImprint` | implemented scaffold | Imprints are candidate-only recognition footprints; morphing supports total or partial semantic overlap, not clinical images or diagnoses |
| Docling document intelligence | `docs/final_treatise_decision_record.md` | `document_processing.py` | contract/local-fallback | Parser is not reasoner |
| Functional areas | `docs/architecture.md` | `areas/*` | implemented | Areas emit structured signals, not final diagnoses |
| Area coherence and mismatch | `docs/neuro_vector_evolution_strategy.md` | `area_coherence.py`, `neuro_dynamics.py` | implemented | Dynamic coherence/mismatch must stay inspectable |
| Intuition | `docs/final_treatise_decision_record.md` | `intuition_engine.py`, `clinical_pipeline.py` | implemented scaffold | Intuition is non-final candidate generation |
| Dream/replay | `docs/dream_self_evolution_governance.md` | `dream_trainer.py`, `VisualImprintMorpher` | implemented scaffold | Dream outputs and visual morph links are candidate-only |
| Differential reasoning | `docs/core_consolidation_map.md` | `differential_engine.py` | implemented scaffold | Requires support/contradiction metadata |
| Policy, abstention and escalation | `docs/architecture.md` | `policy_stack.py`, `abstention.py`, `risk_gate.py`, `diagnostic_orchestrator.py` | implemented | Thresholds require calibration |
| Validation and calibration | `docs/validation/*` | `evaluation/*` | validation-required | No clinical deployment without formal validation |

## Enterprise-grade interpretation

Melampo is enterprise-structured but research-stage. It adopts typed contracts,
audit traces, provenance, deterministic fallback behavior, no-hidden-network-call
boundaries and governance metadata, while remaining non-validated and unsuitable
for autonomous clinical use.

## Neuroscience and AI interpretation boundary

The PI score, precision-weighted coherence, prediction error, mismatch,
inhibitory control, action-potential gate, deep inference and belief-update
metrics are computational abstractions inspired by predictive processing, active
inference, recurrent cross-area integration and inhibitory gating. They are not
literal measurements of neural tissue, membrane potentials or clinical truth.
They must be calibrated and falsified against retrospective and prospective
benchmarks before any regulated context of use.

## Non-negotiable constraints

1. No hidden network calls.
2. External models are not final diagnostic arbiters.
3. Every promoted memory or learning artifact requires provenance.
4. Dream/replay outputs remain candidate-only unless reviewed.
5. Clinical outputs must preserve abstention, escalation and audit metadata.
6. Production clinical use requires formal validation, regulatory review and human oversight.
