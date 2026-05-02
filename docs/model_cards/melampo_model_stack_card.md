# Melampo Model Stack Card

## Status

Research scaffold. Not a validated medical device. Not for autonomous clinical diagnosis.

## Purpose

This card documents the intended multimodal/multimodel stack for Project Melampo and records how each external or optional model is allowed to participate in the system.

## Final authority

Final diagnostic authority belongs to `MelampoDiagnosticOrchestrator`.

External models are specialist signal providers, critics, parsers, retrievers, or explainers. They are not final diagnostic arbiters.

## Model roles

| Component | Primary candidate | Role | Final authority? |
|---|---|---|---|
| Radiology / volumetric imaging | Pillar-0 | CT/MRI/radiology finding signals for `visual_diagnostic_area` | No |
| Clinical text and agentic reasoning | Gemma 4 | Grounded clinical text reasoning, reports, EHR summaries, workflow reasoning | No |
| External critique / research review | Claude Healthcare / Life Sciences style backend | Second-opinion critique, literature review, regulatory/compliance reasoning | No |
| Semantic memory | Weaviate | Object-property semantic memory, ontology-aware RAG, provenance graph | No |
| Document intelligence | Docling | Layout-aware document conversion and RAG chunk preparation | No |
| Final controller | MelampoDiagnosticOrchestrator | Integrates evidence, intuition, differential, policy, critique and dream branch | Yes, within research prototype boundaries |

## Required output properties for model adapters

Every model adapter must return:

- model/provider identity;
- status;
- structured signals;
- confidence;
- uncertainty;
- provenance;
- limitations;
- explicit note that the adapter is not the final diagnostic authority.

## Governance requirements

1. No hidden network calls in core code.
2. Optional live integrations must require explicit configuration.
3. Every clinical source must preserve provenance.
4. Dream-generated traces default to `candidate` learning status.
5. Promotion requires rational-control validation and human review before any clinical use.
6. Every model must be replaceable behind a provider-neutral contract.
7. Outputs must support abstention and escalation.

## Known limitations

- No real Pillar-0, Gemma 4, Claude or Weaviate live inference is executed by default.
- Real-world use requires model-specific validation, local calibration, dataset cards and clinical governance.
- The current implementation is designed to preserve architecture and auditability, not to provide medical diagnosis.

## Future validation checklist

- Add real adapter subclasses in deployment-specific packages.
- Add model-specific calibration tests.
- Add benchmark suites per modality.
- Add dataset cards for every training/evaluation corpus.
- Add red-team tests for hallucination, overconfidence, bias and missing evidence.
- Add monitoring for drift, uncertainty and abstention rates.
