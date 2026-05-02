# Melampo Current Architecture

## Executive summary

Project Melampo is a multimodal, multimodel research architecture for Computer-Aided Intuition. It combines specialist model signals, simulated clinical functional areas, semantic memory, neuro-dynamic metrics, dream/replay, differential reasoning, policy control and an audit-first diagnostic orchestrator.

Melampo is not a single LLM, not a standalone diagnostic model and not a simple RAG wrapper.

## Safety boundary

Melampo is a research scaffold. It is not a validated medical device and must not be used for autonomous clinical diagnosis or patient-care decisions without clinical validation, regulatory review and human specialist oversight.

## High-level architecture

```text
Inputs
  ├── imaging metadata / future CT-MRI volumes
  ├── reports / EHR text / patient narrative
  ├── demographics / provenance / exposures
  └── documents / guidelines / epidemiology

Specialist signal providers
  ├── Pillar-0 contract for radiology / volumetric imaging
  ├── Gemma 4 contract for grounded clinical text and agentic reasoning
  ├── Claude-style critic contract for external scientific review
  ├── Docling contract for document intelligence
  └── Weaviate contract for semantic object-property memory

Melampo core
  ├── functional areas
  ├── area coherence and neuro-dynamic metrics
  ├── dream / replay branch
  ├── intuition engine
  ├── differential engine
  ├── policy stack and critique loop
  └── MelampoDiagnosticOrchestrator

Outputs
  ├── structured research diagnostic result
  ├── differential hypotheses
  ├── support / contradiction / missing evidence
  ├── abstention and escalation policy
  ├── audit trace
  └── validation / calibration records
```

## Canonical modules

### Pipeline and orchestration

- `src/melampo/reasoning/clinical_pipeline.py`
- `src/melampo/reasoning/diagnostic_orchestrator.py`
- `src/melampo/reasoning/pipeline_coordinator.py`
- `src/melampo/orchestration/model_capability_registry.py`
- `src/melampo/orchestration/runtime_services.py`

### Functional areas

- `src/melampo/areas/visual_diagnostic_area.py`
- `src/melampo/areas/language_listening_area.py`
- `src/melampo/areas/case_context_area.py`
- `src/melampo/areas/epidemiology_area.py`

### Reasoning and intuition

- `src/melampo/reasoning/area_coherence.py`
- `src/melampo/reasoning/neuro_dynamics.py`
- `src/melampo/reasoning/intuition_engine.py`
- `src/melampo/reasoning/differential_engine.py`
- `src/melampo/reasoning/critique_loop.py`
- `src/melampo/reasoning/policy_stack.py`

### Memory and RAG

- `src/melampo/memory/vector_memory.py`
- `src/melampo/memory/weaviate_schema.py`
- `src/melampo/memory/weaviate_adapter.py`
- `src/melampo/memory/retriever.py`

### Document processing

- `src/melampo/data/document_processing.py`

### Specialist model contracts

- `src/melampo/models/specialist_adapters.py`

### Validation

- `src/melampo/evaluation/clinical_benchmark.py`
- `src/melampo/evaluation/prospective_validation.py`
- `src/melampo/evaluation/calibration.py`

## Model strategy

| Role | Candidate | Melampo status |
|---|---|---|
| Radiology / volumetric imaging | Pillar-0 | Primary signal-provider candidate for visual diagnostic area |
| Clinical text and agentic reasoning | Gemma 4 | Primary grounded text/workflow candidate |
| External critique / scientific review | Claude Healthcare / Life Sciences style capability | Optional critic, not final arbiter |
| Semantic memory and ontology RAG | Weaviate | Primary object-property memory backend |
| Document intelligence | Docling | Recommended parser with plain-text fallback |
| Final controller | MelampoDiagnosticOrchestrator | Final research authority inside the scaffold |

## Data and memory model

The semantic memory target is an object-property clinical graph:

- `Symptom`
- `Pathology`
- `ClinicalCase`
- `ImagingStudy`
- `ImagingFinding`
- `ClinicalDocument`
- `EpidemiologicalFactor`

The Weaviate schema contract preserves vectors, properties, references, ontology codes and provenance together. This is intentional: in clinical reasoning, a vector without context is unsafe.

## Validation architecture

Melampo separates:

1. retrospective benchmark evaluation;
2. prospective locked-prediction validation;
3. calibration of confidence and abstention;
4. later clinical/regulatory validation.

The codebase implements the first three as research primitives only.

## Enterprise-grade constraints

Every new module should maintain:

- typed public contracts;
- deterministic fallback behavior;
- no hidden network calls;
- provider-neutral interfaces;
- structured outputs;
- provenance and audit metadata;
- explicit limitations;
- tests for empty, normal and adverse paths;
- no automatic promotion of dream-generated material into clinical truth.
