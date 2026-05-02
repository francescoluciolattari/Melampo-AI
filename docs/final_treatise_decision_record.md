# Project Melampo Final Treatise Decision Record

This document records the architectural decisions agreed for the final Melampo treatise and for the enterprise-grade implementation roadmap.

## 1. Foundational identity

Melampo is not a single-model diagnostic assistant and not a simple RAG wrapper.

Melampo is a multimodal, multimodel, neuro-symbolic and neuro-inspired research framework designed to evolve Computer-Aided Diagnosis into Computer-Aided Intuition.

The final treatise should preserve this identity:

```text
specialist models -> functional areas -> coherence/mismatch dynamics -> intuition -> differential reasoning -> policy/critique -> governed memory/dream replay
```

## 2. Final diagnostic authority

The final diagnostic authority belongs to `MelampoDiagnosticOrchestrator`, not to any external model.

External models are specialist signal providers, critics, parsers, retrievers, or explainers. They are not the sole final arbiters of diagnostic output.

The orchestrator must remain:

- auditable;
- policy-aware;
- uncertainty-aware;
- abstention-capable;
- escalation-capable;
- provenance-preserving;
- compatible with deterministic local fallback execution.

## 3. Model strategy

### 3.1 Imaging and radiology

Decision: **Pillar-0 replaces MedGemma as the primary radiology / volumetric imaging foundation model candidate.**

Role:

- CT/MRI volume interpretation;
- radiological finding extraction;
- `visual_diagnostic_area` signal generation;
- imaging salience, uncertainty and mismatch signals.

Limitations:

- not final diagnostic authority;
- research use until validated;
- requires local dataset validation, calibration and governance.

### 3.2 Clinical text and agentic reasoning

Decision: **Gemma 4 replaces MedGemma for non-radiology language, reasoning and agentic workflow functions.**

Role:

- clinical text reasoning;
- report and EHR summarization;
- `language_listening_area` signal generation;
- workflow/tool-use reasoning where locally deployable open-weight behavior is preferred.

Limitations:

- must be grounded by Weaviate RAG and clinical sources;
- not a standalone medical specialist without grounding;
- not final diagnostic authority.

### 3.3 External critic and scientific research layer

Decision: **Claude Healthcare / Life Sciences style capabilities are best used as an optional external critic and scientific research layer.**

Role:

- second-opinion critique;
- literature reasoning;
- regulatory/compliance reasoning;
- MCP/tool workflow review;
- final report quality control.

Limitations:

- not primary imaging model;
- not final diagnostic authority;
- requires trace and human review before clinical use.

### 3.4 Semantic memory and ontology RAG

Decision: **Weaviate is the primary vector/semantic memory backend.**

Reason:

Melampo prioritizes semantic knowledge and relations between concepts. A symptom can suggest multiple pathologies; a pathology can have symptoms, imaging findings, risk factors and documents; a case can link patient context, images, reports, ontology references and differential hypotheses.

Weaviate's object-property model is aligned with this object graph. Milvus/Zilliz remains an alternative for vector-heavy workloads where throughput dominates semantic object modeling.

### 3.5 Document intelligence

Decision: **Docling is the recommended production parser for clinical document ingestion.**

Role:

- PDF, DOCX, PPTX, images, tables, formulas and layout-aware conversion;
- clinical/epidemiological literature chunking;
- provenance-preserving document-to-memory flow.

## 4. Core Melampo areas

The final treatise should describe the following functional areas as canonical:

- `visual_diagnostic_area`;
- `language_listening_area`;
- `case_context_area`;
- `epidemiology_area`;
- semantic memory / Weaviate object graph;
- dream branch / replay branch;
- intuition engine;
- differential engine;
- policy stack;
- critique loop;
- diagnostic orchestrator.

## 5. Intuition definition

The agreed definition for the treatise:

```text
Intuition is an emergent diagnostic candidate produced by multimodal convergence,
semantic memory, coherence between simulated functional areas, controlled mismatch
resolution, inhibition of noisy/bias-prone signals, dream replay and contextual
belief update.
```

It is not an LLM guess. It is a system-level state transition.

## 6. Enterprise-grade implementation principles

Every future module should satisfy:

- typed public contracts;
- provider-neutral interfaces;
- deterministic fallback behavior;
- no hidden network calls;
- structured outputs;
- provenance and audit metadata;
- explicit safety limitations;
- tests for empty, normal and adverse paths;
- no irreversible learning promotion without governance;
- backward-compatible CLI behavior.

## 7. Implementation status after this decision record

Implemented or scaffolded:

- `ModelCapabilityRegistry` for Pillar-0, Gemma 4, Claude, Weaviate and Docling;
- `MelampoWeaviateSchema` object-property ontology contract;
- `MelampoDiagnosticOrchestrator` final controller;
- `ClinicalInferencePipeline` now returns `diagnostic_result`;
- Weaviate-first semantic memory fallback;
- Docling-oriented document processing contract;
- neuro-dynamic metrics and dream auto-evolution plan.

Still required:

- real Weaviate adapter;
- schema materialization command;
- real Pillar-0 adapter;
- real Gemma 4 adapter;
- optional Claude critique adapter;
- Docling converter integration;
- integration tests and CI;
- model cards, dataset cards and clinical governance documents;
- prospective clinical validation before any real-world use.
