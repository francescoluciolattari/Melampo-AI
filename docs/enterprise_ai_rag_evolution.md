# Melampo Enterprise AI/RAG Evolution Plan

This document defines the next architectural evolution of Project Melampo after the enterprise neuro-dynamic and RAG updates.

## Scope and safety boundary

Melampo remains a research framework. The current implementation is not a validated medical device, does not provide clinical diagnosis, and must not be used for patient care without formal clinical validation, regulatory review, prospective testing, monitoring, and human specialist oversight.

The objective is to make the research core more enterprise-ready: auditable, modular, optional-dependency driven, testable, and explicit about uncertainty, provenance, continuous-learning boundaries, and synthetic dream replay.

## Enterprise inspiration model

Melampo should borrow architectural patterns from advanced enterprise AI and RAG systems:

1. **Strict separation of contracts and providers**
   - Keep local dependency-free fallbacks for tests.
   - Put external providers behind explicit interfaces.
   - Track backend, model, collection, source and provenance in every returned object.

2. **Governed RAG over blind generation**
   - All literature, guideline, epidemiological and clinical-document knowledge should enter through a document-processing and vector-memory pipeline.
   - Every answer must be attributable to source, page/section, publication date and license class when available.
   - Dream-generated traces are learning candidates, not medical facts.

3. **Real-time post-training memory**
   - Case traces, reviewed outcomes, diagnostic disagreements and curated literature chunks are upserted into vector memory.
   - Memory records carry `learning_status`: `candidate`, `promoted`, or future states such as `rejected`, `needs_review`, `retired`.
   - Promotion requires rational-control validation and provenance.

4. **Multi-area neuro-dynamic governance**
   - Simulated areas should not merely vote; they should interact through explicit coherence, mismatch, convergence, prediction-error, inhibition and plasticity metrics.
   - These values must modulate intuition, candidate temperature, belief updates and dream replay.

5. **Offline / low-activity self-evolution**
   - During low-activity windows, the dream branch can generate counterfactual combinations of symptoms, exams, epidemiology and imaging metadata.
   - Synthetic traces must be checked by rational-control modules before entering vector memory as promoted learning material.

## Recommended technology choices

### Vector memory backend

Recommended production backend: **Milvus/Zilliz**.

Reasoning:
- Melampo needs multimodal memory, not only text retrieval.
- It needs real-time upserts for post-training memory.
- It needs metadata filtering, hybrid search, reranking and future multi-vector fields for text, image, report, epidemiology and case-trace embeddings.
- The repository now includes a dependency-free local fallback in `src/melampo/memory/vector_memory.py` so tests and offline prototyping remain stable.

Alternative candidates:
- Qdrant: excellent developer ergonomics and filtering; strong alternative for smaller deployments.
- Weaviate: strong schema and hybrid-search ecosystem.
- pgvector: useful when enterprise constraints require PostgreSQL-first deployments.

### Document processing backend

Recommended production parser: **Docling**.

Reasoning:
- Melampo must ingest PDFs, DOCX, PPTX, images, reports, tables and layout-rich documents for clinical and epidemiological RAG.
- Document structure matters: page, section, table, figure and formula provenance must be retained.
- The repository now includes `src/melampo/data/document_processing.py` as a parser-neutral contract with Docling as recommended backend and a plain-text fallback.

Alternative candidates:
- Unstructured: broad ingestion ecosystem.
- LlamaParse: useful in LlamaIndex-centric stacks.
- PyMuPDF/pypdf: lighter-weight fallbacks for PDF-only paths.

### RAG orchestration

Recommended pattern: use Melampo's own `VectorMemoryStore` and `ClinicalDocumentProcessor` as internal contracts, then optionally integrate LlamaIndex or Haystack at the edge.

Reasoning:
- Melampo's core must not depend on one RAG framework.
- LlamaIndex/Haystack can be excellent integration layers, but clinical governance requires Melampo-owned provenance and promotion states.

## Neuro-dynamic evolution

The updated `NeuroDynamicMetrics` module now exposes:

- `pi_score`: predictive-inference quality score.
- `precision_weighted_coherence`: coherence weighted by salience and precision.
- `prediction_error`: mismatch load after pair-prior modulation.
- `cross_area_synchrony`: coherent-pair dominance across areas.
- `convergence_index`: aggregate multimodal agreement index.
- `mismatch_index`: unresolved conflict and mismatch index.
- `inhibitory_control`: suppression of noisy, biased or weak candidates.
- `deductive_gate`: strength of slow, controlled reasoning.
- `revision_pressure`: pressure to re-rank alternatives.
- `dream_plasticity`: intensity of dream replay and synthetic-curriculum exploration.
- `candidate_temperature`: candidate exploration/exploitation temperature.
- `belief_update_rate`: rate used by future belief-update calibration.

These metrics are still computational abstractions. They are inspired by predictive processing, active inference, cross-area synchrony, inhibitory control and recurrent integration, but they are not literal claims about biological neurons or cortical areas.

## Intuition modulation

The intuition engine now uses neuro-dynamic metrics to modulate:

- rapid intuitive candidate score;
- rational revision candidate score;
- contradiction-revision candidate score;
- disagreement penalty;
- area-pair bonus;
- quantum-like belief-update context;
- summary trace for downstream audit.

This makes intuition inspectable: the project can explain why a candidate won, why it was revised, or why the dream branch generated alternatives.

## Dream branch auto-evolution

The dream branch now emits `auto_evolution_plan` with:

- candidate/hold status;
- promotion guardrails;
- learning targets;
- candidate score based on `pi_score`, `convergence_index`, `dream_plasticity` and risk.

Promotion must never be automatic into clinical truth. A promoted dream trace should only become a memory candidate after rational-control validation, provenance labeling and safety review.

## Next engineering tasks

1. Add unit tests for `NeuroDynamicMetrics`, `AreaCoherenceAnalyzer`, `IntuitionEngine`, `DreamTrainer` and `VectorMemoryStore`.
2. Add a Milvus adapter behind the current vector-memory interface.
3. Add a Docling adapter behind `ClinicalDocumentProcessor`.
4. Add a curation workflow: document ingestion -> chunking -> embedding -> vector upsert -> retrieval -> evidence ranking -> audit trace.
5. Add outcome feedback ingestion for reviewed cases.
6. Add scheduled low-activity dream replay jobs with promotion guardrails.
7. Add CI for linting, tests and package installation.
8. Add formal model cards and dataset cards for every embedding/model backend.

## Enterprise-grade definition for Melampo

A module should be considered enterprise-grade only if it has:

- typed public contracts;
- deterministic fallback behavior;
- provider-neutral interfaces;
- structured outputs;
- provenance and audit metadata;
- tests for normal, empty and adverse cases;
- explicit safety limitations;
- no hidden network calls;
- no irreversible learning promotion without governance;
- backward-compatible CLI behavior.

The current update moves the project toward that target. It does not make the whole repository 100% enterprise-grade yet; that requires broader tests, adapters, CI, security review and clinical validation.
