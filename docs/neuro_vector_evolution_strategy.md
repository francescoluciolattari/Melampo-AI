# Neuro-Dynamic Reasoning and Vector Memory Strategy

This document records the current Melampo strategy for neuro-inspired reasoning metrics, controlled dream self-evolution, and vector-memory architecture.

## Research framing

Melampo uses neuroscience as a computational source of abstractions, not as a literal biological simulator.
The project currently maps these ideas into explicit runtime metrics:

- predictive-inference score (`pi_score`)
- precision-weighted coherence
- prediction error
- cross-area synchrony
- conflict load
- deductive gate
- revision pressure
- intuition gain
- bias suppression score

These metrics are implemented in:

`src/melampo/reasoning/neuro_dynamics.py`

They are consumed by:

- `AreaCoherenceAnalyzer`
- `IntuitionEngine`
- `QuantumBeliefLayer`
- `DreamSelfEvolutionLoop`

## Why these metrics

The design takes inspiration from predictive processing and active inference: cognition can be modeled as a process that balances priors, sensory evidence, prediction error, and uncertainty/precision.
Melampo translates that into software-level controls:

- coherent multimodal areas increase precision-weighted coherence
- conflicting or weakly aligned areas increase prediction error
- high precision and low conflict increase intuition gain
- high conflict or dream pressure increases revision pressure
- the belief update is strengthened by precision and dampened by conflict

## Self-evolution guardrail

The dream branch may generate new memory candidates during offline rehearsal.
However, promotion into vector memory is conservative.
A candidate is promoted only if:

- `pi_score >= min_pi_score`
- `prediction_error <= max_prediction_error`
- `bias_suppression_score >= min_bias_suppression`

The implementation is:

`src/melampo/training/self_evolution.py`

This prevents the dream branch from blindly reinforcing noisy, biased, or poorly supported patterns.

## Vector memory strategy

The current repository includes an offline provider-neutral vector memory fallback:

`src/melampo/memory/vector_memory.py`

The fallback uses deterministic hashing embeddings so tests and local research runs are fully executable without network services.
It is not intended to be the final clinical embedding layer.

Recommended future enterprise-grade vector backends:

1. Qdrant — strong candidate for hybrid search, sparse/dense support, filtering, multivectors, and operational simplicity.
2. Milvus — strong candidate for large-scale multi-vector/hybrid search and distributed ANN workloads.
3. pgvector — good candidate when PostgreSQL operational simplicity and transactional integration matter more than advanced vector-native features.

Current recommendation for Melampo:

- default future adapter target: Qdrant for hybrid clinical RAG and multimodal memory
- large distributed research target: Milvus
- simple hospital/institutional deployment target: pgvector

The current `InMemoryVectorStore` should therefore be treated as a local contract/fallback, not the final backend.

## Document ingestion strategy

For clinical documentation, epidemiology, differential diagnosis references, PDFs, and mixed document formats, the preferred future ingestion candidate is Docling-style document conversion, because it targets structured extraction from PDFs and enterprise document formats.

Future ingestion modules should convert each document into:

- canonical text chunks
- page/section provenance
- source metadata
- modality/document type
- citation/trace metadata
- vector records

Candidate future modules:

- `melampo.ingestion.docling_loader`
- `melampo.ingestion.pdf_document_loader`
- `melampo.ingestion.guideline_chunker`
- `melampo.ingestion.epidemiology_corpus_loader`

## Enterprise-grade target checklist

Melampo is not yet production clinical software. To move toward enterprise-grade quality, every new component should provide:

- typed or documented interface
- deterministic test path
- no hidden network dependency in default tests
- source provenance
- research-only safety flags
- clear fallback mode
- runtime configurability
- no real patient data committed to Git
- explicit promotion/evaluation criteria for learned memory
- audit trace for reasoning outputs

## Current status

Implemented in this tranche:

- neuro-dynamic metrics layer
- area coherence integration
- intuition modulation by `pi_score` and prediction error
- belief update modulation by precision/conflict
- provider-neutral vector memory fallback
- semantic memory vector indexing
- controlled dream self-evolution loop
- tests for neuro dynamics, vector memory, and self-evolution

Next implementation step:

- add a Qdrant-compatible adapter interface behind `InMemoryVectorStore`
- add a Docling-compatible ingestion contract for PDF/document conversion
- add calibration tests for `pi_score`, conflict, and candidate selection behavior
