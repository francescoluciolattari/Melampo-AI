# Melampo Core Consolidation Map

This document tracks the canonical core of `src/melampo/` after the enterprise multimodal, multimodel, RAG, validation and diagnostic-orchestration updates.

The goal is to support future consolidation work without duplicating the stabilized clinical reasoning path.

## Canonical clinical reasoning spine

Future reasoning changes should converge into this chain:

```text
clinical_pipeline
  -> area_coherence / neuro_dynamics
  -> dream_trainer
  -> intuition_engine
  -> differential_engine
  -> policy_stack / critique_loop
  -> diagnostic_orchestrator
```

| Module | Classification | Current status | Main issue | Next consolidation action |
|---|---|---|---|---|
| `src/melampo/app.py` | Canonical | Runtime entrypoint | Runtime assembly is clear, but many components are still wired as generic objects rather than persistent typed service contracts. | Keep as canonical entrypoint and gradually tighten dependency injection. |
| `src/melampo/reasoning/clinical_pipeline.py` | Canonical | Main end-to-end reasoning flow | Now returns `diagnostic_result`, but helper construction still happens inside `run()`. | Keep canonical and later move helper construction into runtime assembly. |
| `src/melampo/reasoning/diagnostic_orchestrator.py` | Canonical | Final research diagnostic controller | New module; policy thresholds are still early defaults. | Keep as the only final orchestration authority and calibrate thresholds with validation data. |
| `src/melampo/reasoning/intuition_engine.py` | Canonical | Main intuition core | Neuro-dynamic modulation is in place, but scoring remains heuristic until calibrated. | Evolve scoring here; avoid parallel intuition engines. |
| `src/melampo/reasoning/differential_engine.py` | Canonical | Main differential core | Supports hypotheses, domains, actions, support and contradiction metadata, but domain/test generation remains lightweight. | Expand typed action logic here rather than creating a sibling differential engine. |
| `src/melampo/reasoning/critique_loop.py` | Canonical | Main critique layer | Warnings and priorities exist, but critique still depends heavily on upstream heuristics. | Keep as canonical critique layer; plug optional Claude-style critic behind this path. |
| `src/melampo/reasoning/pipeline_coordinator.py` | Canonical | Main reasoning/policy/differential coordinator | Delegates final output authority to `diagnostic_orchestrator`. | Keep as coordinator, not final authority. |

## Canonical support and runtime-control modules

| Module | Classification | Current status | Main issue | Next consolidation action |
|---|---|---|---|---|
| `src/melampo/reasoning/area_coherence.py` | Canonical support | Inter-area dynamics analyzer | Emits pair profiles, total salience and neuro metrics. | Keep centralized and expand pair semantics here. |
| `src/melampo/reasoning/neuro_dynamics.py` | Canonical support | Computes PI score, mismatch, convergence, prediction error, inhibition and belief-update signals. | Metrics remain computational abstractions and require calibration. | Keep as the canonical neuro-dynamic metric layer. |
| `src/melampo/reasoning/policy_stack.py` | Canonical support | Structured policy layer | Decision semantics remain threshold-driven. | Expand policy semantics without duplicating final orchestration. |
| `src/melampo/reasoning/escalation.py` | Canonical support | Escalation helper | Lightweight by design. | Refine only if escalation paths become more granular. |
| `src/melampo/reasoning/decision_trace.py` | Canonical support | Audit trace helper | Minimal by design. | Keep lightweight. |
| `src/melampo/reasoning/pipeline_state.py` | Canonical support | Shared runtime state | Small and intentionally logic-free. | Avoid embedding reasoning logic here. |
| `src/melampo/evaluation/quantum_gate.py` | Canonical support | Optional quantum-like path gate | Emits allow, margin, level and reasons. | Keep as the only gate for optional quantum-like paths. |

## Canonical orchestration and registry modules

| Module | Classification | Current status | Main issue | Next consolidation action |
|---|---|---|---|---|
| `src/melampo/orchestration/model_capability_registry.py` | Canonical | Enterprise model capability registry | Records Pillar-0, Gemma 4, Claude, Weaviate and Docling roles. | Keep as canonical model-decision registry and update when model strategy changes. |
| `src/melampo/orchestration/contracts.py` | Canonical support | Service contract metadata | Includes name, provider, protocol and role. | Keep as service contract description layer. |
| `src/melampo/orchestration/service_registry.py` | Canonical support | In-memory service registry | Stores role metadata and registry summary. | Keep lightweight and avoid embedding routing semantics here. |
| `src/melampo/orchestration/bootstrap.py` | Canonical support | Baseline service contract inventory | Exposes contract inventory before registry construction. | Keep as baseline registry bootstrap. |
| `src/melampo/orchestration/model_router.py` | Canonical support | Static research router | Still static; superseded strategically by capability registry but useful as simple router. | Later evolve to capability-aware routing without duplicating registry data. |
| `src/melampo/orchestration/runtime_services.py` | Canonical support | Runtime service resolution facade | Returns availability, protocol and resolution mode. | Keep as runtime-service facade. |

## Canonical dream and replay branch

| Module | Classification | Current status | Main issue | Next consolidation action |
|---|---|---|---|---|
| `src/melampo/training/dream_trainer.py` | Canonical | Main dream/replay branch | Emits auto-evolution plan and alternatives, but curriculum remains lightweight. | Keep as single dream branch; grow rare-case, contradiction and mismatch-resolution generation here. |
| `src/melampo/training/replay_filter.py` | Canonical support | Replay assessment helper | Threshold-based replay modes. | Enrich before adding more replay abstractions. |
| `src/melampo/training/counterfactual_sampler.py` | Canonical support | Counterfactual variant helper | Emits focus, perturbation plan and novelty metadata. | Keep active and enrich here. |

## Canonical memory, RAG and ontology modules

| Module | Classification | Current status | Main issue | Next consolidation action |
|---|---|---|---|---|
| `src/melampo/memory/vector_memory.py` | Canonical | Dependency-free semantic vector memory fallback | Local hashing embeddings are only test/research fallback. | Keep fallback; real semantic memory should bind through Weaviate adapter. |
| `src/melampo/memory/weaviate_schema.py` | Canonical | Weaviate object-property schema contract | Schema is not automatically materialized by core. | Keep canonical schema and implement infra-specific subclass for live materialization. |
| `src/melampo/memory/weaviate_adapter.py` | Canonical support | Safe Weaviate adapter boundary | Live calls deliberately require infrastructure subclass. | Keep no-hidden-network-call rule. |
| `src/melampo/memory/retriever.py` | Canonical support | Grounded retrieval helper | Retrieval remains structural and not yet knowledge-rich. | Strengthen grounding quality before adding more retrieval abstractions. |
| `src/melampo/models/evidence_ranker.py` | Canonical support | Ranked evidence helper | Heuristic and not calibrated. | Improve ranking quality here. |

## Canonical document and specialist adapter modules

| Module | Classification | Current status | Main issue | Next consolidation action |
|---|---|---|---|---|
| `src/melampo/data/document_processing.py` | Canonical support | Docling-aware processor with plain-text fallback | Production document catalogs and source governance still external. | Keep as parser-neutral ingestion contract. |
| `src/melampo/models/specialist_adapters.py` | Canonical support | Safe contracts for Pillar-0, Gemma 4 and Claude-style critic | No real inference calls in core. | Implement real adapters as infrastructure-specific subclasses or packages. |

## Canonical area modules

| Module | Classification | Current status | Main issue | Next consolidation action |
|---|---|---|---|---|
| `src/melampo/areas/visual_diagnostic_area.py` | Canonical | Visual/imaging area | Structured carrier rather than rich interpreter. | Future Pillar-0 signal integration should feed this area. |
| `src/melampo/areas/language_listening_area.py` | Canonical | Narrative/listening area | Shallow on narrative temporal structure. | Future Gemma 4 grounded reasoning should feed this area. |
| `src/melampo/areas/case_context_area.py` | Canonical | Context area | Small but aligned with area metadata contract. | Expand only when stronger structured context handling is needed. |
| `src/melampo/areas/epidemiology_area.py` | Canonical | Epidemiology area | Simple exposure/demography grouping. | Grow prevalence and risk-factor reasoning here. |

## Canonical decision-control, validation and belief modules

| Module | Classification | Current status | Main issue | Next consolidation action |
|---|---|---|---|---|
| `src/melampo/models/abstention.py` | Canonical support | Structured abstention assessment | Threshold-driven. | Refine with calibration and prospective validation. |
| `src/melampo/models/risk_gate.py` | Canonical support | Structured risk assessment | Threshold-driven. | Refine with clinical benchmark slices. |
| `src/melampo/models/quantum_belief_layer.py` | Canonical support | Contextual belief-update metadata layer | Lightweight research formalism. | Keep as single quantum-like belief layer. |
| `src/melampo/evaluation/calibration.py` | Canonical validation | ECE, MCE, Brier and threshold suggestions. | Requires real datasets for meaningful values. | Use to calibrate confidence, abstention and orchestrator thresholds. |
| `src/melampo/evaluation/clinical_benchmark.py` | Canonical validation | Dataset-agnostic retrospective benchmark runner. | No bundled clinical datasets. | Use with dataset cards and de-identified labeled JSONL. |
| `src/melampo/evaluation/prospective_validation.py` | Canonical validation | Prediction-lock prospective registry. | File-based research scaffold, not clinical-grade immutable registry. | Replace storage with compliant audit log in deployments. |

## Modules to treat as research scaffold but acceptable

These modules are canonical but not production-complete:

- `intuition_engine.py`
- `differential_engine.py`
- `support_contradiction.py`
- `dream_trainer.py`
- `quantum_belief_layer.py`
- `retriever.py`
- `evidence_ranker.py`
- `model_router.py`
- `runtime_services.py`
- `diagnostic_orchestrator.py`
- `vector_memory.py`
- `weaviate_adapter.py`
- `specialist_adapters.py`

## Consolidation priorities

### Priority 1 - Preserve the canonical chain

All future reasoning changes should converge into:

```text
clinical_pipeline -> intuition_engine -> differential_engine -> critique_loop -> diagnostic_orchestrator
```

### Priority 2 - Avoid parallel reasoning entrypoints

If an experiment duplicates intuition, support/contradiction, differential ranking, critique, dream replay, service orchestration or final diagnostic result construction, fold it back into the canonical modules above.

### Priority 3 - Strengthen helpers before expanding architecture

Highest-value technical work:

- real Pillar-0 adapter behind `specialist_adapters.py`;
- real Gemma 4 adapter behind `specialist_adapters.py`;
- optional Claude critic behind critique loop;
- Weaviate infrastructure subclass;
- richer retrieval and evidence ranking;
- dataset-driven calibration;
- prospective validation storage hardening.

### Priority 4 - Treat this package as the stable research core

Future cleanup should start from this map and remove or merge modules only when they clearly duplicate the canonical core.
