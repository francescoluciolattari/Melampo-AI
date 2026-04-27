# Melampo Core Consolidation Map

This document tracks the canonical core of `src/melampo/` after the recent intuition–dream–differential refactors.
The goal is to support future consolidation work without touching the stabilized clinical reasoning path blindly.

## Final package classification

### Canonical clinical reasoning spine

| Module | Classification | Current status | Main issue | Next consolidation action |
|---|---|---|---|---|
| `src/melampo/app.py` | Canonical | Runtime entrypoint | Runtime assembly is clear, but many components are still wired as generic objects rather than typed service contracts. | Keep as canonical entrypoint and gradually tighten typed interfaces around the assembled components. |
| `src/melampo/reasoning/clinical_pipeline.py` | Canonical | Main end-to-end reasoning flow | The main flow is now rich, but it still constructs several helpers inside `run()` rather than through persistent injectable services. | Keep as canonical path and later move repeated helper construction into runtime-level dependency assembly. |
| `src/melampo/reasoning/intuition_engine.py` | Canonical | Main intuition core | Candidate scoring is strong enough for research scaffolding, but still heuristic-heavy and not yet calibrated by real domain data. | Keep as canonical intuition engine and evolve scoring/calibration here, not in side modules. |
| `src/melampo/reasoning/differential_engine.py` | Canonical | Main differential core | Supports typed hypotheses, domains, actions, and support/contradiction metadata, but domain/test generation still relies on lightweight rules. | Keep as canonical differential core and expand its typed action logic rather than creating parallel differential modules. |
| `src/melampo/reasoning/support_contradiction.py` | Canonical | Central typed signal analyzer | Signal typing is centralized, but remains rule-based and may grow quickly. | Keep as the single place for typed support/contradiction logic and resist duplicating this logic elsewhere. |
| `src/melampo/reasoning/critique_loop.py` | Canonical | Main critique layer | Warnings, suggestions, and priorities are in place, but critique still depends heavily on upstream heuristics. | Keep as canonical critique layer and plug future prioritization/escalation logic here. |
| `src/melampo/reasoning/pipeline_coordinator.py` | Canonical | Main reasoning/policy/differential coordinator | Now exposes richer trace/state metadata, but still delegates all semantics to upstream modules. | Keep canonical and evolve only as the orchestration point, not as a second reasoning engine. |

### Canonical support and runtime-control modules

| Module | Classification | Current status | Main issue | Next consolidation action |
|---|---|---|---|---|
| `src/melampo/reasoning/area_coherence.py` | Canonical support | Inter-area dynamics analyzer | Now emits pair profiles and total salience, but pair semantics are still simplified. | Keep centralized here and only expand if area-dynamics complexity truly grows. |
| `src/melampo/reasoning/policy_stack.py` | Canonical support | Structured policy layer | Now returns reasons and assessments, but decision semantics are still threshold-driven. | Keep as the policy integrator and expand only if policy semantics become richer. |
| `src/melampo/reasoning/escalation.py` | Canonical support | Escalation helper | Now returns reasons and escalation level, but remains intentionally lightweight. | Keep and refine only if escalation paths become more granular. |
| `src/melampo/reasoning/decision_trace.py` | Canonical support | Audit trace helper | Minimal by design; now supports key-value trace entries. | Keep lightweight and avoid turning it into a second state container. |
| `src/melampo/reasoning/pipeline_state.py` | Canonical support | Shared runtime state | Still intentionally small, but now exposes a summary helper. | Keep as lightweight runtime state and avoid embedding reasoning logic here. |
| `src/melampo/evaluation/quantum_gate.py` | Canonical support | Structured quantum-like path gate | Now emits allow, margin, level, and reasons. | Keep as the only gate for optional quantum-like evaluation paths. |

### Canonical orchestration and registry modules

| Module | Classification | Current status | Main issue | Next consolidation action |
|---|---|---|---|---|
| `src/melampo/orchestration/contracts.py` | Canonical support | Service contract metadata | Now includes name, provider, protocol, and role. | Keep as the canonical service contract description layer. |
| `src/melampo/orchestration/service_registry.py` | Canonical support | In-memory service registry | Now stores role metadata and exposes a registry summary. | Keep lightweight and avoid embedding routing semantics here. |
| `src/melampo/orchestration/bootstrap.py` | Canonical support | Baseline service contract inventory | Now exposes a contract inventory before building the registry. | Keep as the single baseline registry bootstrap. |
| `src/melampo/orchestration/model_router.py` | Canonical support | Static research router | Now emits protocol hints and routing mode. | Keep simple until dynamic model routing is introduced. |
| `src/melampo/orchestration/runtime_services.py` | Canonical support | Runtime service resolution layer | Now returns availability, protocol, and resolution mode. | Keep as the canonical runtime-service resolution facade. |

### Canonical dream and replay branch

| Module | Classification | Current status | Main issue | Next consolidation action |
|---|---|---|---|---|
| `src/melampo/training/dream_trainer.py` | Canonical | Main dream/replay branch | Good bridge between replay and revision, but still a research scaffold without a richer generative curriculum. | Keep as the single dream branch and evolve rare-case, contradiction, and mismatch-resolution generation here. |
| `src/melampo/training/replay_filter.py` | Canonical support | Replay assessment helper | Now emits replay assessment metadata, but still relies on threshold-based replay modes. | Keep active and enrich here before adding more replay abstractions. |
| `src/melampo/training/counterfactual_sampler.py` | Canonical support | Counterfactual variant helper | Now emits focus, perturbation plan, and novelty metadata, but remains intentionally lightweight. | Keep active and enrich here before introducing alternative samplers. |

### Canonical evidence and grounding helpers

| Module | Classification | Current status | Main issue | Next consolidation action |
|---|---|---|---|---|
| `src/melampo/memory/retriever.py` | Canonical support | Grounded retrieval helper | Now emits focus and grounding metadata, but retrieval remains structural and not yet knowledge-rich. | Keep active and strengthen grounding quality before adding more retrieval abstractions. |
| `src/melampo/models/evidence_ranker.py` | Canonical support | Ranked evidence helper | Now uses source/kind/grounding scoring, but remains heuristic and not calibrated. | Keep active and improve ranking quality here rather than scattering ranking logic across modules. |

### Canonical area modules

| Module | Classification | Current status | Main issue | Next consolidation action |
|---|---|---|---|---|
| `src/melampo/areas/visual_diagnostic_area.py` | Canonical | Visual/imaging area | Now exposes focus, signal count, and salience, but remains a structured carrier rather than a rich interpreter. | Keep canonical; future enrichment should remain inside this area file. |
| `src/melampo/areas/language_listening_area.py` | Canonical | Narrative/listening area | Now exposes focus and salience, but remains shallow on narrative temporal structure. | Keep canonical; future temporal/narrative enrichment should stay here. |
| `src/melampo/areas/case_context_area.py` | Canonical | Context area | Small but now aligned with the area metadata contract. | Keep canonical and expand only when stronger structured context handling is needed. |
| `src/melampo/areas/epidemiology_area.py` | Canonical | Epidemiology area | Now exposes focus and salience, but still relies on simple exposure/demography grouping. | Keep canonical; grow only if prevalence and exposure reasoning become richer. |

### Canonical decision-control and belief modules

| Module | Classification | Current status | Main issue | Next consolidation action |
|---|---|---|---|---|
| `src/melampo/models/abstention.py` | Canonical support | Structured abstention assessment | Now returns reasons, level, and margin. | Keep and refine only if abstention semantics become more complex. |
| `src/melampo/models/risk_gate.py` | Canonical support | Structured risk assessment | Now returns reasons, band, and margin. | Keep and refine only if risk gating becomes domain-specific. |
| `src/melampo/models/quantum_belief_layer.py` | Canonical support | Contextual belief-update metadata layer | Now exposes contextuality, interference, and belief shift, but remains a lightweight research formalism. | Keep as the single quantum-like belief layer and expand here instead of duplicating belief-update logic. |

## Modules to treat as research scaffold but acceptable

These modules are already part of the canonical core, but should still be treated as research scaffolds rather than production-complete implementations:
- `intuition_engine.py`
- `differential_engine.py`
- `support_contradiction.py`
- `dream_trainer.py`
- `quantum_belief_layer.py`
- `retriever.py`
- `evidence_ranker.py`
- `model_router.py`
- `runtime_services.py`

The right strategy is to evolve these files in place, not to create sibling replacements.

## Consolidation priorities

### Priority 1 - Preserve and strengthen the canonical chain
Future reasoning changes should converge into this path:

`clinical_pipeline -> intuition_engine -> differential_engine -> critique_loop`

### Priority 2 - Avoid parallel reasoning entrypoints
If a new experiment duplicates intuition, support/contradiction, differential ranking, critique behavior, dream replay behavior, or service orchestration behavior, it should be folded back into the canonical modules above instead of creating sibling pipelines.

### Priority 3 - Strengthen helpers before expanding architecture
The most useful next technical work is not adding more top-level abstractions, but improving the existing helpers:
- `retriever`
- `evidence_ranker`
- selected area modules
- dream generation curriculum
- policy semantics
- runtime service contracts
- calibration of intuition/differential weights

### Priority 4 - Treat this package as the stable research core
The current `src/melampo/` core is coherent enough to be treated as the main project spine. Future cleanup should start from this map and only remove or merge modules when they clearly duplicate the canonical core above.
