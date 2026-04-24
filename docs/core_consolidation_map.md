# Melampo Core Consolidation Map

This document tracks the canonical core of `src/melampo/` after the recent intuition–dream–differential refactors.
The goal is to support future consolidation work without touching the stabilized clinical reasoning path blindly.

## Core module map

| Module | Current status | Main issue | Next consolidation action |
|---|---|---|---|
| `src/melampo/app.py` | Canonical runtime entrypoint | Runtime assembly is clear, but many components are still wired as generic objects rather than typed service contracts. | Keep as canonical entrypoint and gradually tighten typed interfaces around the assembled components. |
| `src/melampo/reasoning/clinical_pipeline.py` | Canonical end-to-end reasoning flow | The main flow is now rich, but it still constructs several helpers inside `run()` rather than through persistent injectable services. | Keep as canonical path and later move repeated helper construction into runtime-level dependency assembly. |
| `src/melampo/reasoning/intuition_engine.py` | Canonical intuition core | Candidate scoring is strong enough for research scaffolding, but still heuristic-heavy and not yet calibrated by real domain data. | Keep as canonical intuition engine and evolve scoring/calibration here, not in side modules. |
| `src/melampo/reasoning/differential_engine.py` | Canonical differential core | Now supports typed hypotheses and actions, but domain/test generation still relies on lightweight rules. | Keep as canonical differential core and expand its typed action logic rather than creating parallel differential modules. |
| `src/melampo/reasoning/support_contradiction.py` | Canonical support/contradiction analyzer | Strong centralization exists, but signal typing is still rule-based and may grow quickly. | Keep as the single place for typed support/contradiction logic and resist duplicating this logic elsewhere. |
| `src/melampo/reasoning/critique_loop.py` | Canonical critique layer | Warnings, suggestions, and priorities are in place, but critique still depends heavily on upstream heuristics. | Keep as canonical critique layer and plug future prioritization/escalation logic here. |
| `src/melampo/reasoning/area_coherence.py` | Canonical inter-area dynamics analyzer | Lightweight by design; currently useful, but pair logic is still simplified. | Keep centralized here and expand only if area-dynamics complexity truly grows. |
| `src/melampo/training/dream_trainer.py` | Canonical dream/replay branch | Good bridge between replay and revision, but still a research scaffold without a richer generative curriculum. | Keep as the single dream branch and evolve rare-case, contradiction, and mismatch-resolution generation here. |
| `src/melampo/memory/retriever.py` | Active helper in canonical flow | Retrieval is lightweight and still mostly structural. | Keep active, but strengthen grounding quality before adding more retrieval abstractions. |
| `src/melampo/models/evidence_ranker.py` | Active helper in canonical flow | Ranking is still minimal and largely positional. | Keep active and improve ranking quality here rather than scattering ranking logic across modules. |
| `src/melampo/areas/visual_diagnostic_area.py` | Canonical visual area | Stable as a channel, but still mostly a structured carrier. | Keep canonical; future enrichment should remain inside this area file. |
| `src/melampo/areas/language_listening_area.py` | Canonical narrative/listening area | Stable as a channel, but still shallow on narrative temporal structure. | Keep canonical; future temporal/narrative enrichment should stay here. |
| `src/melampo/areas/case_context_area.py` | Canonical context area | Small but useful; still underpowered compared with the other areas. | Keep canonical and expand only when stronger structured context handling is needed. |
| `src/melampo/areas/epidemiology_area.py` | Canonical epidemiology area | Stable, but still driven by simple exposure/demography grouping. | Keep canonical; grow only if prevalence and exposure reasoning become richer. |

## Consolidation priorities

### Priority 1 — Preserve and strengthen the canonical chain
Future reasoning changes should converge into this path:

`clinical_pipeline -> intuition_engine -> differential_engine -> critique_loop`

### Priority 2 — Avoid parallel reasoning entrypoints
If a new experiment duplicates intuition, support/contradiction, differential ranking, or critique behavior, it should be folded back into the canonical modules above instead of creating sibling pipelines.

### Priority 3 — Strengthen helpers before expanding architecture
The most useful next technical work is not adding more top-level abstractions, but improving the existing helpers:
- `retriever`
- `evidence_ranker`
- selected area modules
- dream generation curriculum

### Priority 4 — Treat this package as the stable research core
The current `src/melampo/` core is already coherent enough to be treated as the main project spine. Future cleanup should start from this map and only remove or merge modules when they clearly duplicate the canonical core above.
