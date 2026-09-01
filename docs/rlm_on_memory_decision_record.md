# RLM-on-Memory: Retrieval Strategy Decision Record

Status: accepted for implementation, foundation tranche landed
Supersedes: the retrieval-strategy sections of `enterprise_ai_rag_evolution.md` (the memory-substrate sections of that document remain in force)

## Context

Melampo's retrieval layer currently performs a single `retrieve(query, top_k=5)`
call and passes the resulting evidence downstream. Recursive Language Models
(RLM) offer an alternative: rather than selecting fragments in advance, the
context is held in an environment and a root model writes code to navigate it,
recursively invoking a smaller model on individual fragments.

The proposal under review was "use RLM instead of RAG".

## Decision

**The migration is scoped to the retrieval strategy only.** The term "RAG" in
this repository covers two separable concerns:

| Concern | Implementation | Affected |
|---|---|---|
| Memory substrate | Weaviate objects and properties, ontologies, knowledge graph, visual imprints | **No.** A recursive strategy has no persistent storage. The substrate becomes more important, not less. |
| Retrieval strategy | `MemoryRetriever.retrieve`, single-shot, `top_k` | **Yes.** This is the layer being replaced. |

The resulting architecture is **RLM-on-Memory**: the governed semantic memory is
the navigable environment, and recursive retrieval is the navigation strategy
over it. Structural navigation affordances — typed relations, entity properties,
`graph_expand` — are what make recursive retrieval accurate; the environment is
therefore built on the semantic memory adapter, not on raw document text.

### Accepted claims

- Recursive retrieval eliminates **a priori lossy compression**: fixed chunk
  boundaries and `top_k` truncation discard information before the question is
  known, and a therapeutic regimen split across two chunks cannot be recovered
  by any downstream model.
- It does **not** eliminate information loss. The loss moves from the summariser
  to the exploration policy: what the root model never queries is never seen,
  and an unissued query leaves no inspectable trace. Coverage is therefore a
  runtime measurement, not a structural guarantee.
- Recursive retrieval **degrades on simple lookups**. A single-fact retrieval
  the model solves directly is replaced by a five-link chain — plan, generate
  code, execute, interpret, terminate — each link a new failure point. Reported
  degradation on single-needle retrieval is roughly 15 points at depth 1 and 30
  at depth 2.
- Recursive retrieval is subject to **overreach**. The root model composes
  fragment summaries into a narrative whose connective tissue originates from no
  source. Every citation remains individually valid, so provenance checks pass
  at 100% while the claim exceeds its evidence. Detecting this requires a
  grounding judge that evaluates whether the conclusion follows from the
  premises, not whether the premises exist.

## Architecture

### Dual-path retrieval

The two strategies fail in opposite directions — one-shot by omission,
recursive by overreach — which is the condition under which running both is
worth its cost. Their divergence is an empirical uncertainty estimate for a
system whose safety architecture is built on uncertainty.

| Outcome | Disposition |
|---|---|
| Found by both | Confirmed, highest confidence |
| One-shot only | Admitted; the fast path reached it first |
| Recursive only, offsets verifiable | Admitted as recall gain |
| Recursive only, no offsets | Discarded as probable overreach |
| Paths contradict | Conflict signal raised; escalation or abstention |

Implemented in `reasoning/retrieval_reconciliation.py`. Reconciliation is
deterministic: no model adjudicates between the paths.

### Cognitive asymmetry

The system contains two reasoning engines with opposite characters —
`IntuitionEngine` (fast, associative) and `DifferentialEngine` (serial,
controlled) — but both are currently fed by the same `top_k=5` retrieval. The
dual-path corrects this: one-shot retrieval feeds intuition, recursive retrieval
feeds the differential. Each engine receives the memory access consistent with
its nature.

### Complexity routing

`ModelRouter` is currently a static 12-line router. It becomes the gate that
decides *how many* paths run, not which one:

| Case profile | Mode |
|---|---|
| Factual lookup, low risk | One-shot only |
| Complex or high risk | Dual path with reconciliation |
| Unresolved area mismatch | Dual path, extended recursive budget |
| Dream branch (low activity) | Recursive only; latency is not binding |

### Authority boundary

`MelampoDiagnosticOrchestrator` remains the sole diagnostic authority, and it is
deterministic Python rather than a model. Recursive retrieval produces a
**grounded case dossier** delivered through `reasoning/workspace.py`; the
orchestrator reads the dossier and decides. A root model that produced the
differential would make an external model the arbiter while leaving the
orchestrator nominally in place.

The following remain outside the recursive perimeter and deterministic:
`safety/rails.py`, `training/promotion_policy.py`,
`evaluation/model_release_gate.py`.

### Ingestion

Ingestion must be reproducible: the same document must yield the same memory
today and in two years, or retrospective validation cannot be repeated. A
recursive strategy is non-deterministic by construction and therefore does not
belong on the write path to permanent memory.

Two layers instead:

| Layer | Producer | Properties |
|---|---|---|
| Source | `data/document_processing.py`, Docling | Deterministic, immutable, versioned; the reference for audit |
| Derived | Recursive synthesis, schema-enforced | Marked `derived`, versioned by model id, re-derivable, never overwrites the source |

### Hypothesis channel

Dream candidates enter the differential as **exclusion hypotheses**, never as
evidence. A differential diagnosis is itself a set of hypotheses to be excluded,
so a candidate framed as "consider also X, synthetically generated, not
observed" is legitimate; the same candidate framed as "memory supports X" is
contamination.

The separation is structural rather than a metadata flag. Candidates sharing a
collection with clinical evidence compete for the same `top_k` slots under the
same similarity ranking, so a single downstream caller that omits a filter
reintroduces them silently. The channel therefore uses a distinct namespace, a
dedicated retrieval path and one authorised consumer.

Hypotheses are gated on diagnostic indeterminacy — low `convergence_index`, high
`conflict_load`, material risk — because additional synthetic alternatives are
informative only when the differential is flat.

Implemented in `training/hypothesis_channel.py`.

## Model selection

Three roles with distinct and partly opposing requirements.

### Root model

Drives the recursive loop. Requires reliable Python generation, strict
instruction following (the `FINAL()` protocol is a common failure point),
multi-step planning, and low per-iteration cost. It does **not** require medical
knowledge: it decides where to look, not what to conclude. Selecting it by
medical benchmark scores optimises the wrong variable.

**Sequencing decision:** start with a frontier API model on synthetic and
de-identified cases to establish a baseline, then measure the loss when moving
to a self-hosted open-weight model. The reverse order risks attributing to the
architecture weaknesses that belong to whichever model could be run locally.

### Sub-model

Small, fast, high-recall extraction over fragments. Sees raw fragments and
therefore sees PHI: self-hosted, no egress. Candidate: MedGemma 1.5 4B.

### Clinical specialists

| Role | Decision |
|---|---|
| Volumetric imaging | **Pillar-0** — confirmed. Pretrained on >155,000 CT/MRI volumes; best in 319 of 366 tasks with a 7.8–15.8 AUROC margin over comparable models. |
| Clinical text | **"Gemma 4" requires replacement.** No verifiable downloadable artefact carries this name, which is a traceability defect for the model card. Candidates: Gemma-3-27B-MeditronFO (58.02 HealthBench, fully open pipeline — an audit advantage) or MedGemma 1.5 27B. Tracked as an open item; identifiers in `models/specialist_adapters.py` and `orchestration/model_capability_registry.py` are unchanged pending that decision. |
| External critique | **Claude** — confirmed, on de-identified input only. |
| Document parsing | **Docling** — confirmed and more central; supplies the character offsets the provenance layer depends on. |

### PHI perimeter

The entire raw case context resides in the recursive environment, and fragments
of it are passed to sub-model calls. Under GDPR and the EU AI Act this requires:
an on-premise sandbox with no network egress, a self-hosted sub-model whenever
text is not de-identified, external APIs only on de-identified or synthetic
data, and retrieval trajectories treated as health data in the audit store.

## Implementation status

Landed in this tranche:

- `memory/context_environment.py` — typed navigation primitives, mandatory
  character offsets, instrumented coverage ledger
- `memory/retrieval_contract.py` — shared contract, mode constants, validator
  for the silent failure modes
- `reasoning/retrieval_reconciliation.py` — dual-path disposition matrix and
  conflict signal
- `training/hypothesis_channel.py` — indeterminacy gate and exclusion-hypothesis
  envelope
- `tests/test_rlm_on_memory_foundation.py` — 15 tests

No existing module was modified: the tranche is additive, and the recursive
strategy joins as an additional `retrieval_mode` value.

Second tranche (coverage semantics and provenance):

- `memory/retrieval_contract.py` — `coverage_basis` declaration and
  `assert_coverage_comparable`, which raises rather than silently comparing a
  selection ratio against a corpus-inspection ratio
- `memory/retriever.py` — all three one-shot branches now declare their basis
- `safety/rails.py` — the trace check accepts character offsets alongside
  `record_id`, page and section. Purely widening; nothing that previously
  carried a trace loses it
- `tests/test_coverage_semantics_and_provenance.py` — 8 tests

## Open items

1. Root model on external API or on-premise — determines whether the engine
   phase can begin before GPU infrastructure exists. Highest schedule impact.
2. Clinical text model selection, replacing the unverifiable registry entry.
3. `mean_grounding_score` still has no recursive equivalent. One-shot retrieval
   sources it from vector similarity; a recursive strategy navigates by code and
   produces no such score. A substitute — sub-model verification, or offset
   coverage of the claim — must exist before the two paths are compared, or the
   A/B measures nothing. *Coverage is resolved: both bases are now declared and
   cross-basis comparison raises.*
4. ~~Mapping character offsets onto the fields `safety/rails.py` accepts as
   traceable.~~ Resolved: the trace check now accepts character offsets.
5. Grounding judge for overreach detection in the release gate.
6. Validation corpus of long longitudinal cases; the current ChestX-ray14 and
   OpenI sets are too short to exercise recursive retrieval.

## Sequencing

| Block | Content | Gate |
|---|---|---|
| A | Credential hygiene, decision record | — |
| B | Environment, contract, metric redefinition | Coverage measurable; no synthetic evidence reaches a validation metric |
| C | Dream-branch pilot: hypothesis channel, grounded scenario enumeration | Reliability data on recursive retrieval where existing guardrails contain the error |
| D | Diagnostic dual path, reconciliation, routing, dossier delivery | Faithfulness not below the one-shot baseline |
| E | Grounding judge, derived ingestion layer, registry correction | Model card populated with measured numbers |

Block C precedes Block D deliberately. Overreach is a direct clinical risk on
the diagnostic path but not in the dream branch, where every output is a
candidate requiring validation and human review before clinical use, and where
low-activity scheduling makes recursive latency irrelevant. The reliability data
is acquired where an error costs nothing.

## References

- Zhang, Kraska, Khattab — *Recursive Language Models*, arXiv:2512.24601
- *Think, But Don't Overthink: Reproducing Recursive Language Models*, arXiv:2603.02615
- *C-RLM: Schema-Enforced Recursive Synthesis for Auditable, Long-Context Clinical Documentation*, medRxiv, January 2026
- *RLM-on-KG*, WordLift
- Agrawal et al. — *Pillar-0: A New Frontier for Radiology Foundation Models*, arXiv:2511.17803
- *An Auditable Pipeline for Clinical LLMs — Fully Open Meditron*, arXiv:2605.16215
