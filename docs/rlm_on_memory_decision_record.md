# RLM-on-Memory: Retrieval Strategy Decision Record

Supersedes: the retrieval-strategy sections of `enterprise_ai_rag_evolution.md` (the memory-substrate sections of that document remain in force)

## Status

This record carries claims of three different kinds, and a single status field
would misrepresent all of them. Each section below is marked with its type.

| Type | Meaning | Reversal cost |
|---|---|---|
| **CONSTRAINT** — accepted | An architectural boundary. Holds because the project chose it; no measurement can refute it. | Redesign |
| **PLAN** — accepted, with a named review trigger | A sequencing decision. Binding until the named event occurs. | Re-plan |
| **CLAIM** — not accepted; registered as falsifiable | An empirical prediction. Not settled by being written down. | None; it was never a commitment |

Claims are registered in `evaluation/falsification_program.py`, each with the
observation that would refute it. Three of them are marked blocking: the
strategy cannot leave research use while they remain open.

Marking an untested prediction "accepted" is precisely the error this project's
governance exists to prevent, and a decision record is a design-control artefact
where that error is legible to an auditor. Marking the whole record "proposed"
would be the opposite failure: code implementing it has already landed, and
implementation without a decision is harder to defend than either alternative.

## Context

Melampo's retrieval layer currently performs a single `retrieve(query, top_k=5)`
call and passes the resulting evidence downstream. Recursive Language Models
(RLM) offer an alternative: rather than selecting fragments in advance, the
context is held in an environment and a root model writes code to navigate it,
recursively invoking a smaller model on individual fragments.

The proposal under review was "use RLM instead of RAG".

## Decision

**Type: CONSTRAINT — accepted.**

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

**Type: CLAIM — registered as falsifiable** (`rlm.dual_path_beats_single_path`, `rlm.disagreement_is_informative`).

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

**Type: PLAN — accepted.** Review trigger: measured cost of recursive retrieval
on simple lookups.

Previously registered as a claim. Withdrawn 2026-09-02: the gate is justified on
cost and latency grounds alone — running recursive retrieval on every trivial
lookup is expensive regardless of accuracy — so it does not depend on the
accuracy claim and the claim need not gate anything. The claim is retained in
the registry with that reason rather than deleted.

`ModelRouter` is currently a static 12-line router. It becomes the gate that
decides *how many* paths run, not which one:

| Case profile | Mode |
|---|---|
| Factual lookup, low risk | One-shot only |
| Complex or high risk | Dual path with reconciliation |
| Unresolved area mismatch | Dual path, extended recursive budget |
| Dream branch (low activity) | Recursive only; latency is not binding |

### Authority boundary

**Type: CONSTRAINT — accepted.**

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

**Type: CONSTRAINT — accepted.**

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

**Type: CONSTRAINT — accepted** for the isolation requirement; **CLAIM** for the value of the hypotheses themselves (`rlm.dream_hypotheses_add_value`).

Hypotheses are found rather than written. A candidate is a path through the
concept graph connecting the observed findings to a condition the case has not
raised; the path is its provenance. The difference from a generative model is
categorical rather than one of quality — a path exists in the graph or it does
not, so a hypothesis cannot be fluent and baseless at once.

Speculation is a gradient, not a category. A short path over well-attested edges
is a connection any clinician would raise; a long path over weak edges is rarely
considered but still traceable. Novelty and support are reported separately,
because collapsing them into one score is what makes a speculative hypothesis
look like a strong one. No path at all is not speculation, and is discarded.

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

A second gate covers graph coverage. On a sparse neighbourhood every unknown
path reaches an upper bound of 1.0, so ranking by the ceiling degenerates: a
three-hop chain of unknowns outranks a two-hop chain of well-attested edges and
the order is decided by tie-breaks. The failure is not noise but loss of
ordering, and emitting the top three of an arbitrary order is worse than
emitting none.

Waiting for a complete graph is not the answer either. A clinical graph never
reaches completeness, so the threshold never arrives, and it deadlocks: the
completion queue is fed by use, and use would be waiting on the queue. Coverage
is therefore assessed locally, per case, and a sparse neighbourhood switches
register rather than silencing the branch — the output becomes questions aimed
at the knowledge base, which is useful from the first day.

Two structural bounds keep sparse traversal honest: at most one unknown edge per
path, since two concatenated unknowns are not a weaker inference but a different
object; and corroboration by two independent findings, which survives a sparse
neighbourhood because a spurious path from one finding is easy while two
converging on the same condition is not.

This has a consequence for `rlm.dream_hypotheses_add_value`: while local density
is low the claim is not testable, because the measurement would reflect graph
coverage rather than hypothesis value. Clinical review must be run on cases
selected from dense regions, or the result is predetermined and negative.

Implemented in `training/hypothesis_channel.py`.

## Model selection

Three roles with distinct and partly opposing requirements.

### Root model

Drives the recursive loop. Requires reliable Python generation, strict
instruction following (the `FINAL()` protocol is a common failure point),
multi-step planning, and low per-iteration cost. It does **not** require medical
knowledge: it decides where to look, not what to conclude. Selecting it by
medical benchmark scores optimises the wrong variable.

**Type: PLAN — accepted.** Review trigger: PHI entering the environment, or
refutation of `rlm.open_weight_root_is_sufficient`.

**Decision.** Two phases.

*Phase 1 — frontier API root model, synthetic and de-identified cases only.*
Establishes the baseline and answers whether recursive retrieval helps at all
before any infrastructure is procured. No PHI enters the environment in this
phase, which is what makes an external API admissible; the constraint is
enforced by the corpus, not by trust.

*Phase 2 — self-hosted open-weight root model.* Required before any real case
enters the environment, since the environment holds the entire raw context.

**Why this order.** Starting on-premise risks attributing to the architecture
weaknesses that belong to whichever model could be run locally. A negative
Phase 1 result is decisive — if recursive retrieval does not help with a strong
root model, it will not help with a weaker one — while a negative on-premise-first
result is uninterpretable.

**The sub-model is self-hosted from Phase 1.** It sees fragments rather than the
whole context, so it could in principle be remote in Phase 1, but keeping it
on-premise throughout means the transition changes exactly one variable. If both
the root and the sub-model move at once, the observed delta cannot be attributed.
It also avoids establishing an egress path that later has to be removed.

**What Phase 1 does not establish.** The Phase 1 baseline is an upper bound on
the architecture, not the system's performance. It must not be cited as a
capability of Melampo, and any model card figure must come from the Phase 2
configuration. This is recorded because a favourable number produced under
relaxed conditions is the kind that survives into a document where it does not
belong.

**Transition criteria.** Phase 2 begins when: a Phase 1 baseline exists on the
same case set to be reused; a pre-registered tolerance for acceptable
degradation has been set *before* the on-premise model is measured; the sandbox
has no network egress; and the audit store treats retrieval trajectories as
health data. The tolerance is pre-registered because setting it afterwards makes
it a description of the result rather than a test of it.

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

### External APIs beyond Phase 1

**Type: PLAN — accepted.** Review trigger: legal review outcome, or evidence of
reconstruction by aggregation.

External APIs remain available past Phase 1 under a predicate oracle: instead of
releasing an attribute, a local agent answers the clinical question that
attribute would have served. Geography is the motivating case — it is almost
never diagnostic in itself, only through a mediating factor such as endemic
exposure, so answering "endemic risk present, band moderate" supplies the
operative variable and withholds the identifier.

This is not an exit from the GDPR and must not be described as one. It is a
defensible transfer position: pseudonymisation as a technical measure, EU-region
endpoints, no training on the data, zero retention, measured de-identification
recall. The controller retains the linkage, so the data remains personal.

Two risks are registered as claims rather than assumed away. Each answer is
low-information alone, but their conjunction may not be
(`privacy.predicate_budget_prevents_reconstruction`, blocking); and withholding
the attribute may cost diagnostic quality
(`privacy.predicate_disclosure_preserves_diagnosis`).

Design constraints: a bounded query budget with cumulative disclosure tracking;
discrete risk bands rather than continuous scores, which carry more bits; and
proactive screening of relevant exposure factors alongside reactive answers,
since an oracle can only be asked about hypotheses the caller has already
formed.

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

Third tranche (root model decision, scoped status, falsification registry):

- `evaluation/falsification_program.py` — claim registry replacing three
  hardcoded strings; resolution requires evidence; three claims are blocking

Eighth tranche (ontology import, coverage measurement):

- `memory/ontology_import.py` — HPO disease-phenotype annotations become
  interval edges. The frequency column is **already published as a range**
  (obligate 100%, very frequent 80–99%, frequent 30–79%, occasional 5–29%, very
  rare 1–4%, excluded 0%), so the interval representation is not a formalism
  imposed on the data: a point estimate would have been the lossy step. Observed
  fractions become Wilson intervals, so epistemic width falls out of sample size
  rather than being assigned
- `memory/graph_coverage.py` — coverage against a reference relation set,
  separating present, reachable-through-a-gap, and absent. The first two feed
  different queues: calibration and completion. `evaluation_is_interpretable`
  guards the case where a measurement would reflect the knowledge base while
  appearing to measure the architecture

Verified against the full annotation file, 285,598 edges: 9.8% documented,
42.6% uncertain positive, 19.9% weak negation, 24.2% gap, 3.4% documented
exclusion. Almost two thirds of real edges carry meaningful epistemic width, and
weak negation — the state a four-item list omits — covers 56,954 relations that
would otherwise have been recorded as documented exclusions.

Seventh tranche (interval edges, density gating, register switch):

- `memory/concept_paths.py` — edges hold an interval, not a point. A single
  number cannot separate "documented as rare" from "nobody has looked": both
  arrive small, and after multiplication along a path both read as near-certain
  denial. Paths carry `strength_lower`, `strength_upper` and `gap_count`;
  `local_density` measures coverage around one case
- `training/mechanism_enumeration.py` — filters and ranks by the plausibility
  ceiling instead of point support. Filtering on support discarded exactly the
  uncertain paths, surfacing conditions the graph already knew well, and it
  contradicted the novelty measure, which rewards low support
- `reasoning/discriminating_tests.py` — ranks by guaranteed gain. A missing edge
  now reads as the full interval rather than a low value

Three consumers read the same interval differently: the diagnostic path takes
the lower bound (what is guaranteed), hypothesis enumeration takes the upper
bound (what could be true), and graph maintenance takes the width (where looking
pays). Exploration is optimistic by construction, decision prudent by
construction.

Sixth tranche (claim revision, route conditionality, discriminating tests):

- `evaluation/falsification_program.py` — claims may be conditional on a route.
  Taking a route substitutes open questions rather than removing them: choosing
  external APIs retires the on-premise sufficiency question and raises two
  privacy ones, and the blocking count stays at three either way
- `reasoning/discriminating_tests.py` — investigations ranked by expected
  information gain over the concept graph, replacing fixed per-domain strings
  that never inspected the hypotheses in contention

Fifth tranche (knowledge-mediated grounding — corrects the fourth):

- `memory/concept_paths.py` — bounded traversal over the clinical concept graph,
  with path strength and shared-mechanism discovery
- `evaluation/grounding_judge.py` — three verdicts instead of two. A relation
  absent from the case is not automatically an error: most clinical inference
  connects a finding to a condition through knowledge outside the case. Checking
  only against case fragments conflates inference with fabrication and rejects
  both
- `training/mechanism_enumeration.py` — hypotheses found by path enumeration
  rather than written by string concatenation, replacing
  `_alternative_hypotheses()` and the `0.2 * len(list)` novelty formula

Fourth tranche (claim `rlm.dual_path_beats_single_path` made testable):

- `evaluation/grounding_judge.py` — structural overreach detection. Complements
  rather than replaces `RAGEvaluator.faithfulness`, which is term overlap and by
  construction cannot see a relation asserted across two fragments: every term
  is supported, every citation is real, and the unsupported part is the relation
- `evaluation/dual_path_ab.py` — paired A/B harness evaluating the registered
  refutation criterion. Within-case pairing, because case difficulty varies more
  than the difference between strategies. Distinguishability decided by a
  deterministic bootstrap interval rather than a mean, so a gain spanning zero
  cannot pass. An inconclusive run leaves the claim open

## Open items

1. ~~Root model on external API or on-premise.~~ Resolved: two-phase, API on
   synthetic cases then self-hosted open-weight. See *Root model* above. The
   sufficiency of the open-weight root is registered as a blocking claim.
2. Clinical text model selection, replacing the unverifiable registry entry.
3. `mean_grounding_score` still has no recursive equivalent. One-shot retrieval
   sources it from vector similarity; a recursive strategy navigates by code and
   produces no such score. A substitute — sub-model verification, or offset
   coverage of the claim — must exist before the two paths are compared, or the
   A/B measures nothing. *Coverage is resolved: both bases are now declared and
   cross-basis comparison raises.*
4. ~~Mapping character offsets onto the fields `safety/rails.py` accepts as
   traceable.~~ Resolved: the trace check now accepts character offsets.
5. ~~Grounding judge for overreach detection.~~ Partially resolved:
   `evaluation/grounding_judge.py` detects overreach structurally — relations
   asserted between entities that never co-occur in a cited fragment, modality
   escalation, unsupported terms, span inflation. This is a lexical floor, not a
   semantic judge: it cannot recognise a paraphrased relation and will flag some
   legitimate synthesis. A semantic judge remains open; where the two would
   disagree, the structural one is the conservative side.
6. Pre-registered degradation tolerance for the Phase 2 root model transition,
   to be set before the on-premise model is measured.
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
