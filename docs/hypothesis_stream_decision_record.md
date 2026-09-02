# Hypothesis Stream Decision Record

How candidate conditions are generated, isolated, gated and delivered. Companion
to `rlm_on_memory_decision_record.md` and `semantic_extraction_decision_record.md`,
using the same three-tier status convention: **CONSTRAINT** (accepted, reversing
is a redesign), **PLAN** (accepted, with a named review trigger), **CLAIM** (not
accepted, registered as falsifiable).

---

## 1. What the stream is for

The Dream branch is a **hypothesis generator**, not part of the evidence path.
Its output answers a question the retrieval layer does not: what should be
considered that the case has not raised.

That places two obligations on it. It must produce candidates that are traceable
rather than invented, and it must be structurally incapable of contributing to
the evidence its own hypotheses would need in order to be confirmed. The second
is the harder one, and the one that failed silently for several iterations.

---

## 2. Two streams, not one

**Type: CONSTRAINT — accepted.**

Two channels generate candidate conditions. They answer different questions, so
they carry different gates and different delivery points.

| | Mechanism enumeration (Dream) | Family history screening |
|---|---|---|
| Question | What else could explain these findings? | What else might this patient have, independent of why they came in? |
| Origin | Concept-graph paths from the observed findings | A condition reported in a relative |
| Gate | Diagnostic indeterminacy, plus local graph density | Inheritance mode, reachable onset, prior assessment |
| Delivered to | Differential, as exclusion hypotheses | Screening list, outside the differential |
| Module | `training/mechanism_enumeration.py` | `reasoning/family_history.py` |

The screening channel deliberately **does not use the indeterminacy gate**. That
gate exists because an alternative explanation is informative only when the
differential is flat; a screening consideration is valid precisely when the
differential has settled, and gating it there would lose it in the clear cases —
which are the ones where a patient leaves without anyone having looked.

Screening items are also reported outside the differential. Placed inside, they
would imply they explain the presenting findings, which they usually do not.

---

## 3. Hypotheses are found, not written

**Type: CONSTRAINT — accepted.**

A candidate is a path through the concept graph connecting an observed finding to
a condition the case has not raised. The path is its provenance.

The difference from a generative model is categorical rather than one of quality:
a path exists in the graph or it does not, so a hypothesis cannot be fluent and
baseless at the same time. This replaced the previous implementation, which built
scenarios by string concatenation (`f"{base_label}_alt_1"`) and computed novelty
as `0.2 * len(perturbation_plan)` — arithmetic on a list length rather than a
measure of anything.

### The cascade

`shared_mechanisms` intersects the causes of one finding with the consequences of
another:

```
causes(bibasilar opacities)   = {pulmonary oedema, pneumonia, ...}
consequences(cardiac failure) = {pulmonary oedema, pleural effusion, ...}
                          ∩   = pulmonary oedema
```

The intersection is not a coincidence of vocabulary, it is a candidate mechanism.
Naming it converts an unsupported assertion of causation into a claim about a
specific pathway that can be examined and rejected.

### Speculation is a gradient

Within the paths that exist, **novelty and support are reported separately**. A
short path over well-attested edges is a connection any clinician would raise; a
long path over weak edges is rarely considered but still traceable. Collapsing
them into one score is what makes a speculative hypothesis look like a strong
one. No path at all is not speculation, and is discarded.

### Three readings of one interval

**Type: CONSTRAINT — accepted.**

| Consumer | Reads | Because |
|---|---|---|
| Diagnostic path | lower bound | what the evidence guarantees |
| Hypothesis enumeration | upper bound | what could be true |
| Graph maintenance | width | where looking pays |

Exploration is optimistic by construction, decision prudent by construction. The
enumerator previously filtered on point support, which discarded exactly the
uncertain paths and surfaced conditions the graph already knew well — and
contradicted its own novelty measure, which rewards low support.

---

## 4. Gating

**Type: CONSTRAINT — accepted** for the mechanism; **PLAN** for the thresholds.

Two gates, both required for the Dream channel.

### Indeterminacy

`convergence_index` low, `conflict_load` high, risk material. Additional
alternatives are informative only when the differential is flat; when one
hypothesis dominates, they add noise and consume review attention.

### Local graph density

Where coverage is thin every unknown path reaches an upper bound of 1.0 and
ranking degenerates: a three-hop chain of unknowns outranks a two-hop chain of
attested edges, and the order is decided by tie-breaks. The failure is not noise
but **loss of ordering**, and emitting the top three of an arbitrary order is
worse than emitting none.

Waiting for a complete graph is not the alternative. A clinical graph never
reaches completeness, so a completeness threshold never arrives, and it
deadlocks: the completion queue is fed by use. Coverage is therefore assessed
**locally, per case**.

Two structural bounds keep sparse traversal honest:

- **at most one unknown edge per path**, because two concatenated unknowns are
  not a weaker inference but a different object, the second conditioned on the
  first being true;
- **corroboration by two independent findings**, which survives a sparse
  neighbourhood because a spurious path from one finding is easy while two
  converging on the same condition is not.

### Register switch

Under low density the branch does not fall silent — it changes register. It emits
**questions aimed at the knowledge base** instead of hypotheses aimed at the
patient, routed to the graph completion queue. This makes the branch useful from
the first day and breaks the deadlock described above.

---

## 5. Isolation from the evidence path

**Type: CONSTRAINT — accepted.**

A synthetic candidate is not evidence. It can legitimately enter the differential
as a *hypothesis to be excluded* — a differential diagnosis is itself a set of
such hypotheses — but never as support for a conclusion.

### Why a metadata flag is insufficient

Candidates sharing a collection with clinical evidence compete for the same
`top_k` slots under the same similarity ranking. `learning_status: "candidate"`
is a property, not an access control: a single downstream caller that omits a
filter reintroduces them silently. In a system on a regulatory path, safety
cannot depend on every caller remembering to filter.

### Enforcement status

Isolation was described as structural for several iterations while being, in
practice, a convention. The two guards written for it — `assert_not_evidence` and
`assert_not_a_finding` — were **invoked zero times** on the production path,
exercised only in tests. A guard that is never called does not guard.

| Layer | Status |
|---|---|
| Logical role markers on every candidate | **Implemented** |
| Enforced boundary calling the guards | **Implemented** — `reasoning/findings_boundary.py` |
| Separate Weaviate collection (physical isolation) | **Open** — block B2 |

`reasoning/findings_boundary.py` is the single point at which patient findings
are assembled. It admits only what is a current, asserted finding of this
patient, and every rejection names where the item belongs instead:

| Rejected | Routes to |
|---|---|
| Negated | Documented exclusion |
| Hypothetical, "rule out" | Open question, discriminating test selection |
| Attributed to a relative | Family history channel |
| Historical | Clinical context, not current state |
| Synthetic hypothesis | Differential, as exclusion hypothesis only |
| Screening consideration | Screening list, outside the differential |

Role markers disqualify an item **before** any assertion check, because the role
already settles the question regardless of how the item is phrased.

### The governing principle

The distinction is **direction, not magnitude**. A finding is an entry point: the
graph is walked *from* it. Anything admitted by mistake generates paths toward
the consequences of something the patient does not have, and those paths are
structurally correct on a false premise — the hardest error to notice, because
nothing downstream looks wrong. Reducing a score does not soften this, because
traversal is binary.

A hypothesis, by contrast, is a **destination**: a candidate to confirm or
exclude, producing no descendants until confirmed.

### Promotion

**Type: CONSTRAINT — accepted.**

`PromotionPolicy` stays deterministic and outside the recursive perimeter. A
candidate admitted to semantic memory becomes retrievable evidence in later
cases, and the system would begin citing its own generated material with
formally intact provenance and substantively circular support.

The same asymmetry governs graph edges and the vocabulary: the system may
**propose** freely, and promotion requires a source that did not participate in
the reasoning — a curated ontology, independent literature, or human
confirmation. Errors in the restrictive direction announce themselves by
generating alarms; errors in the permissive direction conceal themselves by
ceasing to.

---

## 6. Connection to discriminating tests

A hypothesis with nothing in the record to confirm or exclude it is precisely the
input `reasoning/discriminating_tests.py` needs: which observation would most
resolve the uncertainty. Expected information gain is computed over the same
concept graph, ranked by the **guaranteed** gain so a test cannot win on the
strength of what is unknown about it.

This closes the loop: the hypothesis stream proposes, the test selector says how
to settle the proposal, and the boundary keeps neither from being mistaken for
evidence.

---

## 7. Registered claims

| Claim | Blocking | Status |
|---|---|---|
| `rlm.dream_hypotheses_add_value` | No | Open |

Statement: under high diagnostic indeterminacy, hypotheses enumerated from the
concept graph change the differential in a direction a clinician judges correct
more often than they distract.

**Not testable while local density is low.** The measurement would reflect graph
coverage rather than hypothesis value, and the result would be predetermined and
negative. Clinical review must draw cases from dense regions, and that requires
the coverage reference set (block A3.1) to exist first.

Containment establishes that the channel is *safe*. Safe and useful are different
questions: review attention is the scarce resource, and an irrelevant hypothesis
costs it precisely on the complex cases where the reviewer is already loaded.

---

## 8. Modules

| Module | Role |
|---|---|
| `memory/concept_paths.py` | Bounded traversal, interval strength, shared mechanisms, local density |
| `training/mechanism_enumeration.py` | Hypotheses by path enumeration; register switch under sparse coverage |
| `training/hypothesis_channel.py` | Indeterminacy gate, exclusion-hypothesis envelope |
| `reasoning/family_history.py` | Screening hypotheses and prior modifiers |
| `reasoning/findings_boundary.py` | Enforced separation from the evidence path |
| `reasoning/discriminating_tests.py` | Which observation would settle a hypothesis |
| `training/promotion_policy.py` | Deterministic, outside the recursive perimeter |

---

## 9. Open items

1. **Physical isolation** — separate Weaviate collection (B2). Until then
   isolation is enforced at the boundary but not at the store.
2. **Gate thresholds** — indeterminacy and density thresholds are set by
   judgement and need calibration against real cases.
3. **Prior shift table** for family history — a policy artefact requiring
   clinical review.
4. **`MechanismEnumerator` wiring** into `_alternative_hypotheses()` (B1), which
   depends on index-driven extraction (B0b).
