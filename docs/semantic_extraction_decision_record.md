# Semantic Extraction Decision Record

Companion to `rlm_on_memory_decision_record.md`, covering how clinical text
becomes traversable concepts. Uses the same three-tier status convention:
**CONSTRAINT** (accepted, reversing is a redesign), **PLAN** (accepted, with a
named review trigger), **CLAIM** (not accepted, registered as falsifiable).

---

## 1. Context

The concept graph and the case corpus were keyed differently. Graph nodes are
ontology identifiers and disease names; extraction ran against sixteen
hand-written English terms mapped to invented references. Nothing connected the
two, so every traversal began from a concept the graph had never held.

Closing that gap raised three further questions, each of which had a wrong
answer that looked reasonable:

1. how to represent assertion — negated, hypothetical, reported by whom;
2. what to do with modifiers such as *Bilateral* and *Progressive*;
3. where family history belongs.

---

## 2. Assertion representation

**Type: CONSTRAINT — accepted.**

### Decision

Extraction produces an **interval and an epistemic state**, never a scalar
probability, matching the representation edges already carry.

| Assertion | Interval | State |
|---|---|---|
| Affirmed by objective examination | `[0.90, 1.00]` | documented |
| Reported by the patient | `[0.50, 0.90]` | uncertain positive |
| Negated by objective examination | `[0.00, 0.05]` | documented exclusion |
| Negated by the patient | `[0.00, 0.30]` | weak negation |
| Hypothetical, "rule out" | `[0.00, 1.00]` | gap — not evidence |
| Unresolvable | *no finding emitted* | outside the graph |

### Rationale

A scalar reintroduces the collapse removed from edges. Under a multiplicative
score, "denies chest pain", an unresolvable finding, "pneumonia excluded on
imaging" and "patient does not report cough" all arrive as `0.0` — four distinct
epistemic states flattened into one number, and downstream a negated finding
becomes indistinguishable from an unknown one.

The multiplication has a second defect: it assumes independence between factors
that are not independent. Modality depends on source; tense interacts with
polarity ("no longer febrile" is neither pure negation nor pure past). A product
of correlated factors has no probabilistic reading.

The hypothetical case matters most. "Rule out pneumonia" is not evidence of
absence, it is an open question. Scored as a low number it enters the
differential as weak supporting evidence; represented as a full interval it
routes where it belongs — to the hypothesis channel, or to discriminating test
selection.

### Detection mechanism

**Type: CONSTRAINT — accepted.**

Cue lists with bounded scope, following the ConText line. Five axes are read
from the text around a resolved span.

| Axis | Values | Read from |
|---|---|---|
| Polarity | affirmed, negated | Cues before the span, and a smaller set after it |
| Certainty | factual, possible, hypothetical | Cues before the span |
| Experiencer | patient, other | Cues anywhere in the clause |
| Temporality | current, historical | Cues before the span |
| Source | objective, subjective, unspecified | Cues anywhere in the clause |

#### Scope and termination

Scope runs from the cue to the end of the clause or to an adversative
terminator — *but*, *however*, *although*, *except*. Without termination a single
cue negates the remainder of the sentence, which is the classic failure of naive
negation detection:

```
"The patient denies fever but reports cough."
   fever -> negated        (scope ends at "but")
   cough -> affirmed
```

Sentence boundaries also bound the scope, so a negation in one sentence does not
reach into the next.

#### Why source is a separate axis

Objectivity does not override subjectivity uniformly. A clinician can contradict
"I have no arrhythmia", because an ECG is observable. A clinician cannot
contradict "I have no pain", because pain is not observable and the person
experiencing it is the authoritative source. The same holds for nausea, vertigo,
the sensation of breathlessness, itch.

So the rule is not a hierarchy of speakers but a rule of domain: **on observable
signs the observation prevails; on experienced symptoms the experiencer
prevails.** The frequently cited example — patient denies palpitations, ECG shows
arrhythmia — is not a conflict at all: palpitations and arrhythmia are two
different findings, one symptom and one sign, and both readings are correct.

The axis feeds the interval directly: an objective negation is a documented
exclusion at `[0.00, 0.05]`, a reported denial is a weak negation at
`[0.00, 0.30]`. Both are "negated"; only one is strong.

#### Temporality

Temporality does not change whether something happened, it changes whether it is
current. An affirmed historical finding therefore keeps its polarity but has its
**upper bound capped** at 0.35, rather than being negated or dropped: the patient
did have it, and it is context rather than current state.

#### Cue sets are supplied, not built in

English and Italian sets ship with the module and either can be replaced
wholesale. The reference language for concept linking remains an open decision,
and the detector does not pre-empt it.

#### Auditability

Every fired cue is recorded on the result with its category, so a classification
can be explained by pointing at the words that produced it. This is the property
a scalar score cannot offer, and the reason the deterministic layer is the
baseline a model must beat rather than a placeholder for one.

### Model boundary

**Type: CONSTRAINT — accepted.**

```
model     ->  assertion category   (discrete label)
table     ->  category to interval (deterministic, versioned here)
```

A model emitting `0.7` cannot be audited: there is no defensible account of why
a sentence scores 0.7 rather than 0.6, and the same sentence may score
differently on another run. A model emitting `negated_by_patient` can be
audited — a reviewer confirms it against the sentence — and can be measured
against an annotated corpus.

Three consequences: the numbers stay defensible because a person wrote them in a
document; the model becomes falsifiable, since classification accuracy is
measurable while score correctness is not; and the calibration can change
without retraining.

### Category vocabulary

**Type: PLAN — accepted.** Review trigger: annotation of a local corpus.

Adopt the i2b2 2010 assertion categories — *Present, Absent, Possible,
Hypothetical, Conditional, Associated-with-someone-else* — plus *Historical*.
They carry annotated data, a benchmark and comparable models, replacing a
taxonomy that would need validating with one already validated.

**Conditional is unreliable.** Reported accuracy sits near 0.51–0.60 against
0.96 for *Present*, and the reference study excluded it from fine-tuning for
ambiguity with *Hypothetical*. Merge the two rather than acting on the
distinction.

### Implementation status

`memory/assertion.py` implements the deterministic layer: cue lists with bounded
scope and adversative termination, four i2b2 axes plus source, producing an
interval and a state rather than a scalar. English and Italian cue sets are
supplied and either can be replaced, so the corpus language stays open.

`reasoning/findings_boundary.py` is the single point at which patient findings
are assembled. Before it, two guards existed — one rejecting synthetic
hypotheses, one rejecting screening items — and **neither was invoked anywhere
on the production path**. A guard that is never called does not guard; the
isolation was a convention. `assemble` now admits only what is a current,
asserted finding of this patient, and every rejection names where the item
belongs instead:

| Rejected | Routes to |
|---|---|
| Negated | Documented exclusion |
| Hypothetical, "rule out" | Open question, discriminating test selection |
| Attributed to a relative | Family history channel |
| Historical | Clinical context, not current state |
| Synthetic hypothesis | Differential, as exclusion hypothesis only |
| Screening consideration | Screening list, outside the differential |

End-to-end behaviour on the sentences that previously produced false findings:

```
"The patient denies fever and reports no cough."
   admitted : []
   rejected : Fever (negated), Cough (negated)   -> documented_exclusion

"Family history of diabetes mellitus in the mother."
   rejected : Diabetes mellitus (other_experiencer) -> family_history_channel

"Chest CT ordered to rule out pneumonia."
   rejected : Pneumonia (hypothetical) -> open_question

"Presents with progressive dyspnea and bilateral pleural effusion."
   admitted : Dyspnea [Progressive], Pleural effusion [Bilateral]
```

### Sequencing

**Type: PLAN — accepted.** Review trigger: measured comparison.

Deterministic `ConText`-style rules first — negation, experiencer, temporality —
serving both as baseline and as a generator of training annotations. A model
afterwards, measured against that baseline; if it does not beat the rules, the
rules stay, and they are more defensible in a regulatory file. This bootstrap is
the documented strategy behind `NegBERT`: a high-precision rule system generates
additional training material for domain adaptation.

---

## 3. Modifiers

**Type: CONSTRAINT — accepted.**

### Decision

A term descending from `Clinical modifier` (HP:0012823) **never becomes a node**.
It attaches to the nearest finding as an attribute, within a clause-sized
window. A modifier with no finding to qualify collapses, and the discard is
reported rather than silent.

Implemented in `memory/concept_resolution.py`: `TermIndex.role_of` and
`attach_modifiers`.

### Rationale

`Bilateral` and `Progressive` are not findings. Admitted as concepts they attach
to thousands of conditions, so any path crossing them carries no information —
the same degeneracy an unknown edge produces, reached from the other direction.

The distinction is **already in the ontology**: HPO separates clinical modifiers
into their own branch of 357 terms. Deciding whether a term is a finding is
therefore a hierarchy lookup, deterministic and complete over the branch, with
no model involved.

### Why fusion is not the primary mechanism

Composing `Bilateral` + `Cataract` into a pre-coordinated term works only where
that term exists. Across the ontology there are **52** `Bilateral X` forms and
**58** `Progressive X` forms, against roughly 20,000 terms. Composing on the fly
instead yields a concept that is not a graph node and therefore not traversable
— elegant, isolated, and worse than the noise because it appears to work.

The finding keeps its own identifier and its traversability; the modifier travels
with it as an attribute. Fusion into an existing pre-coordinated term remains
available as an optional refinement.

Note: `HP:0000519` is *Developmental cataract*, not *Bilateral cataract*.

---

## 4. Family history

**Type: CONSTRAINT — accepted** for the routing; **PLAN** for the prior table.

### Decision

A condition reported in a relative **never enters patient findings**. It has two
destinations:

1. **Screening hypothesis** — the condition may be present and undiagnosed in
   the patient;
2. **Prior modifier** — it shifts the prior on heritable conditions.

Implemented in `reasoning/family_history.py`.

### Rationale: direction, not magnitude

Recording family history as a finding with a reduced score does not soften the
error, it disguises it. **Traversal is binary**: a finding is an entry point, so
the graph is walked *from* the relative's condition, generating paths toward its
complications in a patient who does not have it. Those paths are correct and the
premise is false, which is the hardest kind of error to notice — the provenance
is intact all the way down.

A score of 0.3 attenuates the number and not the traversal.

As a hypothesis the condition is a **destination**: a candidate to confirm or
exclude, producing no descendants until confirmed.

### Why the screening channel does not use the indeterminacy gate

The two hypothesis channels answer different questions.

| Channel | Question | Gate |
|---|---|---|
| Dream / mechanism enumeration | What else could explain these findings? | Indeterminacy: low convergence, high conflict, material risk |
| Family history screening | What else might this patient have, independent of the presenting complaint? | Inheritance, onset, prior assessment |

A screening consideration is valid precisely when the differential has settled.
Passing it through the indeterminacy gate would lose it in the clear cases —
which are the ones where a patient leaves without anyone having looked.

It is also reported separately from the differential. Placed inside it, the
consideration would imply it explains the presenting findings, which it usually
does not.

### The gate

Three conditions, each readable from the ontology or the record:

1. **Mode of inheritance** must support transmission. HPO encodes 40 terms under
   `Mode of inheritance` (HP:0000005). Sporadic is blocked.
2. **Onset must be reachable** at the patient's age. HPO encodes 22 terms under
   `Onset` (HP:0003674). Family history of an adult-onset condition in a
   paediatric patient is not yet actionable.
3. **Not already assessed.** Without this the same consideration returns at every
   visit, spending the review attention the channel exists to protect.

### Prior shift table

**Type: PLAN — accepted.** Review trigger: clinical review before any use beyond
research.

Intervals rather than points, because the strength of the shift is itself
uncertain.

| Mode of inheritance | Multiplier |
|---|---|
| Autosomal dominant | 2.5 – 6.0 |
| X-linked dominant, mitochondrial | 2.0 – 5.0 |
| Autosomal / X-linked recessive | 1.3 – 2.5 |
| Polygenic | 1.1 – 1.6 |
| Sporadic | 1.0 |
| Unknown | 1.0 – 1.5 |

Attenuated by degree of relatedness: first 1.0, second 0.5, third 0.25.
Attenuation moves the interval **toward 1.0** rather than scaling it, because a
distant relative weakens the inference toward neutrality and never reverses it.

**This table is a policy artefact, not derived truth.** It requires clinical
review, and its values are versioned here so a change is a documented decision
rather than a code edit.

---

## 5. Languages

**Type: CONSTRAINT — accepted.**

### The identifier is the pivot

Clinical text may arrive in English or Italian; the knowledge layer stays in one
language. `HP:0002240` is language-independent, so `Hepatomegaly` and
`Epatomegalia` resolve to the same term and traverse the same graph.

This is not only convenience. Ontologies, terminologies and the literature that
supplies likelihoods are predominantly English, so translating the graph would
mean maintaining **two knowledge bases** rather than two vocabularies. The split
is therefore:

| Layer | Language |
|---|---|
| Input — clinical text | Either |
| Knowledge — graph, edges, likelihoods | English, canonical |
| Output — presented to the clinician | Either, rendered from identifiers |

### The two languages are not symmetric

Measured on the published releases:

| | Surface forms | Curated |
|---|---|---|
| English (`hp.obo`) | 42,045 | ontology labels and EXACT synonyms |
| Italian (`hp-it.babelon.tsv`) | 3,488 terms | **524 official**, 2,964 machine-translated and marked preview |

Italian reaches roughly **15% of the index**, and about 85% of that is
unreviewed DeepL output. Building an Italian resolution path as though it were
equivalent to the English one would produce a recall ceiling far below what the
system appears to offer, and would do so silently.

### Translation status travels with the match

`match_kind` records how a surface form was obtained: ontology label, EXACT
synonym, official translation, or candidate translation. `is_verified_match`
exposes the distinction so a caller can require curated matches where a wrong
resolution costs more than none.

An unreviewed machine translation resolving correctly is common; resolving
incorrectly propagates into a path, a hypothesis and a provenance record that
all look well-formed. That asymmetry is the reason the provenance is preserved
rather than averaged away.

### Coverage is measured per language

`measure_language_coverage` reports translated terms, the official fraction, and
coverage against the index size. A resolution rate measured on English says
nothing about Italian, and reporting coverage turns "the system supports both"
into a statement with a number attached.

### Assertion cues are selected, not merged

Cue sets are chosen per document. Merging them looks convenient and is not: cues
collide across languages — Italian *ma* is a terminator and English *ma* is
nothing, Italian *non* is negation while English *non* appears inside words — so
a merged set fires on text it was not written for, and the failure is silent. An
unsupported language falls back to English rather than guessing.

### Open

- Italian coverage is the limiting factor. Raising it means either contributing
  reviewed translations upstream, or building a local curated lexicon for the
  finding categories that matter most.
- Document language is supplied by the caller. Automatic detection is not
  implemented, and would need its own error budget.

## 6. Vocabulary extension beyond phenotypes

**Type: CONSTRAINT — accepted.**

HPO covers phenotypes. Other categories need other sources, and none is invented:

| Category | Source |
|---|---|
| Laboratory tests and observations | LOINC |
| Drugs | ATC, RxNorm |
| Procedures | SNOMED CT, subject to licence |
| Environmental and occupational exposures | ECTO |
| Diseases | MONDO, Orphanet |
| Inheritance | HPO `Mode of inheritance` |

### Self-extension

The same asymmetry that governs graph edges governs the vocabulary, for the same
reason. A system that adds concepts to its own vocabulary and then uses them as
knowledge widens its notion of what exists every time it meets something it does
not know. After a few cycles the vocabulary holds concepts whose only source is
that the system coined them, with provenance formally traceable and
substantively circular.

Permitted form:

- the system **proposes** missing concepts freely, into a staging vocabulary;
- staging concepts are **not traversable** and generate no hypotheses;
- promotion requires an external source: a mapping to an existing terminology,
  or human confirmation;
- a surveillance metric reports the fraction of vocabulary of internal origin.

---

## 7. Open items

1. **Annotated corpus for validation.** i2b2 requires credentialing and is
   English. An Italian operating corpus needs local annotation, the largest cost
   in this line of work.
2. **Reference language for concept linking** — English with upstream
   normalisation, or Italian using HPO translations. The index is built from
   whichever OBO release is supplied, so the decision is deferred rather than
   pre-empted.
3. **Lexicon breadth** — which finding categories the first version must
   recognise. This parameter determines whether extraction is weeks or months.
4. **Assertion model local or API** — a lightweight few-shot classifier reaches
   roughly 0.93 and runs locally, respecting the PHI perimeter; a fine-tuned
   domain model reaches roughly 0.96 and requires domain training.
5. **Precision over recall.** A missed finding costs one hypothesis; a false
   finding costs a wrong hypothesis in the differential. The two errors are not
   symmetric, so calibration must favour precision — against the instinct that a
   42,000-form index invites using in full.

---

## 8. References

- i2b2 2010 assertion dataset — Present, Absent, Possible, Hypothetical,
  Conditional, Associated-with-someone-else
- Chapman et al. 2001 — NegEx
- Harkema et al. 2009 — ConText: negation, experiencer, temporal status
- Vincze et al. 2008 — BioScope, negation and speculation scope
- arXiv:2503.17425 — comparative assertion detection
- HPO v2026-06-23 — `hp.obo`, branches HP:0012823, HP:0000005, HP:0003674
