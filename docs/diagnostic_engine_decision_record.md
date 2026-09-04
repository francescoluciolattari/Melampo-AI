# Diagnostic Engine Decision Record

How the trained model, the concept graph and the deterministic calibrator divide
the work of producing a diagnosis. Companion to `rlm_on_memory_decision_record.md`,
`semantic_extraction_decision_record.md` and `hypothesis_stream_decision_record.md`,
using the same three-tier status convention.

---

## 1. What expertise research established

Four decades of controlled study of clinical reasoning yield results that
constrain the design, and three of them are counter-intuitive.

**Experts are fast because they know, not slow because they reason.** Speed and
accuracy correlate positively; physicians reach the correct diagnosis faster
than the incorrect one, within the same individual. Analytic reasoning cannot on
its own correct errors, partly because it engages on unfamiliar problems — where
knowledge is already insufficient. Making the system "reason more" is therefore
not the lever.

**Expert knowledge is organised into illness scripts, not facts.** A script has
three components: enabling conditions, the underlying fault, and the clinical
consequences the fault produces. The novice reasons from fault to consequences
slowly; the expert recognises the script from the consequences and descends to
the fault only when it fails to fit.

**The dominant error is premature closure, and it is cognitive rather than a
knowledge deficit.** Cognitive factors are involved in roughly 75% of diagnostic
errors, and most concern data collection, integration and verification rather
than missing knowledge. The same failure is documented in language models as
search satisficing.

**Debiasing strategies do not work; knowledge does.** Telling a clinician to
watch for anchoring does not make them less anchored. The design consequence is
direct: a prompt asking the model to consider alternatives is an exhortation and
does not work; a mechanism that enumerates alternatives regardless of the
model's inclination is a structure and does. This is the justification for the
hypothesis stream being structural rather than prompted.

**The universal model has three components, not two.** Fast recognition, slow
analysis, and a **calibrator** where both outputs meet and the final diagnosis is
produced. The calibrator is where humans are weakest: confidence correlates
poorly with accuracy.

---

## 2. Division of decisions

**Type: CONSTRAINT — accepted.**

| Component | Decides | Why there |
|---|---|---|
| **Trained model** | Which diagnoses, in what order | Pattern recognition over many learned cases is what a model does well and a rule cannot |
| **Concept graph** | What each element rests on: grounded, knowledge-mediated, unsupported | Provenance lives here and nowhere else |
| **Deterministic calibrator** | Whether the result may leave, whether a test is needed, whether to escalate | Metacognition is where the human fails through overconfidence; a rule does not overconfide |

The model **decides** — the phrasing "the model does not decide" used earlier was
wrong. What it may not do is decide *unaccountably*: every entry it places in the
differential must have a path in the graph, or it is a hypothesis to be examined
rather than a proposed diagnosis. The graph does not limit the model, it
classifies it.

Against the human original this architecture has one specific advantage: the
calibrator does not overconfide.

---

## 3. The illness script frame

**Type: CONSTRAINT — accepted.** Implemented in `reasoning/illness_script.py`.

The model emits a structured script rather than free text. This is the stage
that makes everything downstream possible: "probably heart failure, consider
pneumonia" contains no element that can be matched against a node or an edge,
while a script gives every element an address.

| Component | Content | Graph relation |
|---|---|---|
| Enabling conditions | Predisposing epidemiological and structural factors | `enables` |
| Fault | The underlying pathophysiological insult | `causes` |
| Consequences | Signs and symptoms the fault produces | `manifests_as` |
| Differential | Ranked candidates with their discriminating features | — |

`ScriptVerifier` classifies each element against the case findings and the
graph. It does **not** decide the diagnosis: it says what each part rests on.

Grounding uses the findings admitted by the boundary, never every mention: a
negated or hypothetical mention is not an observation and must ground nothing.

The verdict is three-way rather than a score, because a claim absent from the
case is not automatically wrong — most clinical inference connects findings to
conditions through knowledge outside the case. Only "no path in the graph" is a
defect.

---

## 4. Dream branch integration

**Type: CONSTRAINT — accepted.**

`merge_hypotheses` adds channel hypotheses to a script's differential, appended
after the model's own entries and carrying their origin. A synthetic alternative
is therefore never read as the model's reading of the case, and `leading`
ignores channel candidates when reporting what the model proposed.

Conditions the model already raised are skipped: a candidate under consideration
is no longer an alternative, and re-listing it spends the review attention the
channel is rationed to protect.

Merged hypotheses are verified like any other element. Being a hypothesis does
not exempt an entry from having to say what it rests on.

---

## 5. Model and training

**Type: PLAN — accepted.** Review trigger: measured baseline gap.

Four model roles, at different levels and not interchangeable:

| Role | Requirement | Choice |
|---|---|---|
| Root model (recursive retrieval) | Code generation, agentic loop. **Not medical** | Frontier API on synthetic, then self-hosted open weight |
| Sub-model (extraction) | Small, on-premise, sees PHI | MedGemma 1.5 4B |
| **Diagnostic model** | Broad medical knowledge, trainable | Meditron-based, fully open pipeline |
| Imaging specialist | Volumetric domain | Pillar-0 |

A medical fine-tune typically degrades code generation relative to its base, so
the diagnostic model cannot serve as the root model. The requirements are
opposed.

### Adaptation

DoRA rather than LoRA: decomposing magnitude and direction behaves better on
format tasks, which is the primary use here.

**One adapter to begin with, on the script format.** The medical knowledge is
already in the base model; what it must learn is to express that knowledge in
the enabling/fault/consequences structure. Report comprehension and differential
generation should first be attempted with prompting and structure.

Multiple adapters are easy to add later and hard to remove. Each is a separate
configuration item to validate, version and monitor for drift, and that cost
recurs at every cycle. A second adapter is added only when a measurement shows
one is insufficient.

### Functional areas

Visual and linguistic analysis use **different models** — Pillar-0 and the
diagnostic model — so "one adapter per area" is a category error: they are not
the same base. Adapter multiplicity is a question about one model and several
tasks.

**Reading must be blind to the report.** A radiology report is already a human
interpretation, so it is a third source alongside the image and the clinical
narrative, with an assertion status of its own. Disagreement between the visual
area and the report is a signal, not an error to suppress — but only if the
visual reading happened without access to the report. Analysing the image after
reading the report is not re-evaluation, it is confirmation, and it is anchoring:
the bias the literature identifies as dominant and shows cannot be corrected by
exhortation. Independence must be structural.

Two cautions on using the disagreement: the report may be right and the model
wrong, since the radiologist had the whole study, prior comparisons and clinical
context; and flagging too many disagreements consumes the scarcest resource,
reviewer attention. Explicit threshold, and phrasing as "for reconsideration",
never as "the report is wrong".

---

## 6. The learning cycle

**Type: CONSTRAINT — accepted** for independence; **PLAN** for cadence.

| Cadence | What changes | Requires revalidation |
|---|---|---|
| Daily | Graph interval recalibration from confirmed cases | No |
| Quarterly | Model adaptation on new literature and confirmed cases | Yes, under a change control plan |

The two cadences mirror how a clinician learns: yesterday's cases update
frequency estimates immediately, new literature restructures scripts more
slowly.

### Independence of confirmation

**Type: CONSTRAINT — accepted.** Implemented in
`governance/confirmation_registry.py`.

A confirmed case feeding the cycle is legitimate learning. A case "confirmed"
because the system proposed a diagnosis and nobody contradicted it is not: the
system would learn from its own proposals, growing more certain of what it
already believed with every cycle. The mechanism is automation bias, and it is
silent — each step looks correct and the drift is visible only in aggregate.

The safeguard is not "never learn from cases" but **the confirmation must have a
source other than the system's output**, recorded at registration rather than
reconstructed later. Whether a clinician independently reached the same
diagnosis or accepted the one on screen looks identical in the record unless it
was captured at the time.

Admitted sources: histopathology, clinical outcome, reference standard, and
independent review **only when the reviewer was blinded to the suggestion**. An
unrecorded blinding status is treated as unblinded rather than assumed
favourable, because reading the suggestion and agreeing is the failure mode, not
a weaker form of confirmation.

`independence_rate` is worth watching over time: a falling rate means the system
is increasingly being confirmed by agreement with itself.

---

## 7. Open items

1. Baseline gap measurement, to decide whether literature-stage adaptation is
   needed at all — the base model is already medically trained.
2. Case annotation in script format. Hundreds to low thousands of examples
   suffice for a format, and drafting from published case reports with clinical
   correction is far cheaper than writing from scratch. Same skill as the
   coverage reference set.
3. Corpus licences: PubMed abstracts for graph extraction, PMC OA full text for
   training. Publisher agreements are slow and many licences prohibit model
   training outright, which must be checked before negotiating.
4. Open access is unevenly distributed across disciplines, so a corpus drawn
   only from it inherits that skew. Measure coverage by clinical area rather
   than assuming it.
5. Predetermined change control plan for the quarterly cycle.

---

## 8. References

Croskerry P., *Clinical cognition and diagnostic error: applications of a dual
process model of reasoning*, Adv Health Sci Educ 2009 · Norman G.R. et al., *The
causes of errors in clinical reasoning*, Acad Med 2017 · Pelaccia T., Sherbino
J., Norman G., *Dual process models of clinical reasoning: the central role of
knowledge in diagnostic expertise*, J Eval Clin Pract 2024 · Schmidt H.G.,
Norman G.R., Boshuizen H.P.A., illness script theory, 1990 · Staal J. et al.,
BMC Med Educ 2021 · Friedman C.P. et al., *Do physicians know when their
diagnoses are correct?*, J Gen Intern Med 2005 · Nendaz M., Perrier A., Swiss
Med Wkly 2012
