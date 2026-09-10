# Semantic Comparison and Image Hypothesis Generation — Research Report

**Status: research report, not a decision record.** Nothing here has been
implemented. It exists to decide what to build, and it says plainly where the
literature supports a proposal, where it contradicts one, and where it is
silent.

---

## 0. A correction to something stated earlier in this work

`find_paths` was described in conversation as "querying the graph" as though
it returned a yes/no. That was imprecise, and the imprecision matters for
everything below.

Reading `memory/concept_paths.py`: every `ConceptPath` already carries a
**degree**, three ways.

| Property | What it is |
|---|---|
| `strength` | Product of edge weights along the chain. Long chains of weak links score low, as they should. |
| `strength_lower` | What the path *guarantees* — the conservative projection |
| `strength_upper` | What it *could* reach — the optimistic projection |
| `hops` | Path length, available separately |

The interval, not the scalar, is the project's existing answer to "how
strongly are these two concepts connected". Anything built below should read
that interval rather than inventing a parallel score.

---

## 1. The proposal has a name in the literature: spreading activation

The description offered — take the significant terms of a sentence, find
synonyms, build connections, follow the strongest paths as in neural
pathways, select positively and negatively — is **spreading activation**, a
technique with a continuous research literature since Crestani's 1997 survey
in *Artificial Intelligence Review*, and with a medical-ontology
implementation (the ONTOSPREAD framework) applied specifically to RDF graphs
and ontologies.

This is good news in a specific way: it means the design does not have to be
invented from first principles, and three refinements already established in
that literature can be adopted rather than rediscovered.

### 1.1 Refinement already present in this codebase

**Weight degradation with distance.** ONTOSPREAD implements activation decay
proportional to distance from the origin. `ConceptPath.strength` — the
product of edge weights — already does exactly this: each additional hop
multiplies by a number ≤ 1, so a five-hop chain of moderate links scores
below a two-hop chain of strong ones. Nothing to build.

### 1.2 Refinement not present, and directly useful

**Converging paths reward.** ONTOSPREAD explicitly rewards concepts reached
by *multiple independent paths*, above what any single path would score.
This is the graph-level analogue of the cross-check's own principle: two
independent routes to the same conclusion are worth more than one, and worth
more than their individual strengths suggest.

`find_paths` currently returns paths; nothing aggregates *how many distinct
paths* connect two concepts, or rewards convergence. This is a small,
well-defined addition with a clear rationale, and it addresses the exact
question a relevance frame asks: is this link real, or is it one thin thread?

### 1.3 Refinement that is a documented failure mode

**Constrained spreading, not classical spreading.** The literature is
explicit that classical spreading activation *uses all relations of a node,
which adds unsuitable information*. A 2018 paper measuring
query-oriented-constrained spreading against unconstrained reports
improvements of 18.9% and 43.8% in MAP over syntactic and unconstrained
semantic search respectively.

The practical consequence: spreading from "family history" through *every*
HPO relation would surface a large set of weakly-related concepts and drown
the signal. The spread has to be constrained by relation type from the
start — which the project's own architecture already anticipates, since
`ConceptEdge` carries typed relations and epistemic states rather than bare
weights.

---

## 2. The finding that explains an earlier empirical observation

Earlier in this work, comparing answers by character similarity was found to
rank a contradiction above a paraphrase: "pulmonary embolism" vs "pulmonary
oedema" scored 0.706 while "40 mg daily" vs "prednisone 40 mg daily" scored
0.667. The diagnosis offered at the time was informal — that "pulmonary"
discriminates nothing while "embolism" versus "oedema" discriminates
everything.

**That intuition has a formal name and a measured result behind it.**
Information Content — a concept's specificity, formally `-log(p(concept))`
— is precisely the quantity that distinguishes an uninformative shared
prefix from a discriminating term. And the comparison is settled empirically:
evaluating semantic similarity measures on the MSH-WSD disambiguation
dataset, **information-content-based measures achieve higher accuracy than
path-based measures**, including the classical shortest-path and
Leacock-Chodorow measures.

This is directly consequential. `find_paths` is a path-based measure. If it
is used as the sole basis for judging whether two concepts are connected, it
inherits a limitation the literature has already measured. Weighting by
Information Content — rare, specific concepts counting for more than common,
general ones — is the documented improvement, and it is computable from the
HPO annotation frequencies the project has already imported (285,598 arcs,
with published frequency ranges preserved as intervals).

**One qualification worth stating:** these evaluations are on UMLS and MeSH,
not HPO, and on word-sense disambiguation rather than answer comparison. The
direction of the finding is well-supported; the exact magnitude on this
project's graph and task is not established and would need measuring.

---

## 3. Where the literature is more cautious than the proposal

### 3.1 What is established

Approaches that combine graph structure *with* corpus-derived embeddings
outperform either alone: a 2020 JAMIA study generating UMLS concept
embeddings from BioWordVec and BERT, combined with graph convolutional
embeddings over UMLS hierarchical relations, compared favourably against
path-based baselines. Synonym expansion from a medical vocabulary — part of
the proposal — is standard practice and has its own literature (query
expansion with medical ontologies).

### 3.2 What reintroduces a problem this project deliberately avoided

A 2026 framework, UMEval, does something close to what a "comprehension"
step would need: it retrieves and enriches knowledge from UMLS, applies a
noise-aware selection strategy to quantify uncertainty in candidate
definitions and semantic paths, then **uses an LLM to generate similarity
scores with natural-language explanations, verified by a supervisor for
factual alignment**.

The noise-aware selection over semantic paths is a genuinely useful idea and
worth borrowing. The LLM scoring is not: `root_model_cross_check` exists
specifically because no third model should adjudicate between two models'
answers. Adopting an LLM-based similarity judge would reintroduce exactly
the arbitration the cross-check was built to avoid — and would do so at the
point where the system is deciding whether to trust its own two-model
agreement, which is the worst possible place for an unverifiable judgement.

**Recommendation:** take the graph-and-IC path, which is deterministic and
inspectable. Leave LLM-based semantic scoring out.

---

## 4. Image comparison: the literature does not support the proposal as stated

The proposal — morphing between imaging findings *across different reports*
to create hypothetical connections — needs to be separated from what the
literature actually establishes, because the two are different in a way that
matters clinically.

### 4.1 Morphing in medical imaging is real, established, and used within one patient

Image morphing has a legitimate, published role: generating intermediate
Digitally Reconstructed Radiographs for conformal radiotherapy positioning,
and interpolating between breathing phases in 4DCT. In a published
evaluation against real 4DCT data, morphed images agreed closely — fewer
than 2% of voxels misclassified as belonging or not belonging to a lung
section.

**Every one of these applications interpolates between two states of the
same patient's anatomy**, where a true intermediate state physically existed
and the interpolation is estimating it. That is a well-posed problem.

### 4.2 Interpolation produces artifacts that are documented and common

The same literature documents the cost. Interpolation artifacts in 4DCT are
a recognised category; phase-binning artifacts occur in **up to 90% of
scans**, most often near end-inspiration. In cardiac MRI, sub-pixel shifts
change the appearance of Gibbs artifacts enough that the artifact's
appearance varies between patients and between frames — and which
interpolation method is used changes how severe this is.

### 4.3 The specific hazard for hypothesis generation

Radiological practice distinguishes artifact from pathology by a principle
stated directly in the AJR literature: **"Inconsistent visualization of an
imaging finding is clearly suggestive of artifact"** — a finding that does
not persist across images or sequences is suspect.

A morphed image constructed between *two different patients' reports* has
no ground truth at all: the intermediate anatomy it depicts never existed in
any patient. Any structure appearing in it that is not in either source is,
by construction, an artifact — and it would be an artifact that *does*
persist consistently within the generated image, defeating the very
heuristic radiologists use to catch artifacts. A generated hypothesis
grounded in such a feature would be grounded in nothing, while looking
exactly like a finding.

A *Radiology* editorial on synthetic images states the general concern
directly: the biggest risk in these technologies is missed or delayed
diagnosis, and GAN processing **could potentially eliminate key image
features needed for accurate diagnoses**.

### 4.4 What the literature does support, and it is adjacent but different

Interpolation along a *disease axis* within a generative model is
established — composable diffusion generating chest radiographs at
intermediate severity ("0.25 × healthy AND 0.75 × cardiomegaly"). Note what
this is for: **generating training data with controlled properties**, not
generating diagnostic hypotheses about a specific patient.

**Assessment.** Cross-patient morphing for hypothesis generation is not
supported by the literature reviewed and carries a specific, documented
hazard. The underlying goal — finding non-obvious connections between
imaging findings across cases — is legitimate and worth pursuing, but the
mechanism should operate on **extracted findings in the concept graph**,
where a spurious connection is inspectable and can be scored, rather than on
**generated pixels**, where a spurious feature is indistinguishable from a
real one. This keeps the image path aligned with the architecture the project
already committed to: the blind imaging read produces findings, and findings
enter the graph, where the same spreading-activation machinery described in
section 1 applies to them as to any other concept.

---

## 5. Proposed sequence

Ordered by evidence strength and by what unblocks what.

**Implemented (this branch): steps one and two.**
`memory/information_content.py` provides `InformationContentTable`
(`from_frequencies` for real counts, `from_graph_structure` for intrinsic IC
where none exist, `basis` recorded on every score so a measurement is
distinguishable from a default), `score_path`/`rank_paths` carrying both raw
and weighted strength, and `score_convergence` implementing the sub-additive
converging-paths reward. Verified on the Marfan example from the discussion:
"pulmonary" scores IC 0.018 against "marfan syndrome" at 1.000 -- the formal
version of the empirical observation in section 2 -- and two independent
routes lift the connection from 0.604 to 0.905. Step three (constrained
spreading activation) is now also implemented:
`memory/spreading_activation.py` reuses `illness_script.SCRIPT_RELATIONS` as
the default allowed-relation vocabulary (plus HPO's own `has_phenotype`)
rather than inventing a parallel one, applies decay and a threshold to bound
the frontier, and applies Information Content at the destination rather than
along the way, since specificity says how much a concept's activation means,
not how well activation travels through it. `mediating_concepts` spreads
from both ends of a relevance pair and reports what both reach -- verified on
the same Marfan fixture: "connective tissue weakness" is reached from both
origins with weighted activation 0.489, while "pulmonary" is also reached
from both but suppressed to 0.004 by its near-zero IC, exactly the
distinction the whole prior investigation in this document was chasing. The
image work remains as described below.

**First — Information Content weighting on graph paths.** The
best-supported single change here: it has a measured result behind it, it
formalises an intuition this project already arrived at empirically, and it
is computable from HPO frequency data already imported. Scope: an IC score
per concept, used to weight path strength so that a path through
highly-specific concepts outranks one through general ones.

**Second — converging-path reward in `find_paths`.** Small, well-defined,
and directly answers the question a relevance frame asks. Aggregate distinct
paths between two concepts and reward convergence, rather than reporting the
single best path.

**Third — constrained spreading activation for the relevance frame.** Only
after the first two: unconstrained spreading is a documented failure mode,
and the constraint has to be by relation type, which means the edge
typing has to be doing real work first.

**Not now — LLM-based semantic scoring.** Reintroduces adjudication the
cross-check exists to avoid.

**Not as proposed — cross-patient image morphing.** Reconsider as
concept-graph connection between *extracted* imaging findings, where the
existing epistemic machinery applies and a spurious link is inspectable.

---

## 6. Sources

- Crestani, F. (1997). *Application of spreading activation techniques in
  information retrieval.* Artificial Intelligence Review 11(6).
- ONTOSPREAD framework — spreading activation over medical ontologies, with
  distance-based weight degradation and converging-paths reward.
- Query-oriented constrained spreading activation (arXiv 1808.01968) —
  18.9%/43.8% MAP improvement over unconstrained.
- McInnes, Pedersen, Pakhomov (2009). *UMLS-Interface and UMLS-Similarity* —
  path-based measures, open source.
- Evaluating semantic similarity for biomedical WSD (J Biomed Inform, 2013) —
  IC-based measures outperform path-based on MSH-WSD.
- Mao & Fung (2020), JAMIA 27(10) — word and graph embedding for UMLS
  semantic relatedness.
- UMEval (2026) — LLM-augmented medical term semantic evaluation with
  noise-aware path selection.
- Morphing-based interpolation for conformal radiotherapy — <2% voxel
  misclassification vs 4DCT.
- Phase-binning and interpolation artifact detection in 4DCT — artifacts in
  up to 90% of scans.
- Dark rim / Gibbs artifact variability from sub-pixel shifts (JCMR).
- *Synthetic Images Are Here to Stay*, Radiology editorial — risk of
  eliminating diagnostically necessary features.
- Medical diffusion on a budget (arXiv 2303.13430) — healthy/diseased
  interpolation via composable diffusion.
- Application of basic physics principles to clinical neuroradiology (AJR) —
  inconsistent visualisation as the artifact heuristic.
