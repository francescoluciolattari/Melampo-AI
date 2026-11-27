# Compliance and risk

*English · [Italiano](../it/COMPLIANCE_RISK.md) · as of 25 November 2026*

Melampo is research software today, but it is built as if it will become a medical device: MDR class IIa,
IEC 62304 class C, a high-risk AI system under the AI Act. Twenty of the twenty-one recorded hazards have a
control verified by tests; one is open.

## 1. Classification

| Framework | Class | Why |
|---|---|---|
| MDR (EU 2017/745), rule 11 | **IIa** at least | software that gives information used to take diagnostic decisions; IIb if a wrong output could cause a serious deterioration of health |
| IEC 62304 | **C**, provisional | a wrong or missed finding can contribute to serious injury. Class B would need the physician's review to be a risk control external to the software, documented and verified; until the review mask (built on 27 November 2026) is validated, the project takes the stricter class |
| AI Act (EU 2024/1689) | **high-risk**, through Annex I (medical devices of class IIa and above) | the obligations for AI in medical devices apply from 2 August 2028 after Regulation (EU) 2026/1744, according to a secondary source: to verify on EUR-Lex |
| GDPR, art. 9 | health data | pseudonymisation at source, local processing, encrypted stores (ARCHITECTURE §5) |

What class C asks beyond B, and how the project meets it: detailed design of each software unit (module
docstrings and ARCHITECTURE), unit verification (pytest, ~200 files), and acceptance criteria for each unit
written before the measure (VALIDATION §1).

## 2. Hazards, controls, evidence

The register is [`docs/registers/risks.yml`](../registers/risks.yml); `scripts/docs_check.py check` fails when a
control names a test that does not exist. Severity and probability are first estimates (1–5), to be reviewed
with a physician.

| Id | Hazard | Control | Status |
|---|---|---|---|
| R01 | finding attached to the wrong structure | reader and models must agree for a certain finding; otherwise in doubt with every reading; a person only on a discordance of the document; parts read with UBERON | partial: 1 of 24 attributed silently on the last valid fresh sample |
| R02 | right finding removed | neither reader nor models alone set a finding aside | controlled: 0 removed on fresh samples |
| R03 | model states what the sentence does not say | claims must be copied words; other marked words invalidate the answer | controlled |
| R04 | absent or negated finding read as present | assertion detection; negated observations moved to normal | controlled |
| R05 | past fact read as current | history attached as history | controlled |
| R06 | patient's report sent outside the hospital | router refuses external models; the one procedure of reading reads a patient's text only with interpreters on this machine; probes need public corpora | controlled |
| R07 | text not pseudonymised enters review or training | review records refuse it | controlled |
| R08 | learning from unconfirmed labels | clinical basis required; second person confirms reviews | controlled |
| R09 | another patient's records merged | exact identifiers; unrelated records never swept in | controlled |
| R10 | hypothesis promoted without confirmation | minimum confirmations; promotion asks a person | controlled |
| R11 | memory retriever reports evidence from an empty memory | — | **open**: to fix with an explicit abstention |
| R12 | exam and sentence name different sides | a laterality question to a person | controlled |
| R13 | figures measured on tuned data | frozen splits, fresh samples, lost-answer rule | controlled |
| R14 | model does not answer | a lost answer is "cannot say" and goes to a person | controlled |
| R15 | a normal or physiological state read as a finding | the models compare each aspect with the usual one; the reader checks with HPO and PATO | partial: 4 of 169 in the claims2 run, before the control |
| R16 | a measure judged against a remembered value | only a cited reference row, recomputed by the reader | partial: 21 of 29 rows verified; the one procedure does not yet compare a card's measure with the rows |
| R17 | the referral's question read as a current finding | the sections of the report, the conclusion and the sentence before are read; a contradiction goes to a person | partial: 4 of 99 in claims4, before the control |
| R18 | the one procedure reads the patient's documents before it is measured | a card not agreed is a finding stated with uncertainty; readings paired only in one sentence and by every interpreter; the reader before kept to measure the change | partial: the bench of the one procedure is to run |
| R19 | a code given before the readings are compared | no code on the card; only the agreed reading is coded, by the choice of every interpreter | controlled |
| R20 | a patient's sentence enters the shared knowledge | only publications write the thesaurus; a patient's text only on this machine | controlled |
| R21 | a weight of a profile from a misread number | one value per measure; weighs only when both interpreters read it; the judges verify numbers and direction | controlled |
| R22 | a document of another patient, or no medical document, in the case | the identifiers each document writes compared with the patient's, on this machine; another patient's document kept out and its question to a person; a document of no recognised kind to a person | partial: a document without identifiers cannot be checked (it is used unless a person says no in the review mask); the mask is not validated |

## 3. Software of unknown provenance and external models

- **Python packages:** [`docs/registers/soup.md`](../registers/soup.md), generated from `uv.lock` (41 direct
  dependencies, 207 packages in the lock). The CI fails when it is not current.
- **Models, ontologies, corpora:** [`melampo-assets.yaml`](../../melampo-assets.yaml), with licence, pinned
  revision and status (`blocked_licence` where use is not allowed).
- **Licences that constrain use:** PARROT and VoxTell weights are CC BY-NC-SA 4.0 (non-commercial, share-alike);
  VL-JEPA weights are research-only (FAIR licence); NVIDIA models are bench-only until a licence review; the
  Gemma terms for this medical use are to verify; Muse Spark Open asks for the notice "Built with Muse Spark Open"
  and its terms reach the models derived from it (LoRA adapters included), and its user must not be a minor
  (ARCHITECTURE, "Local readers"). A commercial version would need each of these replaced.
- **Melampo's own licence:** GNU AGPLv3 with attribution terms under section 7(b) (see `LICENSE` and the root
  README). A decision record of 2 October 2026 recorded CC BY-NC 4.0 and the legal points of that change; which
  licence holds is for the legal review (open item).

## 4. Where each requirement is met

| Requirement | Document |
|---|---|
| Intended use, users, contraindications (MDR Annex II §1) | [README](README.md) |
| Design, architecture, data flow, human oversight (MDR Annex II §3; AI Act Annex IV §2–3) | [ARCHITECTURE](ARCHITECTURE.md) |
| Training, validation and test data; metrics; verification (MDR Annex II §6; AI Act Annex IV §2) | [VALIDATION](VALIDATION.md) |
| Risk management (ISO 14971); SOUP (IEC 62304 §5.3, §7.1) | this file, the registers |
| Change control, problem resolution, configuration (IEC 62304 §6, §8, §9; AI Act Annex IV §7) | [CONTRIBUTING](CONTRIBUTING.md) |
| Vulnerability reporting (MDR Annex I §17.2, §17.4) | [SECURITY](../SECURITY.md) |
| Post-market surveillance; clinical evaluation | not started: belongs to the certification phase |

## 5. Plan of predetermined changes

Declared on 27 November 2026 (the owner's decision), so that a model trained again is a change foreseen and assessed
with the device, not a new device (AI Act Art. 43(4); MDCG 2025-6, section on substantial modification; the
proposal COM(2025) 1023 would give such plans a basis in the MDR). Nothing has been trained yet.

**What may change.** (a) The reading skill of each interpreter of the one procedure, through an adapter (LoRA) of
that model; (b) the weights and the calibration of the ranking of the diagnoses. The knowledge (synonyms,
equivalences, the concepts the interpreters agreed on) is not trained: it stays in the thesaurus, for every model.

**From which data.** (a) The reading confirmations (`review_feedback.ReadingConfirmation`): the doubt of a card
answered in the review mask with a closed answer, confirmed by a second person, and adjudicated by a third when the
answer is none of the interpreters' readings; the sentence kept as written and de-identified (`data/deidentify.py`:
names, codes, contacts, addresses, places, dates moved by one fixed number of days per case, ages above 89), and
admitted to training only when both interpreters read the de-identified sentence to the confirmed card. (b) The
confirmed diagnoses admitted by `governance/confirmation_registry.py`: a confirmation from a source other than the
system (histology, outcome, independent review); a diagnosis missed because a term was misread goes to (a), a weight
that was wrong to (b).

**How.** One archive, one procedure, one adapter per model: the same rows (the one directive on the de-identified
sentence, the confirmed card as answer) train every model with the same recipe (rank, steps, seed), on a computer
of the hospital; the adapter never leaves it. Methods that move an adapter from one model to another (Trans-LoRA,
Cross-LoRA, CAST, PorTAL) may only start a training: today they are shown between similar architectures, not
between a hybrid Mamba MoE and a dense transformer, nor on clinical tasks, and they keep 85-95% of an adapter
trained on the model at best. **To check again by March 2027: PorTAL (portable task adaptation, July 2026) and
CAST (activation manifold projection), the closest to one training for every model.** The diagnoses are updated by
recalibration first; refitting only when recalibration does not reach the same performance.

**Validation before use.** One case in five (fixed by its key) only measures and never trains. An adapter or a
recalibration is accepted only when, on that part, the errors stay under 2% with no new error against the version
in use, the doubts are counted, and (for an adapter) the model does not return the sentences it was trained on
(memorisation test). Each accepted version is recorded with the fingerprints of its data, its base model and its
recipe; a version that fails is not used, and the version in use stays.

## 6. Open items

1. Fix R11 (the memory retriever abstains explicitly).
2. Measure the one procedure of reading on the benches of the reports (R18), with criteria fixed before the run.
3. The radiologists' gold set: labels, review of the one-labeller labels.
4. Validate the review mask (built on 27 November 2026), which turns the physician's review into a verified external risk control.
5. Legal review of Melampo's licence and of the attribution terms.
6. Verify the AI Act dates on EUR-Lex; decide when a quality management system (ISO 13485) starts.
7. By March 2027: check again the methods that move an adapter between models (PorTAL, CAST), section 5.
