# Compliance and risk

*English · [Italiano](../it/COMPLIANCE_RISK.md) · as of 4 November 2026*

Melampo is research software today, but it is built as if it will become a medical device: MDR class IIa,
IEC 62304 class C, a high-risk AI system under the AI Act. Fifteen of the sixteen recorded hazards have a
control verified by tests; one is open.

## 1. Classification

| Framework | Class | Why |
|---|---|---|
| MDR (EU 2017/745), rule 11 | **IIa** at least | software that gives information used to take diagnostic decisions; IIb if a wrong output could cause a serious deterioration of health |
| IEC 62304 | **C**, provisional | a wrong or missed finding can contribute to serious injury. Class B would need the physician's review to be a risk control external to the software, documented and verified; until the review mask exists and is validated, the project takes the stricter class |
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
| R01 | finding attached to the wrong structure | reader and models must agree; a person on contradiction; parts read with UBERON | partial: 1 of 24 attributed silently on the last valid fresh sample |
| R02 | right finding removed | neither reader nor models alone set a finding aside | controlled: 0 removed on fresh samples |
| R03 | model states what the sentence does not say | claims must be copied words; other marked words invalidate the answer | controlled |
| R04 | absent or negated finding read as present | assertion detection; negated observations moved to normal | controlled |
| R05 | past fact read as current | history attached as history | controlled |
| R06 | patient's report sent outside the hospital | router refuses external models; probes need public corpora | controlled |
| R07 | text not pseudonymised enters review or training | review records refuse it | controlled |
| R08 | learning from unconfirmed labels | clinical basis required; second person confirms reviews | controlled |
| R09 | another patient's records merged | exact identifiers; unrelated records never swept in | controlled |
| R10 | hypothesis promoted without confirmation | minimum confirmations; promotion asks a person | controlled |
| R11 | memory retriever reports evidence from an empty memory | — | **open**: to fix with an explicit abstention |
| R12 | exam and sentence name different sides | a laterality question to a person | controlled |
| R13 | figures measured on tuned data | frozen splits, fresh samples, lost-answer rule | controlled |
| R14 | model does not answer | a lost answer is "cannot say" and goes to a person | controlled |
| R15 | a normal or physiological state read as a finding | the models compare each aspect with the usual one; the reader checks with HPO and PATO | partial: 4 of 169 in the claims2 run, before the control |
| R16 | a measure judged against a remembered value | only a cited reference row, recomputed by the reader | partial: the rows are to verify |

## 3. Software of unknown provenance and external models

- **Python packages:** [`docs/registers/soup.md`](../registers/soup.md), generated from `uv.lock` (41 direct
  dependencies, 207 packages in the lock). The CI fails when it is not current.
- **Models, ontologies, corpora:** [`melampo-assets.yaml`](../../melampo-assets.yaml), with licence, pinned
  revision and status (`blocked_licence` where use is not allowed).
- **Licences that constrain use:** PARROT and VoxTell weights are CC BY-NC-SA 4.0 (non-commercial, share-alike);
  VL-JEPA weights are research-only (FAIR licence); NVIDIA models are bench-only until a licence review; the
  Gemma terms for this medical use are to verify. A commercial version would need each of these replaced.
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

## 5. Open items

1. Fix R11 (the memory retriever abstains explicitly).
2. The radiologists' gold set: labels, review of the one-labeller labels.
3. The input and review mask, which turns the physician's review into a verified external risk control.
4. Legal review of Melampo's licence and of the attribution terms.
5. Verify the AI Act dates on EUR-Lex; decide when a quality management system (ISO 13485) starts.
