# Melampo — what it is and what it is for

*English · [Italiano](../it/README.md) · as of 2 November 2026*

Melampo reads a clinical case (the physician's note, reports, laboratory tables, images) and proposes a
structured differential diagnosis, with the evidence for and against each hypothesis and an explicit
"I cannot say now" when the evidence does not hold and requires more investigation. It is a **research
system**: it is not a medical device and gives no diagnosis to patients.

## Intended use

| | |
|---|---|
| **What it does** | Reads Italian and English clinical documents; links what they say to anatomy (TotalSegmentator classes anchored to UBERON) and to phenotypes (HPO); builds a case graph; ranks hypotheses with their evidence; sends to a person every reading it cannot confirm. |
| **For whom** | Researchers and physicians working on the project, on public corpora and on pseudonymised cases under the project's data rules. |
| **Not for** | Diagnosis or treatment of a patient; use without a physician who reads every output; reports of patients sent to services outside the hospital. |
| **Regulatory status** | Research software, not CE marked. Designed toward MDR class IIa and IEC 62304 class C (see [COMPLIANCE_RISK](COMPLIANCE_RISK.md)). |

## Where things stand

Verified on the code of `main`; the full map is in [ARCHITECTURE](ARCHITECTURE.md).

| Part | State | Note |
|---|---|---|
| Ingestion of documents (PDF, DOCX, images, DICOM) and report router | working | the model says only the kind of document; deterministic cues decide with it |
| Anatomical linker with the Kintsch reader and local models | working | patients' reports are read only by local models |
| "Whose is the finding?" (finding owner) | working, question v1 | 1 of 24 findings of another structure attributed silently; 0 of 184 right findings removed |
| Claim-by-claim reading | built, **off** | first runs did not pass their criteria; second sample ready |
| From finding to HPO term | in probes only | 90% safe against a 95% threshold: not in the graph |
| Concept graph (FalkorDB) and nexus branch | working | 636,733 verified edges |
| Case confirmation and merge | working | a confirmation without a clinical basis is kept out of training |
| Memory retriever | **known defect** | with an empty memory it invents three items of evidence |
| Training (DPO, LoRA) | never run | no GPU, no confirmed labels yet |
| Physician's input and review mask | to build | the questions it will show are ready |

## Repository map

```text
src/melampo/        the system
  data/             ingestion, report router, report reading
  memory/           linker, Kintsch reader, finding owner, claims, ontologies
  imaging/          image interpreter, segmenters, findings
  reasoning/        pipeline, functional areas, intuition, differential, orchestrator
  orchestration/    model router, specialist runtime
  training/         confirmation stores, preference pairs, nexus scheduler
  evaluation/       benches, gold set, LLM reading, review feedback
scripts/            probes and benches (run by the workflows)
tests/              ~200 test files (pytest)
data/               ontologies and references (DVC pins), gold labels (data/gold/)
.github/workflows/  CI and the probes, all on manual trigger except CI
docs/en, docs/it    this documentation, in English and Italian
docs/registers/     risk register and software of unknown provenance (SOUP)
```

## Quick start

```bash
uv sync --extra dev --extra data      # the exact environment in uv.lock (Python 3.12+)
uv run dvc update -R data             # reference data, verified against its pins
uv run pytest -q                      # the whole suite
```

Tests that need FalkorDB, DICOM samples or a network fail without them; CI provides what they need.

## The documentation

| File | Answers |
|---|---|
| [README](README.md) | what it is, for whom, where it stands |
| [ARCHITECTURE](ARCHITECTURE.md) | how it is made and where the data go |
| [VALIDATION](VALIDATION.md) | how we know it works: data, gold sets, probes, criteria, results |
| [COMPLIANCE_RISK](COMPLIANCE_RISK.md) | what can go wrong, how it is stopped, which rules apply |
| [CONTRIBUTING](CONTRIBUTING.md) | how to change it without breaking anything |
| [RADIOLOGIST_GUIDE](RADIOLOGIST_GUIDE.md) | how radiologists label the gold set |

The documents before 2 November 2026 (the research diary, decision records, treatise) are kept unchanged in
the git tag `docs-archive-2026-11`: `git show docs-archive-2026-11:docs/<file>`.
