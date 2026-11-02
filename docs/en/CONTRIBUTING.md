# Contributing: how to change Melampo without breaking anything

*English · [Italiano](../it/CONTRIBUTING.md) · as of 2 November 2026*

Every change goes through a pull request with its tests; a change to how the system reads or decides also
needs a probe on a fresh sample, with criteria written before the run.

## 1. What no change may do

- claim clinical validity without a measure in [VALIDATION](VALIDATION.md);
- send patients' texts to a service outside the hospital, or add a network call that is not configured
  explicitly;
- let one reader alone, or the models alone, remove or promote a finding, a hypothesis or a label;
- bypass abstention, the questions for a person, provenance or the audit trace;
- add a deterministic rule that decides a case: the reader checks with ontologies and grammar, the models read,
  a person decides on contradiction;
- tune a component on the sample that measures it;
- commit patients' data, texts of non-commercial corpora, tokens or passwords.

## 2. Branches, pull requests, commits

| Step | Rule |
|---|---|
| Branch | from `main`, named `feat/…`, `fix/…`, `docs/…`; one subject per branch |
| Commit message | first line: what changed; body: why, and the criterion or measure that supports it. It is the audit trail: say who decided when a decision was the owner's |
| Pull request | description with what changes, the tests, and for a reading change the probe and its criteria; merged by the owner after CI passes |
| Sessions without push rights | deliver a git bundle; the owner applies it (`git fetch <bundle> HEAD`, `git cherry-pick`) and pushes |
| Tags | `docs-archive-2026-11` keeps the documentation before 2 November 2026 |

## 3. Tests and CI

```bash
uv run pytest -q                              # the whole suite
uv run pytest -q tests/test_claims_reading.py # one area
uv run python scripts/docs_check.py check     # registers, links, the two languages
uv run python scripts/docs_check.py soup      # after changing dependencies
```

`ci.yml` runs the suite and checks the DVC pins of the reference data on every push and pull request;
`docs-check.yml` checks the documentation. Every other workflow is started by hand or on a schedule, because it
calls billed models or downloads large data.

## 4. Adding or changing a reading

1. Write the change and its unit tests.
2. Draw a fresh sample (`scripts/owner_probe.py sample --seed <new seed>`): it excludes every report of the
   earlier samples.
3. Label it **before** any reader or model runs on it; mark doubtful labels with "?".
4. Write the criteria in VALIDATION §5 and in the pull request.
5. Keep the change off in ingestion (an environment variable) until the probe passes.
6. Run the workflow; a run that loses more than 5% of the answers does not count.
7. Write the result in VALIDATION §4, passed or not. If not, the change stays off.

## 5. Workflows

| Workflow | Does | Trigger |
|---|---|---|
| `ci` | tests, DVC pins | push, pull request |
| `docs-check` | registers, links, English and Italian in step | push, pull request |
| `owner-probe` | whose is the finding: reader, models, claims (inputs `sets`, `arms`, `models`) | manual |
| `role-probe` | the role of a word: reader and chat models | manual |
| `hpo-probe` | from a finding to an HPO term | manual |
| `external-check` | the linker on CRAFT and MedMentions | manual |
| `public-reports` | public report corpora for the gold-set sample | manual |
| `ci-probe`, `lattice-probe`, `phrase-probe`, `head-probe`, `verify-probe` | readings of the phrase and their checks | manual |
| `linking-bench`, `encoder-bench`, `focused-comparison-bench`, `root-model-bench`, `vetting-bench` | model and encoder benches (billed calls) | manual |
| `collocations`, `longer-names`, `report-memory`, `rad-sprl`, `symptom-coverage` | data for the reader and the linker | manual |
| `classifications-it`, `radiology-kinds`, `data-and-dependency-updates` | weekly updates (CIPI, ICD-9-CM, RadLex, HPO, dependencies) | schedule |
| `pin-model-revisions` | pins model weights to an exact commit and hash | manual |

Secrets used: `OPENROUTER_API_KEY` (public corpora only), `HF_TOKEN`, `UMLS_API_KEY`, `DB_PASSWORD`. They live
in the repository's secrets and in the environment, never in code or documents.

## 6. Problems and changes

A defect is opened as an issue with its reproduction and, if it touches a hazard, the risk id of
[`docs/registers/risks.yml`](../registers/risks.yml). The fix names the issue in its commit; a new hazard
enters the register with its control and the test that verifies it.
