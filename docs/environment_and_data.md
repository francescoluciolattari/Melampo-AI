# Environment and data: uv + DVC

Two tools, two jobs, and a manifest that lists everything they manage.

| What | Tool | Where it is declared |
|---|---|---|
| Python and every Python package | **uv** | `pyproject.toml`, `uv.lock`, `.python-version` |
| Reference data and model weights | **DVC** | `data/*.dvc` |
| Everything the project uses beyond its own code | — | `melampo-assets.yaml` |
| System packages uv cannot install (poppler) | apt | `.github/workflows/ci.yml` |

## Getting started

```bash
uv sync --extra dev --extra data      # the exact environment in uv.lock
uv run dvc update -R data             # reference data, verified against its pins
uv run pytest -q                      # the full suite
```

`uv sync --locked` (what CI runs) fails if `uv.lock` is out of date instead
of silently re-resolving. After changing dependencies in `pyproject.toml`,
run `uv lock` and commit the updated `uv.lock`.

## How reference data is pinned

Each `data/*.dvc` file records a **versioned upstream URL** and the **md5**
of the file it names — for example `hp.obo` from the HPO release
`v2026-09-01`. Public sources need no storage remote: `dvc update`
downloads from the pinned URL, and if the content differed from the pinned
md5 DVC would rewrite the `.dvc` file; CI turns that into a failure with
`git diff --exit-code -- 'data/*.dvc'`.

The HPO files were moved from git to DVC without any change: the md5 of each
file previously in git equals the md5 of the release asset it is now pinned
to. They remain in git history (about 130 MB); purging it requires a history
rewrite and a force push, listed as a pending decision in the manifest.

## Updating HPO

The weekly workflow (`data-and-dependency-updates.yml`) detects a new
release, re-pins every HPO file with `dvc import-url --force` to the new
tag, runs the full suite and opens a pull request whose diff is the `.dvc`
pins and the term-history records — never the data files themselves.
Manually:

```bash
TAG=2026-11-01   # the new release
for f in hp.obo phenotype.hpoa genes_to_phenotype.txt genes_to_disease.txt phenotype_to_genes.txt; do
  uv run dvc import-url --force \
    "https://github.com/obophenotype/human-phenotype-ontology/releases/download/v${TAG}/${f}" "data/${f}"
done
```

All HPO files move together; a test enforces a single release across them.

## Model weights: pinned in the manifest, fetched per melampo-storage.yaml

Which weights: every model in `melampo-assets.yaml` (`kind: model_weights`)
carries its Hugging Face `repo`, the exact commit (`revision`) and the
sha256 of each file. The **Pin model revisions** workflow writes them from
the Hub's own metadata and opens a pull request; nothing is typed by hand
(`scripts/pin_model_revisions.py`). `revision: to_pin` means it has not run
yet for that model. Gated repositories (Pillar-0) need the `HF_TOKEN` secret
from an account that has accepted their conditions: gated access is granted
to users, never to organisations.

Where from: `melampo-storage.yaml`, one key, `backend`:

| Phase | backend | Weights come from |
|---|---|---|
| Now | `upstream` | the original repository, at the pinned commit |
| First deployment | `hf_mirror` | private copies in the project's HF organisation (`scripts/mirror_models_to_hf.py`) |
| First training run | `dvc` | a DVC remote in an EU region (the jurisdiction is enforced) |

```bash
uv run python scripts/configure_model_storage.py          # validate, show where each model comes from
uv run python scripts/configure_model_storage.py --apply  # dvc backend: register the remote
```

Whatever the backend, every downloaded file is verified against its pinned
sha256 (`models/weights.py`). CI never downloads weights: GitHub's standard
runners have no GPU and 14 GB of disk. Git LFS was considered and rejected
(2 GB per file on Free/Pro, 10 GiB/month of storage and bandwidth).

## Symptom sources for all diseases

`data/mondo.obo`, `data/doid.owl` and `data/symp.obo` are DVC-pinned like
HPO (Mondo to its release, DO and SYMP to a commit of their repositories).
`memory/symptom_sources.py` turns HPO, Disease Ontology, NCIt (via UMLS) and
Wikidata into one `SymptomLink` shape; the **Symptom-source coverage**
workflow measures each on `data/common_diseases_reference.tsv` and uploads
per-disease counts and every link as an artifact.

## What never goes into DVC

**Patient data — never, in any remote.** A case's files are processed in
memory (`data/case_attachments.py`); DICOM instances are kept only
de-identified by allowlist (`data/dicom_volume.py`), and only in memory. DVC
is for public reference data and model weights only.
