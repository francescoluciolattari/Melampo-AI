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

## Model weights and licensed data

These cannot be re-downloaded anonymously (gated Hugging Face models,
licensed terminologies), so they need a **DVC storage remote** — a decision
still open (`decisions_pending: dvc-remote` in the manifest). Once chosen:

```bash
uv add --optional data "dvc-s3"          # or dvc-azure / dvc-gs / dvc-ssh
uv run dvc remote add -d storage s3://<bucket>/<prefix>
uv run dvc push
```

Upstream models are pinned by revision (a commit hash on Hugging Face),
downloaded once, and pushed to the remote so the team reproduces a run
without every member holding every upstream licence.

## What never goes into DVC

**Patient data — never, in any remote.** A case's files are processed in
memory (`data/case_attachments.py`); DICOM instances are kept only
de-identified by allowlist (`data/dicom_volume.py`), and only in memory. DVC
is for public reference data and model weights only.
