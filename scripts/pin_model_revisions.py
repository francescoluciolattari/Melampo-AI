"""Pin every Hugging Face model in melampo-assets.yaml to an exact commit and file hashes.

For each ``kind: model_weights`` asset hosted on Hugging Face, asks the Hub
for the current commit of the repository (or of the ``revision`` already
recorded, with ``--keep-revision``) and writes back into the manifest:

    revision: <40-hex commit>
    files:
      - {path: model.safetensors, sha256: <64-hex>, size: 376453120}
    pinned_on: "2026-09-26"

Hashes come from the Hub's own metadata: for files stored in LFS/Xet (the
weights) the Hub publishes the sha256 of the content, so a 60 GB checkpoint
is pinned without downloading it; small non-LFS files (configs, tokenizer
files) are downloaded and hashed here. Nothing is typed by hand.

Run by .github/workflows/pin-model-revisions.yml, which opens a pull request
with the result. HF_TOKEN is used when set: gated repositories (Pillar-0)
need it, from an account that has accepted their conditions.

The manifest is rewritten with ruamel.yaml in round-trip mode, which keeps
its comments and layout -- PyYAML would drop every comment in the file.
"""

import argparse
import datetime as dt
import hashlib
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import requests

from melampo.models.weights import UNPINNED, model_pins, select_files

HF = "https://huggingface.co"


def pin_from_api(
    info: Mapping[str, Any],
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    hash_small_file: Callable[[str], str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """(commit, files) from a ``/api/models/{repo}/revision/{rev}?blobs=true`` response.

    A non-LFS file with no ``hash_small_file`` available is an error, not a
    silent omission: a pin that skips files is a pin of a different model.
    """
    commit = str(info.get("sha", ""))
    siblings = {str(item["rfilename"]): item for item in info.get("siblings", [])}
    files: list[dict[str, Any]] = []
    for path in select_files(sorted(siblings), include, exclude):
        item = siblings[path]
        lfs = item.get("lfs") or {}
        if lfs.get("sha256"):
            files.append({"path": path, "sha256": str(lfs["sha256"]), "size": int(lfs.get("size") or item.get("size") or 0)})
            continue
        if hash_small_file is None:
            raise ValueError(f"{path}: not stored in LFS and no way to hash it")
        files.append({"path": path, "sha256": hash_small_file(path), "size": int(item.get("size") or 0)})
    return commit, files


def _session(token: str | None) -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = "melampo-pin-model-revisions/0.1"
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    return session


def fetch_pin(repo: str, revision: str, include, exclude, session: requests.Session):  # pragma: no cover - network
    response = session.get(f"{HF}/api/models/{repo}/revision/{revision}", params={"blobs": "true"}, timeout=60)
    response.raise_for_status()
    info = response.json()
    commit = info["sha"]

    def hash_small_file(path: str) -> str:
        download = session.get(f"{HF}/{repo}/resolve/{commit}/{path}", timeout=120)
        download.raise_for_status()
        return hashlib.sha256(download.content).hexdigest()

    return pin_from_api(info, include=include, exclude=exclude, hash_small_file=hash_small_file)


def write_pins(manifest_path: Path, pins: Mapping[str, tuple[str, list[dict[str, Any]]]], today: str) -> None:
    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    data = yaml.load(manifest_path.read_text(encoding="utf-8"))
    for asset in data["assets"]:
        if asset["id"] in pins:
            commit, files = pins[asset["id"]]
            asset["revision"] = commit
            asset["files"] = files
            asset["pinned_on"] = today
    with manifest_path.open("w", encoding="utf-8") as handle:
        yaml.dump(data, handle)


def main(argv=None) -> int:  # pragma: no cover - network
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", default="melampo-assets.yaml")
    parser.add_argument("--only", nargs="*", help="asset ids to pin (default: all)")
    parser.add_argument("--keep-revision", action="store_true", help="re-hash the recorded revision instead of the latest")
    args = parser.parse_args(argv)

    import yaml

    manifest_path = Path(args.manifest)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    session = _session(os.environ.get("HF_TOKEN"))
    pins: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    failures: list[str] = []
    for pin in model_pins(manifest):
        if args.only and pin.asset_id not in args.only:
            continue
        revision = pin.revision if args.keep_revision and pin.revision != UNPINNED else "main"
        try:
            pins[pin.asset_id] = fetch_pin(pin.repo, revision, pin.include, pin.exclude, session)
        except Exception as error:  # noqa: BLE001 - reported, the rest still pinned
            failures.append(f"{pin.asset_id} ({pin.repo}): {error}")
            continue
        commit, files = pins[pin.asset_id]
        total = sum(item["size"] for item in files)
        print(f"- {pin.asset_id}: {pin.repo}@{commit[:12]}, {len(files)} files, {total / 1e9:.2f} GB")
    if pins:
        write_pins(manifest_path, pins, dt.datetime.now(dt.UTC).date().isoformat())
    for failure in failures:
        print(f"- NOT PINNED {failure}", file=sys.stderr)
    return 1 if failures and not pins else 0


if __name__ == "__main__":
    raise SystemExit(main())
