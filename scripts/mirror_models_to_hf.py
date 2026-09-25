"""Copy the pinned model revisions into private repositories of the project's HF organisation.

Phase "first deployment" of the storage plan (see src/melampo/models/weights.py):
upstream repositories can be deleted, re-gated or rewritten by their
owners, and a deployed medical device must keep being able to fetch the
exact weights it was validated with. For each pinned model this script

1. downloads exactly the pinned files at the pinned commit,
2. verifies every file against its sha256 in melampo-assets.yaml,
3. uploads them to ``<organisation>/<model name>`` as a **private** repo,
4. records the mirror's commit as ``mirror_revision`` in the manifest.

Nothing is run automatically: it needs ``backend: hf_mirror`` and the
organisation in melampo-storage.yaml, the ``models`` extra, local disk for
the largest model (gpt-oss-120b: over 60 GB), and an HF_TOKEN with write
access to the organisation. The current licences are permissive --
Apache-2.0 (gpt-oss), ECL-2.0 (Pillar-0), OpenMDW-1.1 (Nemotron-Parse 2.0)
-- but each model's terms and gating conditions are re-read before its
first mirror, since a new revision may change them.

    HF_TOKEN=... uv run --extra models python scripts/mirror_models_to_hf.py --only nemotron-parse-2-weights
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import yaml

from melampo.models.weights import (
    BACKEND_HF_MIRROR,
    StorageConfig,
    model_pins,
    verify_files,
)


def main(argv=None) -> int:  # pragma: no cover - network and large downloads
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--storage", default="melampo-storage.yaml")
    parser.add_argument("--manifest", default="melampo-assets.yaml")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args(argv)

    from huggingface_hub import HfApi, snapshot_download

    config = StorageConfig.load(args.storage)
    if config.backend != BACKEND_HF_MIRROR or config.problems():
        print("melampo-storage.yaml must set backend: hf_mirror with an organisation", file=sys.stderr)
        return 2
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    manifest_path = Path(args.manifest)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    mirrored: dict[str, str] = {}
    for pin in model_pins(manifest):
        if args.only and pin.asset_id not in args.only:
            continue
        if not pin.is_pinned:
            print(f"- {pin.asset_id}: skipped, not pinned", file=sys.stderr)
            continue
        with tempfile.TemporaryDirectory() as workdir:
            snapshot_download(
                repo_id=pin.repo,
                revision=pin.revision,
                allow_patterns=[item.path for item in pin.files],
                local_dir=workdir,
                token=token,
            )
            problems = verify_files(workdir, pin.files)
            if problems:
                print(f"- {pin.asset_id}: NOT mirrored, {problems}", file=sys.stderr)
                continue
            target = f"{config.hf_organization}/{pin.mirror_repo_name}"
            api.create_repo(target, private=True, exist_ok=True)
            commit = api.upload_folder(
                folder_path=workdir,
                repo_id=target,
                allow_patterns=[item.path for item in pin.files],
                commit_message=f"Mirror of {pin.repo}@{pin.revision}",
            )
            mirrored[pin.asset_id] = commit.oid
            print(f"- {pin.asset_id}: {target}@{commit.oid[:12]}")

    if mirrored:
        from ruamel.yaml import YAML

        round_trip = YAML()
        round_trip.preserve_quotes = True
        round_trip.width = 4096
        data = round_trip.load(manifest_path.read_text(encoding="utf-8"))
        for asset in data["assets"]:
            if asset["id"] in mirrored:
                asset["mirror_revision"] = mirrored[asset["id"]]
        with manifest_path.open("w", encoding="utf-8") as handle:
            round_trip.dump(data, handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
