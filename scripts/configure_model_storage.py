"""Check melampo-storage.yaml and apply it -- the one place the weights backend is switched.

    uv run python scripts/configure_model_storage.py            # validate, print the plan
    uv run python scripts/configure_model_storage.py --apply    # also register the DVC remote (dvc backend)

For every pinned model it prints where the weights would come from under the
configured backend. With ``--apply`` and ``backend: dvc`` it registers the
EU remote in .dvc/config (``dvc remote add --default``). Credentials are
never read from or written to either file: DVC and huggingface_hub take them
from the environment (AWS_*/AZURE_* variables, HF_TOKEN) of whoever runs it.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from melampo.models.weights import StorageConfig, model_pins, plan_download


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--storage", default="melampo-storage.yaml")
    parser.add_argument("--manifest", default="melampo-assets.yaml")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    config = StorageConfig.load(args.storage)
    problems = config.problems()
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 2
    print(f"backend: {config.backend}")
    manifest = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8"))
    for pin in model_pins(manifest):
        try:
            plan = plan_download(pin, config)
        except ValueError as error:
            print(f"- {pin.asset_id}: {error}")
            continue
        where = plan.dvc_target or f"{plan.repo}@{plan.revision[:12]}"
        print(f"- {pin.asset_id}: {where} ({len(plan.files)} files, sha256-verified)")
    if args.apply:
        for command in config.dvc_remote_commands():
            print("$ " + " ".join(command))
            subprocess.run(["uv", "run", *command], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
