from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .prototype import run_prototype_case


def _load_payload(path: str) -> dict[str, Any]:
    payload_path = Path(path)
    with payload_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Prototype input JSON must contain an object at the top level.")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Melampo clinical research prototype case.")
    parser.add_argument("input_json", help="Path to a JSON file containing the clinical case payload.")
    parser.add_argument(
        "--runtime-profile",
        default="local_research",
        choices=["local_research", "remote_research"],
        help="Runtime profile to use for the prototype run.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include the full raw pipeline result in the JSON output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = _load_payload(args.input_json)
    result = run_prototype_case(payload, runtime_profile=args.runtime_profile)
    if not args.include_raw and "raw_result" in result:
        result = dict(result)
        result.pop("raw_result", None)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
