from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .datasets.chestxray14_loader import ChestXray14CsvLoader
from .datasets.openi_loader import OpenIReportCsvLoader
from .prototype import run_prototype_case


def _load_payload(path: str) -> dict[str, Any]:
    payload_path = Path(path)
    with payload_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Prototype input JSON must contain an object at the top level.")
    return data


def _strip_raw(result: dict, include_raw: bool) -> dict:
    if include_raw or "raw_result" not in result:
        return result
    stripped = dict(result)
    stripped.pop("raw_result", None)
    return stripped


def _run_payloads(payloads: list[dict], runtime_profile: str, include_raw: bool) -> tuple[int, list[dict]]:
    results = []
    exit_code = 0
    for payload in payloads:
        result = run_prototype_case(payload, runtime_profile=runtime_profile)
        if result.get("status") != "completed":
            exit_code = 2
        results.append(_strip_raw(result, include_raw=include_raw))
    return exit_code, results


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


def build_cxr_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Melampo prototype cases from ChestX-ray14-style CSV metadata.")
    parser.add_argument("input_csv", help="Path to a ChestX-ray14-style metadata CSV file.")
    parser.add_argument(
        "--runtime-profile",
        default="local_research",
        choices=["local_research", "remote_research"],
        help="Runtime profile to use for the prototype run.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of CSV rows to process.")
    parser.add_argument("--image-root", default=None, help="Optional local image root used to build series_paths.")
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include each full raw pipeline result in the JSON output.",
    )
    return parser


def build_openi_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Melampo prototype cases from Open-i / Indiana-style CSV metadata.")
    parser.add_argument("input_csv", help="Path to an Open-i / Indiana-style metadata CSV file.")
    parser.add_argument(
        "--runtime-profile",
        default="local_research",
        choices=["local_research", "remote_research"],
        help="Runtime profile to use for the prototype run.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of CSV rows to process.")
    parser.add_argument("--image-root", default=None, help="Optional local image root used to build series_paths.")
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include each full raw pipeline result in the JSON output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = _load_payload(args.input_json)
    result = run_prototype_case(payload, runtime_profile=args.runtime_profile)
    result = _strip_raw(result, include_raw=args.include_raw)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("status") == "completed" else 2


def main_cxr(argv: list[str] | None = None) -> int:
    parser = build_cxr_parser()
    args = parser.parse_args(argv)
    loader = ChestXray14CsvLoader(image_root=args.image_root)
    payloads = loader.load_csv(args.input_csv, limit=args.limit)
    exit_code, results = _run_payloads(payloads, runtime_profile=args.runtime_profile, include_raw=args.include_raw)
    output = {
        "status": "completed" if exit_code == 0 else "completed_with_errors",
        "input_csv": args.input_csv,
        "runtime_profile": args.runtime_profile,
        "case_count": len(results),
        "results": results,
    }
    print(json.dumps(output, indent=2, sort_keys=True, default=str))
    return exit_code


def main_openi(argv: list[str] | None = None) -> int:
    parser = build_openi_parser()
    args = parser.parse_args(argv)
    loader = OpenIReportCsvLoader(image_root=args.image_root)
    payloads = loader.load_csv(args.input_csv, limit=args.limit)
    exit_code, results = _run_payloads(payloads, runtime_profile=args.runtime_profile, include_raw=args.include_raw)
    output = {
        "status": "completed" if exit_code == 0 else "completed_with_errors",
        "input_csv": args.input_csv,
        "runtime_profile": args.runtime_profile,
        "case_count": len(results),
        "results": results,
    }
    print(json.dumps(output, indent=2, sort_keys=True, default=str))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
