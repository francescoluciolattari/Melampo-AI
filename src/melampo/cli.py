from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .datasets.chestxray14_loader import ChestXray14CsvLoader
from .datasets.openi_loader import OpenIReportCsvLoader
from .memory.weaviate_adapter import WeaviateAdapterConfig, WeaviateSemanticMemoryAdapter
from .orchestration.model_capability_registry import ModelCapabilityRegistry
from .prototype import run_prototype_case

IMAGING_STRATEGIES = [
    "local_metadata",
    "local_pixels",
    "remote_radiology_vlm",
    "remote_dicom_3d",
    "hybrid_multimodal",
]


def _load_payload(path: str) -> dict[str, Any]:
    payload_path = Path(path)
    with payload_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Prototype input JSON must contain an object at the top level.")
    return data


def _attach_image_paths(payload: dict[str, Any], image_paths: list[str] | None) -> dict[str, Any]:
    if not image_paths:
        return payload
    updated = dict(payload)
    imaging = list(updated.get("imaging", []))
    if imaging:
        first = dict(imaging[0])
        existing = list(first.get("series_paths", []))
        first["series_paths"] = existing + list(image_paths)
        imaging[0] = first
    else:
        imaging = [
            {
                "study_id": f"local-image-study-{updated.get('case_id', 'unknown')}",
                "modality": "CR",
                "series_paths": list(image_paths),
                "metadata": {"source": "cli_image_path"},
            }
        ]
    updated["imaging"] = imaging
    return updated


def _strip_raw(result: dict, include_raw: bool) -> dict:
    if include_raw or "raw_result" not in result:
        return result
    stripped = dict(result)
    stripped.pop("raw_result", None)
    return stripped


def _run_payloads(payloads: list[dict], runtime_profile: str, include_raw: bool, imaging_strategy: str | None = None) -> tuple[int, list[dict]]:
    results = []
    exit_code = 0
    for payload in payloads:
        result = run_prototype_case(payload, runtime_profile=runtime_profile, imaging_strategy=imaging_strategy)
        if result.get("status") != "completed":
            exit_code = 2
        results.append(_strip_raw(result, include_raw=include_raw))
    return exit_code, results


def _add_imaging_strategy_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--imaging-strategy",
        default=None,
        choices=IMAGING_STRATEGIES,
        help="Optional imaging provider strategy override.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Melampo clinical research prototype case.")
    parser.add_argument("input_json", help="Path to a JSON file containing the clinical case payload.")
    parser.add_argument(
        "--runtime-profile",
        default="local_research",
        choices=["local_research", "remote_research"],
        help="Runtime profile to use for the prototype run.",
    )
    _add_imaging_strategy_argument(parser)
    parser.add_argument(
        "--image-path",
        action="append",
        default=None,
        help="Optional local image path to attach to the first imaging study. May be repeated.",
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
    _add_imaging_strategy_argument(parser)
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
    _add_imaging_strategy_argument(parser)
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of CSV rows to process.")
    parser.add_argument("--image-root", default=None, help="Optional local image root used to build series_paths.")
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include each full raw pipeline result in the JSON output.",
    )
    return parser


def build_weaviate_schema_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect or prepare Melampo's Weaviate semantic-memory schema.")
    parser.add_argument("--endpoint", default=None, help="Optional Weaviate endpoint. No network call is made unless --execute is passed.")
    parser.add_argument("--api-key-env", default=None, help="Optional environment variable name containing the Weaviate API key.")
    parser.add_argument("--execute", action="store_true", help="Attempt live schema materialization. Requires an infrastructure subclass in production.")
    parser.add_argument("--output", default=None, help="Optional path to write schema JSON.")
    return parser


def build_decision_record_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print Melampo's enterprise model decision record.")
    parser.add_argument("--output", default=None, help="Optional path to write decision record JSON.")
    return parser


def _emit_json(payload: dict[str, Any], output_path: str | None = None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if output_path:
        Path(output_path).write_text(text + "\n", encoding="utf-8")
    print(text)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = _load_payload(args.input_json)
    payload = _attach_image_paths(payload, args.image_path)
    result = run_prototype_case(payload, runtime_profile=args.runtime_profile, imaging_strategy=args.imaging_strategy)
    result = _strip_raw(result, include_raw=args.include_raw)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("status") == "completed" else 2


def main_cxr(argv: list[str] | None = None) -> int:
    parser = build_cxr_parser()
    args = parser.parse_args(argv)
    loader = ChestXray14CsvLoader(image_root=args.image_root)
    payloads = loader.load_csv(args.input_csv, limit=args.limit)
    exit_code, results = _run_payloads(
        payloads,
        runtime_profile=args.runtime_profile,
        include_raw=args.include_raw,
        imaging_strategy=args.imaging_strategy,
    )
    output = {
        "status": "completed" if exit_code == 0 else "completed_with_errors",
        "input_csv": args.input_csv,
        "runtime_profile": args.runtime_profile,
        "imaging_strategy": args.imaging_strategy,
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
    exit_code, results = _run_payloads(
        payloads,
        runtime_profile=args.runtime_profile,
        include_raw=args.include_raw,
        imaging_strategy=args.imaging_strategy,
    )
    output = {
        "status": "completed" if exit_code == 0 else "completed_with_errors",
        "input_csv": args.input_csv,
        "runtime_profile": args.runtime_profile,
        "imaging_strategy": args.imaging_strategy,
        "case_count": len(results),
        "results": results,
    }
    print(json.dumps(output, indent=2, sort_keys=True, default=str))
    return exit_code


def main_weaviate_schema(argv: list[str] | None = None) -> int:
    parser = build_weaviate_schema_parser()
    args = parser.parse_args(argv)
    adapter = WeaviateSemanticMemoryAdapter(
        config=WeaviateAdapterConfig(
            endpoint=args.endpoint,
            api_key_env=args.api_key_env,
            enabled=bool(args.execute),
            dry_run=not bool(args.execute),
        )
    )
    payload = adapter.materialize_schema() if args.execute else adapter.prepare_schema_materialization()
    _emit_json(payload, output_path=args.output)
    return 0


def main_decision_record(argv: list[str] | None = None) -> int:
    parser = build_decision_record_parser()
    args = parser.parse_args(argv)
    payload = ModelCapabilityRegistry.build_default().decision_record()
    _emit_json(payload, output_path=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
