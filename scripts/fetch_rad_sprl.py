#!/usr/bin/env python
"""Download Rad-SpRL (2,000 Open-i chest X-ray reports with Spatial Role Labeling annotations) from Mendeley Data.

Rad-SpRL (Datta, Roberts et al.; Mendeley Data, DOI 10.17632/yhb26hfz8n.1) marks, in IU X-ray reports, the
spatial roles Trajector (the finding), Landmark (the anatomical place), Spatial indicator (the preposition),
Diagnosis and Hedge. The Landmark spans are human-annotated anatomical places in real radiology reports:
what our development gold set lacks. The dataset page does not state a licence in plain text: read it on
the page before any use beyond development.

The workspace of the session cannot reach Mendeley; GitHub Actions can (workflow ``rad-sprl``). The script
lists the files through the public API, prints what it found (the layout is not known in advance), downloads
everything into ``--out`` and unpacks archives.

    python scripts/fetch_rad_sprl.py --out rad_sprl
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

DATASET, VERSION = "yhb26hfz8n", 1
API = "https://data.mendeley.com/public-api/datasets"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "melampo-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=300) as response:  # noqa: S310 (fixed https host)
        return response.read()


def files(folder: str = "root") -> list[dict]:
    """Files of a folder, following sub-folders."""
    listing = json.loads(_get(f"{API}/{DATASET}/files?folder_id={folder}&version={VERSION}"))
    out = list(listing)
    try:
        folders = json.loads(_get(f"{API}/{DATASET}/folders/{VERSION}"))
    except Exception as exc:  # noqa: BLE001 - the folder listing is optional
        print(f"no folder listing: {exc}", file=sys.stderr)
        folders = []
    if folder == "root":
        for f in folders:
            fid = f.get("id")
            if fid and fid != "root":
                out += json.loads(_get(f"{API}/{DATASET}/files?folder_id={fid}&version={VERSION}"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("rad_sprl"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        meta = json.loads(_get(f"{API}/{DATASET}?version={VERSION}"))
        (args.out / "dataset.json").write_text(json.dumps(meta, indent=1), "utf-8")
        print("licence:", json.dumps(meta.get("data_licence") or meta.get("licence") or "not in the metadata"))
    except Exception as exc:  # noqa: BLE001
        print(f"no dataset metadata: {exc}", file=sys.stderr)
    found = files()
    print(json.dumps([{k: f.get(k) for k in ("filename", "size", "id")} for f in found], indent=1))
    if not found:
        print("no file listed: read dataset.json and adapt", file=sys.stderr)
        return 2
    for f in found:
        url = (f.get("content_details") or {}).get("download_url") or f.get("download_url")
        if not url:
            print(f"no download url for {f.get('filename')}", file=sys.stderr)
            continue
        target = args.out / f["filename"]
        target.write_bytes(_get(url))
        print(f"{target}: {target.stat().st_size} bytes")
        if zipfile.is_zipfile(target):
            with zipfile.ZipFile(target) as z:
                z.extractall(args.out / target.stem)
            print(f"unpacked {target.name}")
    for p in sorted(args.out.rglob("*")):
        if p.is_file():
            print(p.relative_to(args.out), p.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
