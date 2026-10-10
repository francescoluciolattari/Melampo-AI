#!/usr/bin/env python
"""Build data/linking/procedure_frames.json: the slots of procedure head words (``melampo.memory.frames``).

Sources (see the module): NCIt relations (``--ncit``), text read (``--medmentions``: training abstracts
only; ``--ncit-definitions``; never evaluation documents),
and SNOMED CT RF2 files when a licensed release is available (``--snomed-relationships`` and
``--snomed-descriptions``; Italy needs an affiliate licence through MLDS). The procedure heads are the
words the block memory types as procedures.

    python scripts/build_procedure_frames.py --ncit ncit.obo --ncit-definitions --medmentions ext/mm
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melampo.memory import chunk_lattice as cl  # noqa: E402
from melampo.memory.frames import DEFAULT_PATH, ProcedureFrames  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ncit", type=Path)
    ap.add_argument("--medmentions", type=Path, help="MedMentions clone: its *training* abstracts are read")
    ap.add_argument("--ncit-definitions", action="store_true", help="also read the NCIt definitions as text")
    ap.add_argument("--pubmed", type=Path, nargs="*", default=[], help="PubMed baseline XML .gz files to read")
    ap.add_argument("--exclude-pmids", type=Path, nargs="*", default=[], help="PMIDs never read (MedMentions)")
    ap.add_argument("--snomed-relationships", type=Path)
    ap.add_argument("--snomed-descriptions", type=Path)
    ap.add_argument("--out", type=Path, default=DEFAULT_PATH)
    args = ap.parse_args(argv)
    memory = cl.BlockMemory.load()

    def kind_of(word: str) -> str:
        got = memory.head(word)
        return got[0] if got and got[1] >= cl.MIN_SHARE else ""

    frames = ProcedureFrames()
    sources: dict = {}
    if args.ncit:
        frames.add_ncit(args.ncit)
        sources["ncit"] = {"file": args.ncit.name, "sha256": hashlib.sha256(args.ncit.read_bytes()).hexdigest()}
    texts: list[str] = []
    if args.medmentions:
        sys.path.insert(0, str(ROOT / "scripts"))
        import external_check as ec  # noqa: E402
        import phrase_knowledge as pk  # noqa: E402

        splits = pk.medmentions_splits(args.medmentions)
        texts += [t for d, t, _ in ec.medmentions_documents(args.medmentions) if splits.get(d) == "trng"]
        sources.setdefault("text", []).append({"medmentions_training_abstracts": len(texts)})
    if args.ncit and args.ncit_definitions:
        sys.path.insert(0, str(ROOT / "scripts"))
        import phrase_knowledge as pk  # noqa: E402

        definitions = list(pk._ncit_definitions(args.ncit).values())
        texts += definitions
        sources.setdefault("text", []).append({"ncit_definitions": len(definitions)})
    if texts:
        frames.add_texts(texts, kind_of)
    if args.pubmed:
        sys.path.insert(0, str(ROOT / "scripts"))
        from build_collocations import abstracts, open_source  # noqa: E402

        excluded: set[str] = set()
        for path in args.exclude_pmids:
            excluded.update(path.read_text("utf-8").split())
        n = 0
        for path in args.pubmed:
            with open_source(str(path)) as stream:
                batch = [t for pmid, t in abstracts(stream) if pmid not in excluded]
            frames.add_texts(batch, kind_of)
            n += len(batch)
        sources.setdefault("text", []).append({"pubmed_abstracts": n, "files": len(args.pubmed)})
    if args.snomed_relationships and args.snomed_descriptions:
        frames.add_snomed(args.snomed_relationships, args.snomed_descriptions)
        sources["snomed"] = {"relationships": args.snomed_relationships.name}
    data = frames.to_json()
    out = {
        "note": ("Cornici delle teste di procedura (sito, dispositivo): evidenze per fonte, nessuna scritta a mano. "
                 "Generato da scripts/build_procedure_frames.py; non si modifica a mano."),
        "built": datetime.datetime.now(datetime.UTC).date().isoformat(),
        "sources": sources,
        "frames": data,
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=0) + "\n", "utf-8")
    with_site = sum(1 for h in data if "site" in frames.slots(h))
    with_device = sum(1 for h in data if "device" in frames.slots(h))
    print(f"{len(data)} heads ({with_site} with a site slot, {with_device} with a device slot) -> {args.out}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
