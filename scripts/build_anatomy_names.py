#!/usr/bin/env python
"""Build data/linking/anatomy_names.json: the multi-word names of structures the chunk lattice remembers.

Frank's point 1 (10 October 2026): a reader who knows the domain recognises "inferior vena cava" as one
block already in memory, and joins two blocks, not five words. The lattice remembered the names of other
things (NCIt: procedures, devices, diseases that contain a structure) but not the names of the structures
themselves: an anatomical neighbour of the mention ("of the olfactory bulb") was read word by word, and
its last word could take the kind of another sense ("bulb" is a device in the NCIt heads).

Sources: UBERON names and exact synonyms (pinned release, CC BY 3.0) and the project lexicon (English and
Italian). Kept: names of 2 to 5 words, folded and split as the lattice splits them (hyphens are breaks);
names of one word are the heads' business.

    python scripts/build_anatomy_names.py --uberon data/linking/uberon-basic.obo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melampo.memory.chunk_lattice import _TOKEN  # noqa: E402
from melampo.memory.word_senses import _fold  # noqa: E402

DATA = ROOT / "data" / "linking"
_SYNONYM = re.compile(r'^synonym: "(.*)" EXACT')


def key(name: str) -> str:
    return " ".join(_fold(m.group(0)) for m in _TOKEN.finditer(name))


def uberon_names(path: Path) -> set[str]:
    out: set[str] = set()
    obsolete = False
    current: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line in ("[Term]", "[Typedef]"):
                if current and not obsolete:
                    out.update(current)
                current, obsolete = [], line != "[Term]"
            elif line.startswith("name: "):
                current.append(line[6:].strip())
            elif m := _SYNONYM.match(line):
                current.append(m.group(1))
            elif line == "is_obsolete: true":
                obsolete = True
    if current and not obsolete:
        out.update(current)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--uberon", type=Path, default=DATA / "uberon-basic.obo")
    ap.add_argument("--lexicon", type=Path, default=DATA / "anatomy_lexicon.json")
    ap.add_argument("--out", type=Path, default=DATA / "anatomy_names.json")
    args = ap.parse_args(argv)
    raw = uberon_names(args.uberon)
    lexicon = json.loads(args.lexicon.read_text("utf-8"))
    for entry in lexicon["classes"].values():
        for lang in ("en", "it"):
            raw.update(entry.get(lang, ()))
    names = sorted({k for k in (key(n) for n in raw) if 2 <= len(k.split()) <= 5})
    version = ""
    with args.uberon.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("data-version:"):
                version = line.split(":", 1)[1].strip()
                break
    out = {
        "note": ("Nomi di strutture di 2-5 parole che il reticolo ricorda (UBERON nomi e sinonimi esatti, "
                 "lessico del progetto). Generato da scripts/build_anatomy_names.py; non si modifica a mano. "
                 "UBERON: CC BY 3.0."),
        "sources": {"uberon": {"data_version": version,
                               "sha256": hashlib.sha256(args.uberon.read_bytes()).hexdigest()},
                    "lexicon": args.lexicon.name},
        "count": len(names),
        "names": names,
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=0) + "\n", "utf-8")
    print(f"{len(names)} names -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
