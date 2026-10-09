#!/usr/bin/env python
"""Build data/linking/block_memory.json: what the chunk lattice remembers (9 October 2026).

The lattice (``src/melampo/memory/chunk_lattice.py``) reads a phrase as blocks. A block is either
remembered (a known name with its kind) or composed (modifiers and a head whose kind is known). This
script derives both from the NCI Thesaurus, with the kind of each name taken from the ontology class
(``scripts/ncit_kinds.py``), never from a hand-written word list:

Questionnaire items and data-standard rows (CDISC, "Question", "Response", "Result") are not counted.
Names of a class that sits under two kind roots ("sterol" is a food and a chemical) are left out of
both tables: they cannot tell a reader what kind a word is.

* ``heads``: the words that end NCIt names, with the dominant kind of those names, its share and the
  number of names. Kept when at least ``--min-names`` names end with the word and the dominant kind
  has at least ``--min-share`` of them; where the endings are mixed, the word's own NCIt class is
  used if some name ending with the word has that kind (share 1, count 0). English compounds are right-headed, so the kind of a
  compound is the kind of its last word unless a remembered name says otherwise.
* ``names``: multi-word NCIt names of a kind other than anatomy that contain the name of a structure
  (UBERON names and the project lexicon) -- "inferior vena cava filter", "liver transplantation",
  "brain weight", kept only when the kind is not already given by the last word (a name that is
  composed from its head is not remembered). Names of anatomy are in UBERON already. Names of proteins, genes, chemicals and
  assessment tools are in ``longer_names.json``.

The NCIt file is downloaded by the caller and never committed. The output records the source
version, hash, thresholds and counts.

    python scripts/build_block_memory.py --ncit ncit.obo [--uberon uberon-basic.obo]
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_longer_names import (  # noqa: E402
    _ITEM_CLASS,
    _NOISE,
    anatomical_runs,
    contains_a_run,
    sha256,
    tokens,
    version_of,
)
from ncit_kinds import Kinds, singular  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "linking"
# The kinds a remembered name may have in this file (anatomy is UBERON; the four kinds of
# longer_names.json are not repeated).
NAME_KINDS = frozenset(
    (
        "disease",
        "procedure",
        "device",
        "food",
        "organism",
        "process",
        "property",
        "activity",
        "conceptual",
    )
)


# Own class of a word that is also a gene symbol, a chemical or an organism name ("scar", "air",
# "up") says nothing about the common noun: those kinds do not give a word its kind.
SELF_KINDS = frozenset(
    ("anatomy", "device", "procedure", "property", "process", "activity", "food")
)


def noise(name: str) -> bool:
    """Questionnaire items, codelist rows and data-standard terms are not names a text writes."""
    return bool(_NOISE.search(name) or _ITEM_CLASS.search(name))


def build(kinds: Kinds, runs, min_names: int, min_share: float):
    heads = {}
    for word, counts in kinds.last_single.items():
        total = sum(counts.values())
        kind, n = counts.most_common(1)[0]
        if total >= min_names and n / total >= min_share:
            heads[word] = [kind, round(n / total, 3), total]
        elif total >= min_names:
            # Mixed endings ("filter", "weight"): the word's own NCIt class, if names that end with
            # the word agree with it. Count 0 = the word's own class, not a share of names.
            own = kinds.names_single.get(word) or kinds.names_single.get(word + "s")
            if own in SELF_KINDS and counts.get(own):
                heads[word] = [own, 1.0, 0]
    names, counts = {}, Counter()
    for key, kind in kinds.names_single.items():
        words = tokens(key)
        if kind in NAME_KINDS and len(words) >= 2 and contains_a_run(words, runs):
            # A name whose last word already gives its kind is composed, not remembered.
            if heads.get(singular(words[-1]), (None,))[0] == kind:
                continue
            names[" ".join(words)] = kind
            counts[kind] += 1
    return dict(sorted(heads.items())), dict(sorted(names.items())), counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ncit", type=Path, required=True)
    parser.add_argument("--uberon", type=Path, default=DATA / "uberon-basic.obo")
    parser.add_argument("--lexicon", type=Path, default=DATA / "anatomy_lexicon.json")
    parser.add_argument("--out", type=Path, default=DATA / "block_memory.json")
    parser.add_argument("--min-names", type=int, default=5)
    parser.add_argument("--min-share", type=float, default=0.75)
    args = parser.parse_args(argv)
    kinds = Kinds.from_obo(args.ncit, skip_name=noise)
    runs = anatomical_runs(args.uberon, args.lexicon)
    heads, names, counts = build(kinds, runs, args.min_names, args.min_share)
    out = {
        "note": (
            "Memoria del reticolo di blocchi: 'heads' = ultima parola dei nomi NCIt con il tipo "
            "dominante (quota, numero di nomi); 'names' = nomi NCIt di altro tipo che contengono una "
            "struttura. Generato da scripts/build_block_memory.py; non si modifica a mano."
        ),
        "built": datetime.datetime.now(datetime.UTC).date().isoformat(),
        "sources": {
            "ncit": {"data_version": version_of(args.ncit), "sha256": sha256(args.ncit)}
        },
        "thresholds": {"min_names": args.min_names, "min_share": args.min_share},
        "counts": {"heads": len(heads), **{f"names:{k}": v for k, v in counts.items()}},
        "heads": heads,
        "names": names,
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=0) + "\n", "utf-8")
    print(
        f"{len(heads)} heads, {len(names)} names -> {args.out}: {dict(counts)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
