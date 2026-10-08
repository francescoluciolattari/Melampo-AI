#!/usr/bin/env python
"""Build data/linking/longer_names.json: names of other kinds of things that contain an anatomical name.

"International Prostate Symptom Score", "liver fatty acid binding protein", "TWIK-related spinal cord K+
channel": a mention inside one of these names is a word of the name, not the structure. Syntax cannot
tell them from "liver gene expression" (the experiment `head_probe` measured an AUC of 0.55); a
known longer name of another kind can. The kind comes from the ontology that lists the name, not from
a word list: a protein, a gene, a chemical or biomedical material, a research or clinical assessment
tool. Names of anatomy, diseases, organisms, cells and procedures are not here: those name the
structure ("liver cancer", "mouse brain", "liver biopsy").

Sources (open, downloaded by the caller, never committed): NCI Thesaurus OBO edition
(github.com/NCI-Thesaurus/thesaurus-obo-edition, release asset ncit.obo) and, when given, the Protein
Ontology (every non-obsolete term is a protein). Only multi-word names that contain a name of the
anatomy graph (UBERON names and the project lexicon) are kept, which makes the file a few hundred
kilobytes. The output records the source versions, counts and filters.

    python scripts/build_longer_names.py --ncit ncit.obo [--pr pr.obo] [--uberon uberon-basic.obo]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "linking"

# NCIt classes (OBO ids) that make a name "another kind of thing", and their label in the file.
NCIT_KINDS = {
    "NCIT:C17021": "protein",
    "NCIT:C16612": "gene",
    "NCIT:C1908": "chemical",  # Drug, Food, Chemical or Biomedical Material
    "NCIT:C20993": "assessment_tool",  # Research or Clinical Assessment Tool
}
# Terminology of data standards (CDISC) lists questionnaire items and codelist rows, not names a
# text would write.
_NOISE = re.compile(
    r"\b(cdisc|cdash|sdtm|adam|orres|terminology|test code|test name|response codelist)\b",
    re.IGNORECASE,
)
_ORGANISM_SUFFIX = re.compile(
    r"\s*\((?:human|mouse|rat|bovine|pig|chicken|zebrafish|xenopus)[^)]*\)\s*$",
    re.IGNORECASE,
)
# Questionnaire items and answers are sentences ("have shoulder or arm pain"), not names of an
# instrument: a class whose name says Question, Response or Result is left out with its descendants.
_ITEM_CLASS = re.compile(r"\b(questions?|responses?|results?)\b", re.IGNORECASE)
# The name of an instrument ends in the noun for what it is; a data element ("age at diagnosis of
# congenital heart disease") does not.
_INSTRUMENT_NOUNS = frozenset(
    [
        "scale",
        "scales",
        "score",
        "scores",
        "index",
        "indices",
        "questionnaire",
        "inventory",
        "survey",
        "checklist",
        "rating",
        "ratings",
        "assessment",
        "instrument",
        "module",
        "profile",
        "test",
        "schedule",
        "measure",
        "form",
        "battery",
        "interview",
        "classification",
        "criteria",
        "diary",
        "scoring",
        "staging",
    ]
)
MAX_WORDS = 9


def tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", text.lower().replace("-", " ")))


def read_obo(path: Path):
    """(id, name, synonyms, parents, obsolete) for each term of an OBO file."""
    term = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line == "[Term]":
                if term:
                    yield term
                term = {"id": "", "name": "", "syn": [], "isa": [], "obs": False}
            elif line.startswith("["):
                if term:
                    yield term
                term = None
            elif term is not None:
                if line.startswith("id: "):
                    term["id"] = line[4:]
                elif line.startswith("name: "):
                    term["name"] = line[6:]
                elif line.startswith("is_obsolete: true"):
                    term["obs"] = True
                elif line.startswith("is_a: "):
                    term["isa"].append(line[6:].split(" ")[0])
                elif line.startswith("synonym: "):
                    found = re.match(r'synonym: "(.*)" (\w+)', line)
                    if found:
                        term["syn"].append(found.group(1))
    if term:
        yield term


def anatomical_runs(uberon: Path, lexicon: Path) -> set[tuple[str, ...]]:
    """Word runs (1-4 words) that name a structure: UBERON names and the project lexicon."""
    names: set[str] = set()
    if uberon.exists():
        with uberon.open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("name: "):
                    names.add(line[6:].strip())
    data = json.loads(lexicon.read_text("utf-8"))
    for entry in data["classes"].values():
        for key in ("en", "it"):
            names.update(entry.get(key, ()))
    runs = {tokens(n) for n in names}
    return {r for r in runs if 1 <= len(r) <= 4 and (len(r) > 1 or len(r[0]) >= 4)}


def contains_a_run(words: tuple[str, ...], runs: set[tuple[str, ...]]) -> bool:
    for start in range(len(words)):
        for stop in range(start + 1, min(len(words), start + 4) + 1):
            if words[start:stop] in runs:
                return True
    return False


def ncit_names(path: Path, runs):
    terms = {t["id"]: t for t in read_obo(path)}
    memo: dict[str, frozenset[str]] = {}

    def ancestors(term_id: str, depth: int = 0) -> frozenset[str]:
        if term_id in memo:
            return memo[term_id]
        memo[term_id] = frozenset()
        found = {term_id}
        if depth < 50:
            for parent in terms.get(term_id, {}).get("isa", ()):
                found |= ancestors(parent, depth + 1)
        memo[term_id] = frozenset(found)
        return memo[term_id]

    for term in terms.values():
        if term["obs"]:
            continue
        above = ancestors(term["id"])
        kinds = [label for root, label in NCIT_KINDS.items() if root in above]
        if not kinds or any(
            _ITEM_CLASS.search(terms[a]["name"]) for a in above if a in terms
        ):
            continue
        for name in [term["name"], *term["syn"]]:
            if not name or _NOISE.search(name):
                continue
            if (
                kinds[0] == "assessment_tool"
                and tokens(name)[-1:]
                and tokens(name)[-1] not in _INSTRUMENT_NOUNS
            ):
                continue
            yield name, kinds[0]


def pr_names(path: Path, runs):
    for term in read_obo(path):
        if term["obs"]:
            continue
        for name in [term["name"], *term["syn"]]:
            if name:
                yield _ORGANISM_SUFFIX.sub("", name), "protein"


def build(sources, runs):
    names: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for source, generator in sources:
        for name, kind in generator:
            words = tokens(name)
            if not 2 <= len(words) <= MAX_WORDS or not contains_a_run(words, runs):
                continue
            key = " ".join(words)
            if key not in names:
                names[key] = kind
                counts[f"{source}:{kind}"] += 1
    return names, counts


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def version_of(path: Path) -> str:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("data-version: "):
                return line.split(": ", 1)[1].strip()
            if line.startswith("[Term]"):
                break
    return ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--ncit", type=Path, help="ncit.obo (NCI Thesaurus OBO edition)"
    )
    parser.add_argument("--pr", type=Path, help="Protein Ontology OBO")
    parser.add_argument("--uberon", type=Path, default=DATA / "uberon-basic.obo")
    parser.add_argument("--lexicon", type=Path, default=DATA / "anatomy_lexicon.json")
    parser.add_argument("--out", type=Path, default=DATA / "longer_names.json")
    args = parser.parse_args(argv)
    if not (args.ncit or args.pr):
        parser.error("give at least one of --ncit, --pr")
    runs = anatomical_runs(args.uberon, args.lexicon)
    sources, provenance = [], {}
    if args.ncit:
        sources.append(("ncit", ncit_names(args.ncit, runs)))
        provenance["ncit"] = {
            "data_version": version_of(args.ncit),
            "sha256": sha256(args.ncit),
            "kinds": {v: k for k, v in NCIT_KINDS.items()},
        }
    if args.pr:
        sources.append(("pr", pr_names(args.pr, runs)))
        provenance["pr"] = {
            "data_version": version_of(args.pr),
            "sha256": sha256(args.pr),
        }
    names, counts = build(sources, runs)
    out = {
        "note": (
            "Nomi di altre cose (proteina, gene, sostanza, strumento di valutazione) che contengono "
            "il nome di una struttura. Generato da scripts/build_longer_names.py; non si modifica a mano. "
            "Una menzione dentro uno di questi nomi, scritto cosi nel testo, e una parola del nome, "
            "non la struttura. I nomi di anatomia, malattie, organismi, cellule e procedure non ci "
            "sono: nominano la struttura. Il tipo viene dall'ontologia, non da una lista di parole."
        ),
        "built": datetime.datetime.now(datetime.UTC).date().isoformat(),
        "sources": provenance,
        "counts": dict(counts),
        "names": dict(sorted(names.items())),
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print(f"{len(names)} names -> {args.out}: {dict(counts)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
