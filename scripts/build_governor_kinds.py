#!/usr/bin/env python
"""The kind of the words that govern a mention, from vocabularies that have them as written (14 October 2026).

Why: the Kintsch reader is given the word the mention depends on (``melampo.memory.governor``); the word
decides the reading only when a vocabulary has it with a kind (Frank: deterministic only where the
vocabulary has the correspondence). NCIt types many of these everyday words wrongly or not at all
("pain" and "appearance" as procedures; "radiograph", "view", "extraction" missing). Two sources:

* **RadLex** (RSNA, through the BioPortal API, ``BIOPORTAL_API_KEY``): the radiology lexicon. The kind
  of a word is read from the RadLex branch its exact-match class sits under (procedure, imaging
  modality -> procedure; imaging observation, clinical finding -> finding; anatomical entity -> anatomy;
  device, implant -> device; property -> property).
* **UMLS** (``UMLS_API_KEY``, the UTS REST API): the concepts whose name is the word exactly, and their
  semantic types, mapped to the kinds of the reader (``TUI_KIND``).

A word gets a kind only when every exact match that gives a kind gives the same one, in each source and
across the two; otherwise it is left out (a word with two kinds is not a correspondence). The table is
built where the keys are (GitHub Actions) and is **not committed and not uploaded**: UMLS content is not
redistributed (licence), and RadLex derivatives carry the RSNA attribution. Answers are cached on disk.

    python scripts/build_governor_kinds.py --gold silver/gold.jsonl rad_gold.jsonl --out governor_kinds.json
    python scripts/build_governor_kinds.py --words words.txt --out governor_kinds.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

UMLS_SEARCH = "https://uts-ws.nlm.nih.gov/rest/search/current"
UMLS_CONCEPT = "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}"
BIOPORTAL = "https://data.bioontology.org"

# UMLS semantic types -> the kinds of the reader (chunk_lattice.CLASS_OF_KIND). "disease" stands for
# what a structure has or contains (a finding, a disease, a body substance): the structure stays the
# structure the text speaks of. Types not listed (qualitative, functional, spatial concepts, activities)
# give no kind: too wide to decide a reading.
TUI_KIND = {
    **dict.fromkeys(("T058", "T059", "T060", "T061", "T063"), "procedure"),
    **dict.fromkeys(("T074", "T075", "T203"), "device"),
    **dict.fromkeys(("T032", "T081", "T201"), "property"),
    **dict.fromkeys(("T038", "T039", "T040", "T042", "T043", "T044", "T045"), "process"),
    **dict.fromkeys(("T019", "T020", "T031", "T033", "T037", "T046", "T047", "T048", "T049", "T050",
                     "T184", "T190", "T191"), "disease"),
    **dict.fromkeys(("T017", "T018", "T021", "T022", "T023", "T024", "T025", "T026", "T029", "T030"), "anatomy"),
    **dict.fromkeys(("T001", "T002", "T004", "T005", "T007"), "organism"),
    **dict.fromkeys(("T109", "T116", "T121", "T123", "T126", "T127", "T129", "T131", "T195", "T196",
                     "T197", "T200"), "chemical"),
    **dict.fromkeys(("T087", "T028", "T085", "T086"), "gene"),
}

# RadLex branches (ancestor labels, lower case) -> kinds; the first match in this order wins.
RADLEX_BRANCHES = (
    ("device", "device"), ("implant", "device"),
    ("imaging modality", "procedure"), ("procedure", "procedure"),
    ("imaging observation", "disease"), ("clinical finding", "disease"), ("pathophysiologic", "disease"),
    ("anatomical entity", "anatomy"),
    ("property", "property"),
)


class Cache:
    def __init__(self, path: Path | None):
        self.path, self.data = path, {}
        if path and path.exists():
            self.data = json.loads(path.read_text("utf-8"))

    def get(self, url_key: str, fetch):
        if url_key not in self.data:
            self.data[url_key] = fetch()
            if self.path:
                self.path.write_text(json.dumps(self.data), "utf-8")
        return self.data[url_key]


def _get(url: str, headers: dict | None = None, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {}
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    return {}


def _one(kinds: list[str]) -> str | None:
    """The kind when every match that gives a kind agrees; None otherwise."""
    found = {k for k in kinds if k}
    return found.pop() if len(found) == 1 else None


# UMLS sources searched: no SNOMED CT (Italy is not a SNOMED International member: its use needs an
# affiliate licence), only sources usable under the UMLS licence alone.
UMLS_SABS = "MSH,NCI,HPO,MTH,CHV,LNC,MEDLINEPLUS"


def umls_kinds(word: str, key: str, cache: Cache, get=_get, sabs: str = UMLS_SABS) -> list[str]:
    """The kinds of the UMLS concepts whose name is the word exactly."""
    q = urllib.parse.urlencode({"string": word, "searchType": "exact", "apiKey": key, "pageSize": 25,
                                "sabs": sabs})
    found = cache.get(f"umls:search:{word}", lambda: get(f"{UMLS_SEARCH}?{q}"))
    cuis = [r["ui"] for r in (found.get("result") or {}).get("results", []) if r.get("ui") not in (None, "NONE")]
    out = []
    for cui in cuis:
        concept = cache.get(f"umls:cui:{cui}", lambda cui=cui: get(UMLS_CONCEPT.format(cui=cui) + "?apiKey=" + key))
        tuis = [st.get("uri", "").rsplit("/", 1)[-1] for st in (concept.get("result") or {}).get("semanticTypes", [])]
        out.append(_one([TUI_KIND.get(t) for t in tuis]) or "")
    return out


def radlex_kinds(word: str, key: str, cache: Cache, get=_get) -> list[str]:
    """The kinds of the RadLex classes whose label is the word exactly (from the branch above them)."""
    headers = {"Authorization": f"apikey token={key}"}
    q = urllib.parse.urlencode({"q": word, "ontologies": "RADLEX", "require_exact_match": "true",
                                "include": "prefLabel,synonym"})
    found = cache.get(f"radlex:search:{word}", lambda: get(f"{BIOPORTAL}/search?{q}", headers))
    out = []
    for cls in found.get("collection", []):
        cid = cls.get("@id", "")
        if not cid:
            continue
        enc = urllib.parse.quote(cid, safe="")
        anc = cache.get(f"radlex:anc:{cid}",
                        lambda enc=enc: get(f"{BIOPORTAL}/ontologies/RADLEX/classes/{enc}/ancestors", headers))
        labels = [str(a.get("prefLabel", "")).lower() for a in (anc if isinstance(anc, list) else [])]
        kind = ""
        for needle, k in RADLEX_BRANCHES:
            if any(needle in lab for lab in labels):
                kind = k
                break
        out.append(kind)
    return out


def governor_words(gold_paths: list[Path]) -> list[str]:
    """The lemmas that govern the mentions of the gold rows (their sentences only, never their labels)."""
    from melampo.memory.governor import GovernorParser

    parser = GovernorParser()
    words: Counter[str] = Counter()
    for path in gold_paths:
        for line in Path(path).read_text("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sentence, mention = row.get("sentence", ""), row.get("mention", "")
            at = sentence.find(mention)
            if not sentence or not mention or at < 0:
                continue
            try:
                g = parser.governor(sentence, at, at + len(mention), row.get("language", "en"))
            except (ValueError, OSError):
                continue
            if g is not None and g.pos in ("NOUN", "PROPN", "VERB"):
                words[g.lemma] += 1
    return sorted(words)


def build(words: list[str], umls_key: str | None, bioportal_key: str | None, cache: Cache,
          get=_get) -> dict:
    kinds, evidence, conflicts = {}, {}, []
    for word in words:
        ev = {}
        if bioportal_key:
            ev["radlex"] = radlex_kinds(word, bioportal_key, cache, get)
        if umls_key:
            ev["umls"] = umls_kinds(word, umls_key, cache, get)
        per_source = {s: _one(v) for s, v in ev.items() if any(v)}
        mixed = {s for s, v in ev.items() if any(v) and _one(v) is None}
        evidence[word] = ev
        if mixed or not per_source:
            if mixed:
                conflicts.append(word)
            continue
        kind = _one(list(per_source.values()))
        if kind is None:
            conflicts.append(word)
            continue
        kinds[word] = kind
    return {"kinds": dict(sorted(kinds.items())), "evidence": evidence, "conflicts": sorted(conflicts)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gold", type=Path, nargs="*", default=[], help="gold rows whose sentences give the governing words")
    ap.add_argument("--words", type=Path, help="one word per line, in addition")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("governor_kinds_cache.json"))
    args = ap.parse_args(argv)
    umls_key, bioportal_key = os.environ.get("UMLS_API_KEY"), os.environ.get("BIOPORTAL_API_KEY")
    if not umls_key and not bioportal_key:
        raise SystemExit("set UMLS_API_KEY and/or BIOPORTAL_API_KEY")
    words = governor_words(args.gold) if args.gold else []
    if args.words:
        words = sorted(set(words) | {w.strip().lower() for w in args.words.read_text("utf-8").splitlines() if w.strip()})
    table = build(words, umls_key, bioportal_key, Cache(args.cache))
    table["meta"] = {
        "built": time.strftime("%Y-%m-%d"),
        "sources": [s for s, k in (("RadLex via BioPortal", bioportal_key), ("UMLS (current)", umls_key)) if k],
        "words": len(words), "typed": len(table["kinds"]), "conflicts": len(table["conflicts"]),
        "attribution": "RadLex kinds are based on the RadLex® Ontology developed at the Radiological Society of "
                       "North America; UMLS content is used under the UMLS licence and is not redistributed.",
    }
    args.out.write_text(json.dumps(table, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps({**table["meta"], "kinds_by_type": dict(Counter(table["kinds"].values()))}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
