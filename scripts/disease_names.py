"""The names of the profiled diseases, in English and in Italian (20 November 2026).

    python3 scripts/disease_names.py --profiles profiles.json --mondo mondo.obo --out names.json

A disease of an Italian case is written in Italian ("sospetta appendicite acuta"); its profile is named in English,
by its MONDO id. This writes, for every MONDO id of the profiles, the names that find it
(``memory.profile_library``, ``MELAMPO_DISEASE_NAMES``):

* MONDO's own name and its exact synonyms (CC BY 4.0);
* Wikidata's Italian and English labels and aliases of the item that carries the MONDO id (P5270; CC0).

Only public data are asked: MONDO ids, never a patient's words. A name shared by two diseases is kept for both,
and the library then chooses neither (``ambiguous_name``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from melampo.memory import disease_profiles as dp  # noqa: E402

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
BATCH = 150


def query(ids: list[str]) -> str:
    """The Wikidata items carrying these MONDO ids (P5270, written with or without the prefix), with their Italian
    and English labels and aliases."""
    values = " ".join(f'"{i}" "{i.split(":", 1)[1]}"' for i in ids)
    return ("SELECT ?mondo ?name WHERE { VALUES ?mondo { " + values + " } ?item wdt:P5270 ?mondo . "
            "{ ?item rdfs:label ?name } UNION { ?item skos:altLabel ?name } "
            "FILTER(LANG(?name) = 'it' || LANG(?name) = 'en') }")


def read(payload: dict) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for row in payload.get("results", {}).get("bindings", []):
        mondo = row.get("mondo", {}).get("value", "").strip()
        name = row.get("name", {}).get("value", "").strip()
        if mondo and name:
            out.setdefault(mondo if mondo.startswith("MONDO:") else f"MONDO:{mondo}", set()).add(name)
    return out


def ask(sparql: str) -> dict:
    url = f"{WIKIDATA_SPARQL}?{urllib.parse.urlencode({'query': sparql, 'format': 'json'})}"
    req = urllib.request.Request(url, headers={"User-Agent": "melampo-disease-names/1.0 (open data)",
                                               "Accept": "application/sparql-results+json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            if attempt == 2:
                print(f"wikidata: {error}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profiles", required=True, nargs="+")
    ap.add_argument("--mondo", required=True)
    ap.add_argument("--out", default="names.json")
    ap.add_argument("--no-wikidata", action="store_true")
    args = ap.parse_args(argv)
    ids: set[str] = set()
    for path in args.profiles:
        data = json.loads(Path(path).read_text("utf-8"))
        ids |= {p["disease_id"] for p in data.get("profiles", []) if str(p.get("disease_id", "")).startswith("MONDO:")}
    mondo = dp.Terms.from_obo(Path(args.mondo).read_text("utf-8", errors="replace").splitlines())
    names: dict[str, set[str]] = {i: {mondo.names.get(i, "")} | set(mondo.synonyms.get(i, [])) for i in ids}
    found = 0
    if not args.no_wikidata:
        ordered = sorted(ids)
        for start in range(0, len(ordered), BATCH):
            for mondo_id, words in read(ask(query(ordered[start:start + BATCH]))).items():
                if mondo_id in names:
                    names[mondo_id] |= words
                    found += 1
    out = {i: sorted(w for w in words if w) for i, words in sorted(names.items())}
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), "utf-8")
    print(json.dumps({"diseases": len(out), "with_wikidata": found,
                      "names": sum(len(v) for v in out.values())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
