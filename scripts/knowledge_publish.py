"""The reviewed knowledge (22 November 2026): a run of disease-review merged into the knowledge of the latest Release.

    python3 scripts/knowledge_publish.py --run out --knowledge release

The folder is the content of a knowledge Release (``scripts/knowledge_release.py``): a version of the profiles
that can always be read again; ``main`` receives no automatic commits. A disease the run reviewed replaces its earlier version (profile, measures, tests, scores,
review date); the others are kept as they are. The logs of the readings and verdicts are joined, so the next run
reads and judges nothing twice. ``MELAMPO_DISEASE_PROFILES`` points to ``profiles.json`` of this folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# the texts read are not published: an abstract or a full text belongs to its authors and publisher (the
# repository is public); what is published are the readings, the verdicts and the short sentences cited with
# their source, as any citation. The texts stay in the run's artifact.
LOGS = ("screened.jsonl", "readings.jsonl", "verdicts.jsonl", "questions.jsonl")


def _json(path: Path, default):
    return json.loads(path.read_text("utf-8")) if path.exists() else default


def _lines(path: Path) -> list[str]:
    return [x for x in path.read_text("utf-8").splitlines() if x.strip()] if path.exists() else []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="the out folder of a disease-review run")
    ap.add_argument("--knowledge", required=True)
    args = ap.parse_args(argv)
    run, store = Path(args.run), Path(args.knowledge)
    store.mkdir(parents=True, exist_ok=True)
    reviews_run = _json(run / "reviews.json", {})
    done = set(reviews_run)                         # only the diseases whose review the run finished
    if not done:
        print(json.dumps({"published": [], "why": "the run finished no disease's review"}))
        return 0
    profiles = {p["words"]: p for p in _json(store / "profiles.json", {"profiles": []})["profiles"]}
    profiles.update({p["words"]: p for p in _json(run / "profiles.json", {"profiles": []})["profiles"]
                     if p["words"] in done})
    (store / "profiles.json").write_text(json.dumps({"profiles": [profiles[k] for k in sorted(profiles)]}, indent=1,
                                                    ensure_ascii=False), "utf-8")
    kept = [x for x in _lines(store / "measures.jsonl") if json.loads(x).get("disease") not in done]
    new = [x for x in _lines(run / "measures.jsonl") if json.loads(x).get("disease") in done]
    (store / "measures.jsonl").write_text("".join(x + "\n" for x in kept + new), "utf-8")
    for name in ("tests.json", "scores.json", "curves.json"):
        merged = _json(store / name, {})
        merged.update({k: v for k, v in _json(run / name, {}).items() if k in done})
        (store / name).write_text(json.dumps(merged, indent=1, ensure_ascii=False), "utf-8")
    reviews = _json(store / "reviews.json", {})
    reviews.update(reviews_run)
    (store / "reviews.json").write_text(json.dumps(reviews, indent=1, ensure_ascii=False), "utf-8")
    names = _json(store / "names.json", {})
    for k, v in _json(run / "names.json", {}).items():
        names[k] = sorted(set(names.get(k, [])) | set(v))
    (store / "names.json").write_text(json.dumps(names, indent=1, ensure_ascii=False), "utf-8")
    # the thesaurus (25 November 2026): every label agreed in a publication is knowledge, whichever disease's review
    # it was agreed in, finished or not
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from melampo.reading.thesaurus import FILE, Thesaurus

    thesaurus = Thesaurus.load(store / FILE)
    thesaurus.merge(Thesaurus.load(run / FILE))
    thesaurus.save(store / FILE)
    logs = store / "logs"
    logs.mkdir(exist_ok=True)
    for name in LOGS:
        old = _lines(logs / name)
        seen = set(old)
        add = [x for x in _lines(run / name) if x not in seen]
        (logs / name).write_text("".join(x + "\n" for x in old + add), "utf-8")
    print(json.dumps({"published": sorted(done), "profiles": len(profiles), "concepts": len(thesaurus.concepts)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
