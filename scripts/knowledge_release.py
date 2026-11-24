"""A version of the disease knowledge, ready for a GitHub Release (22 November 2026).

    python3 scripts/knowledge_release.py --previous previous --run out --release release

The versions of the knowledge live in the repository's Releases, never in its history: a Release per version,
its tag ``knowledge-<date>-<version>``, marked as a pre-release so that it is not taken for a release of the
software. Code changes create none: only the two workflows of the knowledge (disease-review, disease-review-update)
call this, and a Release is made only when the content changed.

The folder ``release`` is filled with:

* ``knowledge.db``: the graph, ready, in FalkorDBLite (``memory.disease_knowledge_graph``), the one the diagnosis
  opens (``MELAMPO_DISEASE_PROFILES=<path>/knowledge.db``);
* the files it is built from (``profiles.json``, ``names.json``, ``tests.json``, ``scores.json``, ``curves.json``,
  ``measures.jsonl``, ``reviews.json``, the logs of the readings and verdicts; the texts read are not published:
  an abstract or a full text belongs to its authors);
* ``changes.md``: what changed from the previous version, disease by disease;
* ``version.json``: the version (the hash ``memory.profile_library`` computes) and its tag.

It prints ``changed=true|false`` and ``tag=...`` for the workflow.
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from melampo.clinical import case_graph as cg  # noqa: E402
from melampo.memory import disease_knowledge_graph as kg  # noqa: E402
from melampo.memory.profile_library import ProfileLibrary, norm  # noqa: E402

FILES = ("profiles.json", "names.json", "tests.json", "scores.json", "curves.json", "measures.jsonl", "reviews.json",
         "thesaurus.jsonl")


def _key(m: cg.Manifestation) -> str:
    return m.concept.curie if m.concept is not None and m.concept.code else norm(m.words)


def _profiles(folder: Path) -> dict[str, cg.Disease]:
    path = folder / "profiles.json"
    if not path.exists():
        return {}
    return {p["words"]: cg.Disease.from_dict(p) for p in json.loads(path.read_text("utf-8"))["profiles"]}


def changes(before: dict[str, cg.Disease], after: dict[str, cg.Disease], reviews: dict) -> str:
    """What changed, disease by disease: added and removed manifestations, weights that changed."""
    lines = []
    for name in sorted(set(before) | set(after)):
        old, new = before.get(name), after.get(name)
        if new is None:
            lines.append(f"## {name}\n\nRemoved.\n")
            continue
        r = reviews.get(name, {})
        head = f"## {name}\n\nReviewed {r.get('date', '')}" + (f" (from {r['since']})" if r.get("since") else "") + \
            f", {r.get('texts_read', 0)} texts read, stopped by {r.get('stopped_by', '')}.\n"
        olds = {_key(m): m for m in (old.manifestations if old else [])}
        news = {_key(m): m for m in new.manifestations}
        added = [news[k].words for k in news if k not in olds]
        removed = [olds[k].words for k in olds if k not in news]
        weights = []
        for k in news.keys() & olds.keys():
            a, b = olds[k], news[k]
            if (a.frequency, a.accuracy, a.frequency_basis) != (b.frequency, b.accuracy, b.frequency_basis):
                weights.append(f"{b.words}: {a.frequency} -> {b.frequency} ({b.frequency_basis or 'no basis'})")
        if not (added or removed or weights) and old is not None:
            continue
        lines.append(head)
        if old is None:
            lines.append(f"New: {len(news)} manifestations.\n")
        if added:
            lines.append("Added: " + "; ".join(sorted(added)) + ".\n")
        if removed:
            lines.append("Removed: " + "; ".join(sorted(removed)) + ".\n")
        if weights:
            lines.append("Weights changed:\n" + "\n".join(f"- {w}" for w in sorted(weights)) + "\n")
    return "# Changes\n\n" + ("\n".join(lines) if lines else "Nothing changed.\n")


def build_db(folder: Path, version: str, path: Path) -> dict[str, int]:
    from redislite.falkordb_client import FalkorDB

    db = FalkorDB(str(path))
    graph = db.select_graph(kg.GRAPH)
    reviews = json.loads((folder / "reviews.json").read_text("utf-8")) if (folder / "reviews.json").exists() else {}
    read = lambda n, d: json.loads((folder / n).read_text("utf-8")) if (folder / n).exists() else d  # noqa: E731
    counts = kg.load_version(graph, version, read("profiles.json", {"profiles": []})["profiles"],
                             date=max((r.get("date", "") for r in reviews.values()), default=""),
                             names=read("names.json", {}), tests=read("tests.json", {}),
                             thesaurus=[json.loads(x) for x in (folder / "thesaurus.jsonl").read_text("utf-8")
                                        .splitlines() if x.strip()] if (folder / "thesaurus.jsonl").exists() else [])
    db.close()                     # the file is written whole before it is uploaded
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--previous", default="", help="the files of the latest knowledge Release, downloaded")
    ap.add_argument("--run", required=True, help="the out folder of disease-review")
    ap.add_argument("--release", required=True, help="the folder to fill for the new Release")
    args = ap.parse_args(argv)
    previous, run, release = Path(args.previous) if args.previous else None, Path(args.run), Path(args.release)
    if release.exists():
        shutil.rmtree(release)
    release.mkdir(parents=True)
    if previous and previous.exists():
        for name in FILES + ("logs",):
            src = previous / name
            if src.is_dir():
                shutil.copytree(src, release / name)
            elif src.exists():
                shutil.copy(src, release / name)
    publish = HERE / "knowledge_publish.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("knowledge_publish", publish)
    kp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kp)
    kp.main(["--run", str(run), "--knowledge", str(release)])
    if not (release / "profiles.json").exists():
        print("changed=false")
        print("tag=")
        return 0
    names = release / "names.json"
    version = ProfileLibrary.load(release / "profiles.json", names if names.exists() else None).version
    thesaurus = release / "thesaurus.jsonl"
    if thesaurus.exists() and thesaurus.stat().st_size:      # the thesaurus is part of the version (25 November 2026)
        import hashlib

        version = hashlib.sha256((version + hashlib.sha256(thesaurus.read_bytes()).hexdigest()).encode()).hexdigest()
    old_version = ""
    if previous and (previous / "version.json").exists():
        old_version = json.loads((previous / "version.json").read_text("utf-8")).get("version", "")
    if version == old_version:
        print("changed=false")
        print("tag=")
        return 0
    tag = f"knowledge-{datetime.date.today().isoformat()}-{version[:8]}"
    reviews = json.loads((release / "reviews.json").read_text("utf-8")) if (release / "reviews.json").exists() else {}
    (release / "changes.md").write_text(changes(_profiles(previous) if previous else {}, _profiles(release), reviews),
                                        "utf-8")
    counts = build_db(release, version, release / "knowledge.db")
    (release / "version.json").write_text(json.dumps({"version": version, "tag": tag, "previous": old_version,
                                                      **counts}, indent=1), "utf-8")
    print("changed=true")
    print(f"tag={tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
