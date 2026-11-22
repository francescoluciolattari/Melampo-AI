"""Load a version of the reviewed knowledge into FalkorDB (22 November 2026).

    python3 scripts/knowledge_graph.py --knowledge knowledge/disease_profiles

The version is the one ``memory.profile_library`` computes from ``profiles.json`` and ``names.json`` (the hash of
their content): the same id in the graph, on the branch and in every case read with it. A version already in the
graph is not loaded again; a graph that lost its data is loaded again. The server is the one of
``data/falkordb_config.toml``, or ``MELAMPO_FALKORDB_SERVICE`` ("host:port") with ``MELAMPO_FALKORDB_PASSWORD``.
Only public knowledge is written there, never a patient's case.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from melampo.memory import disease_knowledge_graph as kg  # noqa: E402
from melampo.memory.profile_library import ProfileLibrary  # noqa: E402


def main(argv=None, graph=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--knowledge", required=True, help="the folder with profiles.json, names.json, tests.json")
    args = ap.parse_args(argv)
    folder = Path(args.knowledge)
    profiles, names = folder / "profiles.json", folder / "names.json"
    if not profiles.exists():
        print(json.dumps({"loaded": False, "why": f"{profiles} not found"}))
        return 0
    version = ProfileLibrary.load(profiles, names if names.exists() else None).version
    if graph is None:
        from melampo.memory.falkordb_connection import connect

        graph = connect().select_graph(kg.GRAPH)
    already = kg.has_version(graph, version)
    reviews = json.loads((folder / "reviews.json").read_text("utf-8")) if (folder / "reviews.json").exists() else {}
    date = max((r.get("date", "") for r in reviews.values()), default="") or datetime.date.today().isoformat()
    result = kg.load_version(graph, version, json.loads(profiles.read_text("utf-8"))["profiles"], date=date,
                             names=json.loads(names.read_text("utf-8")) if names.exists() else {},
                             tests=json.loads((folder / "tests.json").read_text("utf-8"))
                             if (folder / "tests.json").exists() else {})
    print(json.dumps({"version": version, "already_there": already, **result,
                      "versions": len(kg.versions(graph))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
