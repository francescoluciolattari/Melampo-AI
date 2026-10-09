"""E1: recount the external check at the level of the concept, not of the identifier.

The external check calls a link an error when the label's identifier is not one of the linked
structure's nodes (or a part or kind of it). Read by hand, the 33 errors of 9 October 2026 include
links that are right and labels that only differ in **how the identifier was chosen**:

* CRAFT labels "right middle lobe" with *anatomical lobe* and "bladder" with *bladder organ*: the
  label is a kind of which our structure is an instance. The text names the finer structure, the
  annotators chose a coarser class (``coarser_label``).
* MedMentions' "aortic arch" is C0003489, and UBERON cross-references that CUI to *pharyngeal arch
  artery*, an embryonic structure. A crosswalk through the FMA or NCIt codes UMLS gives for the CUI
  may land on another UBERON node.

This script recounts the judged links with two rules, each reported apart with every row it moves,
so anyone can disagree with a rule and recount. It changes nothing in the linker and does not touch
the original numbers.

* ``label_coarser``: the label is an ancestor (kind or whole) of the linked structure. The label
  neither confirms nor contradicts the finer structure, and the ontologies themselves disagree on
  such names (UBERON lists "bladder" as a broad synonym of *urinary bladder* and a narrow one of
  *bladder organ*, and "large bowel" as an exact synonym of *colon*, where UMLS has it as *large
  intestine*). These links are taken out of the judged ones (not judged), as the external check
  already does for anatomical CUIs that no UBERON term cross-references; they are not counted as
  right.
* ``crosswalk``: for MedMentions, the label's CUI is mapped to UBERON also through the FMA and NCI
  codes UMLS lists for it (``UMLS_API_KEY``); if that mapping makes the label the structure or a
  part or kind of it, the link agrees by concept. The other direction is checked too: right links
  whose CUI the second crosswalk maps elsewhere are listed as *contradicted*.

Every error row also gets the UMLS name and semantic types of its labels (when the key is set), so
the error list can be read without a UMLS browser.

    python scripts/concept_check.py --rows external_check.rows.json --uberon uberon-basic.obo \
        --out concept_check.json --markdown concept_check.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from melampo.memory.anatomy_graph import AnatomyGraph  # noqa: E402

GOOD = {"same", "finer_label"}


def xref_index(obo: Path) -> dict[str, set[str]]:
    """``FMA:3768`` / ``NCIT:C12345`` / ``UMLS:C0003489`` -> UBERON nodes that cross-reference it."""
    found: dict[str, set[str]] = defaultdict(set)
    current = None
    with open(obo, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("[Term]") or line.startswith("[Typedef]"):
                current = None
            elif line.startswith("id: UBERON:"):
                current = line[4:].strip()
            elif current and line.startswith("xref: "):
                ref = line.split()[1]
                prefix = ref.split(":", 1)[0]
                if prefix in ("FMA", "NCIT", "UMLS"):
                    found[ref].add(current)
    return found


def _ancestors(graph: AnatomyGraph, node: str, depth: int = 14) -> set[str]:
    seen, frontier = set(), {node}
    for _ in range(depth):
        frontier = {p for n in frontier for p, _ in graph.parents.get(n, ()) if p not in seen}
        if not frontier:
            break
        seen |= frontier
    return seen


def compare(graph: AnatomyGraph, cid: str, label_node: str) -> str:
    nodes = graph.nodes_of(cid)
    if label_node in nodes:
        return "same"
    if nodes & _ancestors(graph, label_node):
        return "finer_label"
    if any(label_node in _ancestors(graph, n) for n in nodes):
        return "coarser_label"
    return "other_anatomy"


def _best(verdicts) -> str | None:
    for v in ("same", "finer_label", "coarser_label", "other_anatomy"):
        if v in verdicts:
            return v
    return None


def second_crosswalk(cui: str, lookup, index) -> tuple[set[str], list[str]] | None:
    """UBERON nodes reached from the CUI through its FMA and NCI codes (``None`` if a lookup was lost)."""
    nodes, path = set(), []
    for sab, prefix in (("FMA", "FMA"), ("NCI", "NCIT")):
        codes = lookup.codes(cui, sab)
        if codes is None:
            return None
        for code in codes:
            hit = index.get(f"{prefix}:{code}", set())
            if hit:
                nodes |= hit
                path.append(f"{prefix}:{code}->{','.join(sorted(hit))}")
    return nodes, path


def recount(rows, graph: AnatomyGraph, index, lookup=None) -> dict:
    judged = [r for r in rows if r.get("status") == "accepted" and r.get("by_project_rule") in ("agrees", "error")]
    moved, contradicted, names = [], [], {}
    lost = 0
    for r in judged:
        labels = r.get("labels") or []
        if r["by_project_rule"] == "error":
            if r.get("outcome") == "coarser_label":
                moved.append({**_brief(r), "rule": "label_coarser", "path": labels})
                continue
            if r["corpus"] == "medmentions" and lookup is not None and lookup.available:
                verdicts, paths = set(), []
                for label in labels:
                    cui = label.split("|", 1)[0]
                    got = second_crosswalk(cui, lookup, index)
                    if got is None:
                        lost += 1
                        continue
                    nodes, path = got
                    verdicts |= {compare(graph, r["cid"], n) for n in nodes}
                    paths += path
                best = _best(verdicts)
                if best in GOOD:
                    moved.append({**_brief(r), "rule": "crosswalk", "verdict": best, "path": paths})
                    continue
                if best == "coarser_label":
                    moved.append({**_brief(r), "rule": "label_coarser", "path": paths})
                    continue
        elif r["corpus"] == "medmentions" and lookup is not None and lookup.available:
            # the other direction: a right link the second crosswalk maps to an unrelated structure
            for label in labels:
                cui = label.split("|", 1)[0]
                got = second_crosswalk(cui, lookup, index)
                if got is None:
                    lost += 1
                    continue
                nodes, path = got
                verdicts = {compare(graph, r["cid"], n) for n in nodes}
                if verdicts and not (verdicts & (GOOD | {"coarser_label"})):
                    contradicted.append({**_brief(r), "path": path})
    if lookup is not None and lookup.available:
        for r in judged:
            if r["by_project_rule"] != "error":
                continue
            for label in r.get("labels") or []:
                cui = label.split("|", 1)[0]
                if r["corpus"] == "medmentions" and cui not in names:
                    info = lookup.concept(cui)
                    names[cui] = info if info is not None else {"lost": True}
    by_corpus = {}
    for corpus in sorted({r["corpus"] for r in judged}):
        sub = [r for r in judged if r["corpus"] == corpus]
        errors = sum(r["by_project_rule"] == "error" for r in sub)
        rules = Counter(m["rule"] for m in moved if m["corpus"] == corpus)
        judged_after = len(sub) - rules["label_coarser"]
        errors_after = errors - sum(rules.values())
        by_corpus[corpus] = {
            "judged": len(sub),
            "errors_by_identifier": errors,
            "precision_by_identifier": round(1 - errors / len(sub), 4) if sub else None,
            "moved": dict(rules),
            "judged_by_concept": judged_after,
            "errors_by_concept": errors_after,
            "precision_by_concept": round(1 - errors_after / judged_after, 4) if judged_after else None,
            "contradicted_right_links": sum(c["corpus"] == corpus for c in contradicted),
        }
    return {
        "by_corpus": by_corpus,
        "moved": moved,
        "contradicted": contradicted,
        "label_names": names,
        "lost_lookups": lost,
        "umls": bool(lookup is not None and lookup.available),
    }


def _brief(r) -> dict:
    return {
        "corpus": r["corpus"],
        "mention": r["mention"],
        "cid": r["cid"],
        "labels": r.get("labels"),
        "label_texts": r.get("label_texts"),
        "outcome": r.get("outcome"),
        "sentence": r["sentence"][:200],
    }


def markdown(report: dict) -> str:
    lines = [
        "# E1: errors by identifier and by concept",
        "",
        f"UMLS crosswalk: {'on' if report['umls'] else 'off (UMLS_API_KEY not set)'}; lookups lost: {report['lost_lookups']}.",
        "",
        "| corpus | judged | errors by identifier | label coarser (not judged) | second crosswalk (agrees) | judged by concept | errors by concept | precision: identifier -> concept | right links contradicted |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for corpus, v in report["by_corpus"].items():
        lines.append(
            f"| {corpus} | {v['judged']} | {v['errors_by_identifier']} | {v['moved'].get('label_coarser', 0)} "
            f"| {v['moved'].get('crosswalk', 0)} | {v['judged_by_concept']} | {v['errors_by_concept']} "
            f"| {v['precision_by_identifier']} -> {v['precision_by_concept']} | {v['contradicted_right_links']} |"
        )
    lines += ["", "## Moved rows (read them: each rule can be disputed)", ""]
    for m in report["moved"]:
        lines.append(f"- {m['corpus']} `{m['mention']}` -> {m['cid']}; label {m['labels']} ({m['rule']}; {m.get('path')}): {m['sentence'][:140]}")
    lines += ["", "## Right links the second crosswalk contradicts", ""]
    for c in report["contradicted"][:80]:
        lines.append(f"- {c['corpus']} `{c['mention']}` -> {c['cid']}; label {c['labels']} ({c['path']}): {c['sentence'][:140]}")
    if report["label_names"]:
        lines += ["", "## Names of the labels of the errors (UMLS)", "", "| CUI | name | types |", "|---|---|---|"]
        for cui, info in sorted(report["label_names"].items()):
            lines.append(f"| {cui} | {info.get('name', '?')} | {' '.join(info.get('types', []))} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    from umls_lookup import UmlsLookup

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", required=True, help="external_check rows json")
    ap.add_argument("--uberon", default=str(ROOT / "data" / "linking" / "uberon-basic.obo"))
    ap.add_argument("--umls-cache", default="umls_cache.json")
    ap.add_argument("--out", default="concept_check.json")
    ap.add_argument("--markdown", default="concept_check.md")
    args = ap.parse_args(argv)
    rows = json.loads(Path(args.rows).read_text("utf-8"))
    with open(args.uberon, encoding="utf-8") as handle:
        graph = AnatomyGraph.from_obo(handle)
    lookup = UmlsLookup(cache_path=Path(args.umls_cache))
    report = recount(rows, graph, xref_index(Path(args.uberon)), lookup)
    lookup.save()
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    Path(args.markdown).write_text(markdown(report), "utf-8")
    print(markdown(report)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
