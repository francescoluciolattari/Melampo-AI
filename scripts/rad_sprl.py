#!/usr/bin/env python
"""Rad-SpRL (Datta, Roberts et al.; Mendeley Data, DOI 10.17632/yhb26hfz8n.1, CC BY 4.0): the spatial roles
of 2,000 IU X-ray (Open-i) chest reports, read into our formats.

Each ``RadSpRLRelation`` has a spatial indicator (the preposition: "in", "at", "within"), a Trajector (the
finding, or an anatomical structure) and one or more Landmarks (the anatomical place). Every Landmark is,
by the annotators' definition, a **place in a body where something is**: in our schema its role is
``structure`` (a site of a finding). That makes the 2,084 Landmarks a human-annotated test of two things
our development gold set lacks: the anatomical spans of real radiology reports (where the phrase begins and
ends) and false alarms of every reader that would call a site something else.

    python scripts/rad_sprl.py convert --xml Rad-SpRL.xml --out rad_sprl_landmarks.jsonl \
        --reports rad_sprl_reports.jsonl --gold rad_sprl_gold.jsonl

``--gold`` writes rows for ``role_probe.py`` / ``gold_set.py`` (role ``structure``; the class is not
annotated, so ``structure`` is ``UNLABELLED``, or the class of ``--classes``, a file of labelled distinct
landmark strings: ``landmark|class|relation|note``).
Attribution: Rad-SpRL, S. Datta and K. Roberts, CC BY 4.0; texts from the Indiana University chest X-ray
collection (Open-i, NLM).
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

UNLABELLED = "UNLABELLED"


def _span(tokens: list[tuple[int, int]], ts, tl) -> tuple[int, int]:
    """Character span of a token run; a discontinuous annotation ("47,49" / "1,1": "right ... lung" in
    "right and left lung") gives the span that covers its pieces (see ``_pieces``)."""
    starts, lengths = [int(x) for x in str(ts).split(",")], [int(x) for x in str(tl).split(",")]
    first, last = tokens[starts[0]], tokens[starts[-1] + lengths[-1] - 1]
    return first[0], last[0] + last[1]


def _pieces(tokens: list[tuple[int, int]], ts, tl) -> list[tuple[int, int]]:
    out = []
    for a, n in zip(str(ts).split(","), str(tl).split(","), strict=True):
        out.append(_span(tokens, int(a), int(n)))
    return out


def read(xml: Path):
    """Yield (report_id, text, [landmark dicts])."""
    root = ET.parse(xml).getroot()
    for doc in root.iter("Document"):
        text = doc.findtext("Text") or ""
        ann = doc.find("Annotations")
        tokens = [(int(t.get("cs")), int(t.get("cl"))) for t in ann.iter("Token")]
        sentences = [(int(s.get("ts")), int(s.get("tl"))) for s in ann.iter("Sentence")]
        hedges = {(int(h.get("ts")), int(h.get("tl"))) for h in ann.iter("Hedge")}
        out = []
        for rel in ann.iter("RadSpRLRelation"):
            ind = _span(tokens, rel.get("ts"), rel.get("tl"))
            traj = rel.find("Trajector")
            t_span = _span(tokens, traj.get("ts"), traj.get("tl")) if traj is not None else None
            for lm in rel.iter("Landmark"):
                s, e = _span(tokens, lm.get("ts"), lm.get("tl"))
                pieces = _pieces(tokens, lm.get("ts"), lm.get("tl"))
                ts = int(lm.get("ts").split(",")[0])
                sent = next(((a, b) for a, b in sentences if a <= ts < a + b), None)
                ss, se = _span(tokens, sent[0], sent[1]) if sent else (s, e)
                out.append({
                    "report_id": f"radsprl-{doc.get('id')}", "start": s, "end": e, "mention": text[s:e],
                    "sentence": text[ss:se], "sentence_start": ss, "indicator": text[ind[0]:ind[1]],
                    "trajector": text[t_span[0]:t_span[1]] if t_span else "",
                    "trajector_type": traj.get("type") if traj is not None else "",
                    "pieces": " ".join(text[a:b] for a, b in pieces) if len(pieces) > 1 else "",
                    # a hedge ("may", "possible") in the sentence of the landmark
                    "hedged": bool(sent) and any(sent[0] <= a < sent[0] + sent[1] for a, _ in hedges),
                })
        yield f"radsprl-{doc.get('id')}", text, out


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def load_classes(path: Path | None) -> dict[str, tuple[str, str, str]]:
    if path is None:
        return {}
    out = {}
    for line in path.read_text("utf-8").split("\n"):
        if line.strip() and not line.startswith("#"):
            name, cls, rel, note = (line.split("|") + ["", "", ""])[:4]
            out[norm(name)] = (cls.strip(), rel.strip(), note.strip())
    return out


def convert(args) -> int:
    classes = load_classes(args.classes)
    landmarks, reports = [], []
    for rid, text, lms in read(args.xml):
        reports.append({"report_id": rid, "text": text, "language": "en", "site": "iu-xray (Rad-SpRL)"})
        landmarks += lms
    args.out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in landmarks) + "\n", "utf-8")
    if args.reports:
        args.reports.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in reports) + "\n", "utf-8")
    if args.gold:
        rows = []
        for i, x in enumerate(landmarks):
            cls, rel, note = classes.get(norm(x["mention"]), (UNLABELLED, "", ""))
            rows.append({
                "item_id": f"R{i:05d}", "report_id": x["report_id"], "language": "en", "stratum": "radsprl_en",
                "mention": x["mention"], "sentence": x["sentence"], "start": x["start"], "end": x["end"],
                "structure": cls, "relation": rel or ("equal" if cls not in (UNLABELLED, "NONE_IN_CLASSES", "NOT_ANATOMY", "AMBIGUOUS") else ""),
                "role": "structure", "span_text": x["mention"], "note": note,
                "span_by": "Rad-SpRL annotators (Landmark)", "labelled_by": "Rad-SpRL (role); class: Claude, by distinct string" if cls != UNLABELLED else "Rad-SpRL (role)",
            })
        args.gold.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", "utf-8")
    distinct = Counter(norm(x["mention"]) for x in landmarks)
    print(json.dumps({"reports": len(reports), "landmarks": len(landmarks), "distinct": len(distinct),
                      "hedged": sum(x["hedged"] for x in landmarks),
                      "labelled": sum(1 for x in landmarks if norm(x["mention"]) in classes),
                      "top": distinct.most_common(25)}, indent=1))
    return 0


def build_linker():
    """The linker as the gold set evaluates it (lexicon, part table, UBERON graph, blind reader)."""
    import gold_set as gs_script

    from melampo.memory import anatomy_linker as al
    from melampo.memory.anatomy_graph import AnatomyGraph
    from melampo.memory.blind_reader import BlindReader

    lexicon, parts = gs_script._load()
    obo = ROOT / "data/linking/uberon-basic.obo"
    terms, graph = [], None
    if obo.exists():
        with obo.open(encoding="utf-8") as fh:
            terms = al.load_obo_terms(fh)
        with obo.open(encoding="utf-8") as fh:
            graph = AnatomyGraph.from_obo(fh)
    pool, equivalent = al.build_pool(lexicon, terms)
    parts_json = json.loads((ROOT / "data/linking/anatomy_parts.json").read_text("utf-8"))
    blind = BlindReader.from_sources(lexicon, terms, equivalent, graph, parts_json)
    return al.AnatomyLinker(lexicon, pool, equivalent, parts=parts, graph=graph, blind=blind)


def measure(args) -> int:
    """The linker on every landmark: accepted links against the labelled strings, abstentions, and roles
    other than a site of a finding (each one a false alarm on a place the annotators marked)."""
    from melampo.evaluation.selective_calibration import risk_upper_bound
    from melampo.memory import anatomy_linker as al
    from melampo.memory.report_state import ReportState

    linker = build_linker()
    rows = [json.loads(x) for x in args.gold.read_text("utf-8").split("\n") if x.strip()]
    reports = {r["report_id"]: r["text"] for r in (json.loads(x) for x in args.reports.read_text("utf-8").split("\n") if x.strip())}
    states: dict = {}
    tally: Counter = Counter()
    reasons: Counter = Counter()
    roles: Counter = Counter()
    unlabelled: Counter = Counter()
    wrong = []
    for r in rows:
        state = states.setdefault(r["report_id"], ReportState.parse(reports[r["report_id"]]))
        got = linker.link(r["mention"], r["sentence"], report=state, at=r["start"])
        if got.role:
            roles[got.role] += 1
        if got.status != al.ACCEPTED:
            tally["abstained"] += 1
            reasons[got.reason] += 1
            continue
        tally["accepted"] += 1
        if r["structure"] == UNLABELLED:
            tally["accepted_unlabelled"] += 1
            unlabelled[norm(r["mention"])] += 1
        elif got.cid == r["structure"]:
            tally["accepted_right"] += 1
        else:
            tally["accepted_wrong"] += 1
            wrong.append({"mention": r["mention"], "said": got.cid, "label": r["structure"], "sentence": r["sentence"]})
    judged = tally["accepted_right"] + tally["accepted_wrong"]
    report = {
        "landmarks": len(rows), **tally,
        "error_upper_bound_95": round(risk_upper_bound(tally["accepted_wrong"], judged, 0.05), 4) if judged else None,
        "roles_said": dict(roles), "abstention_reasons": dict(reasons.most_common()),
        "to_label": unlabelled.most_common(), "wrong": wrong,
    }
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), "utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("abstention_reasons", "to_label", "wrong")}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--xml", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--reports", type=Path)
    c.add_argument("--gold", type=Path)
    c.add_argument("--classes", type=Path, help="labelled distinct landmark strings: landmark|class|relation|note")
    m = sub.add_parser("measure")
    m.add_argument("--gold", type=Path, required=True)
    m.add_argument("--reports", type=Path, required=True)
    m.add_argument("--out", type=Path, default=Path("rad_sprl_measure.json"))
    args = ap.parse_args(argv)
    return convert(args) if args.cmd == "convert" else measure(args)


if __name__ == "__main__":
    raise SystemExit(main())
