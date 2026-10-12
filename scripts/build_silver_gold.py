#!/usr/bin/env python
"""Items of the development ("silver") gold set from open, anonymised or fictional texts (12 October 2026).

Strata, each with a fixed seed so that the same sources give the same items:

* ``e3c_it`` / ``e3c_en``: every BODYPART span annotated **by hand** in layer 1 of E3C (clinical cases,
  CC BY-NC). The spans come from people, not from our lexicon: the sample does not depend on what the
  linker knows (no selection bias towards names it already recognises).
* ``parrot_it``: one sentence per report from PARROT v1.0 Italian reports (fictional, written by
  radiologists, CC BY-NC-SA), stratified by modality; every anatomical expression in the sentence is an
  item (annotated by the labeller, not proposed by the linker).
* ``iuxray_en``: one sentence per report from IU X-ray (Open-i, real, anonymised chest X-ray reports).

    python scripts/build_silver_gold.py --e3c E3C-Corpus --parrot PARROT_v1.0 --iuxray iuxray.jsonl \
        --out data/gold/silver_2026-10-12

Writes ``items.jsonl`` (span items of E3C) and ``sentences.jsonl`` (PARROT and IU X-ray sentences to be
annotated in full) plus ``reports.jsonl`` (report id -> text, for the linker's report state).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melampo.evaluation.gold_set import sentence_of  # noqa: E402

XMI = "{http://www.omg.org/XMI}"
CAS = "{http:///uima/cas.ecore}"
CUSTOM = "{http:///webanno/custom.ecore}"
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý])")


def e3c_spans(folder: Path):
    """(doc id, text, [(begin, end)]) of the BODYPART annotations of each layer-1 document."""
    for path in sorted(folder.glob("*.xml")):
        root = ET.parse(path).getroot()
        sofa = root.find(f"{CAS}Sofa")
        if sofa is None:
            continue
        text = sofa.get("sofaString", "")
        spans = sorted({(int(e.get("begin")), int(e.get("end"))) for e in root.iter(f"{CUSTOM}BODYPART")})
        yield path.stem, text, spans


def sentences(text: str) -> list[tuple[int, str]]:
    out, pos = [], 0
    for piece in re.split(r"\n+", text):
        for s in _SENT.split(piece):
            s2 = s.strip()
            if len(s2.split()) >= 4:
                out.append((text.find(s2, pos), s2))
        pos += len(piece) + 1
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--e3c", type=Path, required=True)
    ap.add_argument("--parrot", type=Path, required=True)
    ap.add_argument("--iuxray", type=Path, required=True)
    ap.add_argument("--e3c-en", type=int, default=150, help="English E3C spans to sample")
    ap.add_argument("--sentences", type=int, default=100, help="sentences per report stratum")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    items, reports, to_annotate = [], {}, []
    for lang, take in (("Italian", None), ("English", args.e3c_en)):
        code = "it" if lang == "Italian" else "en"
        pool = []
        for doc, text, spans in e3c_spans(args.e3c / "data_annotation" / lang / "layer1"):
            rid = f"e3c-{doc}"
            reports[rid] = {"text": text, "source": "E3C layer 1", "licence": "CC BY-NC"}
            for b, e in spans:
                pool.append({"report_id": rid, "language": code, "start": b, "end": e, "mention": text[b:e],
                             "sentence": sentence_of(text, b, e), "stratum": f"e3c_{code}",
                             "span_by": "E3C annotators (manual)"})
        if take and len(pool) > take:
            pool = random.Random(20261012).sample(pool, take)
        items += pool
    # PARROT Italian, one sentence per report, stratified by modality
    rows = [json.loads(x) for x in args.parrot.read_text("utf-8").split("\n") if x.strip()]
    italian = [r for r in rows if r["language"] == "Italian"]
    by_mod = defaultdict(list)
    for r in italian:
        by_mod[r["modality"]].append(r)
    rng = random.Random(20261012)
    chosen = []
    while len(chosen) < args.sentences and any(by_mod.values()):
        for mod in sorted(by_mod):
            if by_mod[mod] and len(chosen) < args.sentences:
                chosen.append(by_mod[mod].pop(rng.randrange(len(by_mod[mod]))))
    for r in chosen:
        rid = f"parrot-{r['no']}"
        text = r["report"]
        reports[rid] = {"text": text, "source": "PARROT v1.0", "licence": "CC BY-NC-SA 4.0",
                        "modality": r["modality"], "area": r["area"]}
        sents = [s for s in sentences(text) if not s[1].lower().startswith(("quesito", "dati tecnici", "tecnica"))]
        if sents:
            off, s = sents[rng.randrange(len(sents))]
            to_annotate.append({"report_id": rid, "language": "it", "sentence": s, "offset": off,
                                "stratum": "parrot_it", "modality": r["modality"]})
    # IU X-ray, one sentence per report
    iu = [json.loads(x) for x in args.iuxray.read_text("utf-8").split("\n") if x.strip()]
    rng2 = random.Random(20261012)
    for r in rng2.sample(iu, len(iu)):
        if sum(1 for t in to_annotate if t["stratum"] == "iuxray_en") >= args.sentences:
            break
        sents = sentences(r["text"])
        if not sents:
            continue
        off, s = sents[rng2.randrange(len(sents))]
        reports[r["report_id"]] = {"text": r["text"], "source": "IU X-ray (Open-i)", "licence": "NLM, public"}
        to_annotate.append({"report_id": r["report_id"], "language": "en", "sentence": s, "offset": off,
                            "stratum": "iuxray_en", "modality": "RX"})
    for i, it in enumerate(items):
        it["item_id"] = f"S{i:04d}"
    (args.out / "items.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items) + "\n", "utf-8")
    (args.out / "sentences.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in to_annotate) + "\n", "utf-8")
    (args.out / "reports.jsonl").write_text(
        "\n".join(json.dumps({"report_id": k, **v}, ensure_ascii=False) for k, v in reports.items()) + "\n", "utf-8")
    print(f"{len(items)} span items, {len(to_annotate)} sentences, {len(reports)} reports -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
