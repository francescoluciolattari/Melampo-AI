#!/usr/bin/env python
"""Does the local model help the report router (8 November 2026)? A measure before it is turned on.

The router says which kinds of report a text is (laboratory, electrophoresis, microbiology, pathology, imaging
report, clinical note, other), and the kinds choose the interpreters (``report_router``). Today the deterministic
cues decide alone; with ``MELAMPO_REPORT_ROUTER=model`` the local model's answer is joined to them. This probe reads
public texts of known kinds both ways and counts what matters:

* **an interpreter missed**: a kind of the label whose interpreters are not all run -- what only that interpreter
  reads is lost (the error: every interpreter run that is not needed costs nothing and deletes nothing);
* **an interpreter added** that no kind of the label needs (cost, not error);
* the model's answers that cannot be used, and the decisions the router sends to review.

Texts (public, never patients'): MTSamples (transcribed reports of many specialties, English; scraped from
mtsamples.com and published as CC0 on Kaggle, mirrored on Hugging Face as ``argilla/medical-domain``) and PARROT
(radiology reports in their original language, Italian included). The texts are not committed: labels only, in
``data/gold/router/labels.txt``, written before any reading.

    python scripts/router_probe.py sample --mtsamples medical-domain.parquet --parrot PARROT_v1_0.jsonl \\
        --seed 20261108 --out router_items.jsonl
    python scripts/router_probe.py run --items router_items.jsonl --labels data/gold/router/labels.txt \\
        --models google/gemma-4-31b-it,nvidia/nemotron-3-ultra-550b-a55b \\
        --endpoint https://openrouter.ai/api/v1 --public-data --out router_probe.json

Criteria, fixed before the labels (VALIDATION): with the model, at most 2% of the texts with an interpreter
missed (and the 95% upper bound reported); at most 2% of the model's answers unusable; the router with the model
never misses an interpreter the cues alone ran (by construction: it runs the union). Then
``MELAMPO_REPORT_ROUTER=model`` may be turned on; otherwise it stays off and the reason is written.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from melampo.data.report_router import INTERPRETERS_OF_KIND, KINDS, route  # noqa: E402

LABELS = ROOT / "data/gold/router/labels.txt"
SEED = 20261108
# the specialties of MTSamples, a few texts from each, so that every kind of report is there
PER_SPECIALTY = 6
PARROT_ITALIAN, PARROT_OTHER = 20, 10


def _mtsamples(path: str) -> list[dict]:
    """MTSamples from the Hugging Face mirror (parquet: ``text``, ``prediction`` [{label}], ``id``) or the Kaggle
    CSV (``transcription``, ``medical_specialty``)."""
    p = Path(path)
    rows = []
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq

        for r in pq.read_table(p).to_pylist():
            label = (r.get("prediction") or [{}])[0].get("label", "")
            rows.append({"id": str(r.get("id", len(rows))), "specialty": label.strip(), "text": r.get("text") or ""})
    else:
        import csv

        with p.open(encoding="utf-8") as fh:
            for i, r in enumerate(csv.DictReader(fh)):
                rows.append({"id": str(r.get("") or i), "specialty": (r.get("medical_specialty") or "").strip(),
                             "text": r.get("transcription") or ""})
    return [r for r in rows if len(r["text"].strip()) > 80]


def _lines(text: str) -> str:
    """MTSamples puts headings inline ("FINDINGS: ... IMPRESSION: ..."): each heading on its own line, as in the
    documents a router meets."""
    text = re.sub(r"[ \t]*,?[ \t]*(?=\b[A-Z][A-Z /&-]{2,40}:)", "\n", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def sample(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    items = []
    if args.mtsamples:
        by = {}
        for r in _mtsamples(args.mtsamples):
            by.setdefault(r["specialty"], []).append(r)
        for spec in sorted(by):
            pool = sorted(by[spec], key=lambda r: r["id"])
            for r in rng.sample(pool, min(args.per_specialty, len(pool))):
                items.append({"item_id": f"mts-{r['id']}", "source": "mtsamples", "specialty": spec,
                              "language": "en", "text": _lines(r["text"])})
    if args.parrot:
        reports = [json.loads(x) for x in Path(args.parrot).read_text("utf-8").split("\n") if x.strip()]
        italian = sorted((r for r in reports if r.get("language") == "Italian"), key=lambda r: str(r["no"]))
        other = sorted((r for r in reports if r.get("language") not in ("Italian", "English")),
                       key=lambda r: str(r["no"]))
        for r in rng.sample(italian, min(PARROT_ITALIAN, len(italian))) + rng.sample(other, min(PARROT_OTHER,
                                                                                                len(other))):
            items.append({"item_id": f"parrot-{r['no']}", "source": "parrot", "specialty": r.get("modality", ""),
                          "language": r.get("language", ""), "text": r["report"]})
    Path(args.out).write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n", "utf-8")
    print(f"{len(items)} texts: {dict(Counter(i['source'] for i in items))}")
    return 0


def read_labels(path: Path) -> dict[str, set[str]]:
    """``item_id<TAB>kind+kind<TAB>note``: the kinds the text contains, written before any reading."""
    out = {}
    for line in path.read_text("utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        kinds = {k.strip() for k in parts[1].split("+") if k.strip()}
        unknown = kinds - set(KINDS)
        if unknown:
            raise SystemExit(f"{parts[0]}: unknown kinds {sorted(unknown)}")
        out[parts[0]] = kinds
    return out


def needed(kinds: set[str]) -> set[str]:
    return {i for k in kinds for i in INTERPRETERS_OF_KIND[k]}


def upper_95(errors: int, n: int) -> float:
    from owner_probe import upper_95 as u

    return u(errors, n)


def one(decision, want: set[str]) -> dict:
    ran = set(decision.interpreters)
    return {"kinds": list(decision.kinds), "interpreters": sorted(ran), "missed": sorted(needed(want) - ran),
            "added": sorted(ran - needed(want) - {"imaging"}), "agreement": decision.agreement,
            "needs_review": decision.needs_review, "notes": list(decision.notes)}


def run(args: argparse.Namespace) -> int:
    items = [json.loads(x) for x in Path(args.items).read_text("utf-8").split("\n") if x.strip()]
    labels = read_labels(Path(args.labels))
    items = [i for i in items if i["item_id"] in labels]
    if not items:
        raise SystemExit("no labelled text: the labels are written before the run")
    chats = {}
    models = [m for m in args.models.split(",") if m.strip()]
    if models:
        from role_probe import Cache, check_endpoint, make_chats, probe_models

        check_endpoint(args.endpoint, args.public_data)
        chats = make_chats(models, args.endpoint, Cache(Path(args.cache)), max_tokens=400)
        probe_models(chats)
    rows = []
    for it in items:
        want = labels[it["item_id"]]
        row = {"item_id": it["item_id"], "source": it["source"], "language": it["language"],
               "label": sorted(want), "cues": one(route(it["text"]), want)}
        for name, chat in chats.items():
            row[name] = one(route(it["text"], chat), want)
        rows.append(row)
    n = len(rows)

    def summary(arm: str) -> dict:
        got = [r[arm] for r in rows]
        missed = sum(bool(g["missed"]) for g in got)
        unusable = sum(any(x.startswith("router_model") for x in g["notes"]) for g in got)
        return {"texts": n, "missed_share": round(missed / n, 4), "missed_upper_95": upper_95(missed, n),
                "added_share": round(sum(bool(g["added"]) for g in got) / n, 4),
                "needs_review_share": round(sum(g["needs_review"] for g in got) / n, 4),
                "unusable_answers": unusable, "unusable_upper_95": upper_95(unusable, n),
                "agreement": dict(Counter(g["agreement"] for g in got)),
                "missed_by_kind": dict(Counter(i for g in got for i in g["missed"]))}

    out = {"arms": {arm: summary(arm) for arm in ["cues", *chats]}, "rows": rows,
           "criteria": "missed_share <= 0.02 and unusable_answers/texts <= 0.02 (fixed 8 November 2026)"}
    for arm in chats:
        s = out["arms"][arm]
        s["passes"] = s["missed_share"] <= 0.02 and s["unusable_answers"] / n <= 0.02
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(out["arms"], ensure_ascii=False, indent=1))
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--mtsamples", default="")
    s.add_argument("--parrot", default="")
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--per-specialty", type=int, default=PER_SPECIALTY)
    s.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--items", required=True)
    r.add_argument("--labels", default=str(LABELS))
    r.add_argument("--models", default="")
    r.add_argument("--endpoint", default="http://localhost:8000/v1")
    r.add_argument("--public-data", action="store_true",
                   help="the texts are public (MTSamples, PARROT): they may go to an external service")
    r.add_argument("--cache", default="router_probe_answers.jsonl")
    r.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return sample(args) if args.cmd == "sample" else run(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
