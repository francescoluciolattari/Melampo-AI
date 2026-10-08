"""Measure the context check (``verify``) on real sentences, with the two models.

The linking bench has synthetic sentences; the risk is in real ones ("heart rate", "short axis view",
"thyroid function tests"). This takes the sheets a ``public-reports`` run drew (``items.jsonl``) and
their reports, links every mention with the deterministic stages, and asks the two models, for each
link made from the name alone, whether the marked text names that structure in its sentence. It
changes nothing and certifies nothing: the answer is a list of stopped links to read, and a sample
of confirmed ones to check for misses.

  python scripts/verify_probe.py --items gold_study/items.jsonl --reports reports.jsonl \
      --scope flagged|all --out verify_probe.json --markdown verify_probe.md

Needs OPENROUTER_API_KEY. A model that does not answer (rate limit) makes the row abstain with the
reason ``model_unavailable`` and is counted as such: a run that loses rows says so at the top.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from melampo.memory import anatomy_linker as al  # noqa: E402
from melampo.memory import anatomy_parts as ap  # noqa: E402
from melampo.memory import form_ambiguity as fa  # noqa: E402
from melampo.memory.blind_reader import BlindReader  # noqa: E402
from melampo.memory.anatomy_graph import AnatomyGraph  # noqa: E402
from melampo.memory.report_state import ReportState  # noqa: E402

CHAT_MODELS = {
    "nemotron-3-super": "nvidia/nemotron-3-super-120b-a12b",
    "gemma-3-27b": "google/gemma-3-27b-it",
}
CHAT_PACING = {"retries": 8, "min_interval": 0.5, "max_wait": 60.0}


def _read_jsonl(path: Path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_linkers(uberon: Path, chats: dict | None, scope: str, style: str = "yes_no"):
    lexicon = al.Lexicon.from_json(
        json.loads((ROOT / "data/linking/anatomy_lexicon.json").read_text("utf-8"))
    )
    parts_json = json.loads(
        (ROOT / "data/linking/anatomy_parts.json").read_text("utf-8")
    )
    terms, graph = [], None
    if uberon.exists():
        with open(uberon, encoding="utf-8") as handle:
            terms = al.load_obo_terms(handle)
        with open(uberon, encoding="utf-8") as handle:
            graph = AnatomyGraph.from_obo(handle)
    pool, equivalent = al.build_pool(lexicon, terms)
    parts = ap.PartTable.from_json(parts_json, lexicon)
    flags = fa.audit(
        lexicon,
        [(n, e["whole"]) for e in parts_json["direct"] for n in e["names"]],
    )
    common = {
        "parts": parts,
        "graph": graph,
        "ambiguous": fa.keys(flags),
        "blind": BlindReader.from_sources(
            lexicon, terms, equivalent, graph, parts_json
        ),
    }
    plain = al.AnatomyLinker(lexicon, pool, equivalent, **common)
    asking = (
        al.AnatomyLinker(
            lexicon,
            pool,
            equivalent,
            chats=chats,
            verify=True,
            verify_all=scope == "all",
            verify_style=style,
            **common,
        )
        if chats
        else None
    )
    return plain, asking, flags


def probe(items, texts, plain, asking, flags, workers: int):
    def one(item):
        state = ReportState.parse(texts[item["report_id"]])
        sentence = state.sentence_at(item["start"], item["end"])
        kwargs = {"report": state, "at": item["start"]}
        before = plain.link(item["mention"], sentence, **kwargs)
        row = {
            "item_id": item["item_id"],
            "mention": item["mention"],
            "sentence": sentence,
            "language": item.get("language"),
            "before": {
                "status": before.status,
                "cid": before.cid,
                "reason": before.reason,
            },
            "flagged": al.normalise(item["mention"]) in flags,
        }
        if before.status == al.ACCEPTED and before.stage in ("lexicon", "parts"):
            after = asking.link(item["mention"], sentence, **kwargs)
            row["after"] = {
                "status": after.status,
                "cid": after.cid,
                "reason": after.reason,
                "stage": after.stage,
                "votes": after.votes,
            }
            row["asked"] = any(e.stream == "verify" for e in after.trace)
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, items))


def summarise(rows) -> dict:
    asked = [r for r in rows if r.get("asked")]
    outcome = Counter()
    for r in asked:
        a = r["after"]
        if a["status"] == al.ACCEPTED:
            outcome["confirmed"] += 1
        elif a["reason"] == "model_unavailable":
            outcome["model_unavailable"] += 1
        else:
            outcome[a["reason"]] += 1
    return {
        "items": len(rows),
        "accepted_by_name": sum(
            1 for r in rows if r["before"]["status"] == al.ACCEPTED and "after" in r
        ),
        "asked": len(asked),
        "outcome": dict(outcome),
    }


def render(rows, summary: dict, seed: int = 20261007, sample: int = 25) -> str:
    out = ["# Verify probe (real sentences, two models)", ""]
    lost = summary["outcome"].get("model_unavailable", 0)
    if lost:
        out += [
            f"**WARNING: {lost} of {summary['asked']} questions lost a model answer** "
            "(rate limit or network). Those rows are not a measurement.",
            "",
        ]
    out += [
        f"Items {summary['items']}; accepted from the name alone {summary['accepted_by_name']}; "
        f"asked {summary['asked']}.",
        "",
        "| outcome | count |",
        "|---|---|",
    ]
    out += [f"| {k} | {v} |" for k, v in sorted(summary["outcome"].items())]
    out += ["", "## Stopped by the check (read each one: was the stop right?)", ""]
    for r in rows:
        a = r.get("after")
        if (
            r.get("asked")
            and a["status"] != al.ACCEPTED
            and a["reason"] != "model_unavailable"
        ):
            out.append(
                f"- {r['item_id']} «{r['mention']}» → {r['before']['cid']} "
                f"[{a['reason']}; {a['votes']}] {r['sentence'][:220]}"
            )
    kept = [r for r in rows if r.get("asked") and r["after"]["status"] == al.ACCEPTED]
    random.Random(seed).shuffle(kept)
    out += [
        "",
        f"## Confirmed (random {min(sample, len(kept))} of {len(kept)}: any wrong link left?)",
        "",
    ]
    for r in kept[:sample]:
        out.append(
            f"- {r['item_id']} «{r['mention']}» → {r['before']['cid']} {r['sentence'][:200]}"
        )
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", required=True)
    parser.add_argument("--reports", required=True)
    parser.add_argument("--scope", choices=("flagged", "all"), default="flagged")
    parser.add_argument(
        "--style",
        choices=("yes_no", "choice"),
        default="yes_no",
        help="how the models are asked: a yes/no question, or five balanced options",
    )
    parser.add_argument("--uberon", default=str(ROOT / "data/linking/uberon-basic.obo"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="verify_probe.json")
    parser.add_argument("--markdown", default="verify_probe.md")
    args = parser.parse_args(argv)

    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        print(
            "OPENROUTER_API_KEY is not set: the probe needs the two models.",
            file=sys.stderr,
        )
        return 2
    from melampo.evaluation import linking_bench as lb

    chats = {
        n: lb.OpenRouterChat(s, key, **CHAT_PACING) for n, s in CHAT_MODELS.items()
    }
    items = list(_read_jsonl(Path(args.items)))
    need = {i["report_id"] for i in items}
    texts = {}
    for report in _read_jsonl(Path(args.reports)):
        if report["report_id"] in need:
            texts[report["report_id"]] = report["text"]
    plain, asking, flags = build_linkers(
        Path(args.uberon), chats, args.scope, args.style
    )
    rows = probe(items, texts, plain, asking, flags, args.workers)
    summary = summarise(rows)
    Path(args.out).write_text(
        json.dumps(
            {
                "scope": args.scope,
                "style": args.style,
                "summary": summary,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    Path(args.markdown).write_text(
        f"Scope `{args.scope}`, question style `{args.style}`.\n\n"
        + render(rows, summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
