#!/usr/bin/env python
"""Of what is the finding said? The reader, the model and their agreement on the two HPO samples (31 October 2026).

For every item of the HPO samples (PARROT: development ``hpo_silver_2026-10-31``, held out
``hpo_heldout_2026-10-31``), ``finding_owner.FindingOwner`` reads whether the clause is about the structure the
linker found (``here``) or uses it to place something else (``elsewhere``), with the Kintsch reader alone and with
the reader and the models (the reader's method, ``llm_reading.method_prompt``). The gold: the items of
``data/gold/finding_owner_2026-10-31/elsewhere.txt`` are ``elsewhere``, every other labelled item ``here``.

Criteria fixed before the run (``docs/linker/di_che_cosa_parla_la_frase_2026-10-31.md``):
* every ``elsewhere`` item goes to a person or is set aside (no finding at the structure): none stays ``here``;
* among the ``here`` items no finding is set aside, or at most two, said;
* the role probe (separate workflow) keeps "site or not" where it was.

Texts go to OpenRouter only because PARROT is public (``--public-data``).

    python scripts/owner_probe.py --items dev=hpo_items.jsonl --items heldout=hpo_items_heldout.jsonl \
        --models google/gemma-4-31b-it,nvidia/nemotron-3-ultra-550b-a55b \
        --endpoint https://openrouter.ai/api/v1/chat/completions --public-data --out owner_probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

LABELS = {"dev": ROOT / "data/gold/hpo_silver_2026-10-31/labels.txt",
          "heldout": ROOT / "data/gold/hpo_heldout_2026-10-31/labels.txt"}
ELSEWHERE = ROOT / "data/gold/finding_owner_2026-10-31/elsewhere.txt"


def elsewhere_pairs() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for line in ELSEWHERE.read_text("utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            name, pair = line.split("|")[:2]
            out.setdefault(name, set()).add(pair)
    return out


def gold_of(item: dict, name: str, own_labels: dict[str, str], away: dict[str, set[str]]) -> str:
    """An Italian item with its own label is not the sentence of the English one: it is "here" unless listed."""
    if item["item_id"] in own_labels:
        return "here"
    return "elsewhere" if item["pair"] in away.get(name, set()) else "here"


def table(rows: list[dict], arm: str) -> dict:
    out = {}
    for lang in ("en", "it"):
        rs = [r for r in rows if r["language"] == lang and arm in r]
        c = Counter((r["gold"], r[arm]["verdict"]) for r in rs)
        away = [r for r in rs if r["gold"] == "elsewhere"]
        here = [r for r in rs if r["gold"] == "here"]
        out[lang] = {
            "items": len(rs), "gold_elsewhere": len(away),
            "elsewhere_set_aside": c[("elsewhere", "elsewhere")], "elsewhere_to_a_person": c[("elsewhere", "unsure")],
            "elsewhere_missed": c[("elsewhere", "here")],
            "here_kept": c[("here", "here")], "here_to_a_person": c[("here", "unsure")],
            "here_set_aside_wrongly": c[("here", "elsewhere")], "gold_here": len(here),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--items", action="append", required=True, help="name=path (name: dev or heldout)")
    ap.add_argument("--models", default="")
    ap.add_argument("--endpoint", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--public-data", action="store_true")
    ap.add_argument("--cache", default="owner_probe_answers.jsonl")
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import hpo_probe as hp
    from melampo.data.report_reading import build_linker
    from melampo.memory.finding_owner import FindingOwner

    linker = build_linker()[0]
    from melampo.data.report_reading import finding_owner

    reader_only, note = finding_owner(linker)
    if reader_only is None:
        raise SystemExit(note)
    chats = {}
    models = [m for m in args.models.split(",") if m.strip()]
    if models:
        from role_probe import Cache, check_endpoint, make_chats, probe_models

        check_endpoint(args.endpoint, args.public_data)
        chats = make_chats(models, args.endpoint, Cache(Path(args.cache)), max_tokens=args.max_tokens)
        probe_models(chats)
    with_models = FindingOwner(reader_only.reader, chats) if chats else None
    away = elsewhere_pairs()

    rows = []
    for spec in args.items:
        name, path = spec.split("=", 1)
        labels = hp.read_labels(LABELS[name])
        own = {k: v for k, v in labels.items() if k.endswith(("-it", "-en"))}
        items = [json.loads(x) for x in Path(path).read_text("utf-8").splitlines() if x.strip()]

        def one(item: dict, name: str = name, own: dict = own) -> dict | None:
            at = item["clause"].find(item["mention"])
            if at < 0:
                return None
            row = {"set": name, "item_id": item["item_id"], "language": item["language"],
                   "structure": item["structure"], "gold": gold_of(item, name, own, away)}
            row["reader"] = reader_only(item["mention"], item["clause"], at, item["language"]).as_dict()
            if with_models is not None:
                row["reader+models"] = with_models(item["mention"], item["clause"], at, item["language"]).as_dict()
            return row

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows += [r for r in pool.map(one, items) if r is not None]
    report = {"by_set": {}}
    for name in sorted({r["set"] for r in rows}):
        rs = [r for r in rows if r["set"] == name]
        report["by_set"][name] = {arm: table(rs, arm) for arm in ("reader", "reader+models") if any(arm in r for r in rs)}
    report["rows"] = rows
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(report["by_set"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
