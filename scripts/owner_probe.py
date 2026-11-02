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

    python scripts/owner_probe.py sample --parrot PARROT/data/PARROT_v1_0.jsonl --out owner_items_fresh.jsonl
    python scripts/owner_probe.py --items fresh=owner_items_fresh.jsonl --items dev=hpo_items.jsonl \
        --items heldout=hpo_items_heldout.jsonl --parrot PARROT/data/PARROT_v1_0.jsonl --uberon uberon.obo \
        --models google/gemma-4-31b-it,nvidia/nemotron-3-ultra-550b-a55b \
        --endpoint https://openrouter.ai/api/v1/chat/completions --public-data --out owner_probe.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

LABELS = {"dev": ROOT / "data/gold/hpo_silver_2026-10-31/labels.txt",
          "heldout": ROOT / "data/gold/hpo_heldout_2026-10-31/labels.txt"}
# the sample built for this question (1 November 2026): its own labels, here/elsewhere, written before any run
OWNER_LABELS = ROOT / "data/gold/finding_owner_2026-11-01/labels.txt"
# the sample of the statement reading (1 November 2026): what the sentence says of the structure, by type
STATEMENT_LABELS = ROOT / "data/gold/statement_2026-11-01/labels.txt"
# the sample of the claim-by-claim reading (2 November 2026): what the sentence states of the structure
CLAIMS_LABELS = ROOT / "data/gold/claims_2026-11-02/labels.txt"
# the second claims sample (2 November 2026, after the corrections of the claims run)
CLAIMS2_LABELS = ROOT / "data/gold/claims_2026-11-02b/labels.txt"
SAMPLES = {"fresh": OWNER_LABELS, "statement": STATEMENT_LABELS, "claims": CLAIMS_LABELS, "claims2": CLAIMS2_LABELS}
CLAIMS_FILES = {"claims": CLAIMS_LABELS, "claims2": CLAIMS2_LABELS}
STATEMENT_SEED = 20261102
CLAIMS_SEED = 20261103
CLAIMS2_SEED = 20261104


def owner_labels() -> dict[str, str]:
    out = {}
    if OWNER_LABELS.exists():
        for line in OWNER_LABELS.read_text("utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                item, label = line.split("|")[:2]
                out[item.strip()] = label.strip()
    return out
ELSEWHERE = ROOT / "data/gold/finding_owner_2026-10-31/elsewhere.txt"


def elsewhere_pairs() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for line in ELSEWHERE.read_text("utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            name, pair = line.split("|")[:2]
            out.setdefault(name, set()).add(pair)
    return out


def statement_labels() -> dict[str, tuple[str, bool]]:
    """item -> (type, for a radiologist)."""
    out = {}
    if STATEMENT_LABELS.exists():
        for line in STATEMENT_LABELS.read_text("utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                item, kind, note = (line.split("|") + ["", ""])[:3]
                out[item.strip()] = (kind.strip(), note.rstrip().endswith("?"))
    return out


def claims_labels(name: str = "claims") -> dict[str, tuple[set[str], bool]]:
    """item -> (the claims of the label, for a radiologist)."""
    out = {}
    path = CLAIMS_FILES.get(name, CLAIMS_LABELS)
    if path.exists():
        for line in path.read_text("utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                item, kinds, note = (line.split("|") + ["", ""])[:3]
                out[item.strip()] = (set(kinds.strip().split("+")), note.rstrip().endswith("?"))
    return out


def outcome_of_claims(kinds: set[str]) -> str:
    """What the label attaches: a finding, nothing at the structure (aside), or only the other claims (keep)."""
    if kinds & {"finding", "involved"}:
        return "finding"
    if kinds & {"aside", "other_structure"}:
        return "aside"
    return "keep"


def attached_kinds(owner: dict) -> set[str]:
    """The claims an accepted reading attaches, in the labels' words."""
    claims = owner.get("claims") or {}
    out = set()
    for o in claims.get("findings", []):
        out.add("finding" if o.get("of") == "it" else "involved")
    for key, kind in (("normal", "normal"), ("measure", "measure"), ("past", "history"), ("device", "device"),
                      ("not_assessable", "not_assessable")):
        if claims.get(key):
            out.add(kind)
    return out or {"nothing"}


def claims_table(rows: list[dict], arm: str = "claims") -> dict:
    """The claim-by-claim reading against the labels (criteria fixed on 2 November 2026)."""
    rs = [r for r in rows if arm in r and r.get("gold_claims")]
    c: Counter = Counter()
    out: dict = {"items": len(rs), "for_a_radiologist": [], "kinds_attached_not_in_the_label": Counter(),
                 "kinds_of_the_label_not_attached": Counter(), "dropped_words": 0, "laterality": 0}
    for r in rs:
        gold = set(r["gold_claims"])
        want = outcome_of_claims(gold)
        got = r[arm]
        verdict = got["verdict"]
        c[f"{want}_items"] += 1
        if got.get("review"):
            c[f"{want}_review"] += 1
        if verdict == "unsure":
            c[f"{want}_to_a_person"] += 1
        elif verdict == "elsewhere":
            c[f"{want}_set_aside"] += 1
        else:
            have = got.get("statement", "")
            c[f"{want}_attached"] += 1
            have_outcome = "finding" if have == "finding" else "keep"
            if want != "aside":
                c["attached_as_its_outcome" if have_outcome == want else "attached_as_another_outcome"] += 1
                kinds = attached_kinds(got)
                out["kinds_attached_not_in_the_label"].update(kinds - gold - {"nothing"})
                out["kinds_of_the_label_not_attached"].update(gold - kinds - {"nothing"})
        for a in (got.get("models") or {}).values():
            out["dropped_words"] += int(a.get("dropped") or 0)
        if r.get("gold_doubt"):
            out["for_a_radiologist"].append({"item": r["item_id"], "gold": "+".join(sorted(gold)),
                                             "verdict": verdict, "statement": got.get("statement", "")})
    out.update(c)
    keep = (c["finding_items"] + c["keep_items"]) or 1
    out["not_aside_to_a_person_share"] = round((c["finding_to_a_person"] + c["keep_to_a_person"]) / keep, 4)
    attached = (c["attached_as_its_outcome"] + c["attached_as_another_outcome"]) or 1
    out["attached_as_another_outcome_share"] = round(c["attached_as_another_outcome"] / attached, 4)
    out["kinds_attached_not_in_the_label"] = dict(out["kinds_attached_not_in_the_label"])
    out["kinds_of_the_label_not_attached"] = dict(out["kinds_of_the_label_not_attached"])
    return out


def gold_of(item: dict, name: str, own_labels: dict[str, str], away: dict[str, set[str]]) -> str:
    """An Italian item with its own label is not the sentence of the English one: it is "here" unless listed."""
    if name in CLAIMS_FILES:
        kinds = claims_labels(name).get(item["item_id"], (set(), False))[0]
        return "" if not kinds else "elsewhere" if outcome_of_claims(kinds) == "aside" else "here"
    if name == "statement":
        from melampo.memory.finding_owner import ASIDE, STATEMENT_GROUPS

        kind = statement_labels().get(item["item_id"], ("", False))[0]
        return "" if not kind else "elsewhere" if STATEMENT_GROUPS.get(kind) == ASIDE else "here"
    if name == "fresh":
        return owner_labels().get(item["item_id"], "")
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


def statement_table(rows: list[dict], arm: str) -> dict:
    """The statement reading against the types of the labels (criteria of 1 November 2026): what was attached,
    and whether it is the statement the label says."""
    from melampo.memory.finding_owner import ASIDE, NOTHING, STATEMENT_GROUPS

    rs = [r for r in rows if arm in r and r.get("gold_type")]
    out: dict = {"items": len(rs), "confusion": {}, "pieces_not_in_the_sentence": 0, "answers": 0,
                 "for_a_radiologist": []}
    c = Counter()
    for r in rs:
        gold = STATEMENT_GROUPS.get(r["gold_type"], "")
        got = r[arm]
        verdict, group = got["verdict"], got.get("statement", "")
        side = "aside" if gold == ASIDE else "keep"
        c[f"{side}_items"] += 1
        if verdict == "unsure":
            c[f"{side}_to_a_person"] += 1
        elif verdict == "elsewhere":
            c[f"{side}_set_aside"] += 1
        else:
            c[f"{side}_attached"] += 1
            if side == "keep" and gold != NOTHING and group != NOTHING:
                c["keep_attached_as_its_statement" if group == gold else "keep_attached_as_another_statement"] += 1
        out["confusion"].setdefault(gold, Counter())[group if verdict != "unsure" else "person"] += 1
        for a in (got.get("models") or {}).values():
            out["answers"] += 1
            if a.get("piece") and not a.get("found", True):
                out["pieces_not_in_the_sentence"] += 1
        if r.get("gold_doubt"):
            out["for_a_radiologist"].append({"item": r["item_id"], "gold": r["gold_type"], "verdict": verdict,
                                             "statement": group})
    out.update(c)
    out["confusion"] = {k: dict(v) for k, v in out["confusion"].items()}
    keep = c["keep_items"] or 1
    out["keep_to_a_person_share"] = round(c["keep_to_a_person"] / keep, 4)
    attached = (c["keep_attached_as_its_statement"] + c["keep_attached_as_another_statement"]) or 1
    out["keep_attached_as_another_statement_share"] = round(c["keep_attached_as_another_statement"] / attached, 4)
    return out


FRESH_SEED = 20261101
LOST_LIMIT = 0.05


def build_fresh(parrot: Path, n_en: int, n_it: int, per_report: int, per_structure: int,
                exclude: set[str], seed: int = FRESH_SEED) -> list[dict]:
    """The sample of this question (1 November 2026): findings of PARROT reports never used before (``exclude``:
    the pairs of the HPO samples), up to ``per_report`` per report and ``per_structure`` per structure, English
    from the translations of the reports in other languages than Italian, Italian from the Italian reports."""
    import random

    import hpo_probe as hp
    from melampo.data.report_reading import ReportReaders, build_linker

    readers = ReportReaders(build_linker()[0])        # no owner: the findings as the interpreter makes them
    rows = [json.loads(x) for x in parrot.read_text("utf-8").split("\n") if x.strip()]
    rows = [r for r in rows if f"parrot-{r['no']}" not in exclude and r.get("translation")]
    random.Random(seed).shuffle(rows)
    per: Counter = Counter()
    items: list[dict] = []
    for lang, wanted in (("it", n_it), ("en", n_en)):
        count = 0
        for r in rows:
            if count >= wanted:
                break
            if (r.get("language") == "Italian") != (lang == "it"):
                continue
            text = r["report"] if lang == "it" else r["translation"]
            taken = 0
            for k, f in enumerate(hp._findings(readers, text, f"parrot-{r['no']}-{lang}")):
                if taken >= per_report or count >= wanted or per[f["structure"]] >= per_structure:
                    continue
                per[f["structure"]] += 1
                taken += 1
                count += 1
                items.append({"item_id": f"parrot-{r['no']}-{lang}-{k}", "pair": f"parrot-{r['no']}",
                              "report": r["no"], "language": lang, "modality": r.get("modality"),
                              "area": r.get("area"), "structure": f["structure"], "mention": f["mention"],
                              "clause": hp._line_of(f["clause"], f["mention"])})
    return items


def main(argv: list[str] | None = None) -> int:
    if argv is None and len(sys.argv) > 1 and sys.argv[1] == "sample":
        sp = argparse.ArgumentParser(description="the fresh sample of the owner question")
        sp.add_argument("cmd")
        sp.add_argument("--parrot", required=True)
        sp.add_argument("--out", required=True)
        sp.add_argument("--n-en", type=int, default=200)
        sp.add_argument("--n-it", type=int, default=60)
        sp.add_argument("--per-report", type=int, default=3)
        sp.add_argument("--per-structure", type=int, default=8)
        sp.add_argument("--seed", type=int, default=FRESH_SEED)
        a = sp.parse_args()
        exclude = set()
        # every report already used when the sample was drawn: the two HPO samples, and for the statement sample
        # also the owner sample (so that each seed draws again the items its labels were written on)
        before = {FRESH_SEED: [*LABELS.values()], STATEMENT_SEED: [*LABELS.values(), OWNER_LABELS],
                  CLAIMS_SEED: [*LABELS.values(), OWNER_LABELS, STATEMENT_LABELS],
                  CLAIMS2_SEED: [*LABELS.values(), OWNER_LABELS, STATEMENT_LABELS, CLAIMS_LABELS]}
        for path in before.get(a.seed, [*LABELS.values(), *SAMPLES.values()]):
            if path.exists():
                exclude |= {m.group(0) for line in path.read_text("utf-8").splitlines()
                            if line.strip() and not line.startswith("#")
                            for m in [re.match(r"parrot-\d+", line.strip())] if m}
        items = build_fresh(Path(a.parrot), a.n_en, a.n_it, a.per_report, a.per_structure, exclude, a.seed)
        Path(a.out).write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n", "utf-8")
        print(f"{len(items)} items ({Counter(i['language'] for i in items)}) -> {a.out}")
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--items", action="append", required=True, help="name=path (name: dev or heldout)")
    ap.add_argument("--models", default="")
    ap.add_argument("--endpoint", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--public-data", action="store_true")
    ap.add_argument("--cache", default="owner_probe_answers.jsonl")
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--parrot", help="PARROT_v1_0.jsonl: the whole report, for the exam and the section")
    ap.add_argument("--uberon", help="uberon.obo: what is part of what")
    ap.add_argument("--questions", help="where to write the questions for a person (review_question), one per line")
    ap.add_argument("--skip", default="", help="arms not to run: owner, statement, claims")
    ap.add_argument("--hpo", help="hp-full.obo: what the reader knows of abnormalities (the claims arm)")
    ap.add_argument("--pato", help="pato.obo: the qualities of the abnormalities")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import hpo_probe as hp
    from melampo.data.report_reading import build_linker
    from melampo.memory.finding_owner import FindingOwner, Ownership

    linker = build_linker()[0]
    from melampo.data.report_reading import finding_owner

    reader_only, note = finding_owner(linker)
    if reader_only is None:
        raise SystemExit(note)
    if args.uberon:
        from melampo.memory.finding_owner import Anatomy

        reader_only = FindingOwner(reader_only.reader, {}, Anatomy.from_uberon(Path(args.uberon)))
    reports = {}
    if args.parrot:
        for line in Path(args.parrot).read_text("utf-8").split("\n"):
            if line.strip():
                r = json.loads(line)
                reports[r["no"]] = r
    chats = {}
    models = [m for m in args.models.split(",") if m.strip()]
    if models:
        from role_probe import Cache, check_endpoint, make_chats, probe_models

        check_endpoint(args.endpoint, args.public_data)
        chats = make_chats(models, args.endpoint, Cache(Path(args.cache)), max_tokens=args.max_tokens)
        probe_models(chats)
    with_models = FindingOwner(reader_only.reader, chats, reader_only.anatomy) if chats else None
    # the statement reading (1 November 2026): what the sentence says of the structure
    statement = FindingOwner(reader_only.reader, chats, reader_only.anatomy, question="statement") if chats else None
    # claim by claim (2 November 2026), with what the reader knows of abnormalities (HPO, PATO)
    from melampo.memory.what_is_said import AnomalyWords

    anomalies = AnomalyWords.from_files(Path(args.hpo), Path(args.pato) if args.pato else None,
                                        Path(args.uberon) if args.uberon else None) if args.hpo else AnomalyWords()
    claims = FindingOwner(reader_only.reader, chats, reader_only.anatomy, question="claims",
                          anomalies=anomalies) if chats else None
    from melampo.memory.finding_owner import report_context
    away = elsewhere_pairs()

    rows = []
    for spec in args.items:
        name, path = spec.split("=", 1)
        labels = hp.read_labels(LABELS[name]) if name in LABELS else {}
        own = {k: v for k, v in labels.items() if k.endswith(("-it", "-en"))}
        items = [json.loads(x) for x in Path(path).read_text("utf-8").splitlines() if x.strip()]

        def one(item: dict, name: str = name, own: dict = own) -> dict | None:
            at = item["clause"].find(item["mention"])
            if at < 0:
                return None
            gold = gold_of(item, name, own, away)
            if not gold:
                return None
            row = {"set": name, "item_id": item["item_id"], "language": item["language"],
                   "structure": item["structure"], "gold": gold}
            if name == "statement":
                row["gold_type"], row["gold_doubt"] = statement_labels()[item["item_id"]]
            if name in CLAIMS_FILES:
                kinds, row["gold_doubt"] = claims_labels(name)[item["item_id"]]
                row["gold_claims"] = sorted(kinds)
            report = reports.get(item.get("report"))
            context = {}
            if report is not None:
                text = report["translation"] if item["language"] == "en" else report["report"]
                at_report = text.find(item["clause"])
                context = report_context(text, max(at_report, 0))
            row["context"] = context
            row["_item"] = item
            row["reader"] = reader_only(item["mention"], item["clause"], at, item["language"], item["structure"],
                                        context).as_dict()
            if with_models is not None and "owner" not in args.skip:
                row["reader+models"] = with_models(item["mention"], item["clause"], at, item["language"],
                                                   item["structure"], context).as_dict()
            if statement is not None and "statement" not in args.skip:
                row["statement"] = statement(item["mention"], item["clause"], at, item["language"],
                                             item["structure"], context).as_dict()
            if claims is not None and "claims" not in args.skip:
                row["claims"] = claims(item["mention"], item["clause"], at, item["language"], item["structure"],
                                       context).as_dict()
            return row

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows += [r for r in pool.map(one, items) if r is not None]
    report = {"by_set": {}}
    for name in sorted({r["set"] for r in rows}):
        rs = [r for r in rows if r["set"] == name]
        report["by_set"][name] = {arm: table(rs, arm) for arm in ("reader", "reader+models", "statement", "claims")
                                  if any(arm in r for r in rs)}
        if name in CLAIMS_FILES and any("claims" in r for r in rs):
            report["by_set"][name]["claims_kinds"] = claims_table(rs, "claims")
        if name == "statement" and any("statement" in r for r in rs):
            report["by_set"][name]["statement_types"] = statement_table(rs, "statement")
    if args.questions:
        # the questions a person would be asked, complete in themselves: what the review mask shows
        from melampo.memory.review_question import build_question

        with open(args.questions, "w", encoding="utf-8") as fh:
            for row in rows:
                arm = next(a for a in ("claims", "statement", "reader+models", "reader") if a in row)
                verdict = row[arm]["verdict"]
                if verdict == "here" and not row[arm].get("review"):
                    continue
                item = row["_item"]
                owner = Ownership(**row[arm])
                q = build_question(owner, report=f"parrot-{item.get('report')}", language=item["language"],
                                   sentence=item["clause"], mention=item["mention"],
                                   start=item["clause"].find(item["mention"]), structure=item["structure"],
                                   structure_name=FindingOwner.linked_name(item["structure"]),
                                   context=row.get("context"))
                fh.write(json.dumps({**q.as_dict(), "set": row["set"], "gold": row["gold"]}, ensure_ascii=False) + "\n")
    # a run whose lost answers (a call that failed after its retries) are more than LOST_LIMIT of an arm's does
    # not count for the criteria (2 November 2026: the credit ran out during a run)
    report["lost_answers"] = {}
    for arm in ("reader+models", "statement", "claims"):
        answers = [a for r in rows if arm in r for a in (r[arm].get("models") or {}).values()]
        if answers:
            lost = [a for a in answers if a.get("error")]
            report["lost_answers"][arm] = {
                "answers": len(answers), "lost": len(lost), "share": round(len(lost) / len(answers), 4),
                "messages": dict(Counter(a.get("error_text") or a.get("error") for a in lost).most_common(5))}
    report["valid_for_criteria"] = all(v["share"] <= LOST_LIMIT for v in report["lost_answers"].values())
    if not report["valid_for_criteria"]:
        print(f"WARNING: more than {LOST_LIMIT:.0%} of the answers were lost: the run does not count for the "
              f"criteria {json.dumps(report['lost_answers'])}", file=sys.stderr)
    for row in rows:
        row.pop("_item", None)
    report["rows"] = rows
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps({"valid_for_criteria": report["valid_for_criteria"], "lost_answers": report["lost_answers"],
                      **report["by_set"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
