#!/usr/bin/env python
"""A second, independent labeller (11 November 2026), against errors of labelling.

Double annotation with adjudication is how the annotated corpora of the field are made (BioScope, Vincze et al.
2008: two independent annotators, their disagreements resolved by a chief annotator). Here:

1. the sample is drawn; the second labeller (a model that is none of the readers measured, by default
   Mistral Large, EU) labels every item from the whole sentence alone, before any reading. Since 12 November
   2026 it is given nothing the first labeller or the readers are given (no convention, no manual, no examples,
   no list of words): only what each label means, in plain words, so that both answer the same question;
2. the first labeller labels the sample without opening the second's labels, and commits them;
3. ``compare`` gives the agreement (Cohen's kappa, on what the sentence states and on its uncertainty) and the
   disagreements; each is re-read on the whole sentence and decided by the sources, a change written with its
   reason in the note. All this happens before the readers run: the labels stay written before any reading.

Only public texts go to the external service (--public-data); patients' reports never do.

    python scripts/second_labeller.py label --items owner_items_ctrate6.jsonl --set ctrate6 \\
        --model mistralai/mistral-large-4-0 --endpoint https://openrouter.ai/api/v1/chat/completions \\
        --public-data --out second_labels_ctrate6.txt
    python scripts/second_labeller.py compare --set ctrate6 --second second_labels_ctrate6.txt \\
        --items owner_items_ctrate6.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CLAIMS = ("finding", "involved", "normal", "measure", "history", "device", "not_assessable", "aside",
          "indication", "procedure", "nothing", "uncertain", "uncertain_high")


# Independent of everything the first labeller and the readers are given (12 November 2026, Frank: "assolutamente
# indipendente"): no convention, no manual, no examples, no list of words. Only what each label means, said once
# in plain words, so that the two labellers answer the same question; how to read the sentence is the model's own.
SAYS = {
    "abnormal_in_it": "finding",                 # something abnormal in it or in a part of it
    "abnormal_of_another_reaching_it": "involved",
    "normal_or_absent": "normal",                # it, or something of it, is normal; or something is absent from it
    "its_size": "measure",
    "its_past": "history",
    "a_device_in_it": "device",
    "could_not_be_examined": "not_assessable",
    "only_where_something_else_is": "aside",
    "why_the_exam_was_done": "indication",
    "a_procedure_on_it": "procedure",
    "nothing": "nothing",
}
CERTAINTY = {"certain": "", "very_likely": "uncertain_high", "not_certain": "uncertain"}


def prompt(item: dict) -> str:
    return ("A sentence of a radiology report, and a structure of the body named in it.\n\n"
            f"Sentence: {item['clause']}\nWords that name the structure: \"{item['mention']}\"\n"
            f"Structure: {item['structure'].replace('_', ' ')}\n\n"
            "1. What does the sentence say of this structure? One or more of:\n"
            "- abnormal_in_it: something abnormal in it or in a part of it\n"
            "- abnormal_of_another_reaching_it: something abnormal of another structure that reaches it\n"
            "- normal_or_absent: it, or something of it, is normal, or something is absent from it\n"
            "- its_size: a size of it\n- its_past: something of its past\n- a_device_in_it: a device in or at it\n"
            "- could_not_be_examined: it could not be examined\n"
            "- only_where_something_else_is: it is named only to say where something else is\n"
            "- why_the_exam_was_done: the reason for the exam\n- a_procedure_on_it: a procedure done on it\n"
            "- nothing: nothing of it\n"
            "2. How sure is the writer of what the sentence says? One of: certain; very_likely (very likely, not "
            "quite certain); not_certain (a possibility, a doubt, a hypothesis).\n\n"
            "Answer with JSON only: {\"says\": [\"...\"], \"certainty\": \"...\", \"why\": \"one short sentence\"}.")


def read_answer(text: str) -> tuple[list[str], str]:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    if not isinstance(d, dict):
        d = {}
    if "says" in d:
        claims = {SAYS[str(x).strip()] for x in d.get("says") or [] if str(x).strip() in SAYS}
        sure = str(d.get("certainty") or "").strip()
        if claims and sure not in CERTAINTY:
            claims = set()              # an answer without its certainty is no label
        elif claims and CERTAINTY[sure]:
            claims.add(CERTAINTY[sure])
    else:
        claims = {str(c).strip() for c in d.get("claims") or [] if str(c).strip() in CLAIMS}
    return sorted(claims), str(d.get("why") or "").replace("|", "/").replace("\n", " ")[:200]


def cmd_label(args: argparse.Namespace) -> int:
    from role_probe import Cache, check_endpoint

    check_endpoint(args.endpoint, args.public_data)
    import os
    import time

    from melampo.evaluation import linking_bench as lb

    if "openrouter.ai" in args.endpoint and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is not set")
    key = os.environ.get("OPENROUTER_API_KEY", "") or os.environ.get("LLM_API_KEY", "local")
    # a short wait per question (12 November 2026): the answers are about 200 characters, yet with 180 s and four
    # retries a question that hung held its thread up to 18 minutes, and the run spent 3 h 30 on 161 of 265 items.
    # What it costs (12 November 2026): 63 empty and 16 cut answers in one run, each asked again three times in
    # each of four passes with 1500 tokens: the model spent the tokens before answering, and the credit went. The
    # answer is under 100 tokens: 400 are enough, an empty reply is not asked again by the call, and an answer
    # that cannot be read is never kept in the cache (it was, and it came back from the cache at every pass).
    raw = lb.OpenRouterChat(args.model, key, endpoint=args.endpoint, max_tokens=args.max_tokens,
                            timeout=args.timeout, retries=args.retries, min_interval=0.2, empty_retries=0)
    cache = Cache(Path(args.cache))

    def chat(text: str) -> str:
        k = cache.key(args.model, text)
        if k in cache.data and read_answer(cache.data[k])[0]:
            return cache.data[k]
        answer = raw(text)
        if read_answer(answer)[0]:
            with cache.lock:
                cache.data[k] = answer
                with Path(args.cache).open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"key": k, "model": args.model, "text": answer}, ensure_ascii=False) + "\n")
        return answer

    deadline = time.monotonic() + args.budget_minutes * 60
    items = [json.loads(x) for x in Path(args.items).read_text("utf-8").splitlines() if x.strip()]
    lines = [f"# Second labeller ({args.model}) of {args.set}, written before any reading, from the whole sentence "
             "alone (no convention, no manual). Not a reference: it is compared with the first labeller.",
             "# Format: item|claims|why"]
    # several questions at a time (12 November 2026): one after the other, 114 of 265 items were answered in the
    # two hours of the job. With eight at a time 150 of 265 came back without an answer: what failed is now
    # written for each item, and the items without an answer are asked again, fewer at a time and with longer
    # pauses, until every item has its answer or the passes are spent (a lost answer is never a guess)
    from concurrent.futures import ThreadPoolExecutor

    spent: list[bool] = []

    def one(it: dict) -> tuple[list[str], str]:
        if time.monotonic() > deadline:
            # the budget spent: the item is left without an answer and the file is still written, with the
            # answers kept; the next run asks only what is missing
            return [], "no answer (time budget spent)"
        if spent:
            return [], "no answer (the credit is spent: not asked)"
        try:
            text = chat(prompt(it))
        except Exception as error:  # noqa: BLE001 - a lost answer is a missing label, never a guess
            if "HTTP 402" in str(error):
                spent.append(True)      # payment required: nothing more is asked
            return [], f"no answer ({type(error).__name__}: {str(error)[:120]})"
        claims, why = read_answer(text)
        if not claims:
            return [], ("no answer (empty reply)" if not (text or "").strip()
                        else f"no answer (unreadable reply: {' '.join(text.split())[:120]})")
        return claims, why

    answers: list[tuple[list[str], str]] = [([], "not asked")] * len(items)
    todo = list(range(len(items)))
    for n, workers in enumerate([args.workers, max(1, args.workers // 2), 2, 1][:max(1, args.passes)]):
        if not todo:
            break
        if n:
            if spent or time.monotonic() + args.pause > deadline:
                break
            time.sleep(args.pause)
            print(f"pass {n + 1}: {len(todo)} items asked again, {workers} at a time", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            got = list(pool.map(lambda i: one(items[i]), todo))
        for i, a in zip(todo, got):
            answers[i] = a
        before, todo = len(todo), [i for i in todo if not answers[i][0]]
        if n and len(todo) > before - max(1, before // 20):
            # a pass that answered almost nothing will not do better: stop, instead of paying for it again
            print(f"pass {n + 1} answered {before - len(todo)} of {before}: no further pass", file=sys.stderr)
            break
    reasons = Counter(why.split(":")[0].split(" (")[1].rstrip(")") if why.startswith("no answer (") else why
                      for claims, why in answers if not claims)
    if reasons:
        print(f"without an answer, by reason: {dict(reasons)}", file=sys.stderr)
    lost = 0
    for it, (claims, why) in zip(items, answers):
        lost += not claims
        lines.append(f"{it['item_id']}|{'+'.join(claims) or 'none'}|{why}")
    Path(args.out).write_text("\n".join(lines) + "\n", "utf-8")
    print(f"{len(items)} items labelled by {args.model}, {lost} without an answer -> {args.out}")
    if items and lost / len(items) > 0.02:
        # a second labeller that did not label is no check: the run says so instead of looking done
        print(f"::error::the second labeller left {lost} of {len(items)} items without an answer", file=sys.stderr)
        return 1
    return 0


def _kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    if not n:
        return 0.0
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def cmd_compare(args: argparse.Namespace) -> int:
    import owner_probe as op
    from melampo.memory.reading_memory import label_uncertainty

    first = op.claims_labels(args.set)
    second = {}
    for ln in Path(args.second).read_text("utf-8").splitlines():
        if ln.strip() and not ln.startswith("#"):
            item, claims, why = (ln.split("|", 2) + ["", ""])[:3]
            second[item] = (set() if claims == "none" else set(claims.split("+")), why)
    items = {}
    if args.items:
        items = {json.loads(x)["item_id"]: json.loads(x) for x in Path(args.items).read_text("utf-8").splitlines()
                 if x.strip()}
    both = [i for i in first if i in second and second[i][0]]
    outcome = [(op.outcome_of_claims(first[i][0]), op.outcome_of_claims(second[i][0])) for i in both]
    unc = [(label_uncertainty(first[i][0]) or "none", label_uncertainty(second[i][0]) or "none") for i in both]
    print(json.dumps({"items": len(both), "outcome_agreement": round(sum(a == b for a, b in outcome) / len(both), 3)
                      if both else 0, "outcome_kappa": _kappa(outcome),
                      "uncertainty_agreement": round(sum(a == b for a, b in unc) / len(both), 3) if both else 0,
                      "uncertainty_kappa": _kappa(unc)}, indent=1))
    for i, (o, u) in zip(both, zip(outcome, unc)):
        if o[0] != o[1] or u[0] != u[1]:
            it = items.get(i, {})
            print(f"\n{i}: first {'+'.join(sorted(first[i][0]))} / second {'+'.join(sorted(second[i][0]))}"
                  f"  ({second[i][1]})")
            if it:
                print(f"  [{it['mention']}] {it['clause']}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("label")
    a.add_argument("--items", required=True)
    a.add_argument("--set", required=True)
    a.add_argument("--model", default="mistralai/mistral-large-4-0")
    a.add_argument("--endpoint", default="http://localhost:8000/v1/chat/completions")
    a.add_argument("--public-data", action="store_true")
    a.add_argument("--cache", default="second_labeller_answers.jsonl")
    a.add_argument("--max-tokens", type=int, default=400)
    a.add_argument("--workers", type=int, default=8, help="questions asked at the same time")
    a.add_argument("--passes", type=int, default=3, help="passes over the items still without an answer")
    a.add_argument("--pause", type=float, default=60.0, help="seconds between two passes")
    a.add_argument("--timeout", type=float, default=60.0, help="seconds to wait for one answer")
    a.add_argument("--retries", type=int, default=2)
    a.add_argument("--budget-minutes", type=float, default=170.0,
                   help="minutes after which no question is asked; the labels are written with what was answered")
    a.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--set", required=True)
    c.add_argument("--second", required=True)
    c.add_argument("--items", default="")
    args = ap.parse_args(argv)
    return cmd_label(args) if args.cmd == "label" else cmd_compare(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
