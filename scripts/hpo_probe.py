#!/usr/bin/env python
"""From a report's finding to an HPO term: the candidates of the site, a deterministic choice, a model's choice
(31 October 2026). See ``docs/linker/dal_reperto_al_termine_hpo_2026-10-31.md``.

The findings come from PARROT (radiologists' reports in 14 languages with their English translation; CC BY-NC-SA
4.0): the English translations, read by the narrative reader (``data/report_reading``: linker sites, findings of
the clauses that describe them); the Italian reports also in Italian, paired with their translation by structure.
A sample is drawn with a fixed seed (one finding per report, at most ``--per-structure`` per structure) and
labelled by hand (``data/gold/hpo_silver_2026-10-31/labels.txt``: the most specific HPO term the clause supports,
or NONE). Only the labels are in the repository; the items are rebuilt from PARROT with the same seed.

Methods, on the same items:

* ``candidates`` -- is the label among the HPO terms localised at the site (``hpo_candidates``)? In the shortlist
  a model is shown? (A label that is not there cannot be found by any method below.)
* ``A`` -- deterministic: the most specific candidate named entirely by the clause's words and the site's names
  (English labels and exact synonyms); for an Italian clause, the Italian HPO labels (``hp-it.babelon.tsv``).
* ``B:<model>`` -- three steps. (1) An Italian clause is rendered in English by the model. (2) The model names the
  abnormality the sentence states about the site, in the ontology's style ("Pulmonary nodule"); this name is only
  a query: on the development labels the shortlist built from the sentence's words alone holds the label in 77 of
  107 items, and growing it beyond 15 adds 2 (the misses share no word with the label: "enlarged" and
  "Hepatomegaly"). (3) The model chooses among the shortlist built from the sentence and its name (letters, "none
  of these"), reading the original clause. Every answer is a real term of the site's candidates.
* ``C:<model>`` -- A and B agree, otherwise a person.
* ``first`` -- the baseline a model must beat: always the first term of the shortlist built from the English
  sentence alone.

Scores, against the labels: exact; **too general** (an ancestor of the label: safe, less informative); **too
specific** (a descendant of the label: says more than the report -- the worst error); wrong (anything else);
none/term confusions; answered share. Texts go to OpenRouter only because PARROT is public (``--public-data``).

    python scripts/hpo_probe.py sample --parrot PARROT/data/PARROT_v1_0.jsonl --hpo hp-full.obo \
        --uberon uberon.obo --out hpo_items.jsonl
    python scripts/hpo_probe.py show --items hpo_items.jsonl            # to label
    python scripts/hpo_probe.py probe --items hpo_items.jsonl --labels data/gold/hpo_silver_2026-10-31/labels.txt \
        --hpo hp-full.obo --uberon uberon.obo --italian hp-it.babelon.tsv \
        --models google/gemma-4-31b-it,nvidia/nemotron-3-ultra-550b-a55b \
        --endpoint https://openrouter.ai/api/v1/chat/completions --public-data --out hpo_probe.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from melampo.memory import hpo_candidates as hc  # noqa: E402

SEED = 20261031
NONE = "NONE"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXY"
SHORTLIST = 15
DEFAULT_MODELS = "google/gemma-4-31b-it,nvidia/nemotron-3-ultra-550b-a55b"


# -- the items ------------------------------------------------------------------------------------------------
def _line_of(clause: str, mention: str) -> str:
    """The line of the clause that holds the mention: a report's headers ("Findings", "Posterior fossa:") are not
    part of what the finding says."""
    lines = [x.strip(" -\t") for x in clause.splitlines() if x.strip(" -\t")]
    hit = next((x for x in lines if mention.lower() in x.lower()), None)
    return (hit or " ".join(lines)).strip()


def _findings(readers, text: str, report: str) -> list[dict]:
    out = readers.read(report, text)
    return [f for f in out.get("findings", []) if f.get("polarity") != "negated"
            and "procedure_site" not in (f.get("roles") or [])]


def build_sample(parrot: Path, n_en: int, n_it: int, per_structure: int, seed: int = SEED) -> list[dict]:
    from melampo.data.report_reading import ReportReaders, build_linker

    readers = ReportReaders(build_linker()[0])
    rows = [json.loads(x) for x in parrot.read_text("utf-8").split("\n") if x.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    items: list[dict] = []
    per: Counter = Counter()

    def take(f: dict) -> bool:
        if per[f["structure"]] >= per_structure:
            return False
        per[f["structure"]] += 1
        return True

    italian = [r for r in rows if r.get("language") == "Italian" and r.get("translation")]
    others = [r for r in rows if r.get("language") != "Italian" and r.get("translation")]
    for r in italian:
        if sum(1 for i in items if i["language"] == "it") >= n_it:
            break
        en = _findings(readers, r["translation"], f"parrot-{r['no']}-en")
        it = {f["structure"]: f for f in _findings(readers, r["report"], f"parrot-{r['no']}-it")}
        pair = next((f for f in en if f["structure"] in it), None)
        if pair is None or not take(pair):
            continue
        pid = f"parrot-{r['no']}"
        for lang, f in (("en", pair), ("it", it[pair["structure"]])):
            items.append({"item_id": f"{pid}-{lang}", "pair": pid, "report": r["no"], "language": lang,
                          "source_language": "Italian", "modality": r.get("modality"), "area": r.get("area"),
                          "structure": f["structure"], "mention": f["mention"],
                          "clause": _line_of(f["clause"], f["mention"]), "character": f.get("character", []),
                          "certainty": f.get("certainty", "")})
    count = 0
    for r in others:
        if count >= n_en:
            break
        found = _findings(readers, r["translation"], f"parrot-{r['no']}-en")
        f = next((x for x in found if per[x["structure"]] < per_structure), None)
        if f is None or not take(f):
            continue
        pid = f"parrot-{r['no']}"
        items.append({"item_id": f"{pid}-en", "pair": pid, "report": r["no"], "language": "en",
                      "source_language": r.get("language"), "modality": r.get("modality"), "area": r.get("area"),
                      "structure": f["structure"], "mention": f["mention"],
                      "clause": _line_of(f["clause"], f["mention"]), "character": f.get("character", []),
                      "certainty": f.get("certainty", "")})
        count += 1
    return items


# -- Italian HPO labels ---------------------------------------------------------------------------------------
def italian_labels(path: Path | None) -> dict[str, tuple[str, str]]:
    """HPO id -> (Italian label, status OFFICIAL/CANDIDATE), from the HPO translation project's babelon table."""
    if path is None or not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row.get("predicate_id") == "rdfs:label" and row.get("translation_value"):
                out[row["subject_id"]] = (row["translation_value"], row.get("translation_status", ""))
    return out


_IT_WORD = re.compile(r"[a-zàèéìòù0-9]+")
_IT_STOP = frozenset("di del della dei delle degli il lo la i gli le un una uno in nel nella con per a al alla da "
                     "e o".split())


def _it_stems(text: str) -> set[str]:
    return {w[:-1] if len(w) > 4 and w[-1] in "aeio" else w for w in _IT_WORD.findall(text.lower())
            if w not in _IT_STOP}


def deterministic_it(ph: hc.Phenotypes, anchors: list[str], clause: str, it: dict) -> tuple[str | None, str]:
    """Method A for an Italian clause: the most specific candidate whose Italian label is all in the clause."""
    cands = ph.candidates(anchors)
    words = _it_stems(clause)
    hits = [h for h in cands if h in it and _it_stems(it[h][0]) and _it_stems(it[h][0]) <= words]
    if not hits:
        return None, "no_italian_label_in_the_clause" if any(h in it for h in cands) else "no_italian_label"
    top = [h for h in hits if not (ph.descendants(h) & set(hits))]
    if len(top) > 1:
        return None, "several_terms_fit:" + ",".join(sorted(top)[:4])
    return top[0], "named"


# -- the model's choice ---------------------------------------------------------------------------------------
def choice_prompt(clause: str, site: str, options: list[tuple[str, str]], language: str) -> str:
    lines = [
        "You map a finding of a radiology report to the Human Phenotype Ontology (HPO).",
        f"Sentence of the report ({'Italian' if language == 'it' else 'English'}): \"{clause}\"",
        f"The anatomical site has already been identified: {site}.",
        "Choose the HPO term the sentence states about this site: the most specific term the sentence supports,",
        "never more specific than what is written (a lesion not characterised is not a cyst). If the sentence",
        "states no abnormality at this site (measures alone, a normal finding, the technique, a clinical question),",
        "or no option fits, choose the last option.",
        "Options:",
    ]
    for letter, (_, label) in zip(LETTERS, options):
        lines.append(f"{letter}. {label}")
    lines.append(f"{LETTERS[len(options)]}. none of these")
    lines.append("Answer with the letter only.")
    return "\n".join(lines)


def name_prompt(text: str, site: str) -> str:
    return ("A sentence of a radiology report, about this anatomical site: " + site + ".\n\"" + text + "\"\n"
            "Name the abnormality the sentence states about this site as a short term in the style of the Human "
            "Phenotype Ontology, in English (for example: Pulmonary nodule, Hepatomegaly, Renal cyst). Do not add "
            "what is not written. If the sentence states no abnormality at this site, answer: none. Answer with "
            "the term only.")


def translate_prompt(clause: str) -> str:
    return ("Translate this sentence of an Italian radiology report into English, in the radiologist's register, "
            "keeping every measure and qualifier. Answer with the translation only.\n\n" + clause)


def parse_letter(answer: str, n: int) -> int | None:
    m = re.search(r"\b([A-Z])\b", (answer or "").strip().upper()[:20])
    if not m:
        return None
    i = LETTERS.find(m.group(1))
    return i if 0 <= i <= n else None


# -- scoring --------------------------------------------------------------------------------------------------
def outcome(ph: hc.Phenotypes, answer: str | None, gold: str) -> str:
    if answer is None:
        return "abstained"
    if answer == gold:
        return "exact"
    if gold == NONE:
        return "term_for_none"
    if answer == NONE:
        return "none_for_term"
    if answer in ph.ancestors(gold):
        return "too_general"
    if gold in ph.ancestors(answer):
        return "too_specific"
    return "wrong"


def summarise(rows: list[dict], arm: str) -> dict:
    got = [r[arm] for r in rows if arm in r]
    c = Counter(o["outcome"] for o in got)
    answered = [o for o in got if o["outcome"] != "abstained"]
    good = sum(1 for o in answered if o["outcome"] in ("exact", "too_general"))
    return {"n": len(got), "answered": len(answered), "exact": c["exact"], "too_general": c["too_general"],
            "too_specific": c["too_specific"], "wrong": c["wrong"], "none_for_term": c["none_for_term"],
            "term_for_none": c["term_for_none"], "abstained": c["abstained"],
            "safe_share_of_answered": round(good / len(answered), 3) if answered else None,
            "exact_share_of_all": round(c["exact"] / len(got), 3) if got else None}


def read_labels(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text("utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            item, label = line.split("|")[:2]
            out[item.strip()] = label.strip()
    return out


def run_probe(args) -> dict:
    ph = hc.load(Path(args.hpo), Path(args.uberon))
    it = italian_labels(Path(args.italian) if args.italian else None)
    items = [json.loads(x) for x in Path(args.items).read_text("utf-8").splitlines() if x.strip()]
    labels = read_labels(Path(args.labels))
    if args.limit:
        items = items[: args.limit]
    chats = {}
    models = [m for m in (args.models or "").split(",") if m.strip()]
    if models:
        from role_probe import Cache, check_endpoint, make_chats, probe_models

        check_endpoint(args.endpoint, args.public_data)
        chats = make_chats(models, args.endpoint, Cache(Path(args.cache)), max_tokens=args.max_tokens)
        probe_models(chats)

    def one(item: dict) -> dict:
        gold = labels.get(item["pair"]) or labels.get(item["item_id"])
        anchors = hc.site_anchors(item["structure"])
        site = ", ".join(ph.anatomy.labels.get(a, a) for a in ph.anatomy.site_classes(anchors))
        cands = ph.candidates(anchors)
        row = {"item_id": item["item_id"], "language": item["language"], "structure": item["structure"],
               "gold": gold, "candidates": len(cands), "gold_in_candidates": gold in cands if gold != NONE else None}
        if gold is None:
            row["unlabelled"] = True
            return row
        if item["language"] == "en":
            a, why = ph.deterministic(anchors, item["clause"])
        else:
            a, why = deterministic_it(ph, anchors, item["clause"], it)
        row["A"] = {"answer": a, "why": why, "outcome": outcome(ph, a, gold)}
        if item["language"] == "en":
            short = ph.shortlist(anchors, item["clause"], args.shortlist)
            if gold != NONE:
                row["gold_in_shortlist"] = gold in short
            # the baseline a model must beat: the first term of the shortlist, always
            first = short[0] if short else None
            row["first"] = {"answer": first, "why": "first_of_shortlist", "outcome": outcome(ph, first, gold)}
        row["gold_italian_label"] = it.get(gold, ("", ""))[1] or "missing" if gold != NONE else None
        for name, chat in chats.items():
            text = item["clause"]
            if item["language"] == "it":
                try:
                    text = chat(translate_prompt(item["clause"])).strip()
                except Exception as error:  # noqa: BLE001
                    row[f"B:{name}"] = {"answer": None, "why": f"translation_error:{type(error).__name__}",
                                        "outcome": "abstained"}
                    continue
            try:
                proposed = chat(name_prompt(text, site)).strip().splitlines()[0][:120]
            except Exception as error:  # noqa: BLE001
                proposed = ""
                row[f"name_error:{name}"] = type(error).__name__
            # the model names the abnormality in the ontology's words; the shortlist is built from the sentence and
            # that name (the name is a query, never an answer: the choice below is among real terms only)
            short = ph.shortlist(anchors, f"{text} {proposed if proposed.lower() != 'none' else ''}",
                                 args.shortlist)
            row[f"proposed:{name}"] = proposed
            row[f"gold_in_shortlist:{name}"] = gold in short if gold != NONE else None
            options = [(h, ph.terms[h].name) for h in short]
            try:
                reply = chat(choice_prompt(item["clause"], site, options, item["language"]))
            except Exception as error:  # noqa: BLE001
                row[f"B:{name}"] = {"answer": None, "why": f"model_error:{type(error).__name__}",
                                    "outcome": "abstained"}
                continue
            k = parse_letter(reply, len(options))
            b = None if k is None else (NONE if k == len(options) else options[k][0])
            if proposed.strip().lower().rstrip(".") == "none":
                b = NONE      # the model itself said the sentence states no abnormality here: its choice must agree
            row[f"B:{name}"] = {"answer": b, "why": "unparsed" if k is None else "chosen", "reply": reply[:40],
                                "outcome": outcome(ph, b, gold)}
            c = b if (b is not None and a == b) else None
            row[f"C:{name}"] = {"answer": c, "why": "agree" if c else "to_a_person", "outcome": outcome(ph, c, gold)}
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(one, items))
    if len(chats) >= 2:
        # two different models choosing the same term (otherwise a person): the agreement of two readings, as in
        # the linker's rule; after the first run it was safe in 68% of the English items it accepted
        names = list(chats)
        for row in rows:
            if row.get("unlabelled"):
                continue
            answers = {row.get(f"B:{n}", {}).get("answer") for n in names}
            agreed = answers.pop() if len(answers) == 1 else None
            row["models_agree"] = {"answer": agreed, "why": "agree" if agreed else "to_a_person",
                                   "outcome": outcome(ph, agreed, row["gold"])}
    labelled = [r for r in rows if not r.get("unlabelled")]
    arms = sorted({k for r in labelled for k in r
                   if (k in ("A", "first", "models_agree") or k[:2] in ("B:", "C:")) and isinstance(r[k], dict)})
    report = {"items": len(items), "labelled": len(labelled), "by_language": {}}
    for lang in ("en", "it"):
        rs = [r for r in labelled if r["language"] == lang]
        if not rs:
            continue
        with_term = [r for r in rs if r["gold"] != NONE]
        report["by_language"][lang] = {
            "items": len(rs), "labelled_none": len(rs) - len(with_term),
            "gold_in_candidates": sum(1 for r in with_term if r["gold_in_candidates"]),
            "gold_in_shortlist_en": sum(1 for r in with_term if r.get("gold_in_shortlist")),
            "gold_in_shortlist_by_model": {k.split(":", 1)[1]: sum(1 for r in with_term if r.get(k))
                                           for k in sorted({k for r in rs for k in r
                                                            if k.startswith("gold_in_shortlist:")})},
            "gold_with_italian_label": dict(Counter(r.get("gold_italian_label") for r in with_term)),
            "arms": {arm: summarise(rs, arm) for arm in arms},
        }
    report["rows"] = rows
    return report


def show(items_path: Path, hpo: Path | None, uberon: Path | None) -> None:
    ph = hc.load(hpo, uberon) if hpo and uberon else None
    for x in (json.loads(l) for l in items_path.read_text("utf-8").splitlines() if l.strip()):
        print(f"{x['item_id']}|{x['structure']}|{x['clause'][:200]}")
        if ph is not None and x["language"] == "en":
            anchors = hc.site_anchors(x["structure"])
            short = ph.shortlist(anchors, x["clause"], 8)
            print("   ", "; ".join(f"{h} {ph.terms[h].name}" for h in short))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--parrot", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n-en", type=int, default=100)
    s.add_argument("--n-it", type=int, default=50)
    s.add_argument("--per-structure", type=int, default=6)
    w = sub.add_parser("show")
    w.add_argument("--items", required=True)
    w.add_argument("--hpo")
    w.add_argument("--uberon")
    p = sub.add_parser("probe")
    p.add_argument("--items", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--hpo", required=True)
    p.add_argument("--uberon", required=True)
    p.add_argument("--italian")
    p.add_argument("--models", default="")
    p.add_argument("--endpoint", default="http://localhost:8000/v1/chat/completions")
    p.add_argument("--public-data", action="store_true")
    p.add_argument("--cache", default="hpo_probe_answers.jsonl")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--shortlist", type=int, default=SHORTLIST)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "sample":
        items = build_sample(Path(args.parrot), args.n_en, args.n_it, args.per_structure)
        Path(args.out).write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n", "utf-8")
        print(f"{len(items)} items ({Counter(i['language'] for i in items)}) -> {args.out}")
        return 0
    if args.cmd == "show":
        show(Path(args.items), Path(args.hpo) if args.hpo else None, Path(args.uberon) if args.uberon else None)
        return 0
    report = run_probe(args)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
