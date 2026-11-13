#!/usr/bin/env python
"""Of what is the finding said? The reader, the model and their agreement on the two HPO samples (31 October 2026).

For every item of the HPO samples (PARROT: development ``hpo_silver_2026-10-31``, held out
``hpo_heldout_2026-10-31``), ``finding_owner.FindingOwner`` reads whether the clause is about the structure the
linker found (``here``) or uses it to place something else (``elsewhere``), with the Kintsch reader alone and with
the reader and the models (the reader's method, ``llm_reading.method_prompt``). The gold: the items of
``data/gold/finding_owner_2026-10-31/elsewhere.txt`` are ``elsewhere``, every other labelled item ``here``.

Criteria fixed before the run (``docs-archive-2026-11:docs/linker/di_che_cosa_parla_la_frase_2026-10-31.md``):
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
# the third claims sample (3 November 2026, after the corrections of the second claims run)
CLAIMS3_LABELS = ROOT / "data/gold/claims_2026-11-03/labels.txt"
# the fourth claims sample (4 November 2026, after the reference library); drawn with --per-structure 16: with the
# cap of 8, the reports not yet used gave 74 items
CLAIMS4_LABELS = ROOT / "data/gold/claims_2026-11-04/labels.txt"
# chest CT reports of CT-RATE (6 November 2026): the sample is drawn by the workflow, the labels written before any
# reading, with the labelling convention (VALIDATION §3)
CTRATE_LABELS = ROOT / "data/gold/ctrate/labels.txt"
CTRATE_SEED = 20261107
# the second sample of CT-RATE (7 November 2026), for the principle of doubt and the criteria under 2%: 300 items,
# from reports not used by the first, labelled before any reading
CTRATE2_LABELS = ROOT / "data/gold/ctrate2/labels.txt"
CTRATE2_SEED = 20261108
# the third sample of CT-RATE (9 November 2026), after the corrections ctrate2 led to: 300 items from reports the
# first two did not use, labelled before any reading; the claims reading and the owner question on the same items
CTRATE3_LABELS = ROOT / "data/gold/ctrate3/labels.txt"
CTRATE3_SEED = 20261109
# the fourth (10 November 2026): the uncertainty of the report as one category, with the words it is about, joined
# across the models (ctrate3 lost it in 11 of 43 items); reports none of the first three used
CTRATE4_LABELS = ROOT / "data/gold/ctrate4/labels.txt"
CTRATE4_SEED = 20261110
# the fifth (11 November 2026): every uncertainty with its level (Panicek and Hricak), and what is not uncertainty
# in a place of its own (undetermined, limits); ctrate4 gave the uncertainty to 62 of 250 stated findings
CTRATE5_LABELS = ROOT / "data/gold/ctrate5/labels.txt"
CTRATE5_SEED = 20261111
# the sixth (11 November 2026): the uncertainty as the literature on hedging tells it apart (shields, not
# approximators, sources, time, the nature open or the exam's limits), no word in two places, and the three
# readings (each model, their union) measured on one run
CTRATE6_LABELS = ROOT / "data/gold/ctrate6/labels.txt"
CTRATE6_SEED = 20261112
SAMPLES = {"fresh": OWNER_LABELS, "statement": STATEMENT_LABELS, "claims": CLAIMS_LABELS, "claims2": CLAIMS2_LABELS,
           "claims3": CLAIMS3_LABELS, "claims4": CLAIMS4_LABELS, "ctrate": CTRATE_LABELS, "ctrate2": CTRATE2_LABELS,
           "ctrate3": CTRATE3_LABELS, "ctrate4": CTRATE4_LABELS,
           "ctrate5": CTRATE5_LABELS, "ctrate6": CTRATE6_LABELS}
CLAIMS_FILES = {"claims": CLAIMS_LABELS, "claims2": CLAIMS2_LABELS, "claims3": CLAIMS3_LABELS,
                "claims4": CLAIMS4_LABELS, "ctrate": CTRATE_LABELS, "ctrate2": CTRATE2_LABELS,
                "ctrate3": CTRATE3_LABELS, "ctrate4": CTRATE4_LABELS,
                "ctrate5": CTRATE5_LABELS, "ctrate6": CTRATE6_LABELS}
STATEMENT_SEED = 20261102
CLAIMS_SEED = 20261103
CLAIMS2_SEED = 20261104
CLAIMS3_SEED = 20261105
CLAIMS4_SEED = 20261106


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


def _as_outcome(said: str) -> str:
    """A reading of a doubt as the label's outcome: a finding (with its differential), aside, or the rest."""
    return "finding" if said in ("finding", "differential") else "aside" if said == "aside" else "keep"


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
                      ("not_assessable", "not_assessable"), ("indication", "indication"),
                      ("procedure", "procedure")):
        if claims.get(key):
            out.add(kind)
    if (owner.get("doubt") or {}).get("kind") == "clinical" or claims.get("uncertainty"):
        out.add("uncertain")        # the doubt the sentence states, kept with the finding (7 November 2026)
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
        doubt = got.get("doubt") or {}
        if verdict == "unsure":
            c[f"{want}_to_a_person"] += 1
        elif verdict == "elsewhere":
            c[f"{want}_set_aside"] += 1
        elif verdict == "doubt":
            # in doubt (7 November 2026): kept in the path with its readings, not certain; is the leading
            # reading the label's?
            c[f"{want}_in_doubt"] += 1
            have = got.get("statement", "")
            lead = "aside" if have == "aside" else "finding" if have == "finding" else "keep"
            c["doubt_leading_right" if lead == want else "doubt_leading_wrong"] += 1
            # is the label's reading among those the doubt keeps (8 November 2026)? A doubt without it is an error
            kept = {_as_outcome(v.get("outcome", "")) for v in (doubt.get("readings") or {}).values()} | {lead}
            c["doubt_keeps_the_right_reading" if want in kept else "doubt_without_the_right_reading"] += 1
            c[f"doubt_kind_{doubt.get('kind', '')}"] += 1
        else:
            have = got.get("statement", "")
            c[f"{want}_attached"] += 1
            have_outcome = "finding" if have == "finding" else "keep"
            if doubt.get("closed_by"):
                c[f"closed_doubts_by_{doubt['closed_by']}"] += 1
                c["closed_doubts"] += 1
                if have_outcome != want:
                    c["closed_doubts_wrong"] += 1
            if doubt.get("kind") == "clinical":
                c["clinical_doubts_kept"] += 1
            if want != "aside":
                c["attached_as_its_outcome" if have_outcome == want else "attached_as_another_outcome"] += 1
                kinds = attached_kinds(got)
                out["kinds_attached_not_in_the_label"].update(kinds - gold - {"nothing"})
                out["kinds_of_the_label_not_attached"].update(gold - kinds - {"nothing"})
        for a in (got.get("models") or {}).values():
            out["dropped_words"] += int(a.get("dropped") or 0)
            refs = a.get("measure_refs") or []
            c["measures_compared_with_a_reference"] += len(refs)
            c["measures_without_a_reference"] += max(len(a.get("measure") or []) - len(refs), 0)
        if "measure_" in (got.get("reader") or ""):
            c["reference_check_to_a_person"] += 1   # a comparison the cited row does not support
        read = got.get("claims") or {}
        levels = [u.get("level", "possible") for u in read.get("uncertain_about", [])]
        below_high = bool(read.get("uncertainty")) and (not levels or any(v != "high" for v in levels))
        if "uncertain" in gold and verdict != "unsure":
            # the uncertainty of the report (10 November 2026): kept, whatever the verdict, if the claims carry it;
            # lost, if a finding the report states as uncertain is read as certain; since 11 November an uncertainty
            # below "high" is kept only by an uncertainty below "high"
            c["uncertain_items"] += 1
            if below_high or (doubt.get("kind") == "clinical" and not levels):
                c["uncertainty_kept"] += 1
            else:
                c["uncertainty_lost"] += 1
                out.setdefault("uncertainty_lost_items", []).append(r["item_id"])
        elif verdict != "unsure" and read.get("uncertainty"):
            # (8) the uncertainty given where the report states (11 November 2026): no uncertainty in the label,
            # or one of level "high" (uncertain_high: "consistent with") read lower
            if "uncertain_high" not in gold or below_high:
                c["uncertainty_given"] += 1
                out.setdefault("uncertainty_given_items", []).append(r["item_id"])
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


def upper_95(errors: int, n: int) -> float:
    """The one-sided 95% upper bound of an error share seen as ``errors`` of ``n`` (Clopper-Pearson): what the
    sample allows to say the true share is below. With 0 errors it is about 3/n (the rule of three)."""
    if n <= 0:
        return 1.0
    if errors >= n:
        return 1.0
    lo, hi = errors / n, 1.0
    for _ in range(60):                       # the p at which P(X <= errors | n, p) = 0.05
        mid = (lo + hi) / 2
        tail, term = 0.0, (1 - mid) ** n
        for k in range(errors + 1):
            tail += term
            term *= (n - k) / (k + 1) * mid / (1 - mid) if mid < 1 else 0
        lo, hi = (mid, hi) if tail > 0.05 else (lo, mid)
    return round(hi, 4)


def criteria(rows: list[dict], arm: str = "claims") -> dict:
    """The criteria (VALIDATION §5), computed on the items whose label has no doubt and, apart, on all of them.
    Since 7 November 2026, with the principle of doubt, every error share must stay under 2%, the rate of
    clinically significant errors of radiology reports themselves: (1) an item to set aside attached as certain;
    (2) a finding set aside; (3) items sent to a person, each a discordance of the document; (4) certain readings
    attached otherwise than the label, closed doubts included; (5) answers lost (in the summary); (6) doubts kept
    in the path, at most 10% of the items (a bound of usefulness, not an error), a doubt that does not keep the
    label's reading among its readings counted as an error (under 2% of the items, 8 November 2026), and closed
    doubts wrong under 2% of the closed ones. Each share has its 95% upper bound."""
    def one(rs: list[dict]) -> dict:
        t = claims_table(rs, arm)
        aside = t.get("aside_items", 0)
        n = len(rs)
        person = sum(t.get(f"{w}_to_a_person", 0) for w in ("finding", "keep", "aside"))
        doubts = sum(t.get(f"{w}_in_doubt", 0) for w in ("finding", "keep", "aside"))
        attached = t.get("attached_as_its_outcome", 0) + t.get("attached_as_another_outcome", 0)
        other = t.get("attached_as_another_outcome", 0)
        closed, closed_wrong = t.get("closed_doubts", 0), t.get("closed_doubts_wrong", 0)
        share = lambda a, b: round(a / b, 4) if b else 0.0   # noqa: E731
        return {"items": n, "aside_items": aside,
                "1_aside_attached_as_certain": t.get("aside_attached", 0),
                "2_findings_set_aside": t.get("finding_set_aside", 0),
                "3_to_a_person_share": share(person, n), "3_upper_95": upper_95(person, n),
                "4_certain_attached_otherwise_share": share(other, attached),
                "4_upper_95": upper_95(other, attached),
                "6_in_doubt_share": share(doubts, n),
                "6_doubt_leading_reading_wrong": t.get("doubt_leading_wrong", 0),
                "6_doubt_without_the_right_reading_share": share(t.get("doubt_without_the_right_reading", 0), n),
                "6_doubt_without_upper_95": upper_95(t.get("doubt_without_the_right_reading", 0), n),
                "6_closed_doubts": closed, "6_closed_doubts_wrong_share": share(closed_wrong, closed),
                # the counts behind the shares (11 November 2026: an adapter is decided on them)
                "counts": {"3_to_a_person": person, "4_attached_otherwise": other,
                           "6_doubt_without_the_right_reading": t.get("doubt_without_the_right_reading", 0),
                           "6_closed_doubts_wrong": closed_wrong},
                "clinical_doubts_kept": t.get("clinical_doubts_kept", 0),
                # (7) the uncertainty of the report lost: under 2% of the items, and reported of the items whose
                # label is uncertain (10 November 2026)
                "7_uncertainty_lost": t.get("uncertainty_lost", 0),
                "7_uncertainty_lost_share": share(t.get("uncertainty_lost", 0), n),
                "7_upper_95": upper_95(t.get("uncertainty_lost", 0), n),
                "7_of_the_uncertain_items": share(t.get("uncertainty_lost", 0), t.get("uncertain_items", 0)),
                "7_uncertain_items": t.get("uncertain_items", 0),
                "7_lost_items": t.get("uncertainty_lost_items", []),
                # (8) the uncertainty given where the report states it as certain, or read lower than "high" where
                # the report says "consistent with": under 2% of the items (11 November 2026)
                "8_uncertainty_given": t.get("uncertainty_given", 0),
                "8_uncertainty_given_share": share(t.get("uncertainty_given", 0), n),
                "8_upper_95": upper_95(t.get("uncertainty_given", 0), n),
                "8_given_items": t.get("uncertainty_given_items", []),
                # the earlier names, to compare with the runs before the principle of doubt
                "1_aside_attached_silently": t.get("aside_attached", 0),
                "4_attached_otherwise_share": t["attached_as_another_outcome_share"]}

    rs = [r for r in rows if arm in r and r.get("gold_claims")]
    return {"without_doubt": one([r for r in rs if not r.get("gold_doubt")]), "all": one(rs),
            "doubtful_items": sum(1 for r in rs if r.get("gold_doubt")),
            "uncertainty_by_reading": uncertainty_by_reading([r for r in rs if not r.get("gold_doubt")], arm)}


def fingerprint(item: dict) -> str:
    """The item as labelled (11 November 2026): the clause, the mention and the structure."""
    import hashlib

    return hashlib.sha256((item["clause"] + "\x00" + item["mention"] + "\x00" + item["structure"]).encode(
        "utf-8")).hexdigest()[:16]


def fingerprints(name: str) -> dict[str, str]:
    """The fingerprints of the labelled items of a sample, if taken (data/gold/<sample>/fingerprints.txt)."""
    path = ROOT / "data/gold" / name / "fingerprints.txt"
    if not path.exists():
        return {}
    return dict(ln.split("|", 1) for ln in path.read_text("utf-8").splitlines() if ln.strip() and not
                ln.startswith("#"))


def uncertainty_by_reading(rows: list[dict], arm: str = "claims") -> dict:
    """Criteria 7 and 8 for each reading of the uncertainty (11 November 2026): each model alone and their union,
    from the answers of one run. The rule, fixed before ctrate6 (VALIDATION): the reading used is the one with the
    fewest errors of 7 and 8 together, if both stay under 2%; on a tie, the one that loses less."""
    rows = [r for r in rows if (r.get(arm) or {}).get("verdict") != "unsure"]
    models = sorted({m for r in rows for m in ((r.get(arm) or {}).get("models") or {})})

    def read(r: dict, which: list[str]) -> list[dict]:
        answers = (r.get(arm) or {}).get("models") or {}
        out = []
        for m in which:
            a = answers.get(m) or {}
            out += a.get("uncertain_about") or [{"words": w, "level": "possible"} for w in a.get("uncertainty", [])]
        return out

    n = len(rows)
    result = {}
    for name, which in [*((m, [m]) for m in models), ("union", models)]:
        lost, given, lost_items, given_items = 0, 0, [], []
        for r in rows:
            gold = set(r.get("gold_claims") or [])
            ua = read(r, which)
            below = any(u.get("level", "possible") != "high" for u in ua)
            if "uncertain" in gold:
                if not below:
                    lost += 1
                    lost_items.append(r["item_id"])
            elif ua and ("uncertain_high" not in gold or below):
                given += 1
                given_items.append(r["item_id"])
        result[name] = {"7_lost": lost, "7_share": round(lost / n, 4) if n else 0.0, "7_upper_95": upper_95(lost, n),
                        "8_given": given, "8_share": round(given / n, 4) if n else 0.0,
                        "8_upper_95": upper_95(given, n), "7_items": lost_items, "8_items": given_items,
                        "passes": n > 0 and lost / n <= 0.02 and given / n <= 0.02}
    passing = [k for k, v in result.items() if v["passes"]]
    result["chosen"] = min(passing, key=lambda k: (result[k]["7_lost"] + result[k]["8_given"],
                                                   result[k]["7_lost"])) if passing else ""
    return result


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
LOST_LIMIT = 0.02     # answers lost: at most 2% (7 November 2026, the principle of doubt: every error share under 2%)


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
    def pid(r: dict) -> str:
        # PARROT numbers its reports; another corpus converted to its form names them ("ctrate-17", 6 November)
        return r["no"] if isinstance(r["no"], str) else f"parrot-{r['no']}"

    rows = [r for r in rows if pid(r) not in exclude and r.get("translation")]
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
            for k, f in enumerate(hp._findings(readers, text, f"{pid(r)}-{lang}")):
                if taken >= per_report or count >= wanted or per[f["structure"]] >= per_structure:
                    continue
                per[f["structure"]] += 1
                taken += 1
                count += 1
                items.append({"item_id": f"{pid(r)}-{lang}-{k}", "pair": pid(r),
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
                  CLAIMS2_SEED: [*LABELS.values(), OWNER_LABELS, STATEMENT_LABELS, CLAIMS_LABELS],
                  CLAIMS3_SEED: [*LABELS.values(), OWNER_LABELS, STATEMENT_LABELS, CLAIMS_LABELS, CLAIMS2_LABELS],
                  CLAIMS4_SEED: [*LABELS.values(), OWNER_LABELS, STATEMENT_LABELS, CLAIMS_LABELS, CLAIMS2_LABELS,
                                 CLAIMS3_LABELS],
                  CTRATE_SEED: [],           # the first sample of CT-RATE: no report of it used before
                  CTRATE2_SEED: [CTRATE_LABELS],
                  CTRATE3_SEED: [CTRATE_LABELS, CTRATE2_LABELS],
                  CTRATE4_SEED: [CTRATE_LABELS, CTRATE2_LABELS, CTRATE3_LABELS],
                  CTRATE5_SEED: [CTRATE_LABELS, CTRATE2_LABELS, CTRATE3_LABELS, CTRATE4_LABELS],
                  CTRATE6_SEED: [CTRATE_LABELS, CTRATE2_LABELS, CTRATE3_LABELS, CTRATE4_LABELS, CTRATE5_LABELS]}
        for path in before.get(a.seed, [*LABELS.values(), *SAMPLES.values()]):
            if path.exists():
                exclude |= {m.group(0) for line in path.read_text("utf-8").splitlines()
                            if line.strip() and not line.startswith("#")
                            for m in [re.match(r"(?:parrot|ctrate)-\d+", line.strip())] if m}
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
    ap.add_argument("--corpus", action="append", default=[],
                    help="another corpus in PARROT's form (scripts/ctrate_corpus.py), for the reports' context")
    ap.add_argument("--references", help="the table of reference values (default data/reference/values.yml)")
    ap.add_argument("--passages", help="the local copy of the paragraphs of their sources (jsonl), if any")
    ap.add_argument("--memory", default="", help="the memory of verified examples (default data/reading/memory."
                    "jsonl; 'off' for none)")
    ap.add_argument("--memory-items", default="owner_items_*.jsonl",
                    help="the items whose sentences the memory's examples are read from (a glob)")
    ap.add_argument("--only", default="", help="a diagnosis: only these items (comma-separated ids); its numbers "
                    "are no measure, the items were seen")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    only = {x.strip() for x in args.only.split(",") if x.strip()}

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
    for corpus in [args.parrot, *args.corpus] if args.parrot else args.corpus:
        for line in Path(corpus).read_text("utf-8").split("\n"):
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
    # the reference library (4 November 2026): what a measure is compared with, cited by the models
    from melampo.memory.reference_values import ReferenceValues

    references = ReferenceValues.load(Path(args.references) if args.references else None,
                                      Path(args.passages) if args.passages else None)
    claims = FindingOwner(reader_only.reader, chats, reader_only.anatomy, question="claims",
                          anomalies=anomalies, references=references) if chats else None
    from melampo.memory.finding_owner import report_context
    away = elsewhere_pairs()
    # the manual and the memory of verified examples (11 November 2026); the samples measured here never give
    # examples to themselves
    from melampo.memory import reading_memory as rm

    measured = tuple(spec.split("=", 1)[0] for spec in args.items)
    memory_note = {"manual_version": rm.manual_version(), "examples": 0, "excluded_samples": list(measured)}
    if args.memory.strip().lower() != "off":
        import glob

        texts = {}
        for f in sorted(glob.glob(args.memory_items)):
            sample = Path(f).stem.replace("owner_items_", "")
            known = fingerprints(sample)
            for x in Path(f).read_text("utf-8").splitlines():
                if x.strip():
                    it = json.loads(x)
                    if known.get(it["item_id"], fingerprint(it)) == fingerprint(it):
                        texts[it["item_id"]] = it       # an example whose item changed is not its sentence
        memory = rm.ReadingMemory.load(Path(args.memory) if args.memory else None, texts, measured)
        rm.use(memory)
        memory_note["examples"] = len(memory.examples)

    rows = []
    for spec in args.items:
        name, path = spec.split("=", 1)
        if name in CLAIMS_FILES and not CLAIMS_FILES[name].exists():
            raise SystemExit(f"the labels of {name} are not written yet ({CLAIMS_FILES[name]}): draw the sample with "
                             "sample_only, label it, then run")
        labels = hp.read_labels(LABELS[name]) if name in LABELS else {}
        own = {k: v for k, v in labels.items() if k.endswith(("-it", "-en"))}
        items = [json.loads(x) for x in Path(path).read_text("utf-8").splitlines() if x.strip()]
        # an item drawn again that is not the item labelled is left out, and said (11 November 2026)
        known = fingerprints(name)
        changed = [it["item_id"] for it in items if it["item_id"] in known and known[it["item_id"]] != fingerprint(it)]
        if changed:
            memory_note.setdefault("changed_items", {})[name] = changed
            print(f"{name}: {len(changed)} items are not the items labelled, left out: {changed[:10]}",
                  file=sys.stderr)
            items = [it for it in items if it["item_id"] not in set(changed)]
        if only:
            items = [it for it in items if it["item_id"] in only]

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
            if item.get("modality"):
                context["modality"] = str(item["modality"]).upper()   # the technique the reference values hold for
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
    report = {"by_set": {}, "reading": memory_note}
    if only:
        report["diagnosis_only"] = sorted(only)     # seen items: why, not how much
    for name in sorted({r["set"] for r in rows}):
        rs = [r for r in rows if r["set"] == name]
        report["by_set"][name] = {arm: table(rs, arm) for arm in ("reader", "reader+models", "statement", "claims")
                                  if any(arm in r for r in rs)}
        if name in CLAIMS_FILES and any("claims" in r for r in rs):
            report["by_set"][name]["claims_kinds"] = claims_table(rs, "claims")
            report["by_set"][name]["criteria"] = criteria(rs, "claims")
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
                q = build_question(owner, report=item.get("pair") or f"parrot-{item.get('report')}",
                                   language=item["language"],
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
