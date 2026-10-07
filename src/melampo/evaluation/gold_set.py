"""Gold set for the anatomy linker: sampling, blind annotation sheets, agreement, certification.

**This module builds the instrument, not the data.** A gold set is real report text,
pseudonymised at the source, labelled by two radiologists who did not write the lexicon.
Nothing here invents text or labels. The functions only:

* find the anatomical mentions the lexicon and parts table know in each report (`propose_mentions`);
* draw a sample that is fair to the deployment population: one mention per report, stratified by
  language and structure family (`sample_items`);
* write blind annotation sheets (the system's answer is never shown: a reviewer who sees a wrong
  suggestion first is the failure Dratsch et al., Radiology 2023, measured) (`write_sheets`);
* compute agreement between the two annotators and list what the adjudicator must decide
  (`agreement`, `adjudication_queue`);
* score the linker against the final labels and say what is, and is not, certified (`evaluate`).

A certification claim is a property of the pair (system, population). It holds only if the items
are a random sample of the population the system will see, were never used during development,
and the labels are correct. `evaluate` states those conditions in its verdict.
"""

import csv
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ..memory import anatomy_linker as al
from ..memory.report_state import ReportState
from .linking_bench import upper_error_bound

NONE_IN_CLASSES = (
    "NONE_IN_CLASSES"  # a real anatomical structure that is not one of the classes
)
NOT_ANATOMY = (
    "NOT_ANATOMY"  # not an anatomical structure ("T2", "LM" as sequence or coronary...)
)
AMBIGUOUS = "AMBIGUOUS"  # the text does not say which structure
SPECIAL = (NONE_IN_CLASSES, NOT_ANATOMY, AMBIGUOUS)
RELATIONS = ("equal", "part_of", "contour_of", "approx")
SHEET_FIELDS = (
    "item_id", "report_id", "language", "sentence", "mention", "start", "end",
    "structure", "relation", "side_in_text", "note",
)  # fmt: skip
_TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_SIDE = {
    "destro",
    "destra",
    "dx",
    "sinistro",
    "sinistra",
    "sn",
    "sx",
    "right",
    "left",
    "bilaterale",
    "bilateral",
}
_IT = frozenset(
    "il lo la i gli le un una di del della dei delle nel nella con per non senza si".split()
)
_EN = frozenset("the of and with without no is are there in at to a an".split())


def language_of(text: str) -> str:
    words = {w.lower() for w in _TOKEN.findall(text)}
    it, en = len(words & _IT), len(words & _EN)
    return "it" if it > en else "en" if en > it else "unknown"


_STOP = re.compile(r"[.;:!?]")


def _breaks(text: str, spans: list[tuple[int, int, str]], k: int) -> bool:
    """A sentence ends between token k and token k+1 ("... lobe. Bilateral", "T12. Right", "2. Lumbar").

    A full stop before a lower-case word is an abbreviation ("lobo sup. dx") and does not end
    anything; a mention never crosses a boundary, nor takes a side word from the next sentence.
    """
    if k < 0 or k + 1 >= len(spans):
        return False
    gap = text[spans[k][1] : spans[k + 1][0]]
    word = spans[k + 1][2]
    return bool(_STOP.search(gap)) and (word[0].isupper() or word[0].isdigit())


def propose_mentions(
    text: str, lexicon: al.Lexicon, parts: Any = None, max_words: int = 6
) -> list[dict[str, Any]]:
    """Known names in the text, longest first, no overlaps. Annotators add the ones this misses."""
    spans = [(m.start(), m.end(), m.group()) for m in _TOKEN.finditer(text)]
    taken = [False] * len(spans)
    found: list[dict[str, Any]] = []
    for size in range(max_words, 0, -1):
        for i in range(len(spans) - size + 1):
            if any(taken[i : i + size]) or any(
                _breaks(text, spans, k) for k in range(i, i + size - 1)
            ):
                continue
            start, end = spans[i][0], spans[i + size - 1][1]
            piece = text[start:end]
            if not _known(piece, lexicon, parts):
                continue
            j0, j1 = i, i + size - 1

            # drop edge words that add nothing ("del sigma con" -> "sigma")
            # Trim an edge word that adds nothing ("del", "con"), or that only leniency accepts
            # ("... e dilatazione del"), but never into an exact name ("colon discendente").
            def current() -> str:
                return text[spans[j0][0] : spans[j1][1]]

            def trims(shorter: str) -> bool:
                if not _known(shorter, lexicon, parts):
                    return False
                same = al.normalise(shorter, keep_noise=True) == al.normalise(
                    current(), keep_noise=True
                )
                return same or not _exact(current(), lexicon, parts)

            while j0 < j1 and trims(text[spans[j0 + 1][0] : spans[j1][1]]):
                j0 += 1
            while j1 > j0 and trims(text[spans[j0][0] : spans[j1 - 1][1]]):
                j1 -= 1
            if (
                j0 > 0
                and not taken[j0 - 1]
                and spans[j0 - 1][2].lower() in _SIDE
                and not _breaks(text, spans, j0 - 1)
            ):
                j0 -= 1
            if (
                j1 + 1 < len(spans)
                and not taken[j1 + 1]
                and spans[j1 + 1][2].lower() in _SIDE
                and not _breaks(text, spans, j1)
            ):
                j1 += 1
            for k in range(j0, j1 + 1):
                taken[k] = True
            start, end = spans[j0][0], spans[j1][1]
            found.append({"mention": text[start:end], "start": start, "end": end})
    return sorted(found, key=lambda m: m["start"])


def _exact(piece: str, lexicon: al.Lexicon, parts: Any) -> bool:
    """The piece is a name as written: no tissue or space word dropped, or a whole table entry."""
    kept = al.normalise(piece, keep_noise=True)
    if kept == al.normalise(piece) and kept in lexicon.index:
        return True
    if parts is None:
        return False
    side = {al.SIDE_RIGHT, al.SIDE_LEFT, al.SIDE_BOTH}
    return frozenset(t for t in kept if t not in side) in parts.strict


def _known(piece: str, lexicon: al.Lexicon, parts: Any) -> bool:
    tokens = al.normalise(piece)
    if tokens in lexicon.index:
        return True
    if parts is None:
        return False
    kept = al.normalise(piece, keep_noise=True)
    side = {al.SIDE_RIGHT, al.SIDE_LEFT, al.SIDE_BOTH}
    key = frozenset(t for t in kept if t not in side)
    return key in parts.strict or frozenset(kept) in parts.never


def sentence_of(text: str, start: int, end: int) -> str:
    """The sentence of the mention, without the header label or the neighbouring sentences."""
    return ReportState.parse(text).sentence_at(start, end)


def family(cid: str | None) -> str:
    if not cid:
        return "unknown"
    if cid.startswith(("rib_", "vertebrae_", "liver_segment_")):
        return (
            cid.rsplit("_", 1)[0]
            if cid.startswith("liver_segment_")
            else cid.split("_")[0]
        )
    return re.sub(r"_(left|right)$", "", cid)


def sample_items(
    reports: Iterable[dict[str, Any]],
    lexicon: al.Lexicon,
    parts: Any = None,
    *,
    per_report: int = 1,
    n: int | None = None,
    seed: int = 20261006,
    cap_per_mention: int | None = None,
) -> list[dict[str, Any]]:
    """One (or `per_report`) random mention per report; stratified draw by language when `n` is set.

    `cap_per_mention` keeps at most that many items with the same written mention (case-folded). A
    corpus where "heart" is 62% of the proposals (IU X-ray) would otherwise spend the annotators' time
    on one easy name and say nothing about the rare structures. The capped sample is not
    population-weighted: report error per structure, not one overall rate.

    One mention per report keeps items independent: two mentions of the same report share the
    author's habits and the same dictation errors, and would inflate the sample size.
    """
    rng = random.Random(seed)
    pool: list[dict[str, Any]] = []
    # With a target size the reports are visited in random order and the walk stops once there
    # are enough proposals to draw from (a corpus of 60,000 case reports takes two hours to scan
    # whole, and the sample only needs a few hundred). Without `n`, every report is read.
    wanted = n * (5 if cap_per_mention else 3) if n is not None else None
    if wanted is not None:
        reports = list(reports)
        rng.shuffle(reports)
    for report in reports:
        if wanted is not None and len(pool) >= wanted:
            break
        text = report["text"]
        mentions = propose_mentions(text, lexicon, parts)
        if not mentions:
            continue
        picked = rng.sample(mentions, min(per_report, len(mentions)))
        language = report.get("language") or language_of(text)
        state = ReportState.parse(text)
        for m in picked:
            pool.append(
                {
                    "report_id": report["report_id"],
                    "language": language,
                    "sentence": state.sentence_at(m["start"], m["end"]),
                    "section": state.section_at(m["start"]),
                    **m,
                }
            )
    rng.shuffle(pool)
    if cap_per_mention is not None:
        seen: dict[str, int] = defaultdict(int)
        capped = []
        for item in pool:
            key = item["mention"].casefold()
            if seen[key] < cap_per_mention:
                seen[key] += 1
                capped.append(item)
        pool = capped
    if n is not None and n < len(pool):
        by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in pool:
            by_language[item["language"]].append(item)
        quota = max(1, n // max(1, len(by_language)))
        chosen: list[dict[str, Any]] = []
        for items in by_language.values():
            chosen += items[:quota]
        rest = [i for i in pool if i not in chosen]
        chosen += rest[: n - len(chosen)]
        pool = chosen[:n]
    for index, item in enumerate(pool, start=1):
        item["item_id"] = f"G{index:05d}"
    return pool


def write_sheets(
    items: Sequence[dict[str, Any]],
    out_dir: Path,
    annotators: Sequence[str] = ("A", "B"),
    class_ids: Sequence[str] = (),
) -> list[Path]:
    """Blind sheets: no system output, annotators see the items in different random orders."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, name in enumerate(annotators):
        order = list(items)
        random.Random(1000 + index).shuffle(order)
        path = out_dir / f"annotator_{name}.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=SHEET_FIELDS)
            writer.writeheader()
            for item in order:
                writer.writerow({k: item.get(k, "") for k in SHEET_FIELDS})
        paths.append(path)
    options = out_dir / "valid_structures.txt"
    options.write_text("\n".join([*class_ids, *SPECIAL]) + "\n", encoding="utf-8")
    return [*paths, options]


def read_sheet(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["item_id"]: row for row in csv.DictReader(handle)}


def validate_sheet(
    rows: dict[str, dict[str, str]], class_ids: Iterable[str]
) -> list[str]:
    valid = set(class_ids) | set(SPECIAL)
    problems = []
    for item_id, row in rows.items():
        structure = row.get("structure", "").strip()
        if not structure:
            problems.append(f"{item_id}: structure is empty")
        elif structure not in valid:
            problems.append(f"{item_id}: unknown structure {structure!r}")
        relation = row.get("relation", "").strip()
        if structure in valid - set(SPECIAL) and relation not in RELATIONS:
            problems.append(
                f"{item_id}: relation {relation!r} is not one of {RELATIONS}"
            )
    return problems


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def agreement(
    first: dict[str, dict[str, str]], second: dict[str, dict[str, str]]
) -> dict[str, Any]:
    ids = sorted(set(first) & set(second))
    s1 = [first[i]["structure"].strip() for i in ids]
    s2 = [second[i]["structure"].strip() for i in ids]
    r1 = [first[i].get("relation", "").strip() for i in ids]
    r2 = [second[i].get("relation", "").strip() for i in ids]
    both = [i for i in range(len(ids)) if s1[i] == s2[i] and s1[i] not in SPECIAL]
    return {
        "items": len(ids),
        "structure_agreement": sum(x == y for x, y in zip(s1, s2, strict=True))
        / len(ids)
        if ids
        else 0.0,
        "structure_kappa": cohen_kappa(s1, s2),
        "relation_agreement": (
            sum(r1[i] == r2[i] for i in both) / len(both) if both else 0.0
        ),
    }


def adjudication_queue(
    first: dict[str, dict[str, str]], second: dict[str, dict[str, str]]
) -> list[dict[str, str]]:
    """Items where the two annotators differ in structure or relation. A third reviewer decides."""
    queue = []
    for item_id in sorted(set(first) & set(second)):
        a, b = first[item_id], second[item_id]
        differ = a["structure"].strip() != b["structure"].strip() or (
            a["structure"].strip() not in SPECIAL
            and a.get("relation", "").strip() != b.get("relation", "").strip()
        )
        if differ:
            queue.append(
                {
                    **{
                        k: a.get(k, "")
                        for k in ("item_id", "report_id", "sentence", "mention")
                    },
                    "structure_A": a["structure"].strip(),
                    "relation_A": a.get("relation", "").strip(),
                    "structure_B": b["structure"].strip(),
                    "relation_B": b.get("relation", "").strip(),
                    "structure": "",
                    "relation": "",
                    "note": "",
                }
            )
    return queue


def merge_gold(
    first: dict[str, dict[str, str]],
    second: dict[str, dict[str, str]],
    adjudicated: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Final labels: agreed, else the adjudicator's. Items still open are left out and counted."""
    adjudicated = adjudicated or {}
    gold = []
    for item_id in sorted(set(first) & set(second)):
        a, b = first[item_id], second[item_id]
        agreed = a["structure"].strip() == b["structure"].strip() and (
            a["structure"].strip() in SPECIAL
            or a.get("relation", "").strip() == b.get("relation", "").strip()
        )
        source = a if agreed else adjudicated.get(item_id)
        if not source or not source.get("structure", "").strip():
            continue
        gold.append(
            {
                "item_id": item_id,
                "report_id": a["report_id"],
                "language": a.get("language", ""),
                "sentence": a["sentence"],
                "mention": a["mention"],
                "start": a.get("start", ""),
                "end": a.get("end", ""),
                "structure": source["structure"].strip(),
                "relation": source.get("relation", "").strip() or "equal",
                "agreed": agreed,
            }
        )
    return gold


def evaluate(
    linker: Any,
    gold: Sequence[dict[str, Any]],
    confidence: float = 0.95,
    reports: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Score the linker on final labels. Items labelled NOT_ANATOMY/AMBIGUOUS expect an abstention.

    With ``reports`` (report id -> text) the linker also gets the state of the report and the
    place of the mention, as it will in use; without them it sees the sentence alone.
    """
    states: dict[str, ReportState] = {}
    tally: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    by_language: dict[str, Counter[str]] = defaultdict(Counter)
    errors: list[dict[str, Any]] = []
    for row in gold:
        text = (reports or {}).get(row.get("report_id", ""))
        if text is not None and str(row.get("start", "")).strip().isdigit():
            state = states.setdefault(row["report_id"], ReportState.parse(text))
            result = linker.link(
                row["mention"], row["sentence"], report=state, at=int(row["start"])
            )
        else:
            result = linker.link(row["mention"], row["sentence"])
        label = row["structure"]
        if result.status != al.ACCEPTED:
            outcome = "abstained"
            reasons[result.reason] += 1
        elif label in (NOT_ANATOMY, AMBIGUOUS):
            outcome = "wrong_critical"  # linked something that should not be linked
        elif label == NONE_IN_CLASSES:
            outcome = "accepted_outside_classes"  # an ontology term: checked by hand, not counted as a class error
        elif result.cid != label:
            outcome = "wrong_critical"
        elif result.relation != row["relation"]:
            outcome = "wrong_relation"
        else:
            outcome = "correct"
        tally[outcome] += 1
        by_language[row.get("language", "")][outcome] += 1
        if outcome.startswith("wrong") or outcome == "accepted_outside_classes":
            errors.append(
                {"item_id": row["item_id"], "mention": row["mention"], "gold": label, "gold_relation": row["relation"],
                 "got": result.cid, "got_relation": result.relation, "outcome": outcome, "stage": result.stage}
            )  # fmt: skip
    accepted = tally["correct"] + tally["wrong_critical"] + tally["wrong_relation"]
    wrong = tally["wrong_critical"] + tally["wrong_relation"]
    bound = upper_error_bound(wrong, accepted, confidence)
    return {
        "n": len(gold),
        "outcomes": dict(tally),
        "accepted": accepted,
        "wrong": wrong,
        "error_upper_bound": round(bound, 4),
        "confidence": confidence,
        "critical_upper_bound": round(
            upper_error_bound(tally["wrong_critical"], accepted, confidence), 4
        ),
        "coverage": round(accepted / len(gold), 4) if gold else None,
        "abstention_reasons": dict(reasons),
        "by_language": {k: dict(v) for k, v in by_language.items()},
        "errors": errors,
        "verdict": verdict(accepted, wrong, confidence),
    }


def verdict(
    accepted: int, wrong: int, confidence: float = 0.95, target: float = 0.01
) -> str:
    bound = upper_error_bound(wrong, accepted, confidence)
    needed = cases_needed(target, confidence, wrong)
    conditions = (
        " Valid only if the items are a random sample of the population the system will see, were never "
        "used while developing the lexicon, table, prompts or thresholds, and the labels are correct."
    )
    if accepted == 0:
        return "No accepted links: nothing is certified." + conditions
    if bound <= target:
        return (
            f"Error on accepted links <= {bound:.2%} at {confidence:.0%} confidence ({wrong} errors in {accepted})."
            + conditions
        )
    return (
        f"Not certified: {wrong} errors in {accepted} accepted links bound the error only at {bound:.2%} "
        f"at {confidence:.0%}; certifying <= {target:.0%} with this many errors needs {needed} accepted links."
        + conditions
    )


def cases_needed(target: float, confidence: float, errors: int = 0) -> int:
    """Smallest n of accepted links with at most `errors` wrong whose Clopper-Pearson bound is <= target."""
    lo, hi = max(1, errors), 100000
    while lo < hi:
        mid = (lo + hi) // 2
        if upper_error_bound(errors, mid, confidence) <= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def power(
    true_error: float, n: int, max_errors: int, target_bound_ok: bool = True
) -> float:
    """Probability of passing a test of n accepted links with at most `max_errors` errors."""
    return sum(
        math.comb(n, i) * true_error**i * (1 - true_error) ** (n - i)
        for i in range(max_errors + 1)
    )
