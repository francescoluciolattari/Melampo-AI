"""What a patient's card is compared with, after the one procedure has read it (27 November 2026).

The one procedure gives each document its agreed cards (``reader.read_text``). Three readings of the reader before
it are given to the cards here, so that nothing the anatomy reader did is lost:

1. **the reference library** (``memory.reference_values``): a card with a value, a unit and a structure is shown,
   with the verified rows that apply to its structure and exam in the value's unit, to every interpreter, in one
   question per document. The measure is compared with a row only when every interpreter chooses the same row;
   the comparison is then made here, with the number the sentence writes (never the models'), and the measure
   becomes a statement the case graph turns into a measurement (``case_graph_builder.add_measures``). Interpreters
   that choose differently leave the measure without a row, with the doubt kept and shown to a person;
2. **the pathologist's summary** (``data.pathology_report``): the cards are placed in the report's sections; a
   card of the clinical information is the question asked (``indication``), a card of the material is where the
   sample comes from (``specimen``), neither is a current finding; TNM, grade, margins, nodes, invasion, markers
   go to the card of the diagnosis they belong to (``data.pathology_findings.give_the_diagnosis``, the same rule);
3. **the image** (``imaging.findings.image_evidence``): an imaging report paired with its one volume gives each
   card with a structure the segmenters' reading of its sentence (VoxTell for the lesion, TotalSegmentator for the
   structure). Otherwise each card says why there is no image.

Before them (27 November 2026): the part of the document each card's sentence is in, read by the interpreters on
the whole document, is checked with the headings the report writes (``check_parts``); a card the laboratory
extractor already read as a row is marked and enters the case once (``mark_read_by_the_laboratory``).

Nothing here reads the text again or names a finding: it only places the agreed cards.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .agreement import described
from .normalize import read_choices

# -- 0. the part of the document, and the lines the laboratory extractor reads -------------------------------

BEFORE_THE_EXAM = ("indication", "history", "comparison", "technique")


def position_of(card: dict, text: str) -> int:
    """Where the card's own words are in the text (the sentence's start when they are not in it), or -1."""
    quote = str(card.get("quote") or "")
    start = text.find(quote) if quote else -1
    words = str(card.get("words") or "")
    if start >= 0 and words and words in quote:
        start += quote.find(words)
    return start


def check_parts(cards: list[dict], text: str) -> int:
    """The headings the report writes check the part the interpreters read (``memory.report_sections``, as the
    deterministic cues check the router's model): a card whose heading says another part keeps both, with the
    doubt ``part_differs_from_heading``; a sentence under no heading of a known kind is not checked (the
    interpreters read the whole document, the heading-less question included). Returns how many differ."""
    from ..memory.report_sections import section_at

    differ = 0
    for card in cards:
        at = position_of(card, text)
        section = section_at(text, at) if at >= 0 else None
        if section is None or not section.kind or not section.body <= at < section.end:
            continue
        card["part_heading"] = section.kind
        if card.get("part") and card["part"] != section.kind:
            card["doubts"] = [*card.get("doubts", []), "part_differs_from_heading"]
            differ += 1
    return differ


def lab_lines(extraction: Any, text: str) -> list[str]:
    """The lines of the text whose rows the laboratory extractor read (``lab_results``)."""
    lines = text.splitlines()
    out = []
    for row in getattr(extraction, "results", ()) or ():
        n = int(getattr(row, "line_number", 0) or 0)
        if 0 < n <= len(lines) and lines[n - 1].strip():
            out.append(lines[n - 1].strip())
    return out


def mark_read_by_the_laboratory(cards: list[dict], lines: list[str]) -> int:
    """A card whose sentence is a row the laboratory extractor read is that row (``read_by: lab_table``): the
    value enters the case once, as the extractor's measurement, never also as a card."""
    from .agreement import same_sentence

    marked = 0
    for card in cards:
        if any(same_sentence(card.get("quote", ""), line) for line in lines):
            card["read_by"] = "lab_table"
            marked += 1
    return marked


def is_the_question(card: dict) -> bool:
    """A card of the question asked that the requester only suspects ("sospetta appendicite"): the clinical
    question, a hypothesis of the requester, never a current finding."""
    return card.get("part") == "indication" and card.get("state") == "uncertain"


# -- 1. the reference library -------------------------------------------------------------------------------


def technique_of(text: str) -> str:
    """The technique of the exam the report names (its first sentence naming a modality), or "" when it names
    none: then every technique's rows are shown, each with its technique."""
    import re

    head = " ".join(text[:600].split()).lower()
    for technique, pattern in (("CT", r"\b(tc|ct|tac|tomografia computerizzata|computed tomography)\b"),
                               ("MR", r"\b(rm|mri|rmn|risonanza magnetica|magnetic resonance)\b"),
                               ("US", r"\b(eco|ecografia|ecografico|ultrasound|ultrasonography|sonography)\b"),
                               ("RX", r"\b(rx|radiografia|radiograph|x-ray)\b")):
        if re.search(pattern, head):
            return technique
    return ""


def _measure_of(card: dict, references: Any) -> dict | None:
    """The card's value in the units of the library, when the sentence writes the same number."""
    from ..memory.reference_values import measures_in

    value = card.get("value") or {}
    if not isinstance(value, dict) or value.get("number") is None or not value.get("unit"):
        return None
    given = measures_in(f"{float(value['number']):g} {value['unit']}")
    if not given:
        return None
    m = given[0]
    written = [q for q in measures_in(str(card.get("quote") or ""))
               if q["unit"] == m["unit"] and abs(q["value"] - m["value"]) < 1e-6]
    return {**m, "text": written[0]["text"]} if written else None


def reference_items(cards: list[dict], text: str, references: Any, technique: str = "",
                    sex: str = "") -> list[tuple[int, dict, list]]:
    """The cards that can be compared: agreed by every interpreter, a structure, a value the sentence writes, and
    at least one verified row of the structure, the exam and the value's unit."""
    out = []
    for n, card in enumerate(cards):
        doubts = card.get("doubts") or []
        if "one_interpreter" in doubts or not card.get("site_code") or card.get("read_by"):
            continue
        measure = _measure_of(card, references)
        if measure is None:
            continue
        rows = [r for r in references.rows_for(card["site_code"], technique, sex) if r.unit == measure["unit"]]
        if rows:
            out.append((n, measure, rows))
    return out


def reference_prompt(items: list[tuple[dict, dict, list]]) -> str:
    """One question for a whole document: each measure with its sentence and the rows it may be compared with."""
    blocks = []
    for i, (card, measure, rows) in enumerate(items, 1):
        lines = "\n".join(f"   {k}. {row.line()}" for k, row in enumerate(rows, 1))
        blocks.append(f"{i}. Sentence: \"{card.get('quote', '')}\"\n   Measure: \"{measure['text']}\"\n   Finding: "
                      f"{json.dumps(described(card, 'case'), ensure_ascii=False)}\n{lines}")
    return ("Measures read in a document of one patient. For each, which reference row describes what this measure "
            "measures (the same structure, the same aspect, the same way of measuring)? Do not compare the numbers: "
            "only say which row it is, or none when no row measures the same thing.\n\n"
            + "\n\n".join(blocks) + "\n\n"
            "Answer only JSON: [{\"item\": 1, \"row\": the number of the row, or null when none is}, ...]")


def compare_with_references(cards: list[dict], text: str, references: Any, ask: Callable[[str, str], str] | None,
                            readers: list[str], *, report: str = "", technique: str | None = None,
                            sex: str = "", asked: list | None = None) -> tuple[list[dict], list[dict]]:
    """The measures of one document compared with the library. Each card compared gets ``reference``: the row,
    the measure as written, its value in the row's unit and what the row says of it (below, within, above),
    or, when the interpreters choose differently, the doubt and their answers. Returns the statements of the
    measures compared (for ``add_measures``) and what goes to a person."""
    from ..memory.reference_values import compare

    if references is None or ask is None or not readers:
        return [], []
    technique = technique_of(text) if technique is None else technique
    items = reference_items(cards, text, references, technique, sex)
    if not items:
        return [], []
    prompt = reference_prompt([(cards[n], m, rows) for n, m, rows in items])
    sizes = [len(rows) for _, _, rows in items]
    answers = {r: read_choices(ask(r, prompt), sizes, key="row") for r in readers}
    if asked is not None:
        asked.append({"references": len(items), "answers": {r: len(a) for r, a in answers.items()}})
    statements, to_a_person = [], []
    for k, (n, measure, rows) in enumerate(items):
        card = cards[n]
        chosen = {answers[r].get(k) for r in readers}
        said = {r: (rows[answers[r][k] - 1].id if answers[r].get(k) else None) for r in readers}
        if len(chosen) == 1 and None not in chosen:
            row = rows[chosen.pop() - 1]
            against = compare(measure["value"], row)
            card["reference"] = {"row": row.id, "measure": measure["text"], "value": measure["value"],
                                 "unit": row.unit, "against": against, "answers": said}
            statements.append({"report": report, "statement": "measure", "reference": row.id,
                               "piece": measure["text"], "clause": card.get("quote", ""), "against": against,
                               "structure": card.get("site_code", ""),
                               "mention": card.get("words") or card.get("name", "")})
        elif any(said.values()):
            # the interpreters do not choose the same row: no comparison is made, and the doubt is kept
            card["reference"] = {"row": None, "measure": measure["text"], "doubt": "reference_chosen_differently",
                                 "answers": said}
            to_a_person.append({"report": report, "mention": card.get("words") or card.get("name", ""),
                                "reason": "doubt:reference_chosen_differently", "clause": card.get("quote", "")})
    return statements, to_a_person


# -- 2. the pathologist's summary ---------------------------------------------------------------------------


def give_the_pathology(cards: list[dict], text: str, summary: dict) -> tuple[list[dict], dict]:
    """The cards of a pathology report placed in its sections, with the diagnosis and its items given to the
    cards of the diagnosis (``pathology_findings.give_the_diagnosis``, the rule of the reader before). A card of
    the clinical information or of the material gets ``pathology_statement`` (indication, specimen) and is not a
    current finding. Returns the statements of those cards and what stays with the report."""
    from ..data.pathology_findings import _section, give_the_diagnosis

    sections = {k: list(v) for k, v in (summary.get("sections") or {}).items()}
    rows, specimen = [], []
    for n, card in enumerate(cards):
        quote = str(card.get("quote") or "")
        start = position_of(card, text)     # the card's own words, after the section's heading ("Diagnosi:")
        if start < 0:
            continue
        row = {"start": start, "mention": card.get("words") or card.get("name", ""),
               "structure": card.get("site_code", ""), "clause": quote, "card": n}
        rows.append(row)
        if _section(sections, start) == "material" and card.get("site_code"):
            specimen.append({"cid": card["site_code"]})
    kept, statements, report = give_the_diagnosis(rows, [], text, summary, specimen)
    for row in kept:
        card = cards[row["card"]]
        card["pathology_section"] = row["pathology_section"]
        if row.get("pathology"):
            card["pathology"] = row["pathology"]
    placed = {(s["clause"], s["mention"]): s for s in statements}
    for row in rows:
        s = placed.get((row["clause"], row["mention"]))
        if s is not None:
            cards[row["card"]]["pathology_section"] = s["section"]
            cards[row["card"]]["pathology_statement"] = s["statement"]
    return statements, report


# -- 3. the image -----------------------------------------------------------------------------------------------


def _image_dict(image: Any) -> dict:
    from ..imaging.findings import AGREE, AMBIGUOUS, UNCHECKED

    out = {"status": image.status, "provider": image.provider, "version": image.version,
           "size_check": image.size_check, "notes": list(image.notes), "review": list(image.review)}
    if image.in_organ_share is not None:
        out["in_organ_share"] = image.in_organ_share
    if image.status in (AGREE, UNCHECKED, AMBIGUOUS):
        if image.measures is not None:
            out["measures"] = image.measures.as_dict()
        if image.involvement:
            out["involvement"] = [dict(i) for i in image.involvement]
        if image.frame_of_reference:
            out["frame_of_reference"] = image.frame_of_reference
    return out


def _seen(card: dict) -> bool:
    """A card the image may show: a manifestation of the patient, not absent, at a structure, of what this exam
    observes (not the question asked, the history, the comparison with an earlier exam or the technique)."""
    return (card.get("category") == "manifestation" and card.get("state") != "absent"
            and card.get("subject") != "family_member" and bool(card.get("site_code"))
            and not card.get("read_by") and card.get("part") not in BEFORE_THE_EXAM)


def image_of_cards(cards: list[dict], volume: dict | None, lesion_segmenter: Any = None, organ_segmenter: Any = None,
                   why_not: str = "") -> int:
    """The image side of each card the image may show (``image``): the segmenters' reading of the card's sentence
    in the volume, with the checks of the reader before (``image_evidence``). Without a volume or segmenters
    the card says why. Returns how many cards were looked for in the image."""
    from ..imaging.findings import NO_IMAGE, image_evidence, sizes_in_text
    from ..imaging.segmenters import SegmenterUnavailable
    from ..memory.reference_values import measures_in

    shown = [c for c in cards if _seen(c)]
    if volume is None or lesion_segmenter is None:
        reason = why_not or ("segmenters_not_configured" if volume is not None else "no_volume")
        for card in shown:
            card["image"] = {"status": NO_IMAGE, "notes": [reason]}
        return 0
    array, affine = volume["array"], volume["affine"]
    modality, contrast = volume.get("modality") or "", volume.get("contrast") or "unknown"
    frame = volume.get("frame_of_reference") or ""
    structures, asked = None, False
    for card in shown:
        quote = str(card.get("quote") or card.get("words") or "")
        size = None
        value = card.get("value") or {}
        if isinstance(value, dict) and value.get("number") is not None and value.get("unit"):
            got = [m for m in measures_in(f"{float(value['number']):g} {value['unit']}") if m["unit"] == "mm"]
            size = got[0]["value"] if got else None
        if size is None:
            size = max((s["largest_mm"] for s in sizes_in_text(quote)), default=None)
        try:
            seg = lesion_segmenter.segment(array, affine, [quote])[0]
            if not asked and organ_segmenter is not None and hasattr(organ_segmenter, "all_structures"):
                asked = True
                try:
                    try:
                        structures = organ_segmenter.all_structures(array, affine, modality=modality)
                    except TypeError:
                        structures = organ_segmenter.all_structures(array, affine)
                except SegmenterUnavailable:
                    structures = None
            organ = None
            if structures and card["site_code"] in structures:
                organ = structures[card["site_code"]]
            elif organ_segmenter is not None:
                organ = organ_segmenter.segment(array, affine, [card["site_code"]])[0].mask
            image = image_evidence(seg, affine, array, organ, size, frame, modality=modality,
                                   structure=card["site_code"], contrast=contrast, structures=structures)
            card["image"] = _image_dict(image)
        except SegmenterUnavailable as error:
            card["image"] = {"status": NO_IMAGE, "notes": [f"segmenter_unavailable:{error}"]}
    return len(shown)


__all__ = ["BEFORE_THE_EXAM", "check_parts", "is_the_question", "lab_lines", "mark_read_by_the_laboratory",
           "position_of", "compare_with_references", "give_the_pathology", "image_of_cards", "reference_items",
           "reference_prompt", "technique_of"]
