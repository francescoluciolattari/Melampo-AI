"""External check of the linker on free English corpora labelled by people who are not us.

Two corpora, both free, both with human concept labels, neither written for this project:

* **CRAFT** (Colorado Richly Annotated Full-Text corpus, v4/v5, CC BY 3.0): 97 full-text articles
  of mouse genetics, every mention of an UBERON concept annotated by hand
  (``concept-annotation/UBERON/UBERON/knowtator/*.xml``, ``articles/txt/*.txt``).
* **MedMentions** (full, CC0): 4,392 PubMed abstracts, mentions annotated with UMLS concepts and
  semantic types (``full/data/corpus_pubtator.txt.gz``).

What it does. The mentions are proposed the same way as for the gold set sheets
(``gold_set.propose_mentions``: known names, longest first), each is linked by the deterministic
linker with the graph and the blind reader in the trace, and every **accepted** link is compared with
the human label that overlaps it:

* ``same``: the label is the linked structure (CRAFT: the UBERON id is a node of the class, its
  side family included; MedMentions: the UMLS id is a cross-reference of one of those nodes);
* ``finer_label``: the label is a part or a kind of the structure (linker coarser);
* ``coarser_label``: the structure is a part or a kind of the label (linker finer than the text);
* ``other_anatomy``: an anatomical label of another structure (a wrong structure, or a mapping gap:
  every one is listed for reading);
* ``not_anatomy`` (MedMentions only): the overlapping label has a non-anatomical semantic type
  ("heart rate" as a clinical attribute, "liver function tests" as a laboratory procedure): by the
  project rule the structure is only the modifier of a measurement, so the link is an error;
* ``inside_a_disease_or_procedure`` (MedMentions only): the only label is a longer disease or
  procedure that contains the structure ("prostate cancer"); the structure is named by the project
  rule, but MedMentions does not label it on its own, so the link is not judged;
* ``other_concept`` (MedMentions only): the label is another concept spelled the same ("SVC"
  stromal vascular cells, "PONS" a scale, "IVC" a cancer stage): an error;
* ``anatomy_cui_unmapped`` (MedMentions only): an anatomical label whose UMLS id no UBERON term
  cross-references (MedMentions often labels "brain" with region or part concepts): not judged;
* ``unlabelled``: no label of that kind overlaps (CRAFT: annotators judged it not an UBERON concept,
  or missed it; listed for reading).

Limits, written where the numbers are. The texts are mouse genetics and biomedical abstracts, not
radiology reports; the labels follow those projects' guidelines (CRAFT annotates "renal" as kidney;
MedMentions annotates the longest concept). The numbers are an external check of the lexicon and of
the deterministic stages, **not** a certification: that needs a fresh sample of the population of
use, labelled by radiologists (``docs/gold_set_protocollo.md``).

    python scripts/external_check.py --craft PATH/craft --medmentions PATH/MedMentions \
        --out external_check.json --markdown external_check.md
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from melampo.evaluation import gold_set as gs  # noqa: E402
from melampo.evaluation.selective_calibration import binomial_cdf, certify  # noqa: E402
from melampo.memory import anatomy_linker as al  # noqa: E402
from melampo.memory import anatomy_parts as ap  # noqa: E402
from melampo.memory.anatomy_graph import AnatomyGraph  # noqa: E402
from melampo.memory.blind_reader import BlindReader  # noqa: E402
from melampo.memory.report_state import ReportState  # noqa: E402

DATA = ROOT / "data" / "linking"

# UMLS semantic types of anatomy (the "Anatomy" group of the UMLS Semantic Network, without cells,
# genes and organisms): T017 anatomical structure, T018 embryonic, T021 fully formed, T022 body
# system, T023 body part/organ/component, T024 tissue, T029 body location or region, T030 body space
# or junction, T031 body substance, T190 anatomical abnormality.
ANATOMY_TYPES = frozenset("T017 T018 T021 T022 T023 T024 T029 T030 T031 T190".split())

# Non-anatomical types that make the structure the modifier of a measurement or of a function
# (the project's NOT_ANATOMY rule): clinical attribute, laboratory procedure, laboratory or test
# result, organ or tissue function, physiologic function, organism function, organism attribute.
MEASUREMENT_TYPES = frozenset("T201 T059 T034 T042 T039 T040 T032".split())

# Diseases, findings, injuries, anomalies and procedures: a structure inside them is still named.
DISEASE_OR_PROCEDURE_TYPES = frozenset(
    "T047 T191 T046 T037 T019 T020 T033 T184 T048 T049 T050 T061 T060 T058 T190".split()
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])|\n+")


def sentence_around(text: str, start: int, end: int) -> str:
    left = 0
    for match in _SENTENCE_END.finditer(text, 0, start):
        left = match.end()
    match = _SENTENCE_END.search(text, end)
    right = match.start() if match else len(text)
    return text[left:right].strip()


# -- corpora ---------------------------------------------------------------------------------------


def craft_documents(root: Path):
    """(doc_id, text, [(start, end, uberon_id)]) for every CRAFT article."""
    folder = root / "concept-annotation" / "UBERON" / "UBERON" / "knowtator"
    for path in sorted(folder.glob("*.xml")):
        doc = path.name.split(".")[0]
        text = (root / "articles" / "txt" / f"{doc}.txt").read_text("utf-8")
        tree = ET.parse(path).getroot()
        classes = {
            cm.get("id"): cm.find("mentionClass").get("id")
            for cm in tree.iter("classMention")
        }
        labels = []
        for annotation in tree.iter("annotation"):
            concept = classes.get(annotation.find("mention").get("id"))
            if not concept or not concept.startswith("UBERON:"):
                continue
            spans = annotation.findall("span")
            start = min(int(s.get("start")) for s in spans)
            end = max(int(s.get("end")) for s in spans)
            labels.append((start, end, concept))
        yield doc, text, labels


def medmentions_documents(root: Path):
    """(pmid, title + abstract, [(start, end, (cui, types))]) for every MedMentions abstract."""
    path = root / "full" / "data" / "corpus_pubtator.txt.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        pmid, title, abstract, labels = None, "", "", []
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                if pmid:
                    yield pmid, f"{title} {abstract}", labels
                pmid, title, abstract, labels = None, "", "", []
                continue
            if "|t|" in line[:20]:
                pmid, title = line.split("|t|", 1)
            elif "|a|" in line[:20]:
                abstract = line.split("|a|", 1)[1]
            else:
                parts = line.split("\t")
                if len(parts) >= 6:
                    cui = parts[5].replace("UMLS:", "")
                    labels.append(
                        (
                            int(parts[1]),
                            int(parts[2]),
                            (cui, frozenset(parts[4].split(","))),
                        )
                    )
        if pmid:
            yield pmid, f"{title} {abstract}", labels


# -- ontology helpers ------------------------------------------------------------------------------


def umls_of_nodes(obo: Path) -> dict[str, set[str]]:
    """UBERON id -> UMLS CUIs it cross-references."""
    found: dict[str, set[str]] = defaultdict(set)
    current = None
    with open(obo, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("[Term]"):
                current = None
            elif line.startswith("id: "):
                current = line[4:].strip()
            elif current and line.startswith("xref: UMLS:"):
                found[current].add(line.split()[1].replace("UMLS:", ""))
    return found


def ancestors(graph: AnatomyGraph, node: str, depth: int = 12) -> set[str]:
    seen, frontier = set(), {node}
    for _ in range(depth):
        frontier = {
            parent
            for current in frontier
            for parent, _ in graph.parents.get(current, ())
            if parent not in seen
        }
        if not frontier:
            break
        seen |= frontier
    return seen


def compare_uberon(graph: AnatomyGraph, cid: str, label: str) -> str:
    nodes = graph.nodes_of(cid)
    if label in nodes:
        return "same"
    if nodes & ancestors(graph, label):
        return "finer_label"
    if any(label in ancestors(graph, n) for n in nodes):
        return "coarser_label"
    return "other_anatomy"


# -- the run ---------------------------------------------------------------------------------------


def _lattice():
    from melampo.memory.chunk_lattice import BlockMemory, ChunkLattice
    from melampo.memory.longer_names import LongerNames

    return ChunkLattice(BlockMemory.load(longer_names=LongerNames.load().names))


def build_linker(obo: Path, chats=None, blocks: bool = False):
    lexicon = al.Lexicon.from_json(
        json.loads((DATA / "anatomy_lexicon.json").read_text("utf-8"))
    )
    parts_json = json.loads((DATA / "anatomy_parts.json").read_text("utf-8"))
    parts = ap.PartTable.from_json(parts_json, lexicon)
    with open(obo, encoding="utf-8") as handle:
        terms = al.load_obo_terms(handle)
    with open(obo, encoding="utf-8") as handle:
        graph = AnatomyGraph.from_obo(handle)
    pool, equivalent = al.build_pool(lexicon, terms)
    blind = BlindReader.from_sources(lexicon, terms, equivalent, graph, parts_json)
    # The texts are articles and abstracts, not radiology reports: the linker is told so
    # (``document_type``), as a caller in production says what it is reading.
    linker = al.AnatomyLinker(
        lexicon,
        pool,
        equivalent,
        parts=parts,
        graph=graph,
        blind=blind,
        document_type="literature",
        chunk_lattice=_lattice() if blocks else None,
    )
    if chats:
        # The same linker with the two models reading the sentence for every link made from the
        # name alone (verify, scope all): what the context check adds on text labelled by others.
        linker.asking = al.AnatomyLinker(
            lexicon,
            pool,
            equivalent,
            parts=parts,
            graph=graph,
            blind=blind,
            document_type="literature",
            chats=chats,
            verify=True,
            verify_all=True,
        )
    return lexicon, parts, graph, linker


def _overlapping(labels, start: int, end: int):
    return [lab for lab in labels if lab[0] < end and start < lab[1]]


def run_corpus(
    name, documents, lexicon, parts, graph, linker, umls, limit=None, workers=4
):
    rows = []
    to_ask = []
    for count, (doc, text, labels) in enumerate(documents):
        if limit and count >= limit:
            break
        state = ReportState.parse(text)
        for item in gs.propose_mentions(text, lexicon, parts):
            sentence = sentence_around(text, item["start"], item["end"])
            result = linker.link(
                item["mention"], sentence, report=state, at=item["start"]
            )
            blind = next((e for e in result.trace if e.stream == "blind"), None)
            over = _overlapping(labels, item["start"], item["end"])
            row = {
                "corpus": name,
                "doc": doc,
                "mention": item["mention"],
                "sentence": sentence[:400],
                "status": result.status,
                "cid": result.cid,
                "stage": result.stage,
                "reason": result.reason,
                "relation": result.relation,
                "support": list(getattr(result, "support", ()) or ()),
                "conflicts": list(getattr(result, "conflicts", ()) or ()),
                "convergence": getattr(result, "convergence", None),
                "blind": blind.verdict if blind else None,
                "block": getattr(result, "block", ""),
                "labels": [],
                "outcome": None,
            }
            if name == "craft":
                row["labels"] = sorted({lab[2] for lab in over})
            else:
                row["labels"] = sorted(
                    {f"{lab[2][0]}|{','.join(sorted(lab[2][1]))}" for lab in over}
                )
            if name == "medmentions":
                row["head"] = _head(text, over, item["start"], item["end"])
                row["label_texts"] = sorted({text[lab[0] : lab[1]] for lab in over})
            if result.status == al.ACCEPTED and result.cid:
                row["outcome"] = outcome(
                    name, result.cid, over, graph, umls, item["start"], item["end"]
                )
            elif getattr(result, "role", "") == "inherent_location" and result.about:
                # Not linked, and not lost: the structure whose measure, function or process the
                # text names (SNOMED CT "inherent location"). The label is compared with it.
                row["role"] = result.role
                row["about"] = result.about
                row["about_outcome"] = outcome(
                    name, result.about, over, graph, umls, item["start"], item["end"]
                )
            if row["outcome"]:
                row["by_project_rule"] = by_project_rule(row)
            if (
                getattr(linker, "asking", None) is not None
                and result.status == al.ACCEPTED
                and result.stage in ("lexicon", "parts")
            ):
                to_ask.append((row, item, sentence, state))
            rows.append(row)
    if to_ask:
        from concurrent.futures import ThreadPoolExecutor

        def ask(task):
            row, item, sentence, state = task
            again = linker.asking.link(
                item["mention"], sentence, report=state, at=item["start"]
            )
            row["verified"] = again.status
            row["verify_reason"] = again.reason

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(ask, to_ask))
    return rows


def recorded_summary(rows):
    """The links the linker did not make but recorded as the inherent location of a measure, a
    function or a process ("heart development", "liver function"): how the corpus label compares
    with the recorded structure. ``structure_agrees``: the label is that structure (or finer);
    ``label_is_the_process``: MedMentions labels the process or the disease, no anatomy to
    compare; ``other``: another structure."""
    recorded = [r for r in rows if r.get("about_outcome")]
    if not recorded:
        return None
    process_labels = ("not_anatomy", "other_concept", "inside_a_disease_or_procedure")
    counts = Counter(
        "structure_agrees"
        if r["about_outcome"] in GOOD
        else "label_is_the_process"
        if r["about_outcome"] in process_labels
        else "unlabelled"
        if r["about_outcome"] == "unlabelled"
        else "other"
        for r in recorded
    )
    return {"recorded": len(recorded), **dict(counts)}


def verify_summary(rows):
    """What the models' reading of the sentence did to the judged links: errors it stopped,
    errors it kept, right links it stopped (by the project rule)."""
    asked = [
        r
        for r in rows
        if "verified" in r and r.get("by_project_rule") in ("agrees", "error")
    ]
    if not asked:
        return None
    kept = [r for r in asked if r["verified"] == al.ACCEPTED]
    errors_kept = sum(r["by_project_rule"] == "error" for r in kept)
    return {
        "asked": len(asked),
        "errors_stopped": sum(
            r["by_project_rule"] == "error" and r["verified"] != al.ACCEPTED
            for r in asked
        ),
        "errors_kept": errors_kept,
        "right_links_stopped": sum(
            r["by_project_rule"] == "agrees" and r["verified"] != al.ACCEPTED
            for r in asked
        ),
        "lost_model_answers": sum(
            "model_unavailable" in (r.get("verify_reason") or "") for r in asked
        ),
        "precision_after_verify": round(1 - errors_kept / len(kept), 4)
        if kept
        else None,
        "error_upper_95_after_verify": round(
            upper_error_bound(errors_kept, len(kept)), 4
        )
        if kept
        else None,
    }


# What the project's labelling rule says where MedMentions labels another concept than the structure
# (docs/gold_set_protocollo.md): the structure is named when it is the origin of cells or tissue
# ("liver cells", "brain parenchyma"), the site of a device or a procedure ("IVC filter", "C2
# pedicle screw"), the subject of a disease ("liver disease") or of an imaging measure ("brain
# volume", "heart size"). These are differences of convention, not errors of the linker; they are
# counted apart and listed, so that anyone can disagree with the rule and recount.
_CELL_TYPES = frozenset(("T025", "T026"))
_DEVICE_TYPES = frozenset(("T074", "T073"))
_MORPHOMETRY = re.compile(
    r"\b(?:volumes?|sizes?|weights?|mass|dimensions?)\b", re.IGNORECASE
)
_DISEASE_WORDS = re.compile(
    r"\b(?:disease|diseases|cancer|cancers|tumou?rs?|failure|injur\w*|surgeon|surgery|"
    r"transplant\w*|resection|carcinoma|metasta\w*|patients?)\b",
    re.IGNORECASE,
)


def by_project_rule(row) -> str:
    """``agrees``, ``convention:<why>`` or ``error`` for an accepted MedMentions link; CRAFT and
    the agreeing outcomes are passed through."""
    if row["outcome"] in GOOD:
        return "agrees"
    if row["outcome"] not in BAD:
        return row["outcome"]
    if row["corpus"] != "medmentions":
        # CRAFT labels UBERON ids: no convention to apply, a disagreement is an error
        return "error"
    types = {t for label in row["labels"] for t in label.split("|", 1)[1].split(",")}
    texts = " ".join(row.get("label_texts", ()))
    if types & _CELL_TYPES:
        return "convention:origin_of_cells"
    if _MORPHOMETRY.search(texts):
        return "convention:imaging_measure_of_the_structure"
    if _DISEASE_WORDS.search(texts):
        return "convention:subject_of_a_disease_or_procedure"
    if types & _DEVICE_TYPES and row["outcome"] == "other_concept":
        return "convention:site_of_a_device"
    return "error"


def project_summary(rows):
    accepted = [
        r for r in rows if r["status"] == al.ACCEPTED and r.get("by_project_rule")
    ]
    verdicts = Counter(r["by_project_rule"] for r in accepted)
    judged = verdicts["agrees"] + verdicts["error"]
    return {
        "verdicts": dict(verdicts),
        "judged": judged,
        "errors": verdicts["error"],
        "precision": round(1 - verdicts["error"] / judged, 4) if judged else None,
        "error_upper_95": round(upper_error_bound(verdicts["error"], judged), 4)
        if judged
        else None,
    }


def _head(text, over, start, end):
    """The word next to the mention inside a longer measurement or function label ("brain
    *activity*", "*function* of the liver"): candidates for the data list of measurement heads."""
    for lab in over:
        if not (lab[2][1] & MEASUREMENT_TYPES) or (lab[1] - lab[0]) <= (end - start):
            continue
        after = re.match(r"\W*([A-Za-z]+)", text[end : lab[1]])
        if after:
            return after.group(1).lower()
        before = re.search(r"([A-Za-z]+)\W*$", text[lab[0] : start])
        if before:
            return before.group(1).lower()
    return None


def outcome(name, cid, over, graph, umls, start=None, end=None) -> str:
    if name == "craft":
        if not over:
            return "unlabelled"
        verdicts = {compare_uberon(graph, cid, lab[2]) for lab in over}
    else:
        anatomical = [lab for lab in over if lab[2][1] & ANATOMY_TYPES]
        if not over:
            return "unlabelled"
        if not anatomical:
            if any(lab[2][1] & MEASUREMENT_TYPES for lab in over):
                return "not_anatomy"
            if all(lab[2][1] & DISEASE_OR_PROCEDURE_TYPES for lab in over):
                # "prostate cancer", "liver transplantation", "cancers (prostate, colon)":
                # MedMentions labels the disease or the procedure only; the structure is named
                # (project rule) but the label cannot say which one: not judged, listed.
                return "inside_a_disease_or_procedure"
            # another concept with the same letters: "SVC" stromal vascular cells, "PONS" a
            # scale, "IVC" a cancer stage, "atlas" a book.
            return "other_concept"
        cuis = {c for n in graph.nodes_of(cid) for c in umls.get(n, ())}
        verdicts = set()
        for lab in anatomical:
            if lab[2][0] in cuis:
                verdicts.add("same")
            else:
                nodes = [n for n, cs in umls.items() if lab[2][0] in cs]
                verdicts |= (
                    {compare_uberon(graph, cid, n) for n in nodes}
                    if nodes
                    else {"anatomy_cui_unmapped"}
                )
    for best in (
        "same",
        "finer_label",
        "coarser_label",
        "other_anatomy",
        "anatomy_cui_unmapped",
    ):
        if best in verdicts:
            return best
    return "other_anatomy"


def upper_error_bound(errors: int, n: int, delta: float = 0.05) -> float:
    """One-sided Clopper-Pearson upper bound of the error rate (log-space binomial, any n)."""
    lo, hi = errors / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binomial_cdf(errors, n, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


GOOD = {"same", "finer_label"}
BAD = {"other_anatomy", "not_anatomy", "other_concept", "coarser_label"}


def summarise(rows):
    accepted = [r for r in rows if r["status"] == al.ACCEPTED]
    outcomes = Counter(r["outcome"] for r in accepted)
    judged = [r for r in accepted if r["outcome"] in GOOD | BAD]
    wrong = sum(r["outcome"] in BAD for r in judged)
    blind = Counter((r["outcome"], r["blind"]) for r in accepted)
    certificate = certify(
        [float(r["convergence"] or 0) for r in judged],
        [r["outcome"] in GOOD for r in judged],
        alpha=0.01,
        delta=0.05,
    )
    heads = Counter(r.get("head") for r in rows if r.get("head"))
    return {
        "heads_inside_measurement_labels": heads.most_common(40),
        "proposed": len(rows),
        "accepted": len(accepted),
        "outcomes": dict(outcomes),
        "judged": len(judged),
        "wrong_among_judged": wrong,
        "precision_among_judged": round(1 - wrong / len(judged), 4) if judged else None,
        "error_upper_95": round(upper_error_bound(wrong, len(judged)), 4)
        if judged
        else None,
        "blind_by_outcome": {
            f"{k[0]}|{k[1]}": v for k, v in sorted(blind.items(), key=str)
        },
        "certificate_1pct": {
            k: v for k, v in certificate.as_dict().items() if k != "steps"
        },
    }


def markdown(report, rows, sample=40, seed=20261009) -> str:
    rng = random.Random(seed)
    out = ["# External check (CRAFT, MedMentions)", ""]
    for name, summary in report.items():
        out += [
            f"## {name}",
            "",
            "```",
            json.dumps(summary, indent=1)[:3000],
            "```",
            "",
        ]
        errors = [
            r
            for r in rows
            if r["corpus"] == name and r.get("by_project_rule") == "error"
        ]
        if errors and name == "medmentions":
            out.append(f"### Errors by the project rule ({len(errors)}, all)")
            out.append("")
            out.append("| mention | class | labels | label text | sentence |")
            out.append("|---|---|---|---|---|")
            for r in sorted(errors, key=lambda r: r["mention"].lower()):
                sentence = r["sentence"].replace("|", "/")[:160]
                out.append(
                    f"| {r['mention']} | {r['cid']} | {' '.join(r['labels'])[:40]} "
                    f"| {' / '.join(r.get('label_texts', ()))[:50]} | {sentence} |"
                )
            out.append("")
        bad = [
            r
            for r in rows
            if r["corpus"] == name and r["outcome"] in BAD | {"unlabelled"}
        ]
        rng.shuffle(bad)
        out.append(f"### Links to read ({min(sample, len(bad))} of {len(bad)})")
        out.append("")
        out.append("| outcome | mention | class | labels | blind | sentence |")
        out.append("|---|---|---|---|---|---|")
        for r in bad[:sample]:
            sentence = r["sentence"].replace("|", "/")[:160]
            out.append(
                f"| {r['outcome']} | {r['mention']} | {r['cid']} | {' '.join(r['labels'])[:60]} "
                f"| {r['blind']} | {sentence} |"
            )
        out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--craft", help="CRAFT repository root")
    parser.add_argument("--medmentions", help="MedMentions repository root")
    parser.add_argument("--uberon", default=str(DATA / "uberon-basic.obo"))
    parser.add_argument("--limit", type=int, help="documents per corpus (smoke test)")
    parser.add_argument("--out", default="external_check.json")
    parser.add_argument("--markdown", default="external_check.md")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="also ask the two models (OPENROUTER_API_KEY) about every link made from the name",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--blocks",
        action="store_true",
        help="write the chunk lattice's reading of each phrase in the rows (decides nothing)",
    )
    args = parser.parse_args(argv)

    chats = None
    if args.verify:
        import os

        from melampo.evaluation import linking_bench as lb

        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            print("--verify needs OPENROUTER_API_KEY", file=sys.stderr)
            return 2
        models = {
            "nemotron-3-super": "nvidia/nemotron-3-super-120b-a12b",
            "gemma-3-27b": "google/gemma-3-27b-it",
        }
        chats = {
            n: lb.OpenRouterChat(s, key, retries=8, min_interval=0.5, max_wait=60.0)
            for n, s in models.items()
        }
    obo = Path(args.uberon)
    lexicon, parts, graph, linker = build_linker(obo, chats, args.blocks)
    umls = umls_of_nodes(obo)
    rows = []
    if args.craft:
        rows += run_corpus(
            "craft",
            craft_documents(Path(args.craft)),
            lexicon,
            parts,
            graph,
            linker,
            umls,
            args.limit,
            args.workers,
        )
    if args.medmentions:
        rows += run_corpus(
            "medmentions",
            medmentions_documents(Path(args.medmentions)),
            lexicon,
            parts,
            graph,
            linker,
            umls,
            args.limit,
            args.workers,
        )
    Path(args.out).with_suffix(".rows.json").write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8"
    )
    report = {
        name: summarise([r for r in rows if r["corpus"] == name])
        for name in sorted({r["corpus"] for r in rows})
    }
    for name in report:
        report[name]["by_project_rule"] = project_summary(
            [r for r in rows if r["corpus"] == name]
        )
        report[name]["recorded_as_inherent_location"] = recorded_summary(
            [r for r in rows if r["corpus"] == name]
        )
        report[name]["verify"] = verify_summary(
            [r for r in rows if r["corpus"] == name]
        )
    Path(args.out).write_text(
        json.dumps({"summary": report, "rows": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    Path(args.markdown).write_text(markdown(report, rows), encoding="utf-8")
    print(json.dumps(report, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
