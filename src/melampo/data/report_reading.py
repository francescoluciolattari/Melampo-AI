"""Reading the case's reports: the narrative reader and the imaging interpreter, in the ingestion (30 October 2026).

Until now the report router (``report_router``) named the interpreters a document needs, and two of them --
``narrative_reader`` and ``imaging`` -- were measured in their probes but never run on a case. This module runs them
on what ``ingestion.prepare_payload`` left in the payload: the physician's note and every attachment's text, and
the DICOM series of the attachments.

* **Narrative reader.** The known anatomical names of each text (``gold_set.propose_mentions``: lexicon and part
  table, longest first) are linked in their sentence by the ``AnatomyLinker`` (the report read as a whole,
  ``ReportState``). An accepted link becomes a site row (``report_graph.site_row``: structure, roles,
  qualifiers); an abstention keeps its reason, and is what a person reads. When the reader's resources and a
  local model are configured (below), the linker runs with the Kintsch reader in "review" mode and the model's
  choice: a disagreement is an abstention, never a silent link.
* **Imaging interpreter.** In every text read, the sites become findings (``imaging.interpreter.interpret_exam``:
  a site the clause describes with a size, a character, a degree or a hedge); a text that is not an imaging
  report keeps them on the text (``not_an_imaging_report``). The findings always have their text side. The image
  side is added only to an imaging report's findings, when the exam is paired with that report without
  ambiguity -- exactly one imaging report and exactly one usable volume series among the
  attachments, of a compatible modality -- and the segmenters are configured: VoxTell isolates the lesion the
  clause describes, TotalSegmentator gives the patient's atlas, and ``findings.image_evidence`` checks organ,
  size and character. Otherwise each finding says why there is no image (``exam_not_paired``,
  ``segmenters_not_configured``, ``segmenter_unavailable:...``): never a silent gap.

Nothing here changes the diagnosis: the reading travels with the case (``CaseContext.provenance["report_reading"]``)
with the list of what a person must look at. Patient data stay on this machine: the only model the linker may ask
is a local one (``report_router.local_chat`` refuses any other host).

Configuration (environment, read by ``ReportReaders.from_env``):

* ``MELAMPO_NARRATIVE_READER=off`` -- do not read (default: on; the linker is built from ``data/linking``).
* ``MELAMPO_UBERON`` -- path of ``uberon.obo``: the anatomy graph (neighbour check, parent proposals).
* ``MELAMPO_CI_RESOURCES`` -- directory written by ``scripts/build_ci_resources.py``: the Kintsch reader, in
  "review" mode.
* ``MELAMPO_LOCAL_CHAT_ENDPOINT`` and ``MELAMPO_LOCAL_CHAT_MODELS`` (comma-separated) -- the local models of the
  linker's choice stage and the readers of the claims. The readers chosen by measure (14 November 2026,
  ctrate8 and ctrate9): Nemotron 3 Ultra and Muse Spark Open, both with reasoning, written with the server's
  names and ``+reasoning`` (``nemotron-3-ultra+reasoning,muse-spark-open+reasoning``); Gemma 4 31B is an option.
  ``MELAMPO_LOCAL_CHAT_EXTRA`` (JSON): how this server is asked to reason, when not through the chat template.
* ``MELAMPO_OWNER_QUESTION`` -- ``claims`` (default since 14 November 2026), ``statement`` or ``owner``.
* ``MELAMPO_UNCERTAINTY_FROM`` -- whose reading of the uncertainty is kept (default: Muse's; ``all``: every
  model's).
* ``MELAMPO_IMAGING_SEGMENTERS=voxtell`` -- VoxTell and TotalSegmentator (``MELAMPO_IMAGING_DEVICE``: cuda or cpu).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parents[3] / "data" / "linking"
VOLUME_MODALITIES = ("CT", "MR", "PT")
IMAGING_KIND = "imaging_report"
NARRATIVE = "narrative_reader"
ENV = ("MELAMPO_NARRATIVE_READER", "MELAMPO_UBERON", "MELAMPO_CI_RESOURCES", "MELAMPO_LOCAL_CHAT_ENDPOINT",
       "MELAMPO_LOCAL_CHAT_MODELS", "MELAMPO_IMAGING_SEGMENTERS", "MELAMPO_IMAGING_DEVICE",
       "MELAMPO_OWNER_QUESTION", "MELAMPO_LOCAL_CHAT_EXTRA", "MELAMPO_REPORT_READER", "MELAMPO_HPO_OBO",
       "MELAMPO_THESAURUS", "MELAMPO_DISEASE_PROFILES")
# the one procedure of reading (25 November 2026, the owner's decision): every document of the case is read with the
# same card, the same interpreters and the same comparison as the medical publications (``melampo.reading``);
# "anatomy" is the reader before it, kept only to measure the change on the same benches
ONE, ANATOMY = "one", "anatomy"
_BUILT: dict[tuple[str, ...], "ReportReaders"] = {}


def _language(text: str) -> str:
    from ..evaluation.gold_set import language_of

    found = language_of(text)
    return found if found in ("it", "en") else "it"


def _reader_chat(endpoint: str, spec: str):
    """A local reader of the reports: ``<model>`` or ``<model>+reasoning`` (14 November 2026), with the room its
    answer needs. The claims answer was measured with 900 tokens, and with 8000 when the model reasons first
    (scripts/role_probe.make_chats); 300, the router's, cut it short."""
    import json

    from .report_router import local_chat

    model, _, flag = spec.partition("+")
    reasoning = flag == "reasoning"
    extra = os.environ.get("MELAMPO_LOCAL_CHAT_EXTRA", "").strip()
    return local_chat(endpoint, model, timeout=180.0 if reasoning else 60.0,
                      max_tokens=8000 if reasoning else 900, reasoning=reasoning,
                      extra=json.loads(extra) if extra else None)


def build_linker(uberon: str | None = None, ci_resources: str | None = None, chat_endpoint: str | None = None,
                 chat_models: tuple[str, ...] = ()) -> tuple[Any, list[str]]:
    """The deployment linker from the repository's data, with what is configured. Returns (linker, notes)."""
    import json

    from ..memory import anatomy_linker as al
    from ..memory import anatomy_parts as ap

    notes: list[str] = []
    lexicon = al.Lexicon.from_json(json.loads((DATA / "anatomy_lexicon.json").read_text("utf-8")))
    parts = ap.PartTable.from_json(json.loads((DATA / "anatomy_parts.json").read_text("utf-8")), lexicon)
    terms: list = []
    graph = None
    if uberon and Path(uberon).exists():
        from ..memory.anatomy_graph import AnatomyGraph

        with open(uberon, encoding="utf-8") as handle:
            terms = al.load_obo_terms(handle)
        with open(uberon, encoding="utf-8") as handle:
            graph = AnatomyGraph.from_obo(handle)
        notes.append("anatomy_graph:uberon")
    elif uberon:
        notes.append("anatomy_graph:path_not_found")
    pool, equivalent = al.build_pool(lexicon, terms)
    kwargs: dict[str, Any] = {"parts": parts, "graph": graph}
    if ci_resources:
        try:
            from ..memory import chunk_lattice as cl
            from ..memory.ci_reader import load_reader
            from ..memory.longer_names import LongerNames

            lattice = cl.ChunkLattice(cl.BlockMemory.load(cl.DEFAULT_PATH, LongerNames.load().names,
                                                          cl.load_anatomy_names()))
            kwargs.update(chunk_lattice=lattice, ci_reader=load_reader(Path(ci_resources), lattice),
                          ci_reader_mode="review")
            notes.append("kintsch_reader:review")
        except Exception as error:  # noqa: BLE001 -- missing resources: the linker reads without the reader
            notes.append(f"kintsch_reader_unavailable:{type(error).__name__}")
    if chat_endpoint and chat_models:
        try:
            kwargs["chats"] = {m: _reader_chat(chat_endpoint, m) for m in chat_models}
            notes.append("models:" + ",".join(chat_models))
        except ValueError as error:   # not a local host: refused, patient reports never leave the machine
            notes.append(f"models_refused:{error}")
    return al.AnatomyLinker(lexicon, pool, equivalent, **kwargs), notes


def finding_owner(linker: Any) -> tuple[Any, str]:
    """The two readers of "is the clause about this structure" (``finding_owner``): the linker's Kintsch reader, or
    one built on the chunk lattice alone (it needs no corpus), and the linker's local models if there are any."""
    from ..memory.finding_owner import FindingOwner

    reader = getattr(linker, "ci_reader", None)
    if reader is None:
        try:
            from ..memory import chunk_lattice as cl
            from ..memory.ci_reader import CIReader
            from ..memory.longer_names import LongerNames

            lattice = getattr(linker, "chunk_lattice", None) or cl.ChunkLattice(
                cl.BlockMemory.load(cl.DEFAULT_PATH, LongerNames.load().names, cl.load_anatomy_names()))
            reader = CIReader(lattice)
        except Exception as error:  # noqa: BLE001 -- without the lattice's data the owner is not built
            return None, f"finding_owner_unavailable:{type(error).__name__}"
    chats = dict(getattr(linker, "chats", None) or {})
    anatomy, note = None, ""
    uberon = os.environ.get("MELAMPO_UBERON")
    if uberon and Path(uberon).exists():
        from ..memory.finding_owner import Anatomy

        try:
            anatomy, note = Anatomy.from_uberon(Path(uberon)), "+uberon"
        except Exception as error:  # noqa: BLE001 -- the owner reads without the organs
            note = f"+uberon_unavailable:{type(error).__name__}"
    # the question asked of the models: "is the finding of this structure?" (default) or, once measured, "what
    # does the sentence say of this structure?" (MELAMPO_OWNER_QUESTION=statement, 1 November 2026)
    # ... or claim by claim (MELAMPO_OWNER_QUESTION=claims, 2 November 2026), with what the reader knows of
    # abnormalities from HPO and PATO (MELAMPO_HPO, MELAMPO_PATO)
    # claim by claim is the default since 14 November 2026: on ctrate9 (293 items labelled before any reading)
    # Nemotron 3 Ultra and Muse Spark Open, both with reasoning, read it with 0 errors and 3 doubts (VALIDATION)
    question = os.environ.get("MELAMPO_OWNER_QUESTION", "claims")
    question = question if question in ("owner", "statement", "claims") else "claims"
    anomalies = None
    if question == "claims":
        from ..memory.what_is_said import AnomalyWords

        try:
            anomalies = AnomalyWords.from_env()
        except Exception as error:  # noqa: BLE001 -- the reader checks less without the ontologies
            note += f"+anomalies_unavailable:{type(error).__name__}"
    references = None
    if question == "claims":
        # the reference library (4 November 2026): the table, and the local paragraphs of its sources
        # (MELAMPO_REFERENCE_PASSAGES); without it a measure is only a measure
        from ..memory.reference_values import ReferenceValues

        try:
            references = ReferenceValues.load()
        except Exception as error:  # noqa: BLE001
            note += f"+references_unavailable:{type(error).__name__}"
    phenotypes = None
    if question == "claims":
        # the concept of what is seen, in the same reading (14 November 2026): HPO (MELAMPO_HPO_OBO), UBERON
        # (MELAMPO_UBERON) and PATO (MELAMPO_PATO); without them the words are read and no term is chosen
        hpo, pato = os.environ.get("MELAMPO_HPO_OBO", ""), os.environ.get("MELAMPO_PATO", "")
        if hpo and uberon and Path(hpo).exists() and Path(uberon).exists():
            from ..memory import hpo_candidates as hc

            try:
                phenotypes = hc.load(Path(hpo), Path(uberon), Path(pato) if pato else None)
                note += "+concepts"
            except Exception as error:  # noqa: BLE001
                note += f"+concepts_unavailable:{type(error).__name__}"
    return (FindingOwner(reader, chats, anatomy, question=question, anomalies=anomalies, references=references,
                         phenotypes=phenotypes),
            "finding_owner:reader" + ("+models" if chats else "") + note + (f":{question}" if question != "owner" else ""))


def one_procedure_interpreters() -> tuple[dict[str, Any], list[str]]:
    """The interpreters of the one procedure for a patient's documents: the models of
    ``MELAMPO_LOCAL_CHAT_MODELS`` on ``MELAMPO_LOCAL_CHAT_ENDPOINT`` (this machine only), with the room a whole
    document's reading needs, reasoning included."""
    import json

    from .report_router import local_chat

    endpoint = os.environ.get("MELAMPO_LOCAL_CHAT_ENDPOINT", "").strip()
    models = [m.strip() for m in os.environ.get("MELAMPO_LOCAL_CHAT_MODELS", "").split(",") if m.strip()]
    if not endpoint or not models:
        return {}, ["interpreters_not_configured"]
    extra = os.environ.get("MELAMPO_LOCAL_CHAT_EXTRA", "").strip()
    out = {}
    try:
        for spec in models:
            model, _, flag = spec.partition("+")
            reasoning = flag == "reasoning"
            out[model] = local_chat(endpoint, model, timeout=900.0 if reasoning else 300.0,
                                    max_tokens=60000 if reasoning else 16000, reasoning=reasoning,
                                    extra=json.loads(extra) if extra else None)
    except ValueError as error:   # not a local host: refused, patient reports never leave the machine
        return {}, [f"interpreters_refused:{error}"]
    return out, ["interpreters:" + ",".join(out)]


def _hpo_and_thesaurus(notes: list[str]) -> tuple[Any, Any]:
    from ..reading.normalize import HpoIndex
    from ..reading.thesaurus import from_env

    hpo = HpoIndex()
    path = os.environ.get("MELAMPO_HPO_OBO", "").strip()
    if path and Path(path).exists():
        from ..memory.disease_profiles import Terms

        with open(path, encoding="utf-8") as handle:
            hpo = HpoIndex(Terms.from_obo(handle))
        notes.append("hpo:" + Path(path).name)
    else:
        notes.append("hpo_not_configured")
    thesaurus = from_env()
    notes.append(f"thesaurus:{len(thesaurus.concepts)}")
    return hpo, thesaurus


@dataclass
class ReportReaders:
    """The two interpreters. ``linker``: an ``AnatomyLinker`` (None: the narrative is not read).
    ``lesion_segmenter`` / ``organ_segmenter``: VoxTell and TotalSegmentator, or any segmenter with the same
    interface (None: findings on the text only)."""

    linker: Any = None
    lesion_segmenter: Any = None
    organ_segmenter: Any = None
    notes: tuple[str, ...] = ()
    owner: Any = None        # finding_owner.FindingOwner: is the clause about the structure (31 October 2026)
    # the one procedure (25 November 2026): the interpreters on this machine, HPO for the codes afterwards, the
    # thesaurus of the knowledge; ``reader``: ONE, or ANATOMY to measure the reader before it
    interpreters: dict[str, Any] | None = None
    hpo: Any = None
    thesaurus: Any = None
    reader: str = ONE
    references: Any = None   # reference_values.ReferenceValues: what a card's measure is compared with

    @classmethod
    def from_env(cls) -> ReportReaders:
        """Built once per configuration: the linker's data take about a second to load."""
        key = tuple(os.environ.get(name, "") for name in ENV)
        if key not in _BUILT:
            _BUILT[key] = cls._from_env()
        return _BUILT[key]

    @classmethod
    def _from_env(cls) -> ReportReaders:
        notes: list[str] = []
        linker = None
        if os.environ.get("MELAMPO_NARRATIVE_READER", "on").lower() not in ("off", "0", "false", "no"):
            try:
                models = tuple(m.strip() for m in os.environ.get("MELAMPO_LOCAL_CHAT_MODELS", "").split(",")
                               if m.strip())
                linker, built = build_linker(os.environ.get("MELAMPO_UBERON"),
                                             os.environ.get("MELAMPO_CI_RESOURCES"),
                                             os.environ.get("MELAMPO_LOCAL_CHAT_ENDPOINT"), models)
                notes += built
            except Exception as error:  # noqa: BLE001 -- the case is still read without the linker
                notes.append(f"linker_unavailable:{type(error).__name__}:{error}"[:200])
        else:
            notes.append("narrative_reader_off")
        lesion = organ = None
        if os.environ.get("MELAMPO_IMAGING_SEGMENTERS", "").lower() == "voxtell":
            from ..imaging.segmenters import TotalSegmentatorSegmenter, VoxTellSegmenter

            device = os.environ.get("MELAMPO_IMAGING_DEVICE", "cuda")
            lesion = VoxTellSegmenter(device=device)
            organ = TotalSegmentatorSegmenter(fast=False, device="gpu" if device == "cuda" else "cpu")
            notes.append(f"segmenters:voxtell+totalsegmentator:{device}")
        owner = None
        reader = os.environ.get("MELAMPO_REPORT_READER", ONE).strip().lower() or ONE
        if reader not in (ONE, ANATOMY):
            notes.append(f"report_reader_unknown:{reader}:one_used")
            reader = ONE
        if reader == ANATOMY and linker is not None and os.environ.get(
                "MELAMPO_FINDING_OWNER", "on").lower() not in ("off", "0", "false", "no"):
            owner, why = finding_owner(linker)
            notes.append(why)
        interpreters, hpo, thesaurus = None, None, None
        if reader == ONE:
            interpreters, built = one_procedure_interpreters()
            notes += built
            hpo, thesaurus = _hpo_and_thesaurus(notes)
            notes.append("report_reader:one")
        else:
            notes.append("report_reader:anatomy")
        references = None
        if reader == ONE:
            from ..memory.reference_values import ReferenceValues

            try:
                references = ReferenceValues.load()
            except Exception as error:  # noqa: BLE001 -- without the library a measure is only a measure
                notes.append(f"references_unavailable:{type(error).__name__}")
        return cls(linker, lesion, organ, tuple(notes), owner, interpreters, hpo, thesaurus, reader, references)

    # -- the narrative ---------------------------------------------------------------------------------------
    def read_sites(self, text: str, report_id: str) -> dict[str, Any]:
        """Every known anatomical name of ``text`` linked in its sentence: accepted sites and abstentions."""
        from ..evaluation.gold_set import propose_mentions
        from ..memory.anatomy_linker import ACCEPTED
        from ..memory.report_graph import site_row
        from ..memory.report_state import ReportState

        language = _language(text)
        state = ReportState.parse(text)
        sites: list[dict[str, Any]] = []
        abstentions: list[dict[str, Any]] = []
        for m in propose_mentions(text, self.linker.lexicon, self.linker.parts):
            sentence = state.sentence_at(m["start"], m["end"])
            try:
                result = self.linker.link(m["mention"], sentence, report=state, at=m["start"])
            except Exception as error:  # noqa: BLE001 -- one mention must not stop the report
                abstentions.append({**m, "reason": f"linker_error:{type(error).__name__}"})
                continue
            row = site_row(report_id, m["mention"], m["start"], m["end"], result) \
                if result.status == ACCEPTED else None
            if row is not None:
                row["support"] = list(result.support)
                row["conflicts"] = list(result.conflicts)
                _finding_class_as_site(row)
                sites.append(row)
            else:
                abstentions.append({**m, "reason": result.reason, "options": list(result.options)[:5],
                                    "proposal": result.fallback or ""})
        return {"report": report_id, "language": language, "sites": sites, "abstentions": abstentions}

    # -- the exam --------------------------------------------------------------------------------------------
    def read_exam(self, case_key: str, report: dict[str, Any], text: str, volume: dict[str, Any] | None,
                  why_no_image: str, set_aside: list | None = None,
                  questions: list | None = None, statements: list | None = None) -> list[dict[str, Any]]:
        """The findings of one imaging report, with the image side when ``volume`` (array, affine, modality,
        study) is given and the segmenters are there."""
        from ..imaging.interpreter import interpret_exam

        kwargs: dict[str, Any] = {}
        modality = (volume or {}).get("modality") or ""
        if volume is not None and self.lesion_segmenter is not None:
            kwargs = {"volume": volume["array"], "affine": volume["affine"],
                      "frame_of_reference": volume.get("frame_of_reference") or "",
                      "lesion_segmenter": self.lesion_segmenter, "organ_segmenter": self.organ_segmenter,
                      "contrast": volume.get("contrast") or "unknown"}
        elif volume is not None:
            why_no_image = "segmenters_not_configured"
        findings = interpret_exam(case_key, (volume or {}).get("study") or report["report"], "", modality,
                                  report["report"], text, report["sites"], language=report["language"],
                                  ownership=self.owner, set_aside=set_aside, questions=questions,
                                  statements=statements, **kwargs)
        rows = []
        for f in findings:
            row = f.as_row()
            if not kwargs:
                row["image_notes"] = [*row.get("image_notes", []), why_no_image]
            rows.append(row)
        return rows

    # -- the one procedure ----------------------------------------------------------------------------------
    def site_code(self, site: str) -> str:
        """The structure of a card's site, afterwards: the one class of the lexicon this name is, without a
        context it needs; "" otherwise (the finding keeps the words)."""
        if self.linker is None or not site:
            return ""
        from ..memory.anatomy_linker import normalise

        hits = {cid for cid, region in self.linker.lexicon.index.get(normalise(site), []) if region is None}
        return next(iter(hits)) if len(hits) == 1 else ""

    def read_one(self, case_key: str, physician_note: str, bundle: Any = None,
                 router_chat: Any = None) -> dict[str, Any]:
        """Every document of the case read with the one procedure (``melampo.reading.read_text``, context
        ``case``): the same card, interpreters and comparison as for the publications; the interpreters only on
        this machine."""
        from ..reading import read_text
        from ..reading.case_sides import (
            check_parts,
            compare_with_references,
            give_the_pathology,
            image_of_cards,
            is_the_question,
            lab_lines,
            mark_read_by_the_laboratory,
        )
        from .report_router import route

        out: dict[str, Any] = {"readers": list(self.notes), "procedure": ONE, "texts": [], "findings": [],
                               "to_a_person": [], "questions": [], "statements": []}
        sources: list[tuple[str, str, dict[str, Any]]] = []
        if physician_note.strip():
            sources.append(("note", physician_note, route(physician_note, router_chat).as_dict()))
        for attachment in getattr(bundle, "attachments", ()) or ():
            if attachment.text.strip():
                sources.append((f"attachment-{attachment.index}", attachment.text, dict(attachment.routing or {})))
        if not self.interpreters:
            out["status"] = "interpreters_not_configured"
            for source_id, _, routing in sources:
                out["texts"].append({"report": source_id, "kinds": routing.get("kinds"), "read": False,
                                     "reason": "interpreters_not_configured"})
                out["to_a_person"].append({"report": source_id, "mention": "", "reason":
                                           "not read: no interpreter on this machine (MELAMPO_LOCAL_CHAT_MODELS)"})
            return out
        pathologies = {f"attachment-{a.index}": dict(getattr(a, "pathology", {}) or {})
                       for a in getattr(bundle, "attachments", ()) or ()}
        imaging_reports = [s for s in sources if IMAGING_KIND in (s[2].get("kinds") or ())]
        paired, why = _pair(imaging_reports, _usable_volumes(bundle))
        labs = {f"attachment-{a.index}": getattr(a, "lab", None) for a in getattr(bundle, "attachments", ()) or ()}
        for source_id, text, routing in sources:
            interpreters = routing.get("interpreters") or ()
            if source_id != "note" and _document_to_a_person(source_id, routing, out):
                continue
            if source_id != "note" and not {NARRATIVE, "pathology"} & set(interpreters):
                # what the router sends only to the laboratory extractor (a table of blood tests, a culture) is the
                # extractor's: the interpreters do not read it again
                out["texts"].append({"report": source_id, "kinds": routing.get("kinds"), "read": False,
                                     "reason": "read_by:" + ",".join(interpreters)})
                continue
            try:
                got = read_text([("", text)], "case", self.interpreters, hpo=self.hpo, thesaurus=self.thesaurus)
            except Exception as error:  # noqa: BLE001 -- one document must not stop the case
                out["texts"].append({"report": source_id, "kinds": routing.get("kinds"), "read": False,
                                     "reason": f"reading_failed:{type(error).__name__}"})
                out["to_a_person"].append({"report": source_id, "mention": "",
                                           "reason": f"not read: {type(error).__name__}"})
                continue
            cards = []
            for card in got["cards"]:
                readings = card.pop("readings", None) or {}   # each interpreter's card: not in the case;
                # of a field read differently, what each read stays with the card: the closed answers a person is
                # asked (the review mask, 27 November 2026)
                fields = {d.removesuffix("_given_by_one") for d in card.get("doubts", ())} & set(readings and
                                                                                                   next(iter(readings.values())))
                if fields:
                    card["alternatives"] = {f: {r: rc.get(f) for r, rc in readings.items()} for f in sorted(fields)}
                card["site_code"] = self.site_code(card.get("site") or "")
                cards.append(card)
            # a row the laboratory extractor read enters the case once, as its measurement; the part of the
            # document each card is in, checked with the headings the report writes (27 November 2026)
            by_lab = mark_read_by_the_laboratory(cards, lab_lines(labs.get(source_id), text)) \
                if "lab_table" in interpreters else 0
            differ = check_parts(cards, text)
            language = _language(text)
            entry = {"report": source_id, "kinds": routing.get("kinds"), "read": True, "procedure": ONE,
                     "language": language, "cards": cards, "format": got["format"],
                     "not_read_by": got["not_read"], "questions": len(got["asked"]), "status": got["status"],
                     "read_by_lab_table": by_lab, "parts_differ_from_headings": differ}
            out["texts"].append(entry)
            # what the reader before gave its findings, given to the agreed cards (27 November 2026):
            # the reference library, the pathologist's summary, the image
            ask = (lambda name, question: self.interpreters[name](question))
            measured, doubted = compare_with_references(cards, text, self.references, ask, list(self.interpreters),
                                                        report=source_id, asked=got["asked"])
            out["statements"] += measured
            out["to_a_person"] += doubted
            if "pathology" in interpreters:
                from .pathology_report import read_pathology

                summary = pathologies.get(source_id) or read_pathology(text, language).summary()
                entry["pathology"] = give_the_pathology(cards, text, summary)
            volume, reason = None, "not_an_imaging_report"
            if IMAGING_KIND in (routing.get("kinds") or ()):
                reason = why
                if paired is not None and paired[0] == source_id:
                    volume = _load(paired[1])
                    if isinstance(volume, str):
                        reason, volume = volume, None
                if volume is not None and self.lesion_segmenter is None:
                    reason = "segmenters_not_configured"
            entry["image"] = {"looked_for": image_of_cards(cards, volume, self.lesion_segmenter,
                                                           self.organ_segmenter, reason),
                              "why_not": "" if volume is not None and self.lesion_segmenter is not None else reason}
            for card in cards:
                if card.get("read_by"):
                    continue            # the extractor's row: its doubts are the extractor's
                if card["doubts"]:
                    out["to_a_person"].append({"report": source_id, "mention": card.get("words") or card["name"],
                                               "reason": "doubt:" + ",".join(card["doubts"]),
                                               "clause": card.get("quote", "")})
                if (card.get("image") or {}).get("review"):
                    out["to_a_person"].append({"report": source_id, "mention": card.get("words") or card["name"],
                                               "reason": ",".join(card["image"]["review"]),
                                               "clause": card.get("quote", "")})
                if is_the_question(card):
                    # what the requester suspects is the clinical question, a hypothesis, not a current finding
                    out["statements"].append({"report": source_id, "statement": "indication",
                                              "mention": card.get("words") or card.get("name", ""),
                                              "clause": card.get("quote", ""), "name": card.get("name", "")})
                    continue
                if card.get("category") == "manifestation":
                    out["findings"].append({"report": source_id, "structure": card["site_code"],
                                            "clause": card.get("quote", ""), "mention": card.get("words", ""),
                                            "name": card.get("name", ""), "state": card.get("state"),
                                            "part": card.get("part", ""),
                                            **{k: card[k] for k in ("reference", "pathology", "image")
                                               if card.get(k)}})
        out["status"] = "read"
        return out

    def read(self, case_key: str, physician_note: str, bundle: Any = None, router_chat: Any = None) -> dict[str, Any]:
        """The reading of the whole case: the physician's note and every attachment; with the one procedure
        unless the reader before it is asked for (``MELAMPO_REPORT_READER=anatomy``)."""
        if self.reader == ONE:
            return self.read_one(case_key, physician_note, bundle, router_chat)
        from .report_router import route

        # "questions": what the review mask shows a person, each complete in itself (``review_question``)
        # "statements": what a sentence says of a structure that is not a current finding (its history, that it
        # is normal), when the owner reads the statement (1 November 2026)
        out: dict[str, Any] = {"readers": list(self.notes), "texts": [], "findings": [], "to_a_person": [],
                               "questions": [], "statements": []}
        if self.linker is None:
            out["status"] = "narrative_reader_not_available"
            return out
        sources: list[tuple[str, str, dict[str, Any]]] = []
        if physician_note.strip():
            sources.append(("note", physician_note, route(physician_note, router_chat).as_dict()))
        for attachment in getattr(bundle, "attachments", ()) or ():
            if attachment.text.strip():
                sources.append((f"attachment-{attachment.index}", attachment.text, dict(attachment.routing or {})))
        pathologies = {f"attachment-{a.index}": dict(getattr(a, "pathology", {}) or {})
                       for a in getattr(bundle, "attachments", ()) or ()}
        imaging_reports = [s for s in sources if IMAGING_KIND in (s[2].get("kinds") or ())]
        volumes = _usable_volumes(bundle)
        paired, why = _pair(imaging_reports, volumes)
        for source_id, text, routing in sources:
            if source_id != "note" and _document_to_a_person(source_id, routing, out):
                continue
            narrative = NARRATIVE in (routing.get("interpreters") or ())
            if not narrative and IMAGING_KIND not in (routing.get("kinds") or ()):
                out["texts"].append({"report": source_id, "kinds": routing.get("kinds"), "read": False})
                continue
            reading = self.read_sites(text, source_id)
            entry = {"report": source_id, "kinds": routing.get("kinds"), "read": True,
                     "language": reading["language"], "sites": reading["sites"],
                     "abstentions": reading["abstentions"]}
            out["texts"].append(entry)
            out["to_a_person"] += [{"report": source_id, "mention": a["mention"], "reason": a["reason"]}
                                   for a in reading["abstentions"]]
            # Every text read gives its findings (a site the clause describes); only an imaging report is looked
            # for in an exam. A clinical letter or a histology report that describes "a 32 mm mass in the right
            # upper lobe" states a finding as much as a CT report does (30 October 2026: a report the router read
            # as a clinical note gave no findings at all).
            volume, reason = None, "not_an_imaging_report"
            if IMAGING_KIND in (routing.get("kinds") or ()):
                reason = why
                if paired is not None and paired[0] == source_id:
                    volume = _load(paired[1])
                    if isinstance(volume, str):
                        reason, volume = volume, None
            aside: list[dict[str, Any]] = []
            asked: list[dict[str, Any]] = []
            stated: list[dict[str, Any]] = []
            rows = self.read_exam(case_key, reading, text, volume, reason, aside, asked, stated)
            if "pathology" in (routing.get("interpreters") or ()):
                # the pathologist's diagnosis given to its findings, with its TNM, grade, margins, markers
                # (pathology_findings, 8 November 2026)
                from .pathology_findings import give_the_diagnosis
                from .pathology_report import read_pathology

                summary = pathologies.get(source_id) or read_pathology(text, reading["language"]).summary()
                specimen = [x for x in reading["sites"] if "procedure_site" in (x.get("roles") or ())]
                rows, stated, of_report = give_the_diagnosis(rows, stated, text, summary, specimen)
                entry["pathology"] = of_report
            out["questions"] += asked
            out["statements"] += [{**x, "report": source_id} for x in stated]
            out["findings"] += rows
            out["to_a_person"] += [{"report": source_id, "mention": a["mention"], "reason": a["reason"],
                                    "clause": a["clause"], "placed": a.get("placed", ""),
                                    "question": a.get("question", "")} for a in aside]
            out["to_a_person"] += [{"report": source_id, "mention": r["mention"],
                                    "reason": ",".join(r["review_reasons"]),
                                    "question": next((q["id"] for q in asked if q["mention"] == r["mention"]
                                                      and q["structure"] == r["structure"]
                                                      and q["sentence"] == r["clause"]), "")}
                                   for r in rows if r["needs_review"]]
        out["status"] = "read"
        return out


def _document_to_a_person(source_id: str, routing: dict[str, Any], out: dict[str, Any]) -> bool:
    """A document a person must look at before it enters the case (27 November 2026, the owner's rule), the same
    for both readers: its identifiers are another patient's, or the router recognised no kind of medical document
    in it -- it is not read, and its question goes to the review (``patient_check.document_question``); one
    identifier of the patient and one of another -- it is read with its doubt, and the question goes too. Returns
    whether the document is kept out."""
    from .case_attachments import kind_not_recognised, set_aside_of
    from .patient_check import DOUBT, document_question

    why = set_aside_of(routing)
    check = (routing.get("patient") or {}).get("check", "")
    if routing.get("decision") and why != "another_patient":
        # a person decided in the review mask: the question is answered, the decision holds
        if why:
            out["texts"].append({"report": source_id, "kinds": routing.get("kinds"), "read": False, "reason": why})
        return bool(why)
    system = {"identifiers_compared": list((routing.get("patient") or {}).get("compared", [])),
              "kinds": list(routing.get("kinds") or ())}
    if why:
        out["texts"].append({"report": source_id, "kinds": routing.get("kinds"), "read": False, "reason": why})
        out["to_a_person"].append({"report": source_id, "mention": "", "reason": f"document_set_aside:{why}"})
        out["questions"].append(document_question(source_id, "other" if why == "another_patient" else why, system))
        return True
    if check == DOUBT:
        out["to_a_person"].append({"report": source_id, "mention": "", "reason": "document_identity_in_doubt"})
        out["questions"].append(document_question(source_id, DOUBT, system))
    if kind_not_recognised(routing):
        # recognised by no cue and no model was asked: read, and a person says what it is
        out["to_a_person"].append({"report": source_id, "mention": "", "reason": "document_kind_not_recognised"})
        out["questions"].append(document_question(source_id, "kind_not_recognised", system))
    return False


def _usable_volumes(bundle: Any) -> list[Any]:
    if bundle is None:
        return []
    out = []
    for study in bundle.imaging_studies():
        meta = study.metadata or {}
        vol = meta.get("volume") or {}
        if meta.get("source") == "dicom_attachment" and vol.get("usable") and \
                (vol.get("modality") or "") in VOLUME_MODALITIES and study.dicom_instances:
            out.append(study)
    return out


def _pair(reports: list, volumes: list) -> tuple[tuple[str, Any] | None, str]:
    """The one imaging report with the one volume, or why not: a report is never read on another exam."""
    if not volumes:
        return None, "no_volume_in_the_attachments"
    if len(reports) != 1 or len(volumes) != 1:
        return None, f"exam_not_paired:{len(reports)}_imaging_reports,{len(volumes)}_volumes"
    return (reports[0][0], volumes[0]), ""


def _load(study: Any) -> dict[str, Any] | str:
    """The volume (slice, row, column) with its affine (column, row, slice) -> LPS mm, or why it cannot be read."""
    import numpy as np

    from .dicom_volume import assess_series, load_volume_hu

    try:
        assessment = assess_series(study.dicom_instances)
        array = load_volume_hu(study.dicom_instances, assessment)
    except Exception as error:  # noqa: BLE001
        return f"volume_unreadable:{type(error).__name__}"
    if assessment.affine is None:
        return "volume_without_geometry"
    return {"array": np.asarray(array, dtype=np.float32), "affine": np.asarray(assessment.affine, dtype=float),
            "modality": assessment.modality or "", "study": study.study_id,
            "frame_of_reference": assessment.frame_of_reference_uid or "", "contrast": assessment.contrast}


__all__ = ["ReportReaders", "build_linker"]


def _finding_class_as_site(row: dict[str, Any]) -> None:
    """A class of the atlas that is a finding (``class_kinds``: read from UBERON and HPO, the same for every
    class) gives its site to the structure it is of; the class stays with the finding, for the image (11 November
    2026). The mention becomes the words that name the structure, if the mention holds them; else it stays."""
    import os

    from ..memory.class_kinds import as_site

    if os.environ.get("MELAMPO_FINDER", "") == "2026-11-11":
        return      # the samples labelled before 12 November 2026 are drawn as they were
    site = as_site(row["cid"])
    if "finding_class" not in site:
        return
    row.update(cid=site["cid"], finding_class=site["finding_class"], phenotype=site["phenotype"],
               phenotype_name=site["phenotype_name"])
    import json as _json

    from ..memory.class_kinds import LEXICON

    mention = row["mention"]
    c = _json.loads(LEXICON.read_text("utf-8"))["classes"].get(site["cid"], {})
    names = list(c.get("en", [])) + list(c.get("it", []))
    low = mention.lower()
    for n in sorted(names, key=len, reverse=True):
        at = low.find(n.lower())
        if at >= 0:
            row["mention"] = mention[at:at + len(n)]
            row["start"] = int(row["start"]) + at
            row["end"] = row["start"] + len(n)
            break
