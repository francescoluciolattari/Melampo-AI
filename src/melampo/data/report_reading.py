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
  linker's choice stage.
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
       "MELAMPO_LOCAL_CHAT_MODELS", "MELAMPO_IMAGING_SEGMENTERS", "MELAMPO_IMAGING_DEVICE")
_BUILT: dict[tuple[str, ...], "ReportReaders"] = {}


def _language(text: str) -> str:
    from ..evaluation.gold_set import language_of

    found = language_of(text)
    return found if found in ("it", "en") else "it"


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
        from .report_router import local_chat

        try:
            kwargs["chats"] = {m: local_chat(chat_endpoint, m) for m in chat_models}
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
    return FindingOwner(reader, chats), "finding_owner:reader" + ("+models" if chats else "")


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
        if linker is not None and os.environ.get("MELAMPO_FINDING_OWNER", "on").lower() not in ("off", "0", "false", "no"):
            owner, why = finding_owner(linker)
            notes.append(why)
        return cls(linker, lesion, organ, tuple(notes), owner)

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
                sites.append(row)
            else:
                abstentions.append({**m, "reason": result.reason, "options": list(result.options)[:5],
                                    "proposal": result.fallback or ""})
        return {"report": report_id, "language": language, "sites": sites, "abstentions": abstentions}

    # -- the exam --------------------------------------------------------------------------------------------
    def read_exam(self, case_key: str, report: dict[str, Any], text: str, volume: dict[str, Any] | None,
                  why_no_image: str, set_aside: list | None = None) -> list[dict[str, Any]]:
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
                                  ownership=self.owner, set_aside=set_aside, **kwargs)
        rows = []
        for f in findings:
            row = f.as_row()
            if not kwargs:
                row["image_notes"] = [*row.get("image_notes", []), why_no_image]
            rows.append(row)
        return rows

    def read(self, case_key: str, physician_note: str, bundle: Any = None, router_chat: Any = None) -> dict[str, Any]:
        """The reading of the whole case: the physician's note and every attachment."""
        from .report_router import route

        out: dict[str, Any] = {"readers": list(self.notes), "texts": [], "findings": [], "to_a_person": []}
        if self.linker is None:
            out["status"] = "narrative_reader_not_available"
            return out
        sources: list[tuple[str, str, dict[str, Any]]] = []
        if physician_note.strip():
            sources.append(("note", physician_note, route(physician_note, router_chat).as_dict()))
        for attachment in getattr(bundle, "attachments", ()) or ():
            if attachment.text.strip():
                sources.append((f"attachment-{attachment.index}", attachment.text, dict(attachment.routing or {})))
        imaging_reports = [s for s in sources if IMAGING_KIND in (s[2].get("kinds") or ())]
        volumes = _usable_volumes(bundle)
        paired, why = _pair(imaging_reports, volumes)
        for source_id, text, routing in sources:
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
            rows = self.read_exam(case_key, reading, text, volume, reason, aside)
            out["findings"] += rows
            out["to_a_person"] += [{"report": source_id, "mention": a["mention"], "reason": a["reason"],
                                    "clause": a["clause"], "placed": a.get("placed", "")} for a in aside]
            out["to_a_person"] += [{"report": source_id, "mention": r["mention"],
                                    "reason": ",".join(r["review_reasons"])} for r in rows if r["needs_review"]]
        out["status"] = "read"
        return out


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
