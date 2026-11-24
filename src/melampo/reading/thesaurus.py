"""The thesaurus of Melampo: the concepts the interpreters agreed on, for every model (25 November 2026).

As the UMLS groups the names of one meaning under one concept identifier, the thesaurus groups the labels the
interpreters agreed are the same finding. It is knowledge, not training: any model reads it the same way (its
concepts are the candidates shown in the question), it is versioned with the knowledge (``thesaurus.jsonl`` in
each GitHub Release, ``KConcept``/``KLabel`` in the knowledge graph), and every label carries the sentence it was
agreed in and the answers of the interpreters, so that each can be checked.

A label is the name with the description that makes it what it is (``signature``: interpretation, threshold,
site, side, time, method): "elevated C-reactive protein" above 52 mg/L and without a threshold are two labels, and
only a question in context joins them.

Only publications write it. A patient's reading uses it and never adds to it: a patient's sentence is never
knowledge, and never leaves the machine.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agreement import letters

FILE = "thesaurus.jsonl"


def signature(card: dict) -> dict:
    t = card.get("threshold") or {}
    return {"interpretation": card.get("interpretation") or "",
            "threshold": f"{t.get('comparator', '')}{t.get('number', '')}{t.get('unit', '')}" if t else "",
            "site": letters(card.get("site") or ""), "side": card.get("side") or "",
            "time": letters(card.get("time") or ""), "observed_by": card.get("observed_by") or ""}


def label_key(name: str, sig: dict) -> str:
    return json.dumps([letters(name), sig], sort_keys=True)


@dataclass
class Concept:
    id: str
    name: str
    code: str = ""                         # the HPO term the concept is, when the interpreters chose one
    labels: list[dict[str, Any]] = field(default_factory=list)

    def as_row(self) -> dict:
        return {"id": self.id, "name": self.name, "code": self.code, "labels": self.labels}


@dataclass
class Thesaurus:
    concepts: dict[str, Concept] = field(default_factory=dict)
    by_label: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_rows(cls, rows) -> Thesaurus:
        t = cls()
        for row in rows:
            concept = Concept(row["id"], row["name"], row.get("code", ""), list(row.get("labels") or []))
            t.concepts[concept.id] = concept
            for label in concept.labels:
                t.by_label[label_key(label["label"], label["signature"])] = concept.id
        return t

    @classmethod
    def load(cls, path: str | Path | None) -> Thesaurus:
        p = Path(path) if path else None
        if p is None or not p.exists():
            return cls()
        return cls.from_rows(json.loads(line) for line in p.read_text("utf-8").splitlines() if line.strip())

    def save(self, path: str | Path) -> None:
        Path(path).write_text("".join(json.dumps(c.as_row(), ensure_ascii=False) + "\n"
                                      for c in sorted(self.concepts.values(), key=lambda c: c.id)), "utf-8")

    def find(self, card: dict) -> Concept | None:
        """The concept this label already is (same name, same description), or None."""
        cid = self.by_label.get(label_key(card.get("name", ""), signature(card)))
        return self.concepts.get(cid) if cid else None

    def by_code(self, code: str) -> Concept | None:
        return next((c for c in self.concepts.values() if code and c.code == code), None)

    def add(self, concept: Concept, card: dict, evidence: dict) -> None:
        """A label agreed to be this concept, with its sentence, its source and the interpreters' answers."""
        sig = signature(card)
        key = label_key(card.get("name", ""), sig)
        if key in self.by_label:
            return
        self.concepts.setdefault(concept.id, concept)
        concept.labels.append({"label": card.get("name", ""), "signature": sig, **evidence})
        self.by_label[key] = concept.id

    def new(self, card: dict, code: str = "") -> Concept:
        """A concept for an agreed reading that is no concept yet: its id from its first label."""
        name = card.get("name", "")
        cid = code or "MLP:" + hashlib.sha1(label_key(name, signature(card)).encode()).hexdigest()[:10]
        return self.concepts.get(cid) or Concept(cid, name, code)

    def candidates(self, card: dict, limit: int = 8) -> list[Concept]:
        """The concepts sharing content words with the reading, the closest first (a proposal only: the
        decision is the interpreters')."""
        words = {w for w in letters(f"{card.get('name', '')} {card.get('what', '')}").split() if len(w) > 2}
        scored = []
        for concept in self.concepts.values():
            theirs = {w for lab in [concept.name] + [x["label"] for x in concept.labels]
                      for w in letters(lab).split() if len(w) > 2}
            common = words & theirs
            if common:
                scored.append((len(common) / len(words | theirs), concept.id, concept))
        return [c for _, _, c in sorted(scored, key=lambda x: (-x[0], x[1]))[:limit]]

    def merge(self, other: Thesaurus) -> None:
        for concept in other.concepts.values():
            mine = self.concepts.setdefault(concept.id, Concept(concept.id, concept.name, concept.code))
            for label in concept.labels:
                key = label_key(label["label"], label["signature"])
                if key not in self.by_label:
                    mine.labels.append(label)
                    self.by_label[key] = mine.id


def from_env() -> Thesaurus:
    """The thesaurus of the knowledge in use: ``MELAMPO_THESAURUS`` (a ``thesaurus.jsonl``), else the one beside
    ``MELAMPO_DISEASE_PROFILES`` (its folder, or the folder of its file), else the current version of the
    knowledge graph in that ``.db`` file; empty when there is none."""
    import os

    path = os.environ.get("MELAMPO_THESAURUS", "").strip()
    if path:
        return Thesaurus.load(path)
    profiles = os.environ.get("MELAMPO_DISEASE_PROFILES", "").strip()
    if not profiles or profiles.startswith("falkordb"):
        return Thesaurus()
    p = Path(profiles)
    folder = p if p.is_dir() else p.parent
    if (folder / FILE).exists():
        return Thesaurus.load(folder / FILE)
    if p.suffix == ".db" and p.exists():
        try:
            from redislite.falkordb_client import FalkorDB

            from ..memory import disease_knowledge_graph as kg

            db = FalkorDB(str(p))
            try:
                return Thesaurus.from_rows(kg.read_thesaurus(db.select_graph(kg.GRAPH)))
            finally:
                db.close()
        except Exception:  # noqa: BLE001 -- no thesaurus: the findings keep their words
            return Thesaurus()
    return Thesaurus()


__all__ = ["Concept", "FILE", "Thesaurus", "from_env", "label_key", "signature"]
