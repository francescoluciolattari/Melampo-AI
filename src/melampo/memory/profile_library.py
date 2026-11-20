"""The disease profiles as knowledge the diagnostic path reads (20 November 2026).

A profile is what ``scripts/disease_profiles.py`` writes: a ``clinical.case_graph.Disease`` whose manifestations
each carry the sentence and the citation of the source that states them (HPO annotations, Orphanet, the Disease
Ontology, open publications). This module loads them once and finds the one a disease of the case needs.

* **Knowledge, not training.** The profiles are read from a file and never learnt from patients; the same file
  gives the same profiles. Its version is the hash of its content (``version``): the case's graph records it in
  its snapshot, so a case can be read again with the knowledge it was read with.
* **Several files.** ``MELAMPO_DISEASE_PROFILES`` names one file or a folder of them (``profiles.json`` of each
  run); a disease profiled in two files keeps the manifestations of both, each with its own citation.
* **Finding a profile.** By concept (MONDO id), then by name: the profile's words and the names ``names`` adds
  (MONDO synonyms, the Italian names of an index, as ``{"MONDO:...": ["appendicite acuta", ...]}``), compared
  normalised. Nothing is guessed: no profile, and the disease says so.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..clinical import case_graph as cg

ENV = "MELAMPO_DISEASE_PROFILES"
ENV_NAMES = "MELAMPO_DISEASE_NAMES"


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.rglob("*.json") if p.is_file())
    return [path] if path.is_file() else []


@dataclass
class ProfileLibrary:
    profiles: dict[str, cg.Disease] = field(default_factory=dict)       # by disease id (a MONDO id when known)
    by_name: dict[str, set[str]] = field(default_factory=dict)
    version: str = ""
    files: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path, names: str | Path | None = None) -> ProfileLibrary:
        lib, digest = cls(), hashlib.sha256()
        for f in _files(Path(path)):
            raw = f.read_bytes()
            digest.update(raw)
            lib.files.append(f.name)
            data = json.loads(raw)
            for item in data.get("profiles", []) if isinstance(data, dict) else data:
                lib.add(cg.Disease.from_dict(item))
        if names:
            raw = Path(names).read_bytes()
            digest.update(raw)
            lib.add_names(json.loads(raw))
        lib.version = digest.hexdigest()[:16] if lib.files else ""
        return lib

    @classmethod
    def from_env(cls) -> ProfileLibrary | None:
        path = os.environ.get(ENV, "").strip()
        if not path:
            return None
        lib = cls.load(path, os.environ.get(ENV_NAMES, "").strip() or None)
        return lib if lib.profiles else None

    def add(self, disease: cg.Disease) -> None:
        known = self.profiles.get(disease.disease_id)
        if known is None:
            self.profiles[disease.disease_id] = disease
        else:
            seen = {(m.concept.curie if m.concept and m.concept.code else norm(m.words), json.dumps(
                m.citations, sort_keys=True)) for m in known.manifestations}
            for m in disease.manifestations:
                key = (m.concept.curie if m.concept and m.concept.code else norm(m.words),
                       json.dumps(m.citations, sort_keys=True))
                if key not in seen:
                    known.manifestations.append(m)
                    seen.add(key)
        self.by_name.setdefault(norm(disease.words), set()).add(disease.disease_id)

    def add_names(self, names: dict[str, Iterable[str]]) -> None:
        for disease_id, words in names.items():
            if disease_id in self.profiles:
                for w in words:
                    if norm(w):
                        self.by_name.setdefault(norm(w), set()).add(disease_id)

    def find(self, *, concept: cg.Concept | None = None, words: str = "") -> tuple[cg.Disease | None, str]:
        """The profile and how it was found (``concept``, ``name``), or None and why not (``no_profile``,
        ``ambiguous_name``: two profiles have that name, none is chosen)."""
        if concept is not None and concept.code:
            for key in (concept.curie, concept.code):
                if key in self.profiles:
                    return self.profiles[key], "concept"
        ids = self.by_name.get(norm(_without_status(words)), set())
        if len(ids) == 1:
            return self.profiles[next(iter(ids))], "name"
        return None, "ambiguous_name" if ids else "no_profile"


# the words that say how sure the writer is ("sospetta", "suspected", "in esiti di") are not part of the name
_STATUS = re.compile(r"\b(sospett[aoie]|suspected|possibile|possible|probabile|probable|presunt[aoie]|"
                     r"in esiti di|esiti di|pregress[aoie]|history of|ipotesi di|r/o|rule out|da escludere)\b",
                     re.IGNORECASE)


def _without_status(words: str) -> str:
    return _STATUS.sub(" ", words or "")


__all__ = ["ENV", "ENV_NAMES", "ProfileLibrary", "norm"]
