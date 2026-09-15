"""Every name an HPO term has ever had, kept forever, never lost to a rename.

Corrects a design this project got wrong. An earlier version of the update
workflow treated a rename as a risk to flag for a reviewer -- "the graph no
longer matches the old label" -- and stopped there. That is not sufficient:
a term must never stop being recognisable under a name it used to have.
Renaming is not removal, and this module makes sure the system never treats
it as one.

**What HPO's own release does and does not guarantee.** A renamed term keeps
its id but changes its `name:` line; the previous release's name is not
reliably carried forward as a synonym in the new one. An obsoleted term
sometimes carries `replaced_by:`, sometimes does not, and the old id simply
stops resolving to anything in the new release. Neither case is HPO's
responsibility to solve for this project's own matching -- it is this
project's job to remember what it once matched.

**Append-only, growing across releases, the same discipline as
`graph_store`'s learned layer.** A term's history is discovered once, when a
rename or obsoletion is first detected between two consecutive releases, and
is never rederived from a single release the way the imported ontology layer
is -- there would be nothing to rederive it from, since the old release is
gone. This is the second kind of irreplaceable data this project keeps,
alongside promoted graph edges, and it gets the same treatment: its own file,
appended to, never rewritten.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TermRename:
    """One name change detected for one term id, between two releases."""

    term_id: str
    old_name: str
    new_name: str
    detected_in_release: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "term_id": self.term_id,
            "old_name": self.old_name,
            "new_name": self.new_name,
            "detected_in_release": self.detected_in_release,
        }


@dataclass(frozen=True)
class TermObsoletion:
    """One term retired in a release, with whatever it was replaced by, if anything."""

    term_id: str
    last_known_name: str
    replaced_by: str | None
    detected_in_release: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "term_id": self.term_id,
            "last_known_name": self.last_known_name,
            "replaced_by": self.replaced_by,
            "detected_in_release": self.detected_in_release,
        }


def diff_releases(
    previous_terms: dict[str, str],
    current_terms: dict[str, str],
    *,
    current_obsolete: dict[str, str | None] | None = None,
    release: str,
) -> tuple[list[TermRename], list[TermObsoletion]]:
    """Compare two releases' term_id -> name maps and find what changed.

    Pure comparison, no I/O -- the caller reads both releases' `hp.obo`
    files (the update workflow keeps the previous checkout for exactly this)
    and passes in plain dicts. Kept this way so the diff itself is trivially
    testable without needing two real ontology files on disk.
    """
    current_obsolete = current_obsolete or {}
    renames: list[TermRename] = []
    obsoletions: list[TermObsoletion] = []

    for term_id, old_name in previous_terms.items():
        if term_id in current_obsolete:
            obsoletions.append(
                TermObsoletion(
                    term_id=term_id,
                    last_known_name=old_name,
                    replaced_by=current_obsolete[term_id],
                    detected_in_release=release,
                )
            )
            continue
        new_name = current_terms.get(term_id)
        if new_name and new_name != old_name:
            renames.append(TermRename(term_id=term_id, old_name=old_name, new_name=new_name, detected_in_release=release))

    return renames, obsoletions


@dataclass
class TermHistoryStore:
    """Append-only persistence for every rename and obsoletion ever detected.

    Two files, not one, in the same directory -- renames and obsoletions
    answer different downstream questions (a rename adds a synonym; an
    obsoletion adds a synonym pointing at whatever replaced it, or nothing
    if there was no replacement) and keeping them apart means a reader
    checking one never has to filter out the other.
    """

    directory: Path

    @property
    def renames_path(self) -> Path:
        return self.directory / "term_renames.jsonl"

    @property
    def obsoletions_path(self) -> Path:
        return self.directory / "term_obsoletions.jsonl"

    def append_renames(self, renames: Iterable[TermRename]) -> int:
        return self._append(self.renames_path, [rename.as_dict() for rename in renames])

    def append_obsoletions(self, obsoletions: Iterable[TermObsoletion]) -> int:
        return self._append(self.obsoletions_path, [item.as_dict() for item in obsoletions])

    def _append(self, path: Path, records: list[dict[str, Any]]) -> int:
        if not records:
            return 0
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return len(records)

    def load_renames(self) -> list[TermRename]:
        return [TermRename(**record) for record in self._load(self.renames_path)]

    def load_obsoletions(self) -> list[TermObsoletion]:
        return [TermObsoletion(**record) for record in self._load(self.obsoletions_path)]

    def _load(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
        return records

    def synonyms_by_term_id(self) -> dict[str, list[str]]:
        """Every historical name for each term id, as a synonym lookup.

        Every name a term id has ever carried -- not only the immediately
        previous one -- since a term can be renamed more than once across
        releases and every one of those names may appear in older records,
        cases, or a clinician's memory of what it used to be called.
        """
        by_id: dict[str, list[str]] = {}
        for rename in self.load_renames():
            by_id.setdefault(rename.term_id, []).append(rename.old_name)
        for obsoletion in self.load_obsoletions():
            by_id.setdefault(obsoletion.term_id, []).append(obsoletion.last_known_name)
        return by_id

    def current_id_for(self, name_or_id: str) -> str | None:
        """Where a historical name or a retired id now points, if anywhere.

        Follows one obsoletion hop (retired id -> its replacement) but does
        not chase a chain of several -- a term obsoleted twice over is rare
        enough, and ambiguous enough about which replacement is authoritative,
        that surfacing nothing is safer than guessing which hop to prefer.
        """
        for obsoletion in self.load_obsoletions():
            if obsoletion.term_id == name_or_id or obsoletion.last_known_name == name_or_id:
                return obsoletion.replaced_by
        return None
