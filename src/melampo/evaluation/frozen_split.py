"""The frozen test split: documents that no development step may read.

A certification number is only worth something if the documents it is measured on were never used to
choose a rule, a threshold, a lexicon entry or a model. This module is the guard that makes that a
property of the code and not of anyone's memory:

* ``FrozenSplit`` holds the document ids of the frozen split, the SHA-256 of the sorted list and the
  date it was frozen, and checks the list against its own digest (``verify``);
* ``refuse(docs, purpose)`` raises ``FrozenSplitError`` when a development step is about to read a
  frozen document (writing the memory, building the semantic space, drawing worked examples);
* ``development_rows(rows, frozen, final=False)`` drops the rows of frozen documents from a
  measurement, unless the run is declared ``final``: the single run that writes the certificate.

What the freeze does *not* do: it cannot undo what was already seen. ``seen_before_freeze`` in the
manifest records it (the external check judged links of these documents and their errors were read
before the freeze), so that nobody later takes the split for a virgin test. The test that certifies is
the radiologists' gold set (``docs/gold_set_protocollo.md``), drawn from reports no development step saw.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "linking" / "frozen_test_split.json"


class FrozenSplitError(RuntimeError):
    """A development step tried to read a frozen document."""


def digest(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(set(ids))).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FrozenSplit:
    corpus: str
    ids: frozenset[str]
    sha256: str
    frozen_on: str
    source: str = ""
    seen_before_freeze: dict = field(default_factory=dict)

    @classmethod
    def from_ids(cls, corpus: str, ids: Iterable[str], frozen_on: str, source: str = "",
                 seen_before_freeze: dict | None = None) -> FrozenSplit:
        clean = frozenset(i.strip() for i in ids if i.strip())
        if not clean:
            raise ValueError("a frozen split needs at least one document")
        return cls(corpus, clean, digest(clean), frozen_on, source, dict(seen_before_freeze or {}))

    @classmethod
    def from_json(cls, data: dict) -> FrozenSplit:
        split = cls(data["corpus"], frozenset(data["ids"]), data["sha256"], data["frozen_on"],
                    data.get("source", ""), dict(data.get("seen_before_freeze", {})))
        split.verify()
        return split

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PATH) -> FrozenSplit:
        return cls.from_json(json.loads(Path(path).read_text("utf-8")))

    def to_json(self) -> dict:
        return {"corpus": self.corpus, "count": len(self.ids), "sha256": self.sha256,
                "frozen_on": self.frozen_on, "source": self.source,
                "seen_before_freeze": self.seen_before_freeze, "ids": sorted(self.ids)}

    def save(self, path: Path | str = DEFAULT_PATH) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=1, ensure_ascii=False) + "\n", "utf-8")

    def verify(self) -> None:
        """The list is the one that was frozen (its digest is the recorded one)."""
        if digest(self.ids) != self.sha256:
            raise FrozenSplitError(
                f"the frozen {self.corpus} split no longer matches its digest: the list was edited")

    def is_frozen(self, doc: str, corpus: str | None = None) -> bool:
        return (corpus is None or corpus == self.corpus) and str(doc) in self.ids

    def refuse(self, docs: Iterable[str], purpose: str, corpus: str | None = None) -> None:
        """Raise when any of ``docs`` is frozen: ``purpose`` says what they were about to be used for."""
        hit = sorted(str(d) for d in docs if self.is_frozen(d, corpus))
        if hit:
            raise FrozenSplitError(
                f"{len(hit)} frozen {self.corpus} test document(s) (first: {hit[0]}) cannot be used for {purpose}")

    def open_docs(self, docs: Iterable[str], corpus: str | None = None) -> list[str]:
        return [d for d in docs if not self.is_frozen(d, corpus)]


def development_rows(rows: list[dict], frozen: FrozenSplit | None, final: bool = False) -> tuple[list[dict], int]:
    """``(rows to measure on, rows held back)``. A final run keeps every row; any other run holds back
    the rows of frozen documents of the frozen corpus, so that no development decision sees them."""
    if frozen is None or final:
        return list(rows), 0
    keep = [r for r in rows if not (r.get("corpus") == frozen.corpus and str(r.get("doc")) in frozen.ids)]
    return keep, len(rows) - len(keep)


__all__ = ["DEFAULT_PATH", "FrozenSplit", "FrozenSplitError", "development_rows", "digest"]
