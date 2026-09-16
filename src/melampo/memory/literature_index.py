"""Literature as a retrieval source with citable provenance, never as training data.

The decision this implements, taken after examining four arguments rather
than one preference:

*The literature ages faster than a model can.* A new index entry is an
append; a new fact in a model's weights is a retraining run. This project
already chose DoRA over GaLore because confirmed clinical outcomes arrive
slowly; literature arrives fast, and the same reasoning points the opposite
way -- to a mechanism that keeps up without retraining anything.

*Training would work on one engine and never the other.* Weights can be
fine-tuned only where they are accessible -- never on Claude, a closed API.
A vetting engine enriched with literature on one side and unable to be on
the other would stop being comparable on the same basis, which is the whole
premise of running two.

*A weight has no citation.* Nothing can point to which parameter produced a
claim. A retrieved passage carries its own reference -- title, identifier,
date -- and a reviewer can open it. This is the same reasoning that chose
DPO over full RLVR for regulatory auditability, applied to a different
question.

*The risk is already documented in this project.* Meditron was excluded as a
navigation model because medical fine-tuning erodes format adherence, and
the wider literature suggests medical fine-tuning may not add clinical
accuracy on unseen data either. Training a vetting engine on a literature
corpus risks the same, one role over.

**Separation is the design, as it is in `graph_store`.** Literature lives in
its own index, apart from both the curated concept graph and the patient's
own documents, because the three answer different questions and carry
different authority. A passage from a case report is not an ontology edge
and must never become indistinguishable from one.

**This module is deliberately storage and retrieval only.** No embedding
model is chosen here, no vector database is required, and no ingestion
schedule is set -- those are deployment decisions, and hard-coding one would
repeat the mistake of building against an assumption instead of a decision.
What is fixed here is the shape: what a citation must carry to be checkable,
how a passage is matched, and how retrieved material is handed to a vetting
engine without ever losing its reference.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..memory.concept_paths import (
    ConceptGraphView,
    mentioned_concepts,
    normalise_concept,
)

# Identifier prefixes that make a citation independently checkable. A passage
# whose source cannot be opened by a reviewer is not evidence -- it is an
# assertion with a bibliography-shaped decoration, which is worse than no
# citation because it looks like one.
CHECKABLE_ID_PREFIXES = ("pmid:", "pmcid:", "doi:", "nct:", "dailymed:", "pms-ema:")


@dataclass(frozen=True)
class LiteraturePassage:
    """One retrievable passage, with the reference that makes it checkable."""

    passage_id: str
    text: str
    title: str
    source_id: str
    year: int | None = None
    publication: str | None = None

    @property
    def is_independently_checkable(self) -> bool:
        """Whether a reviewer could open this source and verify the claim.

        The property that distinguishes a citable conjecture from an
        uncitable one, which is the whole point of retrieving literature
        rather than training on it. A passage failing this is still stored --
        removing it would hide that the index contains unverifiable material
        -- but it never qualifies a claim as citation-supported.
        """
        lowered = str(self.source_id).lower()
        return any(lowered.startswith(prefix) for prefix in CHECKABLE_ID_PREFIXES)

    def citation(self) -> str:
        """A short human-readable reference, for showing beside a claim."""
        parts = [self.title]
        if self.publication:
            parts.append(self.publication)
        if self.year:
            parts.append(str(self.year))
        return f"{', '.join(parts)} ({self.source_id})"

    def as_dict(self) -> dict[str, Any]:
        return {
            "passage_id": self.passage_id,
            "title": self.title,
            "source_id": self.source_id,
            "year": self.year,
            "publication": self.publication,
            "is_independently_checkable": self.is_independently_checkable,
            "citation": self.citation(),
        }


@dataclass
class LiteratureIndex:
    """Passages searchable by the concepts they mention.

    Deliberately concept-matched rather than embedding-matched. Not because
    embeddings are wrong -- they would likely retrieve more -- but because
    this project already measured what latent similarity costs in a clinical
    setting: a RAG system asked about heart failure retrieves "acute coronary
    syndrome" because it is close in latent space, not because it answers the
    question. Matching on concepts the graph already names keeps retrieval
    aligned with the same vocabulary the rest of the reasoning uses, and
    leaves an embedding layer as an addition a deployment can make rather
    than an assumption baked in here.
    """

    passages: list[LiteraturePassage] = field(default_factory=list)

    def add(self, passage: LiteraturePassage) -> None:
        self.passages.append(passage)

    def add_many(self, passages: Iterable[LiteraturePassage]) -> int:
        added = list(passages)
        self.passages.extend(added)
        return len(added)

    def __len__(self) -> int:
        return len(self.passages)

    def search(
        self,
        concepts: Sequence[str],
        graph: ConceptGraphView,
        *,
        limit: int = 5,
        checkable_only: bool = True,
    ) -> list["RetrievedPassage"]:
        """Find passages mentioning these concepts, most relevant first.

        ``checkable_only`` defaults to True: material that cannot be opened
        and verified should not reach a vetting engine as though it could,
        and a caller wanting everything has to ask for it explicitly.

        Ranked by how many of the queried concepts a passage mentions, the
        same breadth-before-strength rule `candidate_retrieval` uses -- a
        passage touching three of the case's concepts is more relevant than
        one mentioning a single concept repeatedly.
        """
        wanted = {normalise_concept(item) for item in concepts if normalise_concept(item)}
        if not wanted:
            return []

        hits: list[RetrievedPassage] = []
        for passage in self.passages:
            if checkable_only and not passage.is_independently_checkable:
                continue
            found = set(mentioned_concepts(passage.text, graph, max_results=20)) & wanted
            if found:
                hits.append(RetrievedPassage(passage=passage, matched_concepts=tuple(sorted(found))))

        hits.sort(key=lambda item: (-len(item.matched_concepts), item.passage.passage_id))
        return hits[:limit]


@dataclass(frozen=True)
class RetrievedPassage:
    """A passage the search returned, with what it matched on."""

    passage: LiteraturePassage
    matched_concepts: tuple[str, ...]

    @property
    def breadth(self) -> int:
        return len(self.matched_concepts)

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.passage.as_dict(),
            "matched_concepts": list(self.matched_concepts),
            "breadth": self.breadth,
        }


def as_vetting_context(retrieved: Sequence[RetrievedPassage], *, max_chars: int = 4000) -> str:
    """Format retrieved passages as context for a vetting engine.

    Every passage is prefixed with its citation, not appended with it. A
    model reading context attributes what it reads to whatever is nearest,
    and a reference trailing a long passage is easy to lose; leading with it
    makes the source inseparable from the text it justifies.

    Truncated by whole passages rather than mid-text: half a passage under a
    full citation would attribute to a source something it did not finish
    saying.
    """
    blocks: list[str] = []
    used = 0
    for item in retrieved:
        block = f"[{item.passage.citation()}]\n{item.passage.text}"
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)
