"""Tier 3: compare a claim and a concept's literature description as graphs, not as text.

The last resort in `concept_normalisation`'s cascade, reached only when
lexical matching and embedding similarity both failed. The design was
proposed directly in discussion and is better than the obvious alternative,
for a reason worth stating precisely.

**The obvious alternative, and why it is rejected.** Hand a language model
two phrases and ask "do these mean the same thing?". That is adjudication by
a model whose answer cannot be checked, on exactly the question the concept
graph exists to answer deterministically -- the thing
`root_model_cross_check` was built to refuse and `mechanism_verification`
was built to replace.

**What this does instead.** A model *extracts* structure -- entities and the
relations between them -- from two texts, and the comparison of those two
structures is arithmetic. The model never says "these match"; it says "this
text contains these entities in these relations", and a deterministic
function decides whether two such structures overlap enough. This is the
same division of labour the ingestion role already uses, where a parser
extracts findings from a report without judging them, and it means a
disagreement between two extractions is visible as differing structures
rather than hidden inside a yes/no.

**Asymmetry between the two sides is deliberate.** The concept side's graph
is built once, offline, from curated literature for that concept, and
reused across every case -- so it can be reviewed, corrected, and
regenerated when the literature updates, exactly like the imported ontology
layer it sits beside. The claim side's graph is necessarily built per
answer. Only one of the two is expensive, and only one is re-derivable; the
cache makes that asymmetry pay.

**Nothing here installs a model.** `extractor` is injected. Without one this
tier resolves nothing and the cascade simply reports that, which is the
correct degradation: tier 3 unavailable is a narrower system, not a broken
one.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .concept_paths import normalise_concept

# How much two extracted structures must overlap to count as describing the
# same thing. Deliberately demanding: this tier runs only after two cheaper
# ones failed, so the phrase in hand is already known to be lexically and
# semantically distant from every candidate. A permissive threshold here
# would be the system talking itself into a match it has twice failed to
# find honestly.
DEFAULT_OVERLAP_THRESHOLD = 0.6


@dataclass(frozen=True)
class ExtractedRelation:
    """One subject-relation-object triple pulled out of a text."""

    subject: str
    relation: str
    object: str

    def normalised(self) -> tuple[str, str, str]:
        return (
            normalise_concept(self.subject),
            normalise_concept(self.relation),
            normalise_concept(self.object),
        )

    def as_dict(self) -> dict[str, str]:
        return {"subject": self.subject, "relation": self.relation, "object": self.object}


@dataclass(frozen=True)
class ExtractedStructure:
    """The entities and relations a text was found to contain."""

    source_text: str
    entities: tuple[str, ...] = ()
    relations: tuple[ExtractedRelation, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.entities and not self.relations

    def normalised_entities(self) -> set[str]:
        return {normalise_concept(entity) for entity in self.entities if normalise_concept(entity)}

    def normalised_relations(self) -> set[tuple[str, str, str]]:
        return {relation.normalised() for relation in self.relations}

    def as_dict(self) -> dict[str, Any]:
        return {
            "entities": list(self.entities),
            "relations": [relation.as_dict() for relation in self.relations],
        }


@dataclass(frozen=True)
class StructuralComparison:
    """How much two extracted structures overlap, and on what."""

    entity_overlap: float
    relation_overlap: float
    shared_entities: tuple[str, ...]
    shared_relations: tuple[tuple[str, str, str], ...]

    @property
    def score(self) -> float:
        """One number, weighted toward relations over entities.

        Two texts about the same clinical area share entities easily --
        "calcium", "vitamin d", "macrophage" appear in a great many
        descriptions. Sharing a *relation* between the same two entities is
        a much stronger signal that they describe the same mechanism rather
        than the same subject area. Weighting them equally would let topical
        similarity masquerade as mechanistic agreement, which is the
        distinction this whole tier is trying to draw.
        """
        return 0.35 * self.entity_overlap + 0.65 * self.relation_overlap

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "entity_overlap": round(self.entity_overlap, 4),
            "relation_overlap": round(self.relation_overlap, 4),
            "shared_entities": list(self.shared_entities),
            "shared_relations": [list(relation) for relation in self.shared_relations],
        }


def compare_structures(left: ExtractedStructure, right: ExtractedStructure) -> StructuralComparison:
    """Overlap between two extracted structures -- pure arithmetic, no model involved.

    Jaccard on both entities and relations. Symmetric by construction, so
    which text was the claim and which the description cannot change the
    answer -- a property worth having when one side is cached and the other
    is not, since an asymmetric measure would make the cache observable in
    the results.
    """
    left_entities, right_entities = left.normalised_entities(), right.normalised_entities()
    left_relations, right_relations = left.normalised_relations(), right.normalised_relations()

    shared_entities = left_entities & right_entities
    shared_relations = left_relations & right_relations
    union_entities = left_entities | right_entities
    union_relations = left_relations | right_relations

    return StructuralComparison(
        entity_overlap=len(shared_entities) / len(union_entities) if union_entities else 0.0,
        relation_overlap=len(shared_relations) / len(union_relations) if union_relations else 0.0,
        shared_entities=tuple(sorted(shared_entities)),
        shared_relations=tuple(sorted(shared_relations)),
    )


@dataclass
class ConceptDescriptionStore:
    """Pre-built structures for each graph concept, from curated literature.

    Built offline and cached, because that is what makes this tier
    affordable: a concept's description is extracted once and compared
    thousands of times. The store is deliberately plain -- concept name to
    structure -- so it can be regenerated wholesale when the literature
    updates, the same posture `graph_store` takes toward the derivable
    imported layer.

    Persisted the same way: JSONL, append-only. A description that took a
    model call to extract is exactly as expensive to lose as a promoted
    graph edge, and gets the same discipline -- appending a new description
    never rewrites the file, so a truncated write costs one line rather than
    the whole store.
    """

    structures: dict[str, ExtractedStructure] = field(default_factory=dict)

    def add(self, concept: str, structure: ExtractedStructure) -> None:
        self.structures[normalise_concept(concept)] = structure

    def get(self, concept: str) -> ExtractedStructure | None:
        return self.structures.get(normalise_concept(concept))

    def __len__(self) -> int:
        return len(self.structures)

    def to_record(self, concept: str) -> dict[str, Any] | None:
        structure = self.get(concept)
        if structure is None:
            return None
        return {
            "concept": normalise_concept(concept),
            "source_text": structure.source_text,
            "entities": list(structure.entities),
            "relations": [relation.as_dict() for relation in structure.relations],
        }

    def append_to(self, path: Path, concept: str) -> bool:
        """Write one concept's description to a JSONL file, without rewriting it.

        Returns False and writes nothing if the concept has no stored
        description -- there is nothing to persist, and silently writing an
        empty record would let a later load mistake "never extracted" for
        "extracted as nothing".
        """
        record = self.to_record(concept)
        if record is None:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True

    @classmethod
    def load(cls, path: Path) -> "ConceptDescriptionStore":
        """Read every persisted description. A missing file yields an empty store.

        The normal state of a fresh checkout, before tier 3 has ever run --
        not an error, the same posture `LearnedEdgeStore.load` takes.
        """
        store = cls()
        if not path.exists():
            return store
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            store.structures[record["concept"]] = ExtractedStructure(
                source_text=record.get("source_text", ""),
                entities=tuple(record.get("entities", ())),
                relations=tuple(
                    ExtractedRelation(**relation) for relation in record.get("relations", ())
                ),
            )
        return store

    @property
    def described_concepts(self) -> set[str]:
        return set(self.structures)


@dataclass
class StructuralResolver:
    """Resolve a phrase to a concept by comparing extracted structures.

    Plugs into `NormalisationCascade` as its `structural_resolver`: callable
    with (phrase, candidates), returning a concept name or None.
    """

    store: ConceptDescriptionStore
    extractor: Callable[[str], ExtractedStructure] | None = None
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD
    last_comparison: StructuralComparison | None = None
    # Which route produced the last claim structure: "model_emitted" when the
    # vetting model supplied its own, "extractor" when a model call was made,
    # None when neither could. Recorded because the two are not equally
    # reliable -- a model reporting its own claim is more faithful than a
    # second model re-reading its prose -- and a reader should be able to
    # tell which happened without re-running anything.
    last_structure_source: str | None = None

    def _structure_for(self, phrase: str) -> ExtractedStructure | None:
        """The claim's structure, preferring what the model emitted over re-extraction.

        A vetting model prompted with `canonical_format_example` emits its
        own structure alongside its prose. Using it skips a model call and,
        more importantly, skips a lossy round-trip: the model that formed the
        claim knows its structure better than a second model parsing the
        sentence afterwards. Falling back to the extractor keeps every model
        that was not prompted that way -- or chose not to comply -- working
        exactly as before.
        """
        from .description_population import (
            parse_model_emitted_structure,
        )

        emitted = parse_model_emitted_structure(phrase)
        if emitted is not None and not emitted.is_empty:
            self.last_structure_source = "model_emitted"
            return emitted

        if self.extractor is None:
            self.last_structure_source = None
            return None
        try:
            structure = self.extractor(phrase)
        except Exception:  # noqa: BLE001 - a failing extractor degrades this tier, never breaks the cascade
            self.last_structure_source = None
            return None
        self.last_structure_source = "extractor"
        return structure

    def __call__(self, phrase: str, candidates: Sequence[str]) -> str | None:
        if not phrase:
            return None

        described = [concept for concept in candidates if self.store.get(concept) is not None]
        if not described:
            # No candidate has a cached description, so there is nothing to
            # compare against. Extracting the phrase anyway would spend a
            # model call to learn nothing.
            return None

        claim_structure = self._structure_for(phrase)
        if claim_structure is None or claim_structure.is_empty:
            return None

        best_concept, best_comparison = None, None
        for concept in described:
            comparison = compare_structures(claim_structure, self.store.get(concept))
            if best_comparison is None or comparison.score > best_comparison.score:
                best_concept, best_comparison = concept, comparison

        self.last_comparison = best_comparison
        if best_comparison is not None and best_comparison.score >= self.overlap_threshold:
            return best_concept
        return None
