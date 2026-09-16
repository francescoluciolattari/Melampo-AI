"""Populate the concept description store, and keep both sides of tier 3 on one format.

Two problems solved together, because the second is what makes the first
safe to run at scale.

**Populating from data already on disk.** `hp.obo` carries a curated prose
definition for 17,441 of its terms, each with a PMID attached -- downloaded
with the ontology since the file first arrived, parsed by nothing. That is
exactly the source text tier 3's concept side needs, already licensed,
already versioned with the release, and already covering far more concepts
than promotion or literature retrieval would reach in months of running.
Bulk population from it costs one pass, not a research project.

**Keeping the vetting model and the extractor on one format.** Raised
directly in discussion: rather than having the extractor re-read a vetting
model's free-text answer, ask the model to emit structure alongside its
prose -- and, crucially, derive the format it is asked for *from the
extractor itself*, so a change to the extractor cannot silently desynchronise
the two sides of a comparison.

That last point is the load-bearing one. `compare_structures` matches
entity and relation strings literally. If the concept side's cached
descriptions were extracted under one convention and the claim side arrives
under another -- "required_for" against "requires", "1-alpha-hydroxylase"
against "1α-hydroxylase" -- the arithmetic returns a low score for two
structures that describe the same mechanism, and the failure looks like
disagreement rather than what it is. `canonical_format_example` generates
the prompt fragment from the same `ExtractedStructure` shape the extractor
produces, so the instruction given to a vetting model and the output the
extractor yields cannot drift apart without the example changing too.

**What emitting its own structure does not let a model do.** It does not let
it grade itself. The comparison is still arithmetic, still against a cached
description the model never sees, and `compare_structures` is unchanged. A
model shaping its structure to look agreeable has no target to shape it
toward -- it does not know what the concept side says. What it can do is
report its own claim more faithfully than a second model re-reading its
prose, which is the actual argument for the design.
"""

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .concept_resolution import OntologyTerm
from .structural_comparison import (
    ConceptDescriptionStore,
    ExtractedRelation,
    ExtractedStructure,
)

# A worked example, in the exact shape `ExtractedStructure` round-trips
# through. Written once here rather than duplicated into a prompt string, so
# the format a vetting model is shown and the format the extractor produces
# have a single definition between them.
CANONICAL_EXAMPLE = ExtractedStructure(
    source_text=(
        "Granulomatous macrophages express 1-alpha-hydroxylase, converting 25-hydroxyvitamin D "
        "to calcitriol, which increases intestinal calcium absorption."
    ),
    entities=(
        "granulomatous macrophages",
        "1-alpha-hydroxylase",
        "25-hydroxyvitamin D",
        "calcitriol",
        "intestinal calcium absorption",
    ),
    relations=(
        ExtractedRelation("granulomatous macrophages", "expresses", "1-alpha-hydroxylase"),
        ExtractedRelation("1-alpha-hydroxylase", "converts", "calcitriol"),
        ExtractedRelation("calcitriol", "increases", "intestinal calcium absorption"),
    ),
)


def canonical_format_example() -> str:
    """The structure format, as a prompt fragment, derived from the shape itself.

    Generated rather than hand-written so it cannot fall out of step with
    what `_parse_extraction` accepts and `compare_structures` matches on. A
    caller embedding this in a vetting prompt is showing the model the same
    contract the extractor is held to.
    """
    example = {
        "entities": list(CANONICAL_EXAMPLE.entities),
        "relations": [relation.as_dict() for relation in CANONICAL_EXAMPLE.relations],
    }
    return (
        "Alongside your prose answer, emit the mechanism's structure as JSON in exactly this "
        "shape, on its own line prefixed with STRUCTURE:\n\n"
        f"Text: {CANONICAL_EXAMPLE.source_text}\n"
        f"STRUCTURE: {json.dumps(example, ensure_ascii=False)}\n\n"
        "Use lowercase entity names. Prefer a short verb for the relation. Include only what "
        "your own answer states -- never a relation you did not assert."
    )


def parse_model_emitted_structure(answer: str) -> ExtractedStructure | None:
    """Read a STRUCTURE: line a vetting model emitted alongside its prose.

    Returns None when the model emitted no structure at all, which is a
    normal outcome, not a fault: a model that answers in prose only should
    fall through to the extractor exactly as before, and treating a missing
    line as an error would make the whole cascade depend on every model
    cooperating with a format it may not have been prompted for.
    """
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("STRUCTURE:"):
            continue
        payload = stripped.split(":", 1)[1].strip()
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            return None
        entities = tuple(str(item).strip() for item in data.get("entities", []) if str(item).strip())
        relations = tuple(
            ExtractedRelation(
                subject=str(item.get("subject", "")).strip(),
                relation=str(item.get("relation", "")).strip(),
                object=str(item.get("object", "")).strip(),
            )
            for item in data.get("relations", [])
            if isinstance(item, dict) and item.get("subject") and item.get("relation") and item.get("object")
        )
        if not entities and not relations:
            return None
        return ExtractedStructure(source_text=answer, entities=entities, relations=relations)
    return None


@dataclass
class PopulationReport:
    """What a bulk population run did, and what it could not do."""

    described: int = 0
    skipped_no_definition: int = 0
    skipped_extraction_empty: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def attempted(self) -> int:
        return self.described + self.skipped_no_definition + self.skipped_extraction_empty

    def as_dict(self) -> dict[str, Any]:
        return {
            "described": self.described,
            "skipped_no_definition": self.skipped_no_definition,
            "skipped_extraction_empty": self.skipped_extraction_empty,
            "attempted": self.attempted,
            "errors": list(self.errors[:10]),
        }


def populate_from_ontology(
    terms: Iterable[OntologyTerm],
    store: ConceptDescriptionStore,
    extractor: Any,
    *,
    only_concepts: Sequence[str] | None = None,
    store_path: Path | None = None,
    limit: int | None = None,
) -> PopulationReport:
    """Extract a structure from each term's curated definition and store it.

    ``only_concepts`` restricts the run to concepts that matter now -- the
    vetting bench's own set, or the concepts appearing in confirmed cases --
    rather than all 17,441. Bulk population is cheap in principle and still
    17,441 model calls in practice, and a run scoped to what is actually
    being reasoned about reaches useful coverage first. Passing None means
    everything, which is the right choice for an overnight run and the wrong
    one for a first trial.

    Never overwrites an existing description: a concept already described
    from literature, or from a promotion's justification, keeps what it has.
    Bulk population fills gaps, it does not re-derive what is there.
    """
    report = PopulationReport()
    wanted = {item.strip().lower() for item in only_concepts} if only_concepts is not None else None

    for term in _limited(terms, limit):
        if term.obsolete or not term.name:
            continue
        if wanted is not None and term.name.strip().lower() not in wanted:
            continue
        if store.get(term.name) is not None:
            continue
        if not term.definition:
            report.skipped_no_definition += 1
            continue
        try:
            structure = extractor(term.definition)
        except Exception as error:  # noqa: BLE001 - one bad term must not abort a 17k-term run
            report.errors.append(f"{term.term_id}: {error}")
            continue
        if structure.is_empty:
            report.skipped_extraction_empty += 1
            continue
        store.add(term.name, structure)
        report.described += 1
        if store_path is not None:
            store.append_to(store_path, term.name)

    return report


def _limited(terms: Iterable[OntologyTerm], limit: int | None) -> Iterator[OntologyTerm]:
    if limit is None:
        yield from terms
        return
    for index, term in enumerate(terms):
        if index >= limit:
            return
        yield term
