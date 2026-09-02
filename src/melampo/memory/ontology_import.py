"""Import disease-phenotype annotations into interval-valued concept edges.

The bottleneck for the concept graph was never the topology, it was the numbers:
an ontology states that a relation exists, not how strongly it holds. HPO
annotations are the exception. They carry a frequency column, and that column is
already published as a **range** rather than a point:

    Obligate       100%
    Very frequent  80–99%
    Frequent       30–79%
    Occasional     5–29%
    Very rare      1–4%
    Excluded       0%

So the interval representation is not a formalism imposed on the data. It is
what the data already says, and a point estimate would have been the lossy step.
"Occasional" is not 0.17; it is somewhere between 0.05 and 0.29, and the
difference matters to every consumer that reads a bound.

Annotations also arrive as observed fractions — ``1/1``, ``3/12`` — where the
uncertainty is sampling rather than categorical. These become Wilson score
intervals, so the width falls out of the sample size: one observation of one
patient yields a wide interval, forty-five of fifty a narrow one. Epistemic
uncertainty is measured here, not assigned.

Two annotation forms produce a documented exclusion rather than a gap: the
``Excluded`` frequency term, and the ``NOT`` qualifier. Both mean the phenotype
was looked for and not found, which is the distinction a single weight cannot
carry — and the one that separates "we checked" from "nobody checked".
"""

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from .concept_paths import ConceptEdge, InMemoryConceptGraph

RELATION_HAS_PHENOTYPE = "has_phenotype"

FREQUENCY_TERMS: dict[str, tuple[float, float]] = {
    "HP:0040280": (1.00, 1.00),
    "HP:0040281": (0.80, 0.99),
    "HP:0040282": (0.30, 0.79),
    "HP:0040283": (0.05, 0.29),
    "HP:0040284": (0.01, 0.04),
    "HP:0040285": (0.00, 0.00),
}

FREQUENCY_TERM_NAMES: dict[str, str] = {
    "HP:0040280": "obligate",
    "HP:0040281": "very frequent",
    "HP:0040282": "frequent",
    "HP:0040283": "occasional",
    "HP:0040284": "very rare",
    "HP:0040285": "excluded",
}

EXCLUDED_TERM = "HP:0040285"
NOT_QUALIFIER = "NOT"
WILSON_Z = 1.96

HPOA_COLUMNS = (
    "database_id",
    "disease_name",
    "qualifier",
    "hpo_id",
    "reference",
    "evidence",
    "onset",
    "frequency",
    "sex",
    "modifier",
    "aspect",
    "biocuration",
)


@dataclass(frozen=True)
class Annotation:
    """One disease-phenotype row, before conversion to an edge."""

    disease_id: str
    disease_name: str
    phenotype_id: str
    frequency: str
    qualifier: str
    reference: str
    aspect: str

    @property
    def is_excluded(self) -> bool:
        return self.qualifier.upper() == NOT_QUALIFIER or self.frequency == EXCLUDED_TERM


def wilson_interval(positive: int, total: int, z: float = WILSON_Z) -> tuple[float, float]:
    """Score interval for a proportion, with no external dependency.

    Chosen over the plain proportion because the plain proportion discards the
    sample size: 1/1 and 50/50 both give 1.0, and only one of them is knowledge.
    Here the first stays wide and the second narrows, which is exactly the
    epistemic width the edge is meant to carry.
    """
    if total <= 0:
        return (0.0, 1.0)
    proportion = positive / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = (z / denominator) * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def parse_frequency(value: str) -> tuple[float, float] | None:
    """Convert an HPOA frequency cell to an interval.

    Returns ``None`` when the cell is empty, which is a gap rather than a zero:
    the annotation simply does not state a frequency.
    """
    text = (value or "").strip()
    if not text:
        return None
    if text in FREQUENCY_TERMS:
        return FREQUENCY_TERMS[text]
    if text.endswith("%"):
        try:
            fraction = float(text[:-1]) / 100.0
        except ValueError:
            return None
        return (max(0.0, fraction), min(1.0, fraction))
    if "/" in text:
        head, _, tail = text.partition("/")
        try:
            positive, total = int(head), int(tail)
        except ValueError:
            return None
        if total <= 0 or positive < 0 or positive > total:
            return None
        return wilson_interval(positive, total)
    return None


def parse_hpoa(lines: Iterable[str]) -> Iterator[Annotation]:
    """Parse the HPOA tab-separated format, skipping comments and the header."""
    columns: list[str] | None = None
    for raw in lines:
        line = raw.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if columns is None:
            if fields[0] == "database_id":
                columns = fields
                continue
            columns = list(HPOA_COLUMNS)
        record = dict(zip(columns, fields, strict=False))
        if not record.get("database_id") or not record.get("hpo_id"):
            continue
        yield Annotation(
            disease_id=record.get("database_id", ""),
            disease_name=record.get("disease_name", ""),
            phenotype_id=record.get("hpo_id", ""),
            frequency=record.get("frequency", ""),
            qualifier=record.get("qualifier", ""),
            reference=record.get("reference", ""),
            aspect=record.get("aspect", ""),
        )


def annotation_to_edge(
    annotation: Annotation,
    *,
    label_for: dict[str, str] | None = None,
    unstated_frequency_is_gap: bool = True,
) -> ConceptEdge | None:
    """Convert one annotation to an interval-valued edge.

    An annotation without a stated frequency becomes an explicit unknown edge
    rather than being dropped. Dropping it would make the relation
    indistinguishable from one nobody has recorded, and the traversal needs to
    tell those apart: an unknown edge can be crossed and reported as unknown,
    which is what makes it a candidate for completion.
    """
    source = annotation.disease_name.strip() or annotation.disease_id
    target = (label_for or {}).get(annotation.phenotype_id, annotation.phenotype_id)
    if not source or not target:
        return None

    provenance = f"hpoa:{annotation.disease_id}:{annotation.reference or 'unreferenced'}"

    if annotation.is_excluded:
        return ConceptEdge(source, RELATION_HAS_PHENOTYPE, target, weight=0.0, provenance=provenance, lower=0.0, upper=0.0)

    bounds = parse_frequency(annotation.frequency)
    if bounds is None:
        if not unstated_frequency_is_gap:
            return None
        return ConceptEdge.unknown(source, RELATION_HAS_PHENOTYPE, target, provenance=provenance)

    lower, upper = bounds
    return ConceptEdge(
        source,
        RELATION_HAS_PHENOTYPE,
        target,
        weight=(lower + upper) / 2.0,
        provenance=provenance,
        lower=lower,
        upper=upper,
    )


def build_edges(
    annotations: Iterable[Annotation],
    *,
    label_for: dict[str, str] | None = None,
    unstated_frequency_is_gap: bool = True,
) -> list[ConceptEdge]:
    edges: list[ConceptEdge] = []
    for annotation in annotations:
        edge = annotation_to_edge(
            annotation, label_for=label_for, unstated_frequency_is_gap=unstated_frequency_is_gap
        )
        if edge is not None:
            edges.append(edge)
    return edges


def build_graph(
    lines: Iterable[str],
    *,
    label_for: dict[str, str] | None = None,
    unstated_frequency_is_gap: bool = True,
) -> InMemoryConceptGraph:
    """Read HPOA content into a traversable concept graph."""
    return InMemoryConceptGraph.from_edges(
        build_edges(
            parse_hpoa(lines), label_for=label_for, unstated_frequency_is_gap=unstated_frequency_is_gap
        )
    )


def import_summary(edges: Iterable[ConceptEdge]) -> dict[str, Any]:
    """Describe what an import produced, by epistemic state.

    Reported because a graph built mostly of unknown edges looks the same as a
    populated one from the outside, and behaves very differently.
    """
    counts: dict[str, int] = {}
    widths: list[float] = []
    for edge in edges:
        counts[edge.state] = counts.get(edge.state, 0) + 1
        widths.append(edge.width)
    total = sum(counts.values())
    return {
        "edges": total,
        "by_state": dict(sorted(counts.items())),
        "gap_fraction": round(counts.get("gap", 0) / total, 4) if total else 0.0,
        "mean_width": round(sum(widths) / len(widths), 4) if widths else 0.0,
    }
