"""Load the verification graph from real data, and say plainly when that is not possible.

An audit of three live vetting-bench runs found the bench had been scoring
candidates against a 33-edge, hand-written fixture -- not the 285,598-edge
HPO import the project actually has. Both numbers appear in this codebase and
were conflated: the bench measured how closely a model's phrasing matched
thirty-three lines someone wrote by hand, then reported it as grounding
against the concept graph.

**The silent-fallback trap this module exists to avoid.** The obvious design
-- try the real graph, quietly use the fixture if it is missing -- would
reproduce exactly the failure just found, and worse, would make it
undetectable: every run would look like a real run. `GraphSource` therefore
always reports which source it used and how large it is, and callers that
care (the bench does) can refuse to proceed on a fixture. Falling back is
allowed; falling back *silently* is not.

**A structural limit worth stating even once the real graph is loaded.** HPO
annotations import as a single relation type, `has_phenotype`: disease to
observable phenotype. They do not encode mechanistic causal chains
(granuloma formation -> 1-alpha-hydroxylase activity -> calcitriol excess),
which is precisely what a vetting question asks about. Connecting the bench
to the real graph fixes the scale problem and not the kind problem; a
mechanistic layer is separate work, and `SOURCE_HPOA` reporting honestly is
what keeps that distinction visible rather than buried.
"""

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .concept_paths import ConceptGraphView, InMemoryConceptGraph
from .ontology_import import build_graph

SOURCE_HPOA = "hpoa"
SOURCE_FIXTURE = "hand_written_fixture"
SOURCE_EMPTY = "empty"

# Where a deployment is expected to place the HPO annotation release. Not
# downloaded automatically: the file carries its own licence terms and a
# release date that belongs in the provenance of every edge derived from it,
# so fetching it silently on first run would put undeclared data into a
# clinical knowledge base.
HPOA_PATH_ENV = "MELAMPO_HPOA_PATH"
DEFAULT_HPOA_FILENAMES = ("phenotype.hpoa", "phenotype_annotation.hpoa")

# Where a deployment places the HPO ontology release itself (term names,
# definitions, synonyms including layperson translations) -- a separate file
# from phenotype.hpoa, which carries only disease-to-phenotype associations
# and no synonym data at all.
HP_OBO_PATH_ENV = "MELAMPO_HP_OBO_PATH"
DEFAULT_OBO_FILENAMES = ("hp.obo",)


@dataclass(frozen=True)
class GraphSource:
    """A loaded graph together with where it came from and how big it is.

    Returned instead of a bare graph so no caller can use one without being
    able to see what it is. The three fields answer the three questions the
    audit showed were being skipped: which source, how many edges, and
    whether it is real data or scaffolding.
    """

    graph: ConceptGraphView
    source: str
    edge_count: int
    detail: str = ""

    @property
    def is_real_data(self) -> bool:
        """Whether this graph came from an ontology release rather than a fixture."""
        return self.source == SOURCE_HPOA

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "edge_count": self.edge_count,
            "is_real_data": self.is_real_data,
            "detail": self.detail,
        }


def find_hpoa_file(explicit_path: str | Path | None = None) -> Path | None:
    """Locate an HPO annotation file, or None if there is none to find.

    Checks, in order: an explicitly passed path, the `MELAMPO_HPOA_PATH`
    environment variable, then the conventional filenames in the working
    directory and a `data/` subdirectory. Returns None rather than raising --
    "no real graph available" is a normal state for a checkout without the
    data file, and the caller decides whether that is fatal.
    """
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    from_env = os.environ.get(HPOA_PATH_ENV)
    if from_env:
        candidates.append(Path(from_env))
    for filename in DEFAULT_HPOA_FILENAMES:
        candidates.append(Path(filename))
        candidates.append(Path("data") / filename)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_hpoa_graph(path: str | Path) -> GraphSource:
    """Build the concept graph from a real HPO annotation release."""
    path = Path(path)
    lines: Iterable[str] = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    graph = build_graph(lines)
    edges = sum(len(graph.edges_from(concept)) for concept in graph.concepts())
    return GraphSource(
        graph=graph,
        source=SOURCE_HPOA,
        edge_count=edges,
        detail=f"loaded from {path}",
    )


def find_hp_obo_file(explicit_path: str | Path | None = None) -> Path | None:
    """Locate an hp.obo release, or None if there is none to find.

    Same search order as `find_hpoa_file`, over the ontology file rather
    than the annotation file -- the two are separate downloads from the same
    HPO release and neither implies the other is present.
    """
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    from_env = os.environ.get(HP_OBO_PATH_ENV)
    if from_env:
        candidates.append(Path(from_env))
    for filename in DEFAULT_OBO_FILENAMES:
        candidates.append(Path(filename))
        candidates.append(Path("data") / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_synonym_index(
    explicit_path: str | Path | None = None, **term_index_kwargs: Any
) -> Any:
    """Build a `TermIndex` from a real hp.obo release, or None if there is none to find.

    Returns None rather than an empty index when no file is found, so a
    caller can tell "no synonym data available" (skip the lexical-synonym
    check) from "loaded an index with zero terms" (a genuine parsing
    problem) -- collapsing the two would hide a broken file behind the same
    behaviour as a missing one.
    """
    from .concept_resolution import (
        TermIndex,
    )

    path = find_hp_obo_file(explicit_path)
    if path is None:
        return None
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return TermIndex.from_obo(lines, **term_index_kwargs)


def load_verification_graph(
    *,
    explicit_path: str | Path | None = None,
    fixture_factory: Any = None,
    require_real_data: bool = False,
) -> GraphSource:
    """Load the best available verification graph, always reporting which one.

    ``require_real_data=True`` raises rather than falling back -- the setting
    a bench run making a candidate-selection decision should use, since the
    whole point of the audit that produced this module was that a fixture had
    been mistaken for real data. The default is False so unit tests and
    exploratory work keep running without the data file, but they get a
    `GraphSource` that says plainly what they are using.
    """
    path = find_hpoa_file(explicit_path)
    if path is not None:
        return load_hpoa_graph(path)

    if require_real_data:
        raise FileNotFoundError(
            "no HPO annotation file found: set "
            f"{HPOA_PATH_ENV} or place {DEFAULT_HPOA_FILENAMES[0]} in the working directory "
            "or data/. Refusing to fall back to a hand-written fixture, because a bench run "
            "scored against a fixture is not a measurement of grounding against the concept graph."
        )

    if fixture_factory is None:
        return GraphSource(
            graph=InMemoryConceptGraph.from_edges([]),
            source=SOURCE_EMPTY,
            edge_count=0,
            detail="no HPO file found and no fixture supplied",
        )

    graph = fixture_factory()
    edges = sum(len(graph.edges_from(concept)) for concept in graph.concepts())
    return GraphSource(
        graph=graph,
        source=SOURCE_FIXTURE,
        edge_count=edges,
        detail="hand-written fixture; not a measurement of grounding against real ontology data",
    )
