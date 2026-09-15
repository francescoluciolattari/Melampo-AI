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

from .concept_paths import ConceptGraphView, InMemoryConceptGraph, normalise_concept
from .ontology_import import build_edges, parse_hpoa

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

# Gene annotation files. Optional: they add an edge type phenotype.hpoa
# cannot express, and their absence narrows the graph rather than breaking it.
GENES_TO_PHENOTYPE_ENV = "MELAMPO_GENES_TO_PHENOTYPE_PATH"
DEFAULT_GENES_TO_PHENOTYPE_FILENAMES = ("genes_to_phenotype.txt", "phenotype_to_genes.txt")
GENES_TO_DISEASE_ENV = "MELAMPO_GENES_TO_DISEASE_PATH"
DEFAULT_GENES_TO_DISEASE_FILENAMES = ("genes_to_disease.txt",)

# MAxO medical action annotations. Loaded into their own index, never into
# the concept graph -- see medical_actions.MedicalActionIndex for why.
MAXO_PATH_ENV = "MELAMPO_MAXO_PATH"
DEFAULT_MAXO_FILENAMES = ("maxo-annotations.tsv",)


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


def load_hpoa_graph(
    path: str | Path, *, obo_path: str | Path | None = None, include_gene_annotations: bool = True
) -> GraphSource:
    """Build the concept graph from a real HPO annotation release.

    `phenotype.hpoa` identifies phenotypes only by HPO id (`HP:0001166`), not
    by name, so a graph built from it alone has unreadable targets -- and,
    worse for this project, targets no clinical phrase could ever match:
    every lexical and embedding comparison downstream works on text, and
    `HP:0001166` is not text anyone writes. `hp.obo` carries the id-to-label
    map, so when it is available the labels are resolved here rather than
    leaving an id-shaped graph for every caller to translate.

    A missing hp.obo is not fatal -- the graph still builds, with ids as
    targets -- and `GraphSource.detail` says which happened, so a run on an
    id-shaped graph is visible rather than silently producing nothing that
    matches.
    """
    path = Path(path)
    lines: Iterable[str] = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    label_for: dict[str, str] = {}
    resolved_obo = find_hp_obo_file(obo_path)
    if resolved_obo is not None:
        from .concept_resolution import (
            parse_obo,
        )

        label_for = {
            term.term_id: term.name
            for term in parse_obo(resolved_obo.read_text(encoding="utf-8", errors="ignore").splitlines())
            if term.name
        }

    annotations = list(parse_hpoa(lines))
    hpoa_edges = list(build_edges(annotations, label_for=label_for or None))

    # genes_to_disease.txt identifies diseases only by id; phenotype.hpoa
    # indexes the same ids and carries their names, so the map is built from
    # what was just parsed rather than read from a second source.
    name_for_disease_id = {
        annotation.disease_id: annotation.disease_name
        for annotation in annotations
        if annotation.disease_id and annotation.disease_name
    }
    gene_edges = (
        load_gene_annotation_edges(name_for_disease_id=name_for_disease_id) if include_gene_annotations else []
    )
    graph = InMemoryConceptGraph.from_edges([*hpoa_edges, *gene_edges])

    edges = sum(len(graph.edges_from(concept)) for concept in graph.concepts())
    detail = f"loaded from {path}"
    detail += f"; phenotype labels resolved from {resolved_obo}" if label_for else "; no hp.obo found, targets are HPO ids"
    if gene_edges:
        detail += f"; {len(gene_edges):,} gene-annotation edges included"
    return GraphSource(graph=graph, source=SOURCE_HPOA, edge_count=edges, detail=detail)


def load_gene_annotation_edges(
    *,
    genes_to_phenotype: str | Path | None = None,
    genes_to_disease: str | Path | None = None,
    name_for_disease_id: dict[str, str] | None = None,
) -> list[Any]:
    """Load gene-phenotype and gene-disease edges, if those files are present.

    A separate call from `load_hpoa_graph` rather than folded into it,
    because the two answer different questions and a caller may legitimately
    want one without the other: `phenotype.hpoa` supports "does this disease
    manifest this finding", while the gene files support "is there a genetic
    link between this finding and that condition" -- a connection a vetting
    question can genuinely rest on, and one `has_phenotype` alone can never
    express.

    Missing files yield no edges rather than raising: a deployment without
    the gene annotations is a narrower graph, not a broken one.
    """
    from .gene_annotations import (
        gene_disease_edges,
        gene_phenotype_edges,
        parse_genes_to_disease,
        parse_genes_to_phenotype,
    )

    edges: list[Any] = []

    phenotype_path = _find_file(genes_to_phenotype, GENES_TO_PHENOTYPE_ENV, DEFAULT_GENES_TO_PHENOTYPE_FILENAMES)
    if phenotype_path is not None:
        lines = phenotype_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        edges.extend(gene_phenotype_edges(parse_genes_to_phenotype(lines)))

    disease_path = _find_file(genes_to_disease, GENES_TO_DISEASE_ENV, DEFAULT_GENES_TO_DISEASE_FILENAMES)
    if disease_path is not None:
        lines = disease_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        edges.extend(gene_disease_edges(parse_genes_to_disease(lines), name_for_disease_id=name_for_disease_id))

    return edges


def _find_file(explicit: str | Path | None, env_var: str, filenames: tuple[str, ...]) -> Path | None:
    """Shared search order for every optional data file: explicit, env, cwd, data/."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    from_env = os.environ.get(env_var)
    if from_env:
        candidates.append(Path(from_env))
    for filename in filenames:
        candidates.append(Path(filename))
        candidates.append(Path("data") / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_medical_actions(explicit_path: str | Path | None = None) -> Any:
    """Load the MAxO medical action index, or None if the file is absent.

    Returns None rather than an empty index, the same distinction
    `load_synonym_index` draws: "no MAxO file" and "a file that parsed into
    nothing" are different situations and only the second is a fault.

    Deliberately not folded into `load_hpoa_graph`: these annotations must
    not enter the concept graph, and returning them from the graph loader
    would invite exactly that.
    """
    from .medical_actions import (
        MedicalActionIndex,
        parse_maxo_annotations,
    )

    path = _find_file(explicit_path, MAXO_PATH_ENV, DEFAULT_MAXO_FILENAMES)
    if path is None:
        return None
    lines = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    return MedicalActionIndex.from_annotations(parse_maxo_annotations(lines))


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
    explicit_path: str | Path | None = None, *, include_history: bool = True, **term_index_kwargs: Any
) -> Any:
    """Build a `TermIndex` from a real hp.obo release, or None if there is none to find.

    Returns None rather than an empty index when no file is found, so a
    caller can tell "no synonym data available" (skip the lexical-synonym
    check) from "loaded an index with zero terms" (a genuine parsing
    problem) -- collapsing the two would hide a broken file behind the same
    behaviour as a missing one.

    ``include_history`` is on by default: a term renamed by an HPO release
    must never stop being recognisable under the name it used to have, and
    the update workflow records every such rename permanently. Folding those
    historical names into the index here, rather than requiring every caller
    to remember to attach `TermHistoryStore` separately, is what makes that
    guarantee automatic rather than opt-in.
    """
    from .concept_resolution import TermIndex

    path = find_hp_obo_file(explicit_path)
    if path is None:
        return None
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    index = TermIndex.from_obo(lines, **term_index_kwargs)

    if include_history:
        from .term_history import TermHistoryStore

        history = TermHistoryStore(path.parent)
        for term_id, old_names in history.synonyms_by_term_id().items():
            term = index.by_id.get(term_id)
            if term is None:
                continue
            for old_name in old_names:
                normalised = normalise_concept(old_name)
                if normalised and normalised not in index.by_surface:
                    index.by_surface[normalised] = [term_id]

    return index


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
