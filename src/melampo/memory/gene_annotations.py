"""Parse HPO's gene-phenotype and gene-disease annotation files into new relation types.

`phenotype.hpoa` gives the graph exactly one relation, `has_phenotype`, and
that ceiling was named directly as a structural limit: HPO's own annotation
release does not encode mechanistic causal chains, only disease-to-observable
mappings. It does not follow that HPO has nothing else to offer -- it
publishes three further files (`genes_to_phenotype.txt`,
`phenotype_to_genes.txt`, `genes_to_disease.txt`) that add a different edge
type, gene involvement, which the concept graph has never carried at all.

**Column detection is header-driven, deliberately.** `genes_to_phenotype.txt`
is confirmed from a working parse (`entrez_gene_id`, `entrez_gene_symbol`,
`hpo_term_id`, `hpo_term_name`, `frequency_raw`, `frequency_hpo`); the exact
column names for `phenotype_to_genes.txt` and `genes_to_disease.txt` were not
independently confirmed to the same standard, and hard-coding a guessed
position would fail silently on a header that differs by one column. Every
parser here reads its own header line and maps field names to positions the
same way `ontology_import.parse_hpoa` already does, so a file whose header
does not match what a parser expects raises rather than silently
misattributing a gene to the wrong field.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from .concept_paths import ConceptEdge

RELATION_ASSOCIATED_GENE = "associated_gene"
RELATION_CAUSES_DISEASE = "causes_disease"

# The confirmed genes_to_phenotype.txt header uses entrez_gene_symbol and
# hpo_term_id; some releases and phenotype_to_genes.txt (the same
# association from the other direction) have used gene_symbol and hpo_id.
# Both variants are accepted via the same alias mechanism genes_to_disease
# uses below, rather than committing to one and failing on the other.
GENE_PHENOTYPE_FIELD_ALIASES = {
    "gene_symbol": ("entrez_gene_symbol", "gene_symbol"),
    "hpo_id": ("hpo_term_id", "hpo_id"),
    "hpo_term_name": ("hpo_term_name", "hpo_name"),
}

# genes_to_disease.txt's exact header was not independently confirmed to the
# same standard as the other two files. The parser below is deliberately
# tolerant of the two column-name variants seen in different HPO release
# generations rather than committing to one; verify against the header of
# whatever release is actually in use before relying on this in production.
GENE_DISEASE_FIELD_ALIASES = {
    "gene_symbol": ("gene_symbol", "entrez_gene_symbol"),
    "disease_id": ("disease_id", "database_id"),
    "disease_name": ("disease_name",),
}


@dataclass(frozen=True)
class GenePhenotypeAssociation:
    """One row: a gene implicated in a phenotype, from either direction file."""

    gene_symbol: str
    hpo_id: str
    hpo_term_name: str


@dataclass(frozen=True)
class GeneDiseaseAssociation:
    """One row: a gene implicated in a disease."""

    gene_symbol: str
    disease_id: str
    disease_name: str


def _split_header(line: str) -> dict[str, int]:
    """Map a tab-separated header's field names to their column index.

    A `#`-prefixed header (as some HPO releases use) has its comment marker
    stripped before splitting, so `#hpo_id` and `hpo_id` resolve to the same
    field name -- a file using either convention parses identically.
    """
    fields = line.lstrip("#").rstrip("\n").split("\t")
    return {name.strip(): index for index, name in enumerate(fields)}


def parse_genes_to_phenotype(lines: Iterable[str]) -> Iterator[GenePhenotypeAssociation]:
    """Parse genes_to_phenotype.txt or phenotype_to_genes.txt.

    Both files carry the same association in the two directions HPO
    generates it from; the fields this project needs (gene symbol, HPO term
    id and name) are present in both, so one parser reads either.
    """
    columns: dict[str, int] | None = None
    resolved_names: dict[str, str] | None = None
    for raw in lines:
        line = raw.rstrip("\n")
        if not line:
            continue
        if columns is None:
            columns = _split_header(line)
            resolved_names = _resolve(columns, GENE_PHENOTYPE_FIELD_ALIASES, required=("gene_symbol", "hpo_id"))
            if resolved_names is None:
                # Not the header row this parser expects -- HPO releases have
                # used more than one column-naming convention over time.
                # Raising here, rather than silently treating a data row as
                # the header, is what makes a genuinely unrecognised file
                # visible instead of producing zero associations with no
                # explanation.
                raise ValueError(
                    f"unrecognised genes_to_phenotype header: {line!r}; "
                    f"expected variants of {list(GENE_PHENOTYPE_FIELD_ALIASES)}"
                )
            continue
        fields = line.split("\t")
        gene_symbol = _field(fields, columns, resolved_names["gene_symbol"])
        hpo_id = _field(fields, columns, resolved_names["hpo_id"])
        hpo_term_name = _field(fields, columns, resolved_names.get("hpo_term_name", ""))
        if not gene_symbol or not hpo_id:
            continue
        yield GenePhenotypeAssociation(gene_symbol=gene_symbol, hpo_id=hpo_id, hpo_term_name=hpo_term_name)


def parse_genes_to_disease(lines: Iterable[str]) -> Iterator[GeneDiseaseAssociation]:
    """Parse genes_to_disease.txt.

    Accepts either observed column-naming convention for gene symbol and
    disease id (see `GENE_DISEASE_FIELD_ALIASES`) rather than committing to
    one, since this file's exact header was not confirmed to the same
    standard as genes_to_phenotype.txt's.
    """
    columns: dict[str, int] | None = None
    resolved_names: dict[str, str] | None = None
    for raw in lines:
        line = raw.rstrip("\n")
        if not line:
            continue
        if columns is None:
            columns = _split_header(line)
            resolved_names = _resolve(columns, GENE_DISEASE_FIELD_ALIASES, required=("gene_symbol", "disease_id"))
            if resolved_names is None:
                raise ValueError(
                    f"unrecognised genes_to_disease header: {line!r}; "
                    f"expected one of the field-name variants this project has seen for "
                    f"{list(GENE_DISEASE_FIELD_ALIASES)}"
                )
            continue
        fields = line.split("\t")
        gene_symbol = _field(fields, columns, resolved_names["gene_symbol"])
        disease_id = _field(fields, columns, resolved_names["disease_id"])
        disease_name = _field(fields, columns, resolved_names.get("disease_name", "disease_name"))
        if not gene_symbol or not disease_id:
            continue
        yield GeneDiseaseAssociation(gene_symbol=gene_symbol, disease_id=disease_id, disease_name=disease_name)


def _resolve(columns: dict[str, int], aliases_by_field: dict[str, tuple[str, ...]], *, required: tuple[str, ...]) -> dict[str, str] | None:
    """Map each canonical field name to whichever of its known aliases is present.

    Optional fields (not in ``required``) that match nothing are simply
    absent from the result rather than failing the whole header -- a file
    missing an optional column, such as a term's plain-language name, should
    still parse the fields it does have.
    """
    resolved: dict[str, str] = {}
    for canonical, aliases in aliases_by_field.items():
        found = next((alias for alias in aliases if alias in columns), None)
        if found is not None:
            resolved[canonical] = found
        elif canonical in required:
            return None
    return resolved


def _field(fields: list[str], columns: dict[str, int], name: str) -> str:
    index = columns.get(name)
    if index is None or index >= len(fields):
        return ""
    return fields[index].strip()


def gene_phenotype_edges(associations: Iterable[GenePhenotypeAssociation]) -> Iterator[ConceptEdge]:
    """Turn gene-phenotype associations into graph edges.

    Weight is deliberately uniform rather than invented: HPO's own
    gene-phenotype files carry no per-association strength the way
    phenotype.hpoa's frequency column does, and assigning one here would be
    fabricating precision the source data does not have. A caller wanting
    graded strength should compute it from something real -- how many
    diseases link this gene to this phenotype, for instance -- not from a
    number this function invented.
    """
    for association in associations:
        if not association.hpo_term_name:
            continue
        yield ConceptEdge(
            source=association.gene_symbol,
            relation=RELATION_ASSOCIATED_GENE,
            target=association.hpo_term_name,
            weight=1.0,
            provenance=f"hpo_gene_annotation:{association.hpo_id}",
        )


def gene_disease_edges(associations: Iterable[GeneDiseaseAssociation]) -> Iterator[ConceptEdge]:
    """Turn gene-disease associations into graph edges, on the same uniform-weight basis."""
    for association in associations:
        if not association.disease_name:
            continue
        yield ConceptEdge(
            source=association.gene_symbol,
            relation=RELATION_CAUSES_DISEASE,
            target=association.disease_name,
            weight=1.0,
            provenance=f"hpo_gene_annotation:{association.disease_id}",
        )
