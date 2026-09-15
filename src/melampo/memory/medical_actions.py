"""Parse MAxO annotations: what is done *about* a disease, not how to recognise it.

Added after a direct question -- is MAxO enough to discriminate between
conditions and reach a diagnosis -- and the honest answer, from the shipped
file itself, is no. Two measurements decide it.

**MAxO annotations here carry no diagnostic relation at all.** The 438 rows
break down as 401 TREATS, 34 PREVENTS, 2 NO_OBSERVED_BENEFIT, 1
CONTRAINDICATED. Not one says "this procedure distinguishes A from B". The
file answers "what do you do once you know it is A", which is a different
question from the one a differential asks, and the earlier suggestion that
MAxO could answer `OpenQuestion`'s "which test would settle this" was
wrong.

**Coverage is 1.6%.** 202 diseases out of the 12,880 in `phenotype.hpoa`.
Even for the questions it does answer, it answers them for one disease in
sixty.

**So why parse it at all.** A CONTRAINDICATED or NO_OBSERVED_BENEFIT
annotation is clinical information nothing else in this project has, and it
is curated with a PMID and an ORCID attached. It is worth having *where it
exists*, and the 1.6% figure is exactly why it must never become a premise:
absence of a MAxO annotation says nothing whatsoever about a disease, and
any code treating "no annotation" as "no treatment" or "not contraindicated"
would be reading a gap as a fact. `MedicalActionIndex.coverage_note` exists
to keep that visible to anything that uses this.

The edges this produces are deliberately kept *out* of the diagnostic
concept graph. A treatment relation sitting alongside `has_phenotype` would
let a differential traverse "disease -> treated by -> physical therapy ->
treats -> other disease" and surface two conditions as related because they
share a therapy, which is not a diagnostic connection at all.
"""

import csv
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

RELATION_TREATS = "TREATS"
RELATION_PREVENTS = "PREVENTS"
RELATION_CONTRAINDICATED = "CONTRAINDICATED"
RELATION_NO_OBSERVED_BENEFIT = "NO_OBSERVED_BENEFIT"

# Relations that warn against an action rather than recommend one. Separated
# because they are the asymmetric case: a missing TREATS row is
# uninformative at 1.6% coverage, but a present CONTRAINDICATED row is a
# positive statement worth surfacing every time it applies.
CAUTIONARY_RELATIONS = frozenset({RELATION_CONTRAINDICATED, RELATION_NO_OBSERVED_BENEFIT})


@dataclass(frozen=True)
class MedicalAction:
    """One curated statement about an action taken for a disease."""

    disease_id: str
    disease_name: str
    maxo_id: str
    maxo_name: str
    relation: str
    hpo_id: str = ""
    evidence: str = ""
    source_id: str = ""

    @property
    def is_cautionary(self) -> bool:
        return self.relation in CAUTIONARY_RELATIONS

    @property
    def is_citable(self) -> bool:
        """Whether this row names a source a reviewer could open.

        The same standard `literature_index` applies: a clinical claim with
        no resolvable reference is an assertion, and MAxO rows do usually
        carry a PMID, so the ones that do not are worth telling apart.
        """
        return self.source_id.lower().startswith(("pmid:", "doi:", "pmc"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "disease_id": self.disease_id,
            "disease_name": self.disease_name,
            "action": self.maxo_name,
            "maxo_id": self.maxo_id,
            "relation": self.relation,
            "is_cautionary": self.is_cautionary,
            "source_id": self.source_id,
            "is_citable": self.is_citable,
        }


def parse_maxo_annotations(lines: Iterable[str]) -> Iterator[MedicalAction]:
    """Read maxo-annotations.tsv into MedicalAction records.

    Uses csv.DictReader against the file's own header rather than fixed
    positions, for the reason the gene-annotation parsers already
    established: a release that adds or reorders a column should not
    silently misattribute a field. Rows missing a disease or an action are
    skipped -- there is nothing to attach them to.
    """
    reader = csv.DictReader((line.lstrip("\ufeff") for line in lines), delimiter="\t")
    for row in reader:
        disease_id = (row.get("disease_id") or "").strip()
        maxo_id = (row.get("maxo_id") or "").strip()
        if not disease_id or not maxo_id:
            continue
        yield MedicalAction(
            disease_id=disease_id,
            disease_name=(row.get("disease_name") or "").strip(),
            maxo_id=maxo_id,
            maxo_name=(row.get("maxo_name") or "").strip(),
            relation=(row.get("relation") or "").strip().upper(),
            hpo_id=(row.get("hpo_id") or "").strip(),
            evidence=(row.get("evidence") or "").strip(),
            source_id=(row.get("source_id") or "").strip(),
        )


@dataclass
class MedicalActionIndex:
    """Medical actions looked up by disease, kept apart from the concept graph.

    Deliberately not a `ConceptGraphView`. Treatment relations in the same
    graph as `has_phenotype` would let a differential traverse
    "disease -> treated by -> physical therapy -> treats -> other disease"
    and surface two conditions as related because they share a therapy. That
    is not a diagnostic connection, and the cleanest way to prevent it is to
    give these edges nowhere to be traversed from.
    """

    by_disease_id: dict[str, list[MedicalAction]] = field(default_factory=dict)
    by_disease_name: dict[str, list[MedicalAction]] = field(default_factory=dict)

    @classmethod
    def from_annotations(cls, actions: Iterable[MedicalAction]) -> "MedicalActionIndex":
        index = cls()
        for action in actions:
            index.by_disease_id.setdefault(action.disease_id, []).append(action)
            if action.disease_name:
                index.by_disease_name.setdefault(action.disease_name.strip().lower(), []).append(action)
        return index

    def __len__(self) -> int:
        return sum(len(items) for items in self.by_disease_id.values())

    @property
    def disease_count(self) -> int:
        return len(self.by_disease_id)

    def for_disease(self, disease: str) -> list[MedicalAction]:
        """Actions for a disease, by id or by name. Empty means unannotated, not untreatable."""
        return self.by_disease_id.get(disease) or self.by_disease_name.get(disease.strip().lower()) or []

    def cautions_for(self, disease: str) -> list[MedicalAction]:
        """Only the contraindicated / no-observed-benefit rows for a disease.

        Separated from `for_disease` because these are the rows worth
        surfacing unprompted: a treatment suggestion is one of many possible,
        while a documented contraindication is a specific warning that
        applies whenever that disease is under consideration.
        """
        return [action for action in self.for_disease(disease) if action.is_cautionary]

    def coverage_note(self, total_diseases: int | None = None) -> str:
        """A plain statement of how little this index covers.

        Returned as text meant to travel with any output derived from this
        index. At 1.6% coverage, the absence of an annotation is
        uninformative, and a reader who does not know the coverage figure
        cannot tell "nothing is recommended" from "nobody has annotated
        this".
        """
        base = f"{len(self)} curated actions across {self.disease_count} diseases"
        if total_diseases:
            base += f" ({self.disease_count / total_diseases:.1%} of {total_diseases:,} known diseases)"
        return base + "; absence of an annotation carries no clinical meaning"
