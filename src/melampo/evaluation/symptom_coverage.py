"""Measure how well each symptom source covers a reference list of common diseases.

The decision this feeds (`decisions_pending: knowledge-all-diseases` in
melampo-assets.yaml) is which source(s) should give the concept graph its
symptom links for diseases HPO does not annotate. Every candidate describes
itself as broad; this module counts instead, over one fixed list of common
diseases (`data/common_diseases_reference.tsv`), per source:

- how many reference diseases have at least one link, and at least
  ``usable_threshold`` distinct findings (one symptom is not a profile);
- how many of those findings could be mapped onto HPO, the graph's own
  phenotype vocabulary -- a link that cannot be mapped cannot enter the graph;
- how many links sit only on narrower diseases (Mondo descendants), which is
  a different and weaker kind of coverage, so it is reported separately;
- the evidence tiers the links come from.

Coverage is not correctness. A source can cover every disease with wrong or
trivial findings; precision needs a clinician-reviewed sample, which this
report is designed to make cheap to draw (every link is written out with its
source, tier and reference), not to replace.
"""

import csv
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..memory.symptom_sources import DiseaseIndex, SymptomLink

DEFAULT_USABLE_THRESHOLD = 5


@dataclass(frozen=True)
class ReferenceDisease:
    mondo_id: str
    mondo_label: str
    query_name: str
    icd10: str
    specialty: str
    resolution: str


def load_reference(path: str | Path) -> list[ReferenceDisease]:
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    return [
        ReferenceDisease(
            mondo_id=row["mondo_id"],
            mondo_label=row["mondo_label"],
            query_name=row["query_name"],
            icd10=row["icd10"],
            specialty=row["specialty"],
            resolution=row["resolution"],
        )
        for row in csv.DictReader(lines, delimiter="\t")
    ]


def _symptom_key(link: SymptomLink) -> str:
    return link.hpo_id or link.symptom_id or link.symptom_label.lower()


@dataclass
class DiseaseCoverage:
    present: int = 0
    excluded: int = 0
    mapped_to_hpo: int = 0
    via_descendants: int = 0
    tiers: Counter = field(default_factory=Counter)


@dataclass
class SourceSummary:
    source: str
    diseases_with_links: int
    diseases_usable: int
    diseases_only_via_descendants: int
    total_present: int
    total_excluded: int
    hpo_mapping_rate: float
    median_present_when_covered: float
    tiers: dict[int, int]

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__ | {"tiers": dict(self.tiers)}


def measure(
    reference: Sequence[ReferenceDisease],
    links: Iterable[SymptomLink],
    index: DiseaseIndex,
    *,
    sources: Sequence[str],
    usable_threshold: int = DEFAULT_USABLE_THRESHOLD,
) -> dict[str, Any]:
    """Per-disease and per-source coverage of ``reference`` by ``links``."""
    by_source_disease: dict[str, dict[str, list[SymptomLink]]] = defaultdict(lambda: defaultdict(list))
    for link in links:
        if link.mondo_id:
            by_source_disease[link.source][link.mondo_id].append(link)

    per_disease: dict[str, dict[str, DiseaseCoverage]] = {}
    union_hpo: dict[str, set[str]] = {}
    for disease in reference:
        descendants = index.descendants(disease.mondo_id)
        per_disease[disease.mondo_id] = {}
        union_hpo[disease.mondo_id] = set()
        for source in sources:
            own = by_source_disease[source].get(disease.mondo_id, [])
            present = {_symptom_key(link): link for link in own if not link.is_exclusion}
            excluded = {_symptom_key(link) for link in own if link.is_exclusion}
            narrower = {
                _symptom_key(link)
                for child in descendants
                for link in by_source_disease[source].get(child, [])
                if not link.is_exclusion
            }
            coverage = DiseaseCoverage(
                present=len(present),
                excluded=len(excluded),
                mapped_to_hpo=sum(1 for link in present.values() if link.hpo_id),
                via_descendants=len(narrower - set(present)),
                tiers=Counter(link.tier for link in present.values()),
            )
            per_disease[disease.mondo_id][source] = coverage
            union_hpo[disease.mondo_id].update(link.hpo_id for link in present.values() if link.hpo_id)

    summaries = [_summarise(source, per_disease, usable_threshold) for source in sources]
    union_covered = sum(1 for hpo_ids in union_hpo.values() if hpo_ids)
    union_usable = sum(1 for hpo_ids in union_hpo.values() if len(hpo_ids) >= usable_threshold)

    specialties: dict[str, dict[str, int]] = defaultdict(lambda: {"diseases": 0, "usable_union_hpo": 0})
    for disease in reference:
        specialties[disease.specialty]["diseases"] += 1
        if len(union_hpo[disease.mondo_id]) >= usable_threshold:
            specialties[disease.specialty]["usable_union_hpo"] += 1

    uncovered = [
        disease.mondo_id
        for disease in reference
        if all(per_disease[disease.mondo_id][source].present == 0 for source in sources)
    ]
    return {
        "reference_size": len(reference),
        "usable_threshold": usable_threshold,
        "sources": [summary.as_dict() for summary in summaries],
        "union_hpo": {"covered": union_covered, "usable": union_usable},
        "specialties": dict(specialties),
        "uncovered_everywhere": uncovered,
        "per_disease": {
            mondo_id: {
                source: {
                    "present": cov.present,
                    "excluded": cov.excluded,
                    "mapped_to_hpo": cov.mapped_to_hpo,
                    "via_descendants": cov.via_descendants,
                    "tiers": dict(cov.tiers),
                }
                for source, cov in by_source.items()
            }
            | {"union_hpo": len(union_hpo[mondo_id])}
            for mondo_id, by_source in per_disease.items()
        },
    }


def _summarise(source: str, per_disease: Mapping[str, Mapping[str, DiseaseCoverage]], usable: int) -> SourceSummary:
    rows = [by_source[source] for by_source in per_disease.values()]
    covered = [row for row in rows if row.present > 0]
    total_present = sum(row.present for row in rows)
    tiers: Counter = Counter()
    for row in rows:
        tiers.update(row.tiers)
    return SourceSummary(
        source=source,
        diseases_with_links=len(covered),
        diseases_usable=sum(1 for row in rows if row.present >= usable),
        diseases_only_via_descendants=sum(1 for row in rows if row.present == 0 and row.via_descendants > 0),
        total_present=total_present,
        total_excluded=sum(row.excluded for row in rows),
        hpo_mapping_rate=round(sum(row.mapped_to_hpo for row in rows) / total_present, 3) if total_present else 0.0,
        median_present_when_covered=statistics.median([row.present for row in covered]) if covered else 0.0,
        tiers=dict(sorted(tiers.items())),
    )


def render_markdown(
    result: Mapping[str, Any],
    reference: Sequence[ReferenceDisease],
    *,
    notes: Sequence[str] = (),
) -> str:
    """The step summary a reviewer reads first; the JSON and JSONL carry the rest."""
    size = result["reference_size"]
    threshold = result["usable_threshold"]
    lines = [
        "### Symptom-source coverage of common diseases",
        "",
        (
            f"Reference list: {size} common diseases (Mondo-resolved). "
            f"'Usable' = at least {threshold} distinct findings on the disease itself."
        ),
        "",
        "| Source | With ≥1 link | Usable | Only via narrower diseases | Links | Exclusions | Mapped to HPO | Median links | Tiers |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for summary in result["sources"]:
        tiers = ", ".join(f"T{tier}: {count}" for tier, count in summary["tiers"].items()) or "–"
        lines.append(
            f"| {summary['source']} | {summary['diseases_with_links']}/{size} | {summary['diseases_usable']}/{size} "
            f"| {summary['diseases_only_via_descendants']} | {summary['total_present']} | {summary['total_excluded']} "
            f"| {summary['hpo_mapping_rate']:.0%} | {summary['median_present_when_covered']} | {tiers} |"
        )
    union = result["union_hpo"]
    lines += [
        "",
        (
            f"**All sources together, counting only HPO-mapped findings:** {union['covered']}/{size} diseases "
            f"with at least one, {union['usable']}/{size} usable."
        ),
        "",
        "| Specialty | Diseases | Usable (all sources, HPO-mapped) |",
        "|---|---|---|",
    ]
    for specialty, counts in sorted(result["specialties"].items()):
        lines.append(f"| {specialty} | {counts['diseases']} | {counts['usable_union_hpo']} |")
    label_for = {disease.mondo_id: disease.mondo_label for disease in reference}
    uncovered = result["uncovered_everywhere"]
    lines += ["", f"**No link from any source ({len(uncovered)}):** " + (", ".join(label_for.get(m, m) for m in uncovered) or "none")]
    if notes:
        lines += ["", "**Run notes**", "", *[f"- {note}" for note in notes]]
    return "\n".join(lines) + "\n"
