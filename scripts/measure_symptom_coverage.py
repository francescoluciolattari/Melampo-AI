"""Measure symptom-source coverage of the common-disease reference list.

Run by .github/workflows/symptom-coverage.yml, where the network and
UMLS_API_KEY are available; runnable locally the same way:

    uv run dvc update data/mondo.obo.dvc data/doid.owl.dvc data/symp.obo.dvc \
        data/hp.obo.dvc data/phenotype.hpoa.dvc
    UMLS_API_KEY=... uv run python scripts/measure_symptom_coverage.py --out-dir coverage

Offline sources (HPO, Disease Ontology) are always measured. NCIt (through
UMLS) needs UMLS_API_KEY; Wikidata needs network access. A source that could
not be reached is reported as "not measured" with the reason -- never as
zero coverage, which would read as a finding about the source.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from melampo.connectors.umls import UmlsConfig, UmlsConnector
from melampo.connectors.wikidata import KEY_DOID, KEY_MONDO, WikidataConnector
from melampo.evaluation.symptom_coverage import load_reference, measure, render_markdown
from melampo.memory.concept_resolution import TermIndex, parse_obo
from melampo.memory.ontology_import import parse_hpoa
from melampo.memory.symptom_sources import (
    NCIT_FINDING_LABELS,
    SOURCE_DO,
    SOURCE_DO_TEXT,
    SOURCE_HPO,
    SOURCE_NCIT,
    SOURCE_WIKIDATA,
    DiseaseIndex,
    assign_mondo,
    hpo_xref_map,
    infer_orientation,
    label_counts,
    links_from_doid,
    links_from_hpoa,
    links_from_ncit_relations,
    links_from_wikidata,
    normalise_to_hpo,
    parse_doid_owl,
    parse_mondo,
)

UMLS_SECONDS_BETWEEN_CALLS = 0.1  # UTS allows 20 requests/second per IP; this stays at half.


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def measure_ncit(reference, index, notes):
    config = UmlsConfig.from_env()
    connector = UmlsConnector(config=config)
    if not connector.availability().available:
        notes.append("NCIt not measured: UMLS_API_KEY not configured.")
        return None, {}
    raw: list[tuple[str, str, str, list[dict]]] = []
    errors: list[str] = []
    for disease in reference:
        codes = [xref.split(":", 1)[1] for xref in index.xrefs_of(disease.mondo_id, "NCIT")]
        if not codes:
            for cui_xref in index.xrefs_of(disease.mondo_id, "UMLS"):
                try:
                    codes.extend(connector.source_codes_for_cui(cui_xref.split(":", 1)[1], "NCI"))
                except Exception as error:  # noqa: BLE001 - recorded, reported
                    errors.append(f"{disease.mondo_id} atoms {cui_xref}: {error}")
                time.sleep(UMLS_SECONDS_BETWEEN_CALLS)
        for code in dict.fromkeys(codes):
            try:
                rows = connector.source_relations("NCI", code, additional_labels=NCIT_FINDING_LABELS)
            except Exception as error:  # noqa: BLE001
                errors.append(f"{disease.mondo_id} NCI:{code}: {error}")
                continue
            finally:
                time.sleep(UMLS_SECONDS_BETWEEN_CALLS)
            raw.append((disease.mondo_id, disease.mondo_label, code, rows))
    counts = label_counts(row for *_, rows in raw for row in rows)
    orientation = infer_orientation(counts)
    links = [
        link
        for mondo_id, label, code, rows in raw
        for link in links_from_ncit_relations(rows, queried_code=code, disease_label=label, orientation=orientation, mondo_id=mondo_id)
    ]
    queried = len({mondo_id for mondo_id, *_ in raw})
    notes.append(
        f"NCIt: {len(raw)} NCI codes queried for {queried} reference diseases; "
        f"label counts {dict(counts)}; orientation inferred: {orientation}."
    )
    if errors:
        notes.append(f"NCIt: {len(errors)} call(s) failed and are excluded, first: {errors[0]}")
    return links, {"label_counts": dict(counts), "orientation": orientation, "errors": errors}


def measure_wikidata(reference, index, notes):
    connector = WikidataConnector()
    mondo_ids = [disease.mondo_id for disease in reference]
    mondo_for_doid = {
        xref: disease.mondo_id for disease in reference for xref in index.xrefs_of(disease.mondo_id, "DOID")
    }
    try:
        rows = connector.symptom_bindings(KEY_MONDO, mondo_ids)
        rows += connector.symptom_bindings(KEY_DOID, list(mondo_for_doid))
    except Exception as error:  # noqa: BLE001
        notes.append(f"Wikidata not measured: {error}")
        return None
    links = list(links_from_wikidata(rows, mondo_for_doid=mondo_for_doid))
    unique = {(link.mondo_id, link.symptom_id): link for link in links}
    notes.append(f"Wikidata: {len(rows)} result rows, {len(unique)} distinct disease-symptom statements.")
    return list(unique.values())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--reference", default="data/common_diseases_reference.tsv")
    parser.add_argument("--out-dir", default="symptom-coverage")
    parser.add_argument("--skip-ncit", action="store_true")
    parser.add_argument("--skip-wikidata", action="store_true")
    args = parser.parse_args(argv)

    data = Path(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    index = DiseaseIndex.from_mondo(parse_mondo(_read_lines(data / "mondo.obo")))
    reference = load_reference(args.reference)
    missing = [disease.mondo_id for disease in reference if disease.mondo_id not in index.diseases]
    if missing:
        notes.append(f"Reference ids absent or obsolete in this Mondo release: {missing}")
        reference = [disease for disease in reference if disease.mondo_id in index.diseases]

    hpo_terms = list(parse_obo(_read_lines(data / "hp.obo")))
    term_index = TermIndex.from_terms(hpo_terms)
    ncit_to_hpo = hpo_xref_map(hpo_terms, "NCIT")

    links = assign_mondo(links_from_hpoa(parse_hpoa(_read_lines(data / "phenotype.hpoa"))), index)
    symptom_labels = {term.term_id: term.name for term in parse_obo(_read_lines(data / "symp.obo"))}
    links += assign_mondo(links_from_doid(parse_doid_owl(data / "doid.owl"), symptom_labels), index)
    sources = [SOURCE_HPO, SOURCE_DO, SOURCE_DO_TEXT]

    extra: dict = {}
    if args.skip_ncit:
        notes.append("NCIt skipped (--skip-ncit).")
    else:
        ncit_links, extra["ncit"] = measure_ncit(reference, index, notes)
        if ncit_links is not None:
            links += ncit_links
            sources.append(SOURCE_NCIT)
    if args.skip_wikidata:
        notes.append("Wikidata skipped (--skip-wikidata).")
    else:
        wikidata_links = measure_wikidata(reference, index, notes)
        if wikidata_links is not None:
            links += wikidata_links
            sources.append(SOURCE_WIKIDATA)

    links = normalise_to_hpo(links, term_index=term_index, xref_map=ncit_to_hpo)
    result = measure(reference, links, index, sources=sources)
    result["run"] = extra | {"sources_measured": sources}

    relevant = {disease.mondo_id for disease in reference}
    for disease in reference:
        relevant |= index.descendants(disease.mondo_id)
    with (out / "links.jsonl").open("w", encoding="utf-8") as handle:
        for link in links:
            if link.mondo_id in relevant:
                handle.write(json.dumps(link.as_dict(), ensure_ascii=False) + "\n")
    (out / "coverage.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = render_markdown(result, reference, notes=notes)
    (out / "coverage.md").write_text(markdown, encoding="utf-8")
    sys.stdout.write(markdown)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
