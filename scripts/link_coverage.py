"""How many disease -> finding links each source gives the common diagnoses of imaging (14 November 2026): the
second count, before the diagnostic graph is enriched. No language model is asked.

The first count (source_coverage.py) found the concepts of 49 of the 50 diagnoses in SNOMED CT, and phenotypes
for 6 of them in the HPO annotations, the only links of the graph of today. This one counts, for the same
diagnoses (source_coverage.DIAGNOSES), the links each candidate source gives:

- **UMLS relations** of the diagnosis's concept, by source and label: SNOMED CT's defining relationships (finding
  site, associated morphology), NCIt's (disease has finding, may have finding, anatomic site), MED-RT, and any
  other "manifestation" relation. Counts only: no SNOMED CT content is written.
- **Disease Ontology**: ``has_symptom`` (doid.obo).
- **Monarch**: disease-to-phenotype associations of the MONDO concept (Monarch API v3).
- **StatPearls** (NCBI Bookshelf, CC BY 4.0): is there a chapter for the diagnosis, and does it describe imaging
  (its "Evaluation" with CT, MRI, ultrasound or radiography). The chapter's text is not kept.

SemMedDB (relations read from PubMed, under the UMLS licence) needs its own download: not counted here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("source_coverage", HERE / "source_coverage.py")
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)

UMLS = sc.UMLS
MONARCH = "https://api-v3.monarchinitiative.org/v3/api/association"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BOOKS = "https://www.ncbi.nlm.nih.gov/books"
# relations that link a disease to what is found in it (either direction's name)
FINDING_LABELS = re.compile(r"finding|manifestation|morphology|symptom|sign|anatomic_site|finding_site|"
                            r"associated_with|pathological_process|clinical_course", re.I)
IMAGING = re.compile(r"\b(CT|computed tomography|MRI|magnetic resonance|ultrasound|ultrasonograph\w*|"
                     r"radiograph\w*|x-ray|sonograph\w*|imaging)\b", re.I)


class Fetch:
    def __init__(self, pause: float = 0.12):
        self.pause, self.calls = pause, 0

    def json(self, url: str) -> dict | None:
        text = self.text(url)
        return json.loads(text) if text else None

    def text(self, url: str) -> str:
        time.sleep(self.pause)
        self.calls += 1
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "melampo-link-coverage/1.0"})
                with urllib.request.urlopen(req, timeout=40) as response:
                    return response.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    return ""
                if attempt == 2:
                    return ""
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    return ""
            time.sleep(2 * (attempt + 1))
        return ""


def umls_relations(fetch: Fetch, key: str, cui: str) -> Counter:
    """(source, label) -> count of the concept's relations; the related concepts' names are not kept."""
    counts: Counter = Counter()
    page = 1
    while page <= 20:
        q = {"apiKey": key, "pageSize": 200, "pageNumber": page}
        got = fetch.json(f"{UMLS}/content/current/CUI/{cui}/relations?{urllib.parse.urlencode(q)}") or {}
        rows = got.get("result") or []
        for r in rows:
            label = r.get("additionalRelationLabel") or r.get("relationLabel") or ""
            counts[(r.get("rootSource", ""), label)] += 1
        if len(rows) < 200:
            break
        page += 1
    return counts


def doid_symptoms(path: Path) -> tuple[dict[str, set[str]], dict[str, int]]:
    """name or exact synonym -> DOID ids; DOID id -> number of has_symptom relations."""
    by_name: dict[str, set[str]] = {}
    symptoms: Counter = Counter()
    current = ""
    for line in path.read_text("utf-8", errors="replace").splitlines():
        if line.startswith("id: "):
            current = line[4:].strip()
        elif line.startswith("name: ") and current:
            by_name.setdefault(sc._norm(line[6:]), set()).add(current)
        elif line.startswith("synonym: ") and " EXACT" in line and current:
            m = re.match(r'synonym: "(.*?)"', line)
            if m:
                by_name.setdefault(sc._norm(m.group(1)), set()).add(current)
        elif line.startswith("relationship: has_symptom") and current:
            symptoms[current] += 1
    return by_name, symptoms


def monarch_phenotypes(fetch: Fetch, mondo: str) -> int:
    q = {"subject": mondo, "category": "biolink:DiseaseToPhenotypicFeatureAssociation", "limit": 1}
    got = fetch.json(f"{MONARCH}?{urllib.parse.urlencode(q)}") or {}
    return int(got.get("total") or 0)


def statpearls(fetch: Fetch, name: str) -> dict:
    """The StatPearls chapter of the diagnosis, and whether it describes imaging."""
    q = {"db": "books", "term": f'statpearls[book] AND "{name}"[title]', "retmode": "json", "retmax": 3}
    got = fetch.json(f"{EUTILS}/esearch.fcgi?{urllib.parse.urlencode(q)}") or {}
    ids = got.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return {"chapter": "", "imaging": False}
    summary = fetch.json(f"{EUTILS}/esummary.fcgi?db=books&retmode=json&id={ids[0]}") or {}
    doc = summary.get("result", {}).get(ids[0], {})
    nbk = doc.get("chapteraccessionid") or doc.get("accessionid") or ""
    if not nbk:
        return {"chapter": "", "imaging": False}
    page = fetch.text(f"{BOOKS}/{nbk}/")
    evaluation = page.split("Evaluation", 1)[1][:20000] if "Evaluation" in page else ""
    return {"chapter": nbk, "title": doc.get("title", ""), "imaging": bool(IMAGING.search(evaluation)),
            "imaging_mentions": len(IMAGING.findall(evaluation))}


def count(fetch: Fetch, umls_key: str, mondo: Path, doid: Path, diagnoses=None) -> dict:
    _, mondo_by_name, _ = sc.read_obo_names(mondo)
    doid_by_name, doid_symptom_counts = doid_symptoms(doid)
    labels_total: Counter = Counter()
    rows = []
    for name in diagnoses or sc.DIAGNOSES:
        client = sc.Client(umls_key, "")
        cuis = client.cuis(name)
        relations = umls_relations(fetch, umls_key, cuis[0]) if cuis else Counter()
        labels_total.update(relations)
        finding_like = {f"{s}:{label}": n for (s, label), n in relations.items() if FINDING_LABELS.search(label)}
        by_source: Counter = Counter()
        for key, n in finding_like.items():
            by_source[key.split(":", 1)[0]] += n
        mondo_ids = sorted(mondo_by_name.get(sc._norm(name), set()))
        doids = sorted(doid_by_name.get(sc._norm(name), set()))
        rows.append({
            "diagnosis": name,
            "umls_concept": bool(cuis),
            "umls_finding_links": dict(by_source),
            "snomed_defining": sum(n for k, n in finding_like.items() if k.startswith("SNOMEDCT_US:")),
            "nci_finding": sum(n for k, n in finding_like.items() if k.startswith("NCI:")),
            "doid": doids, "doid_symptoms": sum(doid_symptom_counts.get(d, 0) for d in doids),
            "mondo": mondo_ids, "monarch_phenotypes": sum(monarch_phenotypes(fetch, m) for m in mondo_ids),
            "statpearls": statpearls(fetch, name),
        })

    def share(test):
        return f"{sum(1 for r in rows if test(r))}/{len(rows)}"

    summary = {
        "diagnoses": len(rows),
        "with_any_umls_finding_link": share(lambda r: sum(r["umls_finding_links"].values()) > 0),
        "with_snomed_defining_relations": share(lambda r: r["snomed_defining"] > 0),
        "with_nci_finding_relations": share(lambda r: r["nci_finding"] > 0),
        "with_doid_symptoms": share(lambda r: r["doid_symptoms"] > 0),
        "with_monarch_phenotypes": share(lambda r: r["monarch_phenotypes"] > 0),
        "with_statpearls_chapter": share(lambda r: r["statpearls"]["chapter"]),
        "with_statpearls_imaging": share(lambda r: r["statpearls"]["imaging"]),
        "median_monarch_phenotypes": sorted(r["monarch_phenotypes"] for r in rows)[len(rows) // 2] if rows else 0,
        "calls": fetch.calls,
    }
    top = Counter()
    for (source, label), n in labels_total.items():
        if FINDING_LABELS.search(label):
            top[f"{source}:{label}"] += n
    return {"summary": summary, "finding_relation_labels": dict(top.most_common(40)), "diagnoses": rows,
            "note": "counts and open identifiers only; no SNOMED CT content and no chapter text is written"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mondo", required=True)
    ap.add_argument("--doid", required=True)
    ap.add_argument("--out", default="link_coverage.json")
    args = ap.parse_args(argv)
    key = os.environ.get("UMLS_API_KEY", "")
    if not key:
        print("UMLS_API_KEY is not set", file=sys.stderr)
        return 1
    result = count(Fetch(), key, Path(args.mondo), Path(args.doid))
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({"summary": result["summary"], "finding_relation_labels": result["finding_relation_labels"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
