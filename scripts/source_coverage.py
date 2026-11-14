"""How much of the findings and of the diagnoses each medical source covers (14 November 2026): the count that
chooses the sources of the concept of what is seen, before anything is built on them. No language model is asked.

Three counts, from what is already labelled and from a list written before any source was looked at:

1. **The labelled findings** (``data/gold/hpo_*_2026-10-31``): for every HPO term a label gives, whether the same
   concept is in SNOMED CT, NCIt and MeSH (through its UMLS concept) and in RadLex (BioPortal, exact name or
   exact synonym).
2. **The concepts HPO lacks**: the labels whose note says HPO has no term for the finding ("no renal mass term");
   whether another source has it (UMLS exact name; RadLex exact name).
3. **The common diagnoses of imaging** (``DIAGNOSES``, written before the count): whether each is a concept of
   MONDO, SNOMED CT, NCIt, ICD-10-CM and MeSH, and how many phenotypes the HPO annotations give it (the graph
   Melampo reasons on today: disease -> phenotype, mostly rare diseases).

Licences: SNOMED CT through the UMLS licence, for research (the owner's request); its content is never written:
the results hold only which sources have a concept, never a SNOMED CT description or code, and the open sources'
identifiers. Keys: UMLS_API_KEY, BIOPORTAL_API_KEY (environment).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = [ROOT / "data/gold/hpo_silver_2026-10-31/labels.txt", ROOT / "data/gold/hpo_heldout_2026-10-31/labels.txt"]
UMLS = "https://uts-ws.nlm.nih.gov/rest"
BIOPORTAL = "https://data.bioontology.org"
UMLS_SOURCES = {"SNOMEDCT_US": "SNOMED CT", "NCI": "NCIt", "MSH": "MeSH", "ICD10CM": "ICD-10-CM", "HPO": "HPO"}

# Written before the count (14 November 2026): diagnoses a CT, MRI or ultrasound report commonly concludes with
DIAGNOSES = (
    "pneumonia", "pulmonary embolism", "pneumothorax", "pleural effusion", "atelectasis", "emphysema",
    "pulmonary fibrosis", "lung cancer", "pulmonary nodule", "bronchiectasis", "pulmonary edema",
    "acute appendicitis", "acute cholecystitis", "cholelithiasis", "choledocholithiasis", "acute pancreatitis",
    "diverticulitis", "small bowel obstruction", "colorectal cancer", "hepatic steatosis", "cirrhosis",
    "hepatocellular carcinoma", "liver metastasis", "hepatic hemangioma", "renal cyst", "nephrolithiasis",
    "hydronephrosis", "renal cell carcinoma", "pyelonephritis", "benign prostatic hyperplasia",
    "abdominal aortic aneurysm", "aortic dissection", "deep vein thrombosis", "atherosclerosis",
    "ischemic stroke", "intracerebral hemorrhage", "subdural hematoma", "multiple sclerosis", "brain metastasis",
    "glioblastoma", "herniated disc", "spinal stenosis", "osteoarthritis", "vertebral fracture", "osteoporosis",
    "breast cancer", "thyroid nodule", "lymphoma", "ovarian cyst", "uterine leiomyoma",
)


class Client:
    def __init__(self, umls_key: str, bioportal_key: str, pause: float = 0.06):
        self.umls_key, self.bioportal_key, self.pause = umls_key, bioportal_key, pause
        self.calls = 0

    def _get(self, url: str) -> dict | None:
        time.sleep(self.pause)
        self.calls += 1
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    return None
                if attempt == 2:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
            time.sleep(2 * (attempt + 1))
        return None

    def cuis(self, string: str, *, source_ui: str = "") -> list[str]:
        """The UMLS concepts with this exact name, or the concept of this code of a source."""
        q = {"string": source_ui or string, "searchType": "exact", "apiKey": self.umls_key, "pageSize": 10}
        if source_ui:
            q.update(inputType="sourceUi", sabs="HPO")
        got = self._get(f"{UMLS}/search/current?{urllib.parse.urlencode(q)}") or {}
        return [r["ui"] for r in got.get("result", {}).get("results", []) if r.get("ui") not in (None, "NONE")]

    def sources_of(self, cui: str) -> set[str]:
        q = {"apiKey": self.umls_key, "sabs": ",".join(UMLS_SOURCES), "pageSize": 200}
        got = self._get(f"{UMLS}/content/current/CUI/{cui}/atoms?{urllib.parse.urlencode(q)}") or {}
        return {a.get("rootSource", "") for a in got.get("result", []) if isinstance(a, dict)}

    def radlex(self, name: str) -> bool:
        if not self.bioportal_key:
            return False
        q = {"q": name, "ontologies": "RADLEX", "require_exact_match": "true", "apikey": self.bioportal_key}
        got = self._get(f"{BIOPORTAL}/search?{urllib.parse.urlencode(q)}") or {}
        return bool(got.get("collection"))


def read_labels(paths) -> tuple[list[str], list[str]]:
    """The HPO terms the labels give, and the findings their notes say HPO has no term for."""
    terms, missing = [], []
    for path in paths:
        for line in Path(path).read_text("utf-8").splitlines():
            if not line.strip() or line.startswith("#") or line.count("|") < 2:
                continue
            _, label, note = line.split("|", 2)
            terms += [t for t in label.split(",") if t.startswith("HP:")]
            missing += [m.strip() for m in re.findall(r"no ([a-z][a-z \-]+?) term", note)]
    return sorted(set(terms)), sorted(set(missing))


def read_obo_names(path: Path) -> tuple[dict[str, str], dict[str, set[str]], dict[str, set[str]]]:
    """id -> name, normalised name or exact synonym -> ids, id -> xrefs (MONDO, HPO)."""
    names, by_name, xrefs, current = {}, {}, {}, ""
    for line in path.read_text("utf-8", errors="replace").splitlines():
        if line.startswith("id: "):
            current = line[4:].strip()
        elif line.startswith("name: ") and current:
            names[current] = line[6:].strip()
            by_name.setdefault(_norm(line[6:]), set()).add(current)
        elif line.startswith("synonym: ") and " EXACT" in line and current:
            m = re.match(r'synonym: "(.*?)"', line)
            if m:
                by_name.setdefault(_norm(m.group(1)), set()).add(current)
        elif line.startswith("xref: ") and current:
            xrefs.setdefault(current, set()).add(line[6:].split()[0])
    return names, by_name, xrefs


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def hpoa_counts(path: Path) -> Counter:
    """Disease (OMIM:/ORPHA:/DECIPHER:) -> number of phenotype annotations."""
    counts: Counter = Counter()
    for line in path.read_text("utf-8", errors="replace").splitlines():
        if line.startswith("#") or line.startswith("database_id"):
            continue
        parts = line.split("\t")
        if len(parts) > 3 and parts[3].startswith("HP:"):
            counts[parts[0]] += 1
    return counts


def count(client: Client, hpo: Path, mondo: Path, hpoa: Path, labels=LABELS) -> dict:
    hpo_names, _, _ = read_obo_names(hpo)
    mondo_names, mondo_by_name, mondo_xrefs = read_obo_names(mondo)
    annotations = hpoa_counts(hpoa)
    terms, missing = read_labels(labels)

    findings = []
    for term in terms:
        cuis = client.cuis("", source_ui=term)
        present = set().union(*[client.sources_of(c) for c in cuis]) if cuis else set()
        findings.append({"hpo": term, "name": hpo_names.get(term, ""), "umls": bool(cuis),
                         "in": sorted(UMLS_SOURCES[s] for s in present if s in UMLS_SOURCES and s != "HPO"),
                         "radlex": client.radlex(hpo_names.get(term, ""))})
    gaps = []
    for name in missing:
        cuis = client.cuis(name)
        present = set().union(*[client.sources_of(c) for c in cuis]) if cuis else set()
        gaps.append({"finding": name, "in": sorted(UMLS_SOURCES[s] for s in present if s in UMLS_SOURCES),
                     "radlex": client.radlex(name)})
    diseases = []
    for name in DIAGNOSES:
        cuis = client.cuis(name)
        present = set().union(*[client.sources_of(c) for c in cuis]) if cuis else set()
        mondo_ids = sorted(mondo_by_name.get(_norm(name), set()))
        linked = sorted({x for m in mondo_ids for x in mondo_xrefs.get(m, set())
                         if x.startswith(("OMIM:", "Orphanet:", "ORPHA:"))})
        phenotypes = sum(annotations.get(x.replace("Orphanet:", "ORPHA:"), 0) for x in linked)
        diseases.append({"diagnosis": name, "mondo": mondo_ids, "in": sorted(UMLS_SOURCES[s] for s in present
                                                                          if s in UMLS_SOURCES and s != "HPO"),
                         "hpoa_phenotypes": phenotypes})

    def share(rows, test):
        return f"{sum(1 for r in rows if test(r))}/{len(rows)}"

    summary = {
        "findings_labelled": {
            "terms": len(findings),
            **{src: share(findings, lambda r, s=src: s in r["in"]) for src in ("SNOMED CT", "NCIt", "MeSH")},
            "RadLex": share(findings, lambda r: r["radlex"]),
            "any_other_source": share(findings, lambda r: r["in"] or r["radlex"]),
        },
        "concepts_hpo_lacks": {
            "findings": len(gaps),
            **{src: share(gaps, lambda r, s=src: s in r["in"]) for src in ("SNOMED CT", "NCIt", "MeSH")},
            "RadLex": share(gaps, lambda r: r["radlex"]),
            "any_source": share(gaps, lambda r: r["in"] or r["radlex"]),
        },
        "common_diagnoses": {
            "diagnoses": len(diseases),
            "MONDO": share(diseases, lambda r: r["mondo"]),
            **{src: share(diseases, lambda r, s=src: s in r["in"]) for src in ("SNOMED CT", "NCIt", "ICD-10-CM",
                                                                                "MeSH")},
            "with_hpo_annotations": share(diseases, lambda r: r["hpoa_phenotypes"] > 0),
        },
        "umls_and_bioportal_calls": client.calls,
    }
    return {"summary": summary, "findings": findings, "gaps": gaps, "diagnoses": diseases,
            "note": "no SNOMED CT content is written: only which sources hold a concept"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hpo", required=True)
    ap.add_argument("--mondo", required=True)
    ap.add_argument("--hpoa", required=True)
    ap.add_argument("--out", default="source_coverage.json")
    args = ap.parse_args(argv)
    umls, bioportal = os.environ.get("UMLS_API_KEY", ""), os.environ.get("BIOPORTAL_API_KEY", "")
    if not umls:
        print("UMLS_API_KEY is not set", file=sys.stderr)
        return 1
    result = count(Client(umls, bioportal), Path(args.hpo), Path(args.mondo), Path(args.hpoa))
    if not bioportal:
        result["summary"]["note"] = "BIOPORTAL_API_KEY not set: RadLex not counted"
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
