"""The open sources of the drug branch (15 November 2026): Italian names to active ingredients, ingredients to
their US label, each with its licence.

The chain, every link with its source:

1. **AIFA** (Agenzia Italiana del Farmaco, "Liste dei farmaci", CSV; open data CC BY 4.0 per aifa.gov.it/open-data):
   the commercial name or the AIC code -> the active ingredient as AIFA writes it (Italian) and the ATC code.
   The page gives no column names, so the reader recognises them by their words (``AIFA_COLUMNS``) and says which
   it found; a file whose columns it cannot recognise is reported, never guessed.
2. **ATC -> international name in English**: Wikidata (P267, CC0) and, for the centrally authorised medicines,
   the EMA medicines table (``medicines-output-medicines-report_en.xlsx``). A single-substance ATC code (level
   5) names one substance; where the two sources disagree, both are kept and the ingredient is marked in doubt.
3. **openFDA drug labels** (CC0, open.fda.gov/license): the label of the ingredient, its sections
   ``boxed_warning``, ``contraindications``, ``warnings_and_cautions`` (or ``warnings``), ``drug_interactions``,
   ``adverse_reactions``, ``dosage_and_administration``, with ``set_id``, ``version`` and ``effective_time``. The
   label is a US label: what it says is cited as such.

Nothing of a patient goes through here: the cache is built from the public lists for every ingredient they carry
(``scripts/drug_sources.py``, GitHub workflow ``drug-sources``), and a case reads only the cache
(``clinical.drug_branch``). DDInter (CC BY-NC-SA) and StatPearls (CC BY-NC-ND) are not used here: non-commercial.
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

AIFA_LISTS = {
    "anagrafica": "https://drive.aifa.gov.it/farmaci/confezioni_fornitura.csv",
    "principi_attivi": "https://drive.aifa.gov.it/farmaci/PA_confezioni.csv",
    "atc": "https://drive.aifa.gov.it/farmaci/atc.csv",
}
AIFA_LICENCE = "CC BY 4.0 (AIFA open data)"
EMA_MEDICINES = "https://www.ema.europa.eu/en/documents/report/medicines-output-medicines-report_en.xlsx"
EMA_LICENCE = "EMA website content, reproduction authorised with acknowledgement of the source (to be confirmed)"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
WIKIDATA_LICENCE = "CC0"
OPENFDA_LABEL = "https://api.fda.gov/drug/label.json"
OPENFDA_LICENCE = "CC0 (openFDA)"
LABEL_SECTIONS = ("boxed_warning", "contraindications", "warnings_and_cautions", "warnings", "drug_interactions",
                  "adverse_reactions", "dosage_and_administration")
USER_AGENT = "Melampo-drug-sources/1.0 (https://github.com/francescoluciolattari/Melampo-AI)"

# the words a column name carries, for each field; the first column that holds them is taken
AIFA_COLUMNS = {
    "aic": ("codice_aic", "aic"),
    "name": ("denominazione", "nome_commerciale", "farmaco", "medicinale", "nome"),
    "package": ("confezione", "descrizione_confezione"),
    "ingredient": ("principio_attivo", "principi_attivi", "pa_associati", "principio", "sostanza",
                   "descrizione_pa"),
    "atc": ("codice_atc", "atc"),
    "holder": ("ragione_sociale", "titolare", "azienda", "ditta"),
    "status": ("stato_amministrativo", "stato"),
    "form": ("forma_farmaceutica", "forma"),
}


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _header_key(name: str) -> str:
    return "_".join(norm(name).split())


def read_csv(text: str) -> tuple[list[str], list[list[str]]]:
    """Header and rows of a CSV whose separator is ';' or ',' (as the first line shows)."""
    text = text.lstrip("﻿")
    first = text.split("\n", 1)[0]
    sep = ";" if first.count(";") >= first.count(",") else ","
    rows = list(csv.reader(io.StringIO(text), delimiter=sep))
    return (rows[0], rows[1:]) if rows else ([], [])


def decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


@dataclass
class AifaProduct:
    aic: str = ""
    name: str = ""
    package: str = ""
    ingredient: str = ""
    atc: str = ""
    holder: str = ""
    status: str = ""
    form: str = ""


def map_columns(header: list[str]) -> dict[str, int]:
    """Field -> column index, by the words of the column names."""
    keys = [_header_key(h) for h in header]
    found: dict[str, int] = {}
    for field_name, words in AIFA_COLUMNS.items():
        for word in words:
            index = next((i for i, k in enumerate(keys) if k == word), None)
            if index is None:
                index = next((i for i, k in enumerate(keys) if word in k and i not in found.values()), None)
            if index is not None:
                found[field_name] = index
                break
    return found


def _samples(rows: list[list[str]], columns: dict[str, int], header: list[str]) -> dict[str, list[str]]:
    """The first three distinct values of each recognised column: what the column really holds, for the report."""
    out: dict[str, list[str]] = {}
    for name, i in columns.items():
        seen: list[str] = []
        for row in rows:
            value = row[i].strip() if i < len(row) else ""
            if value and value not in seen:
                seen.append(value)
            if len(seen) == 3:
                break
        out[f"{name} ({header[i]})"] = seen
    return out


def _holds_names(rows: list[list[str]], i: int, n: int = 200) -> bool:
    """Most of the first ``n`` filled cells of the column are words of four letters or more."""
    values = [row[i].strip() for row in rows[: n * 3] if i < len(row) and row[i].strip()][:n]
    named = sum(1 for v in values if re.search(r"[A-Za-z]{4,}", v))
    return bool(values) and named >= 0.8 * len(values)


def read_aifa(text: str, ingredients: dict[str, str] | None = None) -> tuple[list[AifaProduct], dict[str, Any]]:
    """The products of an AIFA list, and what the reader recognised (the header, the columns it mapped, three values
    of each). Without an ingredient column the ingredients come from ``ingredients`` (``read_aifa_ingredients``:
    AIC code or package code -> ingredients), when given (15 November 2026: the "Anagrafica farmaci" has none)."""
    header, rows = read_csv(text)
    columns = map_columns(header)
    samples = _samples(rows, columns, header)
    if "ingredient" in columns and not _holds_names(rows, columns["ingredient"]):
        # a column called like the ingredients that holds flags or counts ("S", "2"): not taken as names
        samples["ingredient_not_names"] = samples.pop(f"ingredient ({header[columns.pop('ingredient')]})")
    report = {"header": header, "columns": {k: header[i] for k, i in columns.items()}, "rows": len(rows),
              "samples": samples}
    if not ({"name", "aic"} & set(columns)) or ("ingredient" not in columns and not ingredients):
        report["problem"] = "no ingredient column (and no ingredients file), or neither a name nor an AIC column"
        return [], report
    products, joined = [], 0
    for row in rows:
        values = {k: (row[i].strip() if i < len(row) else "") for k, i in columns.items()}
        if not values.get("ingredient") and ingredients:
            values["ingredient"] = ingredients.get(values.get("aic", "").zfill(9)) or ingredients.get(
                values.get("package", ""), "")
            joined += bool(values["ingredient"])
        if values.get("ingredient"):
            products.append(AifaProduct(**values))
    if ingredients is not None:
        report["joined_from_ingredients_file"] = joined
    report["products"] = len(products)
    return products, report


def read_aifa_ingredients(text: str) -> tuple[dict[str, str], dict[str, Any]]:
    """"Principi attivi per confezione": AIC code (nine digits) and package code -> the ingredients of the package,
    joined by "/" in the order of the file."""
    header, rows = read_csv(text)
    columns = map_columns(header)
    report = {"header": header, "columns": {k: header[i] for k, i in columns.items()}, "rows": len(rows),
              "samples": _samples(rows, columns, header)}
    if "ingredient" not in columns or not ({"aic", "package"} & set(columns)):
        report["problem"] = "no ingredient column, or neither an AIC nor a package column"
        return {}, report
    parts: dict[str, list[str]] = {}
    def cell(row: list[str], name: str) -> str:
        i = columns.get(name)
        return row[i].strip() if i is not None and i < len(row) else ""

    for row in rows:
        ingredient = cell(row, "ingredient")
        if not ingredient:
            continue
        for key in (cell(row, "aic").zfill(9) if cell(row, "aic") else "", cell(row, "package")):
            if key and ingredient not in parts.setdefault(key, []):
                parts[key].append(ingredient)
    report["keys"] = len(parts)
    return {k: "/".join(v) for k, v in parts.items()}, report


def wikidata_atc_query() -> str:
    """Every Wikidata item with an ATC code (P267) and its English label."""
    return ("SELECT ?item ?atc ?label WHERE { ?item wdt:P267 ?atc . "
            "?item rdfs:label ?label . FILTER(LANG(?label) = 'en') }")


def read_wikidata_atc(payload: dict[str, Any]) -> dict[str, list[str]]:
    """ATC code -> English names (an ATC code may be carried by more than one item)."""
    out: dict[str, list[str]] = {}
    for row in payload.get("results", {}).get("bindings", []):
        atc = row.get("atc", {}).get("value", "").strip().upper()
        label = row.get("label", {}).get("value", "").strip()
        if atc and label and label not in out.setdefault(atc, []):
            out[atc].append(label)
    return out


def read_ema_medicines(path: Path) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """ATC code -> INN / common names from the EMA medicines table (human medicines)."""
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    header_row, header = None, []
    for index, row in enumerate(ws.iter_rows(max_row=40, values_only=True)):
        cells = [str(c or "") for c in row]
        keys = [_header_key(c) for c in cells]
        if any("atc" in k for k in keys) and any("inn" in k or "active_substance" in k for k in keys):
            header_row, header = index, cells
            break
    report: dict[str, Any] = {"header": header}
    if header_row is None:
        report["problem"] = "no header with an ATC column and an INN / active substance column"
        return {}, report
    keys = [_header_key(c) for c in header]
    atc_i = next(i for i, k in enumerate(keys) if "atc" in k)
    inn_i = next((i for i, k in enumerate(keys) if "inn" in k), None)
    if inn_i is None:
        inn_i = next(i for i, k in enumerate(keys) if "active_substance" in k)
    cat_i = next((i for i, k in enumerate(keys) if k == "category"), None)
    report["columns"] = {"atc": header[atc_i], "inn": header[inn_i]}
    out: dict[str, list[str]] = {}
    for row in ws.iter_rows(min_row=header_row + 2, values_only=True):
        if cat_i is not None and str(row[cat_i] or "").strip().lower() not in ("", "human"):
            continue
        atc, inn = str(row[atc_i] or "").strip().upper(), str(row[inn_i] or "").strip()
        if atc and inn:
            for code in re.split(r"[,;\s]+", atc):
                if re.fullmatch(r"[A-Z]\d\d[A-Z]{2}\d\d", code) and inn not in out.setdefault(code, []):
                    out[code].append(inn)
    report["codes"] = len(out)
    return out, report


@dataclass
class DrugLabel:
    """The parts of a US label the drug branch reads, with what identifies its version."""

    query: str
    set_id: str = ""
    version: str = ""
    effective_time: str = ""
    generic_names: list[str] = field(default_factory=list)
    substance_names: list[str] = field(default_factory=list)
    rxcui: list[str] = field(default_factory=list)
    sections: dict[str, list[str]] = field(default_factory=dict)
    source: str = "openFDA drug label"
    licence: str = OPENFDA_LICENCE

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DrugLabel:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def citation(self) -> dict[str, str]:
        return {"source": self.source, "set_id": self.set_id, "version": self.version,
                "effective_time": self.effective_time, "licence": self.licence,
                "url": f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={self.set_id}" if self.set_id else ""}


def label_query(name: str) -> str:
    """The openFDA search for a single-ingredient label of this generic name."""
    return f'openfda.generic_name.exact:"{name.upper()}"'


def pick_label(name: str, results: list[dict[str, Any]]) -> DrugLabel | None:
    """Of the labels found, one whose generic name is exactly the ingredient (not a combination), the most recent;
    ties broken by set id so the choice is the same every time."""
    wanted = norm(name)
    fitting = []
    for r in results:
        generic = [norm(g) for g in r.get("openfda", {}).get("generic_name", [])]
        if wanted in generic and (r.get("drug_interactions") or r.get("adverse_reactions")):
            fitting.append(r)
    if not fitting:
        return None
    best = max(fitting, key=lambda r: (str(r.get("effective_time", "")), str(r.get("set_id", ""))))
    openfda = best.get("openfda", {})
    return DrugLabel(
        query=name, set_id=str(best.get("set_id", "")), version=str(best.get("version", "")),
        effective_time=str(best.get("effective_time", "")), generic_names=list(openfda.get("generic_name", [])),
        substance_names=list(openfda.get("substance_name", [])), rxcui=list(openfda.get("rxcui", [])),
        sections={s: [str(x) for x in best.get(s, [])] for s in LABEL_SECTIONS if best.get(s)})


class Http:
    """GET with a User-Agent, a pause between calls and three tries; 404 is an empty answer."""

    def __init__(self, pause: float = 0.3, opener: Callable[..., Any] | None = None):
        self.pause, self.calls, self.opener = pause, 0, opener or urllib.request.urlopen

    def get(self, url: str, *, accept: str = "application/json", timeout: int = 60) -> bytes:
        time.sleep(self.pause)
        self.calls += 1
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
                with self.opener(req, timeout=timeout) as response:
                    return response.read()
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    return b""
                if attempt == 2:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
            time.sleep(3 * (attempt + 1))
        return b""


def fetch_label(http: Http, name: str, api_key: str = "") -> DrugLabel | None:
    q = {"search": label_query(name), "limit": 20}
    if api_key:
        q = {"api_key": api_key, **q}
    raw = http.get(f"{OPENFDA_LABEL}?{urllib.parse.urlencode(q)}")
    return pick_label(name, json.loads(raw).get("results", [])) if raw else None


# Italian active-ingredient forms carry salts and hydration the INN does not ("atorvastatina calcio triidrato")
_SALT_WORDS = {
    "calcio", "sodio", "sodica", "sodico", "potassio", "potassica", "magnesio", "cloridrato", "idrocloruro",
    "bromidrato", "maleato", "fumarato", "besilato", "mesilato", "succinato", "tartrato", "citrato", "fosfato",
    "solfato", "acetato", "dipropionato", "propionato", "triidrato", "diidrato", "monoidrato", "emiidrato",
    "sesquiidrato", "anidro", "anidra", "idrato", "pentaidrato", "eptaidrato", "esaidrato", "tosilato", "valerato",
    "furoato", "butirrato", "pivalato", "dicloridrato", "trometamolo", "lisina", "arginina", "meglumina",
    "nitrato", "lattato", "gluconato", "carbonato", "ossido", "bitartrato", "dimesilato", "etabonato",
}


def ingredient_core(ingredient: str) -> str:
    """The ingredient without its salt and hydration words, lower case."""
    words = [w for w in norm(ingredient).split() if w not in _SALT_WORDS]
    return " ".join(words)


@dataclass
class DrugCache:
    """The local cache a case reads: products, ATC names, labels, and the manifest of their versions."""

    root: Path

    def path(self, name: str) -> Path:
        return self.root / name

    def write_json(self, name: str, data: Any) -> None:
        p = self.path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True), "utf-8")

    def read_json(self, name: str, default: Any = None) -> Any:
        p = self.path(name)
        return json.loads(p.read_text("utf-8")) if p.exists() else default

    def label(self, inn: str) -> DrugLabel | None:
        data = self.read_json(f"labels/{_header_key(inn)}.json")
        return DrugLabel.from_dict(data) if data else None

    def put_label(self, inn: str, label: DrugLabel | None) -> None:
        self.write_json(f"labels/{_header_key(inn)}.json", label.as_dict() if label else {})

    def has_label_entry(self, inn: str) -> bool:
        return self.path(f"labels/{_header_key(inn)}.json").exists()


def ingredients_of(products: Iterable[AifaProduct]) -> dict[str, set[str]]:
    """Each AIFA ingredient (as written) -> its ATC codes."""
    out: dict[str, set[str]] = {}
    for p in products:
        if p.ingredient:
            out.setdefault(p.ingredient, set())
            if p.atc:
                out[p.ingredient].add(p.atc.strip().upper())
    return out


__all__ = [n for n in dir() if not n.startswith("_")]
