"""Out-of-range laboratory rows as HPO phenotypes -- graph entry points -- deterministically: step 4 of the document-processing plan.

Step 3 (lab_results.py) turned a report's rows into ClinicalObservation,
each assessed only against the range printed beside it. Nothing downstream
could use that yet: the diagnostic graph is built from HPO annotations, its
nodes are HPO phenotype labels ("Elevated circulating creatinine
concentration"), and "Creatinina 1,45 mg/dL (0,50 - 1,20)" is not one. This
module is the bridge: an observation assessed "above" or "below" its own
printed range becomes the matching HPO phenotype, which the pipeline then
uses as a finding exactly as if the physician had typed it.

**A curated table, verified against the ontology file, never recalled.**
Every rule below names its HPO term by id AND by the exact label it carries
in the bundled data/hp.obo (release 2026-09-01); a test checks both against
that file on every run, and that each term is annotated to at least one
disease in data/phenotype.hpoa -- a term no disease carries is a graph node
with nothing behind it. HPO renames terms (the ones used here were mostly
"Elevated serum X" until they became "Elevated circulating X
concentration"); a rename fails that test instead of silently breaking the
match. Terms were chosen by reading their definitions, not only their
names: HPO's "Polycythemia" requires red cell count, hemoglobin and red
cell volume all above range, so a high red cell count alone maps to nothing.

**What is refused, each with its reason returned -- never dropped:**

- anything not assessed above/below its printed range (within, not
  assessable, reference not parsed, no range, number format ambiguous);
- a row whose printed flag contradicts its range (H printed, value below):
  either the extraction or the report is wrong, and a phenotype must not be
  built on either;
- an unknown specimen. Leukocytes, glucose, proteins, hemoglobin, bilirubin
  are measured in blood AND urine, with different meanings (80 leukocytes/µL
  in urine is pyuria, not leukocytosis). The specimen comes from the
  section header (EMOCROMO, CHIMICA CLINICA -> blood; ESAME URINE -> urine)
  or from the analyte's own name ("glucosio urinario"); a row under no
  recognised header is not mapped, not assumed to be blood;
- a percentage where the HPO term means an absolute count or concentration
  ("Neutrofili 78,5 %" is not "Increased total neutrophil count"; an
  electrophoresis "Albumina 58,4 %" is not hypoalbuminemia);
- the prothrombin time as a percentage: "Tempo di Quick 45 %" is BELOW range
  when clotting is PROLONGED -- the direction inverts. Only INR and seconds;
- a qualitative urine result of "tracce": borderline by definition;
- an analyte with no rule, or matching more than one.

What this does not do: combine results (anemia with a low MCV is not turned
into "Microcytic anemia" -- that is reasoning, not reading), grade severity,
or use the physician's own knowledge of the patient. Each phenotype stays
traceable to the exact attachment line it came from.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..types import ClinicalObservation

SPECIMEN_BLOOD = "blood"
SPECIMEN_URINE = "urine"
SPECIMEN_CSF = "csf"

WITHHELD_NOT_OUT_OF_RANGE = "not_out_of_range"
WITHHELD_FLAG_DISAGREEMENT = "printed_flag_contradicts_range"
WITHHELD_SPECIMEN_UNKNOWN = "specimen_unknown"
WITHHELD_SPECIMEN_MISMATCH = "specimen_not_covered_by_rule"
WITHHELD_PERCENT = "percentage_not_absolute_value"
WITHHELD_UNIT_NOT_ACCEPTED = "unit_not_accepted_by_rule"
WITHHELD_NO_RULE = "no_phenotype_rule"
WITHHELD_AMBIGUOUS_RULE = "matches_more_than_one_rule"
WITHHELD_NO_TERM_FOR_DIRECTION = "no_hpo_term_for_this_direction"
WITHHELD_QUALITATIVE_BORDERLINE = "qualitative_borderline"
WITHHELD_NOT_LABORATORY = "not_a_laboratory_observation"
WITHHELD_SUSCEPTIBILITY = "susceptibility_is_not_a_phenotype"

_CANDIDATE_STATUSES = ("above", "below", "qualitative_differs_from_reference")


@dataclass(frozen=True)
class HpoTerm:
    term_id: str
    label: str


@dataclass(frozen=True)
class PhenotypeRule:
    """One analyte, how it is printed on Italian reports, and its HPO terms per direction."""

    key: str
    names: frozenset[str]
    specimen: str
    high: HpoTerm | None = None
    low: HpoTerm | None = None
    exclude_percent: bool = False
    # When set, only these units (normalised, see _unit_key) are accepted --
    # e.g. the prothrombin time, where a percentage inverts the direction.
    units: frozenset[str] | None = None
    # Qualitative results that count as "present" for this rule (urine dipstick).
    qualitative_positive: bool = False


def _rule(key: str, names: Iterable[str], specimen: str, high: tuple[str, str] | None = None,
          low: tuple[str, str] | None = None, **options: Any) -> PhenotypeRule:
    return PhenotypeRule(
        key=key,
        names=frozenset(_normalise(name) for name in names),
        specimen=specimen,
        high=HpoTerm(*high) if high else None,
        low=HpoTerm(*low) if low else None,
        **options,
    )


def _normalise(text: str) -> str:
    """Lower case, accents off, punctuation to spaces: "Gamma-GT", "γ GT" and "gamma gt" compare equal."""
    text = unicodedata.normalize("NFKD", text.replace("γ", "gamma ").replace("α", "alfa ").replace("β", "beta "))
    text = "".join(char for char in text if not unicodedata.combining(char)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


B = SPECIMEN_BLOOD
U = SPECIMEN_URINE
_SECONDS = frozenset({"sec", "s", "secondi"})
_RATIO = frozenset({"", "ratio", "inr"})

# Every id and label below was read from data/hp.obo (2026-09-01) and checked
# by tests/test_lab_phenotypes.py -- see the module docstring.
RULES: tuple[PhenotypeRule, ...] = (
    # --- Blood count -------------------------------------------------------
    _rule("wbc", ["globuli bianchi", "leucociti", "wbc", "gb", "conta leucocitaria"], B,
          high=("HP:0001974", "Increased total leukocyte count"), low=("HP:0001882", "Decreased total leukocyte count"),
          exclude_percent=True),
    _rule("hemoglobin", ["emoglobina", "hb", "hgb", "emoglobina totale"], B,
          high=("HP:0001900", "Increased circulating hemoglobin concentration"), low=("HP:0001903", "Anemia")),
    _rule("hematocrit", ["ematocrito", "hct", "ht"], B,
          high=("HP:0001899", "Increased hematocrit"), low=("HP:0031851", "Reduced hematocrit")),
    _rule("mcv", ["mcv", "volume corpuscolare medio", "volume globulare medio"], B,
          high=("HP:0005518", "Increased mean corpuscular volume"), low=("HP:0025066", "Decreased mean corpuscular volume")),
    _rule("platelets", ["piastrine", "plt", "conta piastrinica", "trombociti"], B,
          high=("HP:0001894", "Thrombocytosis"), low=("HP:0001873", "Thrombocytopenia"), exclude_percent=True),
    _rule("neutrophils", ["neutrofili", "neutrofili assoluti", "neu", "neut"], B,
          high=("HP:0011897", "Increased total neutrophil count"), low=("HP:0001875", "Decreased total neutrophil count"),
          exclude_percent=True),
    _rule("lymphocytes", ["linfociti", "linfociti assoluti", "lym", "linf"], B,
          high=("HP:0100827", "Increased total lymphocyte count"), low=("HP:0001888", "Decreased total lymphocyte count"),
          exclude_percent=True),
    _rule("monocytes", ["monociti", "monociti assoluti", "mono"], B,
          high=("HP:0012311", "Increased total monocyte count"), low=("HP:0012312", "Decreased total monocyte count"),
          exclude_percent=True),
    _rule("eosinophils", ["eosinofili", "eosinofili assoluti", "eos"], B,
          high=("HP:0001880", "Increased total eosinophil count"), low=("HP:0031891", "Decreased total eosinophil count"),
          exclude_percent=True),
    _rule("basophils", ["basofili", "basofili assoluti", "baso"], B,
          high=("HP:0031807", "Increased total basophil count"), low=("HP:0031808", "Decreased total basophil count"),
          exclude_percent=True),
    # --- Kidney, glucose, electrolytes ------------------------------------
    _rule("creatinine", ["creatinina", "creatininemia", "creatinina sierica", "crea"], B,
          high=("HP:0003259", "Elevated circulating creatinine concentration")),
    _rule("egfr", ["egfr", "gfr", "filtrato glomerulare", "egfr ckd epi", "velocita di filtrazione glomerulare"], B,
          low=("HP:0012213", "Decreased glomerular filtration rate")),
    _rule("urea", ["urea", "azotemia", "bun", "azoto ureico"], B,
          high=("HP:0003138", "Increased blood urea nitrogen"), low=("HP:0031969", "Reduced blood urea nitrogen")),
    _rule("glucose", ["glucosio", "glicemia", "glucosio sierico", "glicemia a digiuno"], B,
          high=("HP:0003074", "Hyperglycemia"), low=("HP:0001943", "Hypoglycemia")),
    _rule("sodium", ["sodio", "natremia", "na"], B, high=("HP:0003228", "Hypernatremia"), low=("HP:0002902", "Hyponatremia")),
    _rule("potassium", ["potassio", "kaliemia", "potassiemia", "k"], B,
          high=("HP:0002153", "Hyperkalemia"), low=("HP:0002900", "Hypokalemia")),
    _rule("calcium", ["calcio", "calcemia", "calcio totale", "ca"], B,
          high=("HP:0003072", "Hypercalcemia"), low=("HP:0002901", "Hypocalcemia")),
    _rule("magnesium", ["magnesio", "magnesemia", "mg"], B,
          high=("HP:0002918", "Hypermagnesemia"), low=("HP:0002917", "Hypomagnesemia")),
    _rule("phosphate", ["fosforo", "fosfato", "fosfatemia", "fosforemia", "fosforo inorganico"], B,
          high=("HP:0002905", "Hyperphosphatemia"), low=("HP:0002148", "Hypophosphatemia")),
    _rule("chloride", ["cloro", "cloruri", "cloremia", "cl"], B,
          high=("HP:0011423", "Hyperchloremia"), low=("HP:0003113", "Hypochloremia")),
    _rule("uric_acid", ["acido urico", "uricemia"], B, high=("HP:0002149", "Hyperuricemia"), low=("HP:0003537", "Hypouricemia")),
    # --- Liver, pancreas, muscle ------------------------------------------
    _rule("alt", ["alt", "gpt", "alt gpt", "gpt alt", "alanina aminotransferasi", "alanina transaminasi"], B,
          high=("HP:0031964", "Elevated circulating alanine aminotransferase concentration")),
    _rule("ast", ["ast", "got", "ast got", "got ast", "aspartato aminotransferasi", "aspartato transaminasi"], B,
          high=("HP:0031956", "Elevated circulating aspartate aminotransferase concentration")),
    _rule("ggt", ["ggt", "gamma gt", "gamma glutamiltransferasi", "gamma glutamil transferasi", "gamma glutamiltranspeptidasi"], B,
          high=("HP:0030948", "Elevated gamma-glutamyltransferase level"), low=("HP:0034445", "Reduced gamma-glutamyltransferase level")),
    _rule("alp", ["fosfatasi alcalina", "alp", "fal", "fosfatasi alcalina totale"], B,
          high=("HP:0003155", "Elevated circulating alkaline phosphatase concentration"),
          low=("HP:0003282", "Decreased circulating alkaline phosphatase activity")),
    _rule("bilirubin_total", ["bilirubina totale", "bilirubina", "bilirubinemia totale"], B,
          high=("HP:0002904", "Hyperbilirubinemia")),
    _rule("bilirubin_direct", ["bilirubina diretta", "bilirubina coniugata"], B,
          high=("HP:0002908", "Conjugated hyperbilirubinemia")),
    _rule("bilirubin_indirect", ["bilirubina indiretta", "bilirubina non coniugata"], B,
          high=("HP:0008282", "Unconjugated hyperbilirubinemia")),
    _rule("albumin", ["albumina", "albuminemia", "albumina sierica"], B,
          low=("HP:0003073", "Hypoalbuminemia"), exclude_percent=True),
    _rule("total_protein", ["proteine totali", "protidemia", "protidemia totale"], B,
          high=("HP:0002152", "Hyperproteinemia"), low=("HP:0003075", "Hypoproteinemia")),
    _rule("gamma_globulins", ["gamma", "gamma globuline", "frazione gamma"], B,
          high=("HP:0010702", "Increased circulating immunoglobulin concentration"),
          low=("HP:0004313", "Decreased circulating immunoglobulin concentration"), exclude_percent=True),
    _rule("ldh", ["ldh", "lattato deidrogenasi", "latticodeidrogenasi"], B,
          high=("HP:0025435", "Increased circulating lactate dehydrogenase concentration"),
          low=("HP:0045041", "Reduced circulating lactate dehydrogenase concentration")),
    _rule("ck", ["ck", "cpk", "creatinchinasi", "creatin chinasi", "creatinfosfochinasi", "ck totale"], B,
          high=("HP:0003236", "Elevated circulating creatine kinase activity")),
    _rule("amylase", ["amilasi", "amilasemia", "amilasi totale"], B,
          high=("HP:0410288", "Hyperamylasemia"), low=("HP:0410289", "Hypoamylasemia")),
    # --- Lipids, glucose control ------------------------------------------
    _rule("cholesterol_total", ["colesterolo totale", "colesterolo", "colesterolemia"], B,
          high=("HP:0003124", "Hypercholesterolemia"), low=("HP:0003146", "Hypocholesterolemia")),
    _rule("ldl", ["colesterolo ldl", "ldl", "ldl colesterolo", "colesterolo ldl calcolato"], B,
          high=("HP:0003141", "Elevated circulating LDL-C concentration"),
          low=("HP:0003563", "Decreased circulating LDL-C concentration")),
    _rule("hdl", ["colesterolo hdl", "hdl", "hdl colesterolo"], B,
          high=("HP:0012184", "Elevated circulating HDL-C concentration"),
          low=("HP:0003233", "Decreased circulating HDL-C concentration")),
    _rule("triglycerides", ["trigliceridi", "trigliceridemia"], B, high=("HP:0002155", "Hypertriglyceridemia")),
    _rule("hba1c", ["hba1c", "emoglobina glicata", "emoglobina glicosilata", "hb glicata"], B,
          high=("HP:0040217", "Elevated hemoglobin A1c")),
    # --- Inflammation, iron, vitamins -------------------------------------
    _rule("crp", ["proteina c reattiva", "pcr", "crp", "proteina c reattiva quantitativa"], B,
          high=("HP:0011227", "Elevated circulating C-reactive protein concentration")),
    _rule("esr", ["ves", "velocita di eritrosedimentazione", "eritrosedimentazione"], B,
          high=("HP:0003565", "Elevated erythrocyte sedimentation rate")),
    _rule("ferritin", ["ferritina", "ferritinemia"], B,
          high=("HP:0003281", "Increased circulating ferritin concentration"),
          low=("HP:0012343", "Decreased circulating ferritin concentration")),
    _rule("iron", ["sideremia", "ferro", "ferro sierico"], B,
          high=("HP:0003452", "Elevated circulating iron concentration"),
          low=("HP:0040303", "Decreased circulating iron concentration")),
    _rule("transferrin_saturation", ["saturazione transferrina", "saturazione della transferrina", "indice di saturazione della transferrina"], B,
          high=("HP:0012463", "Elevated transferrin saturation"), low=("HP:0012464", "Decreased transferrin saturation")),
    _rule("vitamin_b12", ["vitamina b12", "cobalamina", "b12"], B,
          high=("HP:6000016", "Elevated circulating vitamin B12 concentration"),
          low=("HP:0100502", "Decreased circulating vitamin B12 concentration")),
    _rule("folate", ["folati", "acido folico", "folato", "folati sierici"], B,
          high=("HP:0032164", "Increased circulating folate concentration"),
          low=("HP:0100507", "Decreased circulating folate concentration")),
    # --- Thyroid, heart -----------------------------------------------------
    _rule("tsh", ["tsh", "tireotropina", "ormone tireostimolante"], B,
          high=("HP:0002925", "Elevated circulating thyroid-stimulating hormone concentration"),
          low=("HP:0031098", "Decreased thyroid-stimulating hormone level")),
    _rule("troponin_t", ["troponina t", "troponina t hs", "hs tnt", "tnt"], B,
          high=("HP:0410174", "Increased circulating troponin T concentration")),
    _rule("troponin_i", ["troponina i", "troponina i hs", "hs tni", "tni"], B,
          high=("HP:0410173", "Increased circulating troponin I concentration")),
    _rule("bnp", ["bnp", "peptide natriuretico tipo b", "peptide natriuretico cerebrale"], B,
          high=("HP:0033534", "Increased circulating brain natriuretic peptide concentration")),
    # --- Coagulation ----------------------------------------------------------
    # One rule, INR or seconds only: as a percentage ("Tempo di Quick 45 %")
    # the prothrombin time is BELOW range exactly when clotting is prolonged.
    _rule("prothrombin_time", ["inr", "pt inr", "pt", "tempo di protrombina", "tempo di quick", "tempo di protrombina inr",
                               "tempo di quick inr"], B,
          high=("HP:0008151", "Prolonged prothrombin time"), units=_SECONDS | _RATIO),
    _rule("aptt", ["aptt", "ptt", "tempo di tromboplastina parziale attivata", "aptt ratio", "ptt ratio"], B,
          high=("HP:0003645", "Prolonged partial thromboplastin time"), units=_SECONDS | _RATIO),
    _rule("fibrinogen", ["fibrinogeno", "fibrinogenemia"], B,
          high=("HP:0011899", "Hyperfibrinogenemia"), low=("HP:0011900", "Hypofibrinogenemia")),
    _rule("d_dimer", ["d dimero", "d dimeri", "ddimero"], B,
          high=("HP:0033106", "Elevated circulating D-dimer concentration")),
    # --- Urine ------------------------------------------------------------------
    _rule("urine_glucose", ["glucosio", "glucosio urinario", "glicosuria"], U,
          high=("HP:0003076", "Glycosuria"), qualitative_positive=True),
    _rule("urine_protein", ["proteine", "proteine urinarie", "proteinuria", "albumina"], U,
          high=("HP:0000093", "Proteinuria"), qualitative_positive=True),
    _rule("urine_blood", ["emoglobina", "sangue", "emazie", "eritrociti", "globuli rossi", "sangue occulto"], U,
          high=("HP:0000790", "Hematuria"), qualitative_positive=True),
    _rule("urine_ketones", ["chetoni", "corpi chetonici", "chetonuria"], U,
          high=("HP:0002919", "Ketonuria"), qualitative_positive=True),
    _rule("urine_leukocytes", ["leucociti", "globuli bianchi", "esterasi leucocitaria", "leucocituria"], U,
          high=("HP:0012085", "Pyuria"), qualitative_positive=True),
)

# Section headers that name the specimen. Matched as whole words after
# normalisation; urine and CSF win over blood when a header names both.
_URINE_WORDS = ("urine", "urina", "urinario", "urinaria", "urinocoltura", "sedimento")
_CSF_WORDS = ("liquor", "lcr", "cerebrospinale", "liquido cefalorachidiano")
_BLOOD_WORDS = (
    "emocromo", "ematologia", "emocromocitometrico", "chimica", "biochimica", "chimico clinica", "siero",
    "plasma", "sangue", "coagulazione", "emostasi", "protidogramma", "elettroforesi", "proteine sieriche",
    "lipidi", "assetto lipidico", "profilo lipidico", "funzionalita", "ormoni", "endocrinologia",
    "tiroide", "emogas", "emogasanalisi", "marcatori", "sierologia", "immunologia", "enzimi", "elettroliti",
    "metabolismo", "sideremia", "assetto marziale", "infiammazione", "cardiaci",
)


def _has_word(text: str, words: Sequence[str]) -> bool:
    padded = f" {text} "
    return any(f" {word} " in padded for word in words)


def specimen_of(section: str | None, analyte: str) -> str | None:
    """Blood, urine or CSF -- from the analyte's own name first, then its section header; None when neither says."""
    analyte_key = _normalise(analyte)
    if _has_word(analyte_key, _URINE_WORDS) or analyte_key.endswith("uria"):
        return SPECIMEN_URINE
    if _has_word(analyte_key, _CSF_WORDS):
        return SPECIMEN_CSF
    if not section:
        return None
    section_key = _normalise(section)
    if _has_word(section_key, _URINE_WORDS):
        return SPECIMEN_URINE
    if _has_word(section_key, _CSF_WORDS):
        return SPECIMEN_CSF
    if _has_word(section_key, _BLOOD_WORDS):
        return SPECIMEN_BLOOD
    return None


def _name_keys(analyte: str) -> set[str]:
    """The printed name, the name without its parenthetical, and the parenthetical alone: "Globuli bianchi (WBC)"."""
    keys = {_normalise(analyte)}
    without = re.sub(r"\([^)]*\)", " ", analyte)
    keys.add(_normalise(without))
    for inner in re.findall(r"\(([^)]*)\)", analyte):
        keys.add(_normalise(inner))
    return {key for key in keys if key}


def _unit_key(unit: str | None) -> str:
    return (unit or "").strip().lower()


@dataclass(frozen=True)
class LabPhenotype:
    term_id: str
    label: str
    direction: str
    rule: str
    sources: tuple[str, ...]
    analytes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "term_id": self.term_id, "label": self.label, "direction": self.direction, "rule": self.rule,
            "sources": list(self.sources), "analytes": list(self.analytes),
        }


@dataclass(frozen=True)
class WithheldObservation:
    source: str | None
    analyte: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"source": self.source, "analyte": self.analyte, "reason": self.reason}


@dataclass(frozen=True)
class PhenotypeMapping:
    phenotypes: tuple[LabPhenotype, ...] = ()
    withheld: tuple[WithheldObservation, ...] = field(default_factory=tuple)

    def finding_labels(self) -> list[str]:
        return [item.label for item in self.phenotypes]

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe provenance: which phenotypes, from which lines, and why every other row was not used."""
        reasons: dict[str, int] = {}
        for item in self.withheld:
            reasons[item.reason] = reasons.get(item.reason, 0) + 1
        return {
            "phenotypes": [item.as_dict() for item in self.phenotypes],
            "withheld_count": len(self.withheld),
            "withheld_by_reason": dict(sorted(reasons.items())),
            "withheld": [item.as_dict() for item in self.withheld],
        }


def _direction(observation: ClinicalObservation, rule: PhenotypeRule | None) -> tuple[str | None, str | None]:
    """("high"|"low", None) or (None, withheld reason)."""
    status = observation.interpretation
    details = observation.details
    if status == "above":
        return "high", None
    if status == "below":
        return "low", None
    if status == "qualitative_differs_from_reference" and rule is not None and rule.qualitative_positive:
        qualitative = details.get("qualitative_value")
        if qualitative in ("positive", "present"):
            return "high", None
        if qualitative == "traces":
            return None, WITHHELD_QUALITATIVE_BORDERLINE
    return None, WITHHELD_NOT_OUT_OF_RANGE


def map_observations_to_phenotypes(observations: Iterable[ClinicalObservation]) -> PhenotypeMapping:
    """Each laboratory observation outside its printed range -> its HPO phenotype, or the reason it was not used."""
    found: dict[str, dict[str, Any]] = {}
    withheld: list[WithheldObservation] = []

    for observation in observations:
        details = observation.details or {}
        analyte = str(observation.code)
        source = observation.source
        if details.get("kind") == "antimicrobial_susceptibility":
            withheld.append(WithheldObservation(source, analyte, WITHHELD_SUSCEPTIBILITY))
            continue
        if details.get("kind") != "laboratory_result":
            withheld.append(WithheldObservation(source, analyte, WITHHELD_NOT_LABORATORY))
            continue
        # The most basic reason first, so the per-reason counts say what
        # they mean: a row inside its range is "not out of range", whatever
        # else might also be true of it.
        if observation.interpretation not in _CANDIDATE_STATUSES:
            withheld.append(WithheldObservation(source, analyte, WITHHELD_NOT_OUT_OF_RANGE))
            continue
        if details.get("flag_agreement") == "disagrees":
            withheld.append(WithheldObservation(source, analyte, WITHHELD_FLAG_DISAGREEMENT))
            continue

        specimen = specimen_of(details.get("section"), analyte)
        keys = _name_keys(analyte)
        named = [rule for rule in RULES if rule.names & keys]
        if not named:
            withheld.append(WithheldObservation(source, analyte, WITHHELD_NO_RULE))
            continue
        if specimen is None:
            withheld.append(WithheldObservation(source, analyte, WITHHELD_SPECIMEN_UNKNOWN))
            continue
        candidates = [rule for rule in named if rule.specimen == specimen]
        if not candidates:
            withheld.append(WithheldObservation(source, analyte, WITHHELD_SPECIMEN_MISMATCH))
            continue
        if len(candidates) > 1:
            withheld.append(WithheldObservation(source, analyte, WITHHELD_AMBIGUOUS_RULE))
            continue
        rule = candidates[0]

        direction, reason = _direction(observation, rule)
        if direction is None:
            withheld.append(WithheldObservation(source, analyte, reason or WITHHELD_NOT_OUT_OF_RANGE))
            continue
        unit = _unit_key(observation.unit)
        is_qualitative = details.get("qualitative_value") is not None
        if not is_qualitative:
            if rule.exclude_percent and "%" in unit:
                withheld.append(WithheldObservation(source, analyte, WITHHELD_PERCENT))
                continue
            if rule.units is not None and unit not in rule.units:
                withheld.append(WithheldObservation(source, analyte, WITHHELD_UNIT_NOT_ACCEPTED))
                continue
        term = rule.high if direction == "high" else rule.low
        if term is None:
            withheld.append(WithheldObservation(source, analyte, WITHHELD_NO_TERM_FOR_DIRECTION))
            continue

        entry = found.setdefault(term.term_id, {"term": term, "direction": direction, "rule": rule.key, "sources": [], "analytes": []})
        if source:
            entry["sources"].append(source)
        entry["analytes"].append(analyte)

    phenotypes = tuple(
        LabPhenotype(
            term_id=term_id, label=entry["term"].label, direction=entry["direction"], rule=entry["rule"],
            sources=tuple(entry["sources"]), analytes=tuple(dict.fromkeys(entry["analytes"])),
        )
        for term_id, entry in found.items()
    )
    return PhenotypeMapping(phenotypes=phenotypes, withheld=tuple(withheld))
