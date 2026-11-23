"""The diseases of a case ranked by likelihood ratios, absences included (14 November 2026): step 7 of
"Il grafo del caso".

The method is LIRICAL's (Robinson et al., AJHG 2020; https://lirical.readthedocs.io/en/latest/phenotype-score.html),
carried from rare diseases to every disease with a profile: each manifestation weighs as the ratio between how often
it is seen with the disease and how often without it,

    LR(present) = f / b,        LR(absent) = (1 - f) / (1 - b),

with ``f`` the manifestation's frequency in the disease and ``b`` its frequency across the diseases of the profiles
(the background). The posterior odds are the prior odds times the product of the ratios, written as a sum of
log10. What the existing ``memory.differential_ranking`` cannot do and this does: an absence counts. A hallmark
stated absent divides the odds by about ten; a manifestation that "excludes" the disease, found present, by a
thousand (LIRICAL's figure for an excluded feature found).

The weights that are not measured yet are named constants, not hidden numbers, to be replaced by measure:

- ``TYPICALITY_FREQUENCY``: the frequency a typicality stands for when the source gives no number (hallmark 0.9,
  common 0.5, rare 0.1). GraphDx uses the same four levels with fixed weights.
- ``SILENCE_WEIGHT``: how much of an absence a silent covering exam is worth, in log space (0.5: half). A report
  that does not mention a finding is weaker evidence than one that says it is not there.
- ``EVIDENCE_WEIGHT``: a manifestation stated by one source only (``reported``) weighs half, in log space, of one
  two independent sources state (``established``).
- ``VERBAL_WEIGHT``: a frequency given only by a word ("often", "rarely") weighs half, in log space (21 November
  2026): such words are read very differently by different readers.
- A manifestation with ``accuracy`` (sensitivity, specificity or likelihood ratios measured by studies of
  diagnostic accuracy) weighs by those, its background being the patients without the disease the study measured,
  not the library (21 November 2026).
- ``FLOOR``: no frequency is taken as exactly 0 or 1, so no single fact decides alone (except ``EXCLUDES``).

Not assessed and doubtful manifestations weigh nothing: the first become the exams to propose, the second stay in
the doubts of the result (principle of doubt). Each disease comes back with every contribution, so the ranking can
be read manifestation by manifestation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..clinical import case_graph as cg

TYPICALITY_FREQUENCY = {"hallmark": 0.9, "common": 0.5, "rare": 0.1, "excluding": 0.0}
SILENCE_WEIGHT = 0.5
VERBAL_WEIGHT = 0.5
# a manifestation one source only states weighs less than one two independent sources state (15 November 2026)
EVIDENCE_WEIGHT = {"established": 1.0, "reported": 0.5, "": 1.0}
FLOOR = 0.01
EXCLUDES = 1e-3
BACKGROUND_DEFAULT = 0.05


def frequency(m: cg.Manifestation) -> float | None:
    if m.frequency is not None:
        return float(m.frequency)
    return TYPICALITY_FREQUENCY.get(m.typicality)


def _key(m: cg.Manifestation) -> str:
    return m.concept.curie if m.concept is not None else " ".join(m.words.lower().split())


def background(profiles: Iterable[cg.Disease]) -> dict[str, float]:
    """Each manifestation's mean frequency across the diseases of the profiles (absent from a profile = 0)."""
    profiles = list(profiles)
    if not profiles:
        return {}
    sums: dict[str, float] = {}
    for d in profiles:
        seen: dict[str, float] = {}
        for m in d.manifestations:
            f = frequency(m)
            if f is not None:
                seen[_key(m)] = max(seen.get(_key(m), 0.0), f)
        for k, f in seen.items():
            sums[k] = sums.get(k, 0.0) + f
    return {k: s / len(profiles) for k, s in sums.items()}


def _clip(x: float) -> float:
    return min(1 - FLOOR, max(FLOOR, x))


# (23 November 2026) a statement of direction without a number ("makes appendicitis more likely") weighs by these
# provisional likelihood ratios, to be calibrated on published cases (step 9); a strong statement doubles the
# effect in log space; sources that disagree weigh nothing
DIRECTION_LR = {"more_likely": 2.0, "less_likely": 0.5}


def _direction_lr(m: cg.Manifestation, state: str) -> tuple[float, str] | None:
    when = {cg.PRESENT: "present", cg.ABSENT: "absent"}.get(state)
    said = (m.direction or {}).get(when or "")
    if not said or said.get("effect") not in DIRECTION_LR:
        return None
    lr = DIRECTION_LR[said["effect"]] ** (2 if said.get("strong") else 1)
    return lr, (f"direction stated by {said.get('sources', 1)} source(s), {said['effect']}"
                f"{', strongly' if said.get('strong') else ''}: provisional LR {lr:g}, to be calibrated (step 9)")


def _measured_lr(m: cg.Manifestation, state: str) -> tuple[float, str] | None:
    """LR+ for a present manifestation, LR- for an absent one, from ``accuracy``: the likelihood ratio a study gives,
    or the one its sensitivity and specificity give; None when the studies did not measure it."""
    acc = m.accuracy or {}
    sens, spec = acc.get("sensitivity"), acc.get("specificity")
    where = f"measured ({acc.get('design', 'study')}, {acc.get('setting', 'setting not stated')})"
    if state == cg.PRESENT:
        lr = acc.get("lr_positive")
        if lr is None and sens is not None and spec is not None:
            lr = _clip(float(sens)) / _clip(1 - float(spec))
        return (max(FLOOR, float(lr)), f"LR+ {float(lr):.2f} {where}") if lr is not None else None
    if state == cg.ABSENT:
        lr = acc.get("lr_negative")
        if lr is None and sens is not None and spec is not None:
            lr = _clip(1 - float(sens)) / _clip(float(spec))
        return (max(FLOOR, float(lr)), f"LR- {float(lr):.2f} {where}") if lr is not None else None
    return None


@dataclass
class Contribution:
    manifestation_id: str
    words: str
    state: str
    basis: str
    log10_lr: float
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"manifestation": self.manifestation_id, "words": self.words, "state": self.state,
                "basis": self.basis, "log10_lr": round(self.log10_lr, 4), "note": self.note}


@dataclass
class Ranked:
    disease_id: str
    words: str
    branch: str
    log10_odds: float
    contributions: list[Contribution] = field(default_factory=list)
    doubts: list[str] = field(default_factory=list)
    not_assessed: list[str] = field(default_factory=list)

    @property
    def posterior(self) -> float:
        """The posterior probability against the other diseases ranked with it (``rank``)."""
        return getattr(self, "_posterior", float("nan"))

    def as_dict(self) -> dict[str, Any]:
        return {"disease": self.disease_id, "words": self.words, "branch": self.branch,
                "log10_odds": round(self.log10_odds, 4), "posterior": round(self.posterior, 4),
                "contributions": [c.as_dict() for c in self.contributions], "doubts": self.doubts,
                "not_assessed": self.not_assessed}


def score(disease: cg.Disease, bg: dict[str, float], prior: float | None = None) -> Ranked:
    """The log10 odds of one disease from the states of its manifestations (``clinical.manifestation_state``)."""
    out = Ranked(disease.disease_id, disease.words, disease.branch,
                 math.log10(prior / (1 - prior)) if prior else 0.0)
    for m in disease.manifestations:
        state = disease.state_of(m.manifestation_id)
        if state is None:
            out.not_assessed.append(m.manifestation_id)
            continue
        f = frequency(m)
        if state.state == cg.NOT_ASSESSED:
            out.not_assessed.append(m.manifestation_id)
            continue
        if state.state == cg.DOUBTFUL:
            out.doubts.append(m.manifestation_id)
            continue
        measured = _measured_lr(m, state.state)
        if measured is not None:
            # the likelihood ratio a study of diagnostic accuracy measured: its background is the patients without
            # the disease it was measured against, not the library (21 November 2026)
            lr, note = measured
            if state.basis == cg.SILENT and state.state != cg.PRESENT:
                lr, note = lr ** SILENCE_WEIGHT, note + f"; silent covering exam, weight {SILENCE_WEIGHT}"
            out.contributions.append(Contribution(m.manifestation_id, m.words, state.state, state.basis,
                                                  math.log10(lr), note))
            out.log10_odds += math.log10(lr)
            continue
        directed = _direction_lr(m, state.state) if m.frequency_basis not in ("pooled_studies", "one_study") else None
        if directed is not None:
            lr, note = directed
            out.contributions.append(Contribution(m.manifestation_id, m.words, state.state, state.basis,
                                                  math.log10(lr), note))
            out.log10_odds += math.log10(lr)
            continue
        if f is None:
            out.contributions.append(Contribution(m.manifestation_id, m.words, state.state, state.basis, 0.0,
                                                  "no frequency or typicality: not weighed"))
            continue
        b = _clip(bg.get(_key(m), BACKGROUND_DEFAULT))
        if state.state == cg.PRESENT:
            lr = EXCLUDES if m.typicality == "excluding" else _clip(f) / b
            note = "excluding manifestation found" if m.typicality == "excluding" else ""
        else:
            lr = (1 - _clip(f)) / (1 - b)
            note = ""
            if state.basis == cg.SILENT:
                lr = lr ** SILENCE_WEIGHT
                note = f"silent covering exam, weight {SILENCE_WEIGHT}"
        weight = EVIDENCE_WEIGHT.get(m.evidence, 1.0)
        if m.frequency_basis == "verbal":
            # a word of frequency ("often") is read very differently by different readers: the weakest evidence
            weight *= VERBAL_WEIGHT
            note = (note + "; " if note else "") + f"frequency from a word, weight {VERBAL_WEIGHT}"
        if weight != 1.0:
            lr = lr ** weight
            note = (note + "; " if note else "") + f"{m.evidence} by one source, weight {weight}"
        value = math.log10(lr)
        out.log10_odds += value
        out.contributions.append(Contribution(m.manifestation_id, m.words, state.state, state.basis, value, note))
    return out


def rank(diseases: Iterable[cg.Disease], *, profiles: Iterable[cg.Disease] | None = None,
         background_frequencies: dict[str, float] | None = None,
         priors: dict[str, float] | None = None) -> list[Ranked]:
    """The diseases from the most to the least likely, each with its contributions. ``profiles`` (default: the
    diseases themselves) give the background unless ``background_frequencies`` is passed; ``priors`` by disease
    id (default: equal)."""
    diseases = list(diseases)
    bg = background_frequencies if background_frequencies is not None else background(profiles or diseases)
    ranked = [score(d, bg, (priors or {}).get(d.disease_id)) for d in diseases]
    if ranked:
        top = max(r.log10_odds for r in ranked)
        weights = [10 ** (r.log10_odds - top) for r in ranked]
        total = sum(weights)
        for r, w in zip(ranked, weights):
            r._posterior = w / total
    return sorted(ranked, key=lambda r: (-r.log10_odds, r.disease_id))


__all__ = ["BACKGROUND_DEFAULT", "EVIDENCE_WEIGHT", "EXCLUDES", "FLOOR", "SILENCE_WEIGHT", "TYPICALITY_FREQUENCY", "Contribution",
           "Ranked", "background", "frequency", "rank", "score"]
