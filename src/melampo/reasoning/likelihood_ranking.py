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
# (28 November 2026) an observation of the case that no manifestation of the hypothesis accounts for weighs as
# LIRICAL's "no match": a fixed likelihood ratio of 1/100 (https://lirical.readthedocs.io/en/latest/phenotype-score.html).
# Until then the ranking ignored such observations: a disease explaining two findings of ten was not asked about the
# other eight (explanatory completeness)
NO_MATCH = 0.01
# (28 November 2026) the prior of each disease beyond the first in a hypothesis of several: about 2 to 7.5% of the
# patients with a diagnosis have more than one (rare genetic disease; Genes 2022, 13:1275, citing Posey et al., NEJM
# 2017); measured in cohorts of rare disease, not for common disease: a declared constant, to be calibrated
SECOND_DIAGNOSIS = 0.05


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
    explains: list[str] = field(default_factory=list)       # the facts of the case its present manifestations rest on
    unexplained: list[str] = field(default_factory=list)    # the observations it does not account for
    masked: list[str] = field(default_factory=list)         # absences a drug of the patient can hide, not weighed

    @property
    def posterior(self) -> float:
        """The posterior probability against the other diseases ranked with it (``rank``)."""
        return getattr(self, "_posterior", float("nan"))

    @property
    def energy(self) -> float:
        """The energy of the hypothesis, ``-log10`` of its odds: the lower, the better it explains the case."""
        return -self.log10_odds

    def as_dict(self) -> dict[str, Any]:
        return {"disease": self.disease_id, "words": self.words, "branch": self.branch,
                "log10_odds": round(self.log10_odds, 4), "posterior": round(self.posterior, 4),
                "contributions": [c.as_dict() for c in self.contributions], "doubts": self.doubts,
                "not_assessed": self.not_assessed, "unexplained": self.unexplained, "masked": self.masked}


_NOT_WEIGHED = "not_weighed"


def _contribution(m: cg.Manifestation, state: cg.ManifestationState, bg: dict[str, float]) -> Contribution | None:
    """What one manifestation in one state weighs (None: doubtful or not assessed, it weighs nothing)."""
    if state.state in (cg.NOT_ASSESSED, cg.DOUBTFUL):
        return None
    measured = _measured_lr(m, state.state)
    if measured is not None:
        # the likelihood ratio a study of diagnostic accuracy measured: its background is the patients without
        # the disease it was measured against, not the library (21 November 2026)
        lr, note = measured
        if state.basis == cg.SILENT and state.state != cg.PRESENT:
            lr, note = lr ** SILENCE_WEIGHT, note + f"; silent covering exam, weight {SILENCE_WEIGHT}"
        return Contribution(m.manifestation_id, m.words, state.state, state.basis, math.log10(lr), note)
    directed = _direction_lr(m, state.state) if m.frequency_basis not in ("pooled_studies", "one_study") else None
    if directed is not None:
        lr, note = directed
        return Contribution(m.manifestation_id, m.words, state.state, state.basis, math.log10(lr), note)
    f = frequency(m)
    if f is None:
        return Contribution(m.manifestation_id, m.words, state.state, state.basis, 0.0,
                            "no frequency or typicality: not weighed")
    return _from_frequency(m.manifestation_id, m.words, state, f, _clip(bg.get(_key(m), BACKGROUND_DEFAULT)),
                           excluding=m.typicality == "excluding", evidence=m.evidence,
                           verbal=m.frequency_basis == "verbal")


def _from_frequency(mid: str, words: str, state: cg.ManifestationState, f: float, b: float, *, excluding: bool,
                    evidence: str, verbal: bool, note: str = "") -> Contribution:
    if state.state == cg.PRESENT:
        lr = EXCLUDES if excluding else _clip(f) / b
        if excluding:
            note = (note + "; " if note else "") + "excluding manifestation found"
    else:
        lr = (1 - _clip(f)) / (1 - b)
        if state.basis == cg.SILENT:
            lr = lr ** SILENCE_WEIGHT
            note = (note + "; " if note else "") + f"silent covering exam, weight {SILENCE_WEIGHT}"
    weight = EVIDENCE_WEIGHT.get(evidence, 1.0)
    if verbal:
        # a word of frequency ("often") is read very differently by different readers: the weakest evidence
        weight *= VERBAL_WEIGHT
        note = (note + "; " if note else "") + f"frequency from a word, weight {VERBAL_WEIGHT}"
    if weight != 1.0:
        lr = lr ** weight
        note = (note + "; " if note else "") + f"{evidence} by one source, weight {weight}"
    return Contribution(mid, words, state.state, state.basis, math.log10(lr), note)


# the states of one manifestation in several diseases of a hypothesis: the most informative one counts
_INFORMATIVE = {cg.PRESENT: 0, cg.DOUBTFUL: 1, cg.ABSENT: 2, cg.NOT_ASSESSED: 4}


def _rank_of_state(state: cg.ManifestationState) -> int:
    if state.state == cg.ABSENT and state.basis == cg.SILENT:
        return 3
    return _INFORMATIVE.get(state.state, 4)


def score(disease: cg.Disease, bg: dict[str, float], prior: float | None = None, *,
          observed: dict[str, str] | None = None, masked: dict[str, str] | None = None) -> Ranked:
    """The log10 odds of one disease from the states of its manifestations (``clinical.manifestation_state``);
    ``observed`` (fact id -> words) are the observations of the case to account for, ``masked`` (manifestation key
    -> why) the absences a drug of the patient can hide."""
    return score_set([disease], bg, prior, observed=observed, masked=masked)


def score_set(diseases: list[cg.Disease], bg: dict[str, float], prior: float | None = None, *,
              observed: dict[str, str] | None = None, masked: dict[str, str] | None = None) -> Ranked:
    """The log10 odds of a hypothesis of one or more diseases together (28 November 2026).

    A manifestation two diseases share is expected with the probability that either produces it, the noisy-OR
    of their frequencies (Shwe et al., Methods Inf Med 1991, QMR-DT): ``1 - (1 - f1)(1 - f2)``. Each disease beyond the
    first weighs ``SECOND_DIAGNOSIS``; each observation none of them accounts for weighs ``NO_MATCH``. One disease
    alone is ranked exactly as before, with the observations it does not account for added.
    """
    diseases = list(diseases)
    masked = masked or {}
    head = diseases[0] if diseases else None
    out = Ranked("+".join(d.disease_id for d in diseases) or "none",
                 " + ".join(d.words for d in diseases) or "no disease of the catalogue",
                 head.branch if head is not None and len(diseases) == 1 else "hypothesis_set",
                 math.log10(prior / (1 - prior)) if prior else 0.0)
    if len(diseases) > 1:
        extra = (len(diseases) - 1) * math.log10(SECOND_DIAGNOSIS)
        out.log10_odds += extra
        out.contributions.append(Contribution("prior:second_diagnosis", f"{len(diseases)} diseases together", "", "",
                                              extra, f"prior {SECOND_DIAGNOSIS} for each disease beyond the first"))
    groups: dict[str, list[tuple[cg.Manifestation, cg.ManifestationState]]] = {}
    for d in diseases:
        for m in d.manifestations:
            state = d.state_of(m.manifestation_id) or cg.ManifestationState(m.manifestation_id, cg.NOT_ASSESSED)
            groups.setdefault(_key(m), []).append((m, state))
    explains: set[str] = set()
    for key, members in groups.items():
        members.sort(key=lambda ms_: _rank_of_state(ms_[1]))
        m, state = members[0]
        if state.state in (cg.PRESENT, cg.DOUBTFUL):
            explains |= set(state.evidence)
        if state.state == cg.NOT_ASSESSED:
            out.not_assessed.append(m.manifestation_id)
            continue
        if state.state == cg.DOUBTFUL:
            out.doubts.append(m.manifestation_id)
            continue
        if state.state == cg.ABSENT and key in masked:
            out.masked.append(m.manifestation_id)
            out.contributions.append(Contribution(m.manifestation_id, m.words, state.state, state.basis, 0.0,
                                                  f"absence not weighed: {masked[key]}"))
            continue
        if len(members) == 1:
            c = _contribution(m, state, bg)
        else:
            c = _shared(key, members, state, bg)
        if c is not None:
            out.log10_odds += c.log10_lr
            out.contributions.append(c)
    out.explains = sorted(explains)
    for fact_id, words in (observed or {}).items():
        if fact_id in explains:
            continue
        value = math.log10(NO_MATCH)
        out.unexplained.append(fact_id)
        out.log10_odds += value
        out.contributions.append(Contribution(f"observed:{fact_id}", words, cg.PRESENT, "observed", value,
                                              "an observation of the case no manifestation accounts for: "
                                              f"LIRICAL's no-match ratio {NO_MATCH}"))
    return out


def _shared(key: str, members: list[tuple[cg.Manifestation, cg.ManifestationState]],
            state: cg.ManifestationState, bg: dict[str, float]) -> Contribution | None:
    """A manifestation several diseases of the hypothesis share: the noisy-OR of their frequencies; without any
    frequency, the strongest of what they weigh alone."""
    m = members[0][0]
    if state.state == cg.PRESENT and any(x.typicality == "excluding" for x, _ in members):
        return Contribution(m.manifestation_id, m.words, state.state, state.basis, math.log10(EXCLUDES),
                            "excluding manifestation found")
    freqs = [(frequency(x), x) for x, _ in members if frequency(x) is not None]
    if freqs:
        f = 1.0
        for value, _ in freqs:
            f *= 1 - _clip(value)
        f = 1 - f
        lead = max(freqs, key=lambda fx: fx[0])[1]
        return _from_frequency(m.manifestation_id, m.words, state, f, _clip(bg.get(key, BACKGROUND_DEFAULT)),
                               excluding=False, evidence=lead.evidence, verbal=lead.frequency_basis == "verbal",
                               note=f"noisy-OR of {len(freqs)} diseases")
    alone = [c for c in (_contribution(x, state, bg) for x, _ in members) if c is not None]
    return max(alone, key=lambda c: abs(c.log10_lr)) if alone else None


def rank(diseases: Iterable[cg.Disease], *, profiles: Iterable[cg.Disease] | None = None,
         background_frequencies: dict[str, float] | None = None,
         priors: dict[str, float] | None = None, observed: dict[str, str] | None = None,
         masked: dict[str, str] | None = None) -> list[Ranked]:
    """The diseases from the most to the least likely, each with its contributions. ``profiles`` (default: the
    diseases themselves) give the background unless ``background_frequencies`` is passed; ``priors`` by disease
    id (default: equal); ``observed`` and ``masked`` as in ``score_set``."""
    diseases = list(diseases)
    bg = background_frequencies if background_frequencies is not None else background(profiles or diseases)
    ranked = [score(d, bg, (priors or {}).get(d.disease_id), observed=observed, masked=masked) for d in diseases]
    posteriors(ranked)
    return sorted(ranked, key=lambda r: (-r.log10_odds, r.disease_id))


def posteriors(ranked: list[Ranked]) -> None:
    """Each hypothesis's posterior against the others of the list (softmax of the log10 odds)."""
    if ranked:
        top = max(r.log10_odds for r in ranked)
        weights = [10 ** (r.log10_odds - top) for r in ranked]
        total = sum(weights)
        for r, w in zip(ranked, weights):
            r._posterior = w / total


def entropy_bits(ranked: list[Ranked]) -> float:
    """The Shannon entropy, in bits, of the posterior over the hypotheses of the list: how undecided they leave
    the case."""
    if not ranked:
        return 0.0
    top = max(r.log10_odds for r in ranked)
    weights = [10 ** (r.log10_odds - top) for r in ranked]
    total = sum(weights)
    return -sum((w / total) * math.log2(w / total) for w in weights if w > 0)


__all__ = ["BACKGROUND_DEFAULT", "EVIDENCE_WEIGHT", "EXCLUDES", "FLOOR", "NO_MATCH", "SECOND_DIAGNOSIS", "SILENCE_WEIGHT",
           "TYPICALITY_FREQUENCY", "Contribution", "Ranked", "background", "entropy_bits", "frequency", "posteriors",
           "rank", "score", "score_set"]
