"""The intuition engine, lateral part (28 November 2026): when the diagnosis is unclear, a wider spectrum.

The rapid part (``reasoning.intuition_engine``) passes the candidates of ``rank_differential`` to the diagnostic path.
This part looks beside them, in four steps that follow the owner's design ("Rete di Salienza e DMN") and the
literature it rests on:

1. **Salience** (the salience network: what deserves attention). The observations of the case, findings affirmed
   and laboratory values beyond their range, are compared with the best explanation the case has so far (the
   diseases named or proposed, or none): salient is an observation it leaves *unexplained*, one stated in *doubt*,
   and one that *contradicts* it (an excluding manifestation found). An observation a present disease accounts
   for, or a drug of the patient whose label names it (taken before it appeared), is explained, and said so.
2. **Generation** (the default mode network: new connections). Every profile of the library that accounts for at
   least one observation is a single hypothesis, all of them, not a sample. When the best single leaves salient
   observations unexplained, the case is *unclear* and pairs are generated: *waves* of activation bind a part of
   the observations together and leave the rest inhibited, and the best explanation of each part is joined with
   the best explanation of the other (two diseases, each accounting for what the other does not). The waves are
   deterministic oscillators, ``sin(2π(phase + t·frequency))`` with a phase from the observation and a frequency
   from the golden ratio: periodic, non-linear, never repeating the same pattern, the same for the same case.
3. **Evaluation** (prefrontal comparison with the models in memory). Each hypothesis is weighed against the whole
   case by its *energy*, ``-log10`` of its odds (``likelihood_ranking.score_set``): the profiles from certain
   sources, absences included, the observations it does not account for (LIRICAL's no-match ratio), a second
   disease at its prior (``SECOND_DIAGNOSIS``), and the absences a drug of the patient can hide not weighed. The
   drop of the energy against the best explanation before is the "aha" of the owner's design, measured: the
   insight that simplifies the problem is the Bayesian model reduction of Friston et al. (Neural Computation
   2017). The entropy of the posterior over the hypotheses, before and after, is reported with it.
4. **The guard** (the eureka heuristic: an insight feels true whether or not it is; Laukkonen et al.). A proposal
   enters the case only as a *hypothesis to exclude*: it must explain the case better than no disease of the
   catalogue, account for at least one salient observation the best explanation leaves, and not be dominated by
   another proposal that accounts for as many salient observations with a lower energy (or for more, with no higher
   one); a pair must also be better than each of its
   diseases alone and than the best single explanation of the case. It is then ranked with the others by the
   diagnostic path; what would discriminate it is in its exams to propose.

Why pairs: phenotype-driven systems recover both diagnoses of a dual-diagnosis case in their first five for only
4.9-9.7% of synthetic and 0-4.2% of literature cases ("Finding One, Missing Another", NeurIPS 2026); the phenotype of
two diagnoses can be discrete, synergistic or antagonistic (AJHG 2024, 111:2382).

Not yet here, and why (no branch without its data): the comparison of the case's lesions with other reports,
confirmed cases and the literature through what the imaging interpreter already measures (``imaging.findings``:
shape, density from core to border, margins, the structures involved) and the words of the reports, in two stages
as MAC/FAC (Forbus, Gentner & Law, Cognitive Science 1995): a cheap content vector first, then the structural match
of the graphs. No learned latent space, JEPA or other (verified 28 November 2026: it cannot be matched to the words
of the literature or to a certain source, and it is weakest on CT and MR volumes); the relations between
manifestations (reinforcing, attenuating, exclusive:
from validated clinical scores and from opposite qualities in the logical definitions of HPO), the evolution in
time (the onset of each manifestation), and what a drug can hide, read from its label by the interpreters
(``DrugProfile.masks``: weighed here when present, not yet filled).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ..clinical import case_graph as cg
from ..clinical import manifestation_state as ms
from ..memory.profile_library import ProfileLibrary
from . import likelihood_ranking as lr

LATERAL = "lateral_intuition"
WAVES = 64                      # a search budget, as the saturation of a review: how many waves; reported, not a weight
_GOLDEN = (math.sqrt(5) - 1) / 2
_LAB = ("above", "below")


@dataclass(frozen=True)
class Observation:
    fact_id: str
    words: str
    concept: str            # the curie, "" when the fact has no code
    kind: str               # finding, measurement
    doubtful: bool = False


def observations(graph: cg.CaseGraph) -> list[Observation]:
    """The facts of the case a hypothesis is asked to account for: the findings affirmed (in doubt or not) and the
    laboratory values beyond their range."""
    out = []
    for f in graph.findings:
        if f.polarity == cg.NEGATED or str(f.temporality).lower() in ms.HYPOTHETICAL:
            continue
        out.append(Observation(f.finding_id, f.words, f.concept.curie if f.concept else "", "finding",
                               str(f.certainty).lower() in ms.UNCERTAIN))
    for m in graph.measurements:
        if m.interpretation not in _LAB:
            continue
        ref = m.reference or {}
        code = str(ref.get(f"{m.interpretation}_hpo") or "")
        out.append(Observation(m.measurement_id, f"{m.analyte} {m.interpretation}", code, "measurement"))
    return out


def drug_explanations(graph: cg.CaseGraph) -> dict[str, list[dict[str, str]]]:
    """The observations a drug of the patient can explain: its label names them among its effects and the drug was
    started before they appeared (``clinical.drug_branch``). Timing unknown is a doubt: not an explanation."""
    names = {m.medication_id: m.name or m.words for m in graph.medications}
    out: dict[str, list[dict[str, str]]] = {}
    for e in graph.drug_effects:
        if e.timing == "after_start":
            out.setdefault(e.finding, []).append({"medication": names.get(e.medication, e.medication),
                                                  "section": e.section, "sentence": e.sentence})
    return out


def masked_by_drugs(graph: cg.CaseGraph) -> dict[str, str]:
    """The manifestations a drug of the patient can hide, by key, with the label's sentence."""
    out: dict[str, str] = {}
    for p in graph.drug_profiles:
        for mk in p.masks:
            key = str(mk.get("concept") or "") or " ".join(str(mk.get("words", "")).lower().split())
            if key:
                out[key] = f"{p.inn or p.ingredient} can hide it ({mk.get('section', 'label')}: {mk.get('sentence', '')})"
    return out


# -- the knowledge, once per library ------------------------------------------------------------------------------
_INDEX: dict[tuple[int, str], dict[str, Any]] = {}


def _index(library: ProfileLibrary) -> dict[str, Any]:
    key = (id(library), library.version)
    if key not in _INDEX:
        by_key: dict[str, set[str]] = {}
        by_lab: dict[str, set[str]] = {}
        for pid, p in library.profiles.items():
            for m in p.manifestations:
                by_key.setdefault(lr._key(m), set()).add(pid)
                for v in m.verified_by:
                    if v.get("kind") == "laboratory":
                        for k in (v.get("code"), ms._norm(str(v.get("analyte", "")))):
                            if k:
                                by_lab.setdefault(str(k), set()).add(pid)
        _INDEX.clear()
        _INDEX[key] = {"by_key": by_key, "by_lab": by_lab, "background": lr.background(library.profiles.values())}
    return _INDEX[key]


def _reachable(library: ProfileLibrary, graph: cg.CaseGraph, obs: list[Observation],
               below: Callable[[str, str], bool] | None) -> set[str]:
    """The profiles with at least one manifestation that may account for an observation (the states decide)."""
    index = _index(library)
    out: set[str] = set()
    curies = {o.concept for o in obs if o.concept}
    words = {ms._norm(o.words) for o in obs if not o.concept}
    for key, pids in index["by_key"].items():
        if key in curies or key in words or (below and any(below(c, key) for c in curies)):
            out |= pids
    measured = {o.fact_id for o in obs if o.kind == "measurement"}
    for m in graph.measurements:
        if m.measurement_id in measured:
            for k in ((m.code.curie if m.code else ""), ms._norm(m.analyte)):
                out |= index["by_lab"].get(k, set())
    return out


class _States:
    """The state of each manifestation in this case, assessed once for every manifestation that reads the same."""

    def __init__(self, graph: cg.CaseGraph, below: Callable[[str, str], bool] | None):
        self.graph, self.below, self.cache = graph, below, {}

    def of(self, m: cg.Manifestation) -> cg.ManifestationState:
        sig = (lr._key(m), m.region, json.dumps(m.verified_by, sort_keys=True, default=str))
        if sig not in self.cache:
            self.cache[sig] = ms.assess(m, self.graph, below=self.below)
        s = self.cache[sig]
        return cg.ManifestationState(m.manifestation_id, s.state, s.basis, list(s.evidence), s.note)

    def view(self, disease: cg.Disease) -> cg.Disease:
        """The disease with the states of its manifestations in this case (the profile itself is not changed)."""
        return cg.Disease(disease.disease_id, disease.branch, disease.words, disease.origin,
                          concept=disease.concept, manifestations=disease.manifestations,
                          states=[self.of(m) for m in disease.manifestations])


def waves(fact_ids: list[str], n: int = WAVES) -> list[frozenset[str]]:
    """The active part of the observations at each wave: one oscillator per observation, its phase from the
    observation, its frequency from the golden ratio; active while the sine is not negative. Deterministic."""
    osc = []
    for i, fid in enumerate(sorted(fact_ids)):
        phase = int(hashlib.sha256(fid.encode()).hexdigest()[:8], 16) / 2 ** 32
        osc.append((fid, phase, ((i + 1) * _GOLDEN) % 1.0))
    out = []
    for t in range(n):
        out.append(frozenset(fid for fid, phase, freq in osc if math.sin(2 * math.pi * (phase + t * freq)) >= 0))
    return out


@dataclass
class _Hyp:
    diseases: tuple[cg.Disease, ...]
    ranked: lr.Ranked
    origin: str

    @property
    def energy(self) -> float:
        return self.ranked.energy

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(d.disease_id for d in self.diseases)

    def explains(self, facts: set[str]) -> set[str]:
        return set(self.ranked.explains) & facts


def _concept_ids(graph: cg.CaseGraph) -> set[str]:
    return {d.concept.curie for d in graph.diseases if d.concept is not None and d.concept.code} | \
        {d.disease_id for d in graph.diseases}


def lateral_cycle(graph: cg.CaseGraph, library: ProfileLibrary, *,
                  below: Callable[[str, str], bool] | None = None, n_waves: int = WAVES) -> dict[str, Any]:
    """The lateral part of the intuition engine on one case's graph. Returns what it saw, what it proposes (as
    hypotheses to exclude, with the profiles to add as candidates) and what the ranking must weigh (``observed``,
    ``masked``). The graph is not changed here."""
    states = _States(graph, below)
    bg = _index(library)["background"]
    obs = observations(graph)
    by_id = {o.fact_id: o for o in obs}
    drugs = drug_explanations(graph)
    masked = masked_by_drugs(graph)
    present_diseases = [states.view(d) for d in graph.branch(cg.PRESENT_DISEASES) if d.manifestations]
    by_present: dict[str, str] = {}
    for d in present_diseases:
        for s in d.states:
            if s.state == cg.PRESENT:
                for e in s.evidence:
                    by_present.setdefault(e, d.words)
    # the observations to account for: not in doubt, not explained by a drug or by a disease already present
    observed = {o.fact_id: o.words for o in obs
                if not o.doubtful and o.fact_id not in drugs and o.fact_id not in by_present}

    def score(diseases: Iterable[cg.Disease], origin: str) -> _Hyp:
        views = tuple(diseases)
        return _Hyp(views, lr.score_set(list(views), bg, observed=observed, masked=masked), origin)

    null = score((), "none")
    current = [score((states.view(d),), "case") for d in graph.diseases
               if d.branch in (cg.HYPOTHESIZED_DISEASES, cg.CANDIDATE_DISEASES) and d.manifestations]
    best0 = min([null, *current], key=lambda h: (h.energy, h.ids))

    # 1. salience
    salient: dict[str, list[str]] = {}
    for fid in observed:
        if fid not in best0.ranked.explains:
            salient.setdefault(fid, []).append("unexplained")
    for o in obs:
        if o.doubtful:
            salient.setdefault(o.fact_id, []).append("doubtful")
    for d in best0.diseases:
        for m in d.manifestations:
            s = d.state_of(m.manifestation_id)
            if s is not None and s.state == cg.PRESENT and m.typicality == "excluding":
                for e in s.evidence:
                    salient.setdefault(e, []).append(f"contradicts {d.words}")
    target = {f for f, why in salient.items()
              if f in by_id and (f not in best0.ranked.explains or any(w.startswith("contradicts") for w in why))}

    # 2. generation: every reachable profile, then pairs when the best single leaves salient observations
    known = _concept_ids(graph)
    singles = []
    for pid in sorted(_reachable(library, graph, obs, below)):
        p = library.profiles[pid]
        concept = p.concept.curie if p.concept is not None else pid
        if pid in known or concept in known:
            continue
        h = score((states.view(p),), "every_reachable_profile")
        if h.ranked.explains and set(h.ranked.explains) & set(by_id):
            singles.append(h)
    pool = sorted([*current, *singles], key=lambda h: (h.energy, h.ids))
    eligible = [h for h in singles if h.energy < null.energy and h.explains(target)]
    best_single = min([best0, *eligible], key=lambda h: (h.energy, h.ids))
    # unclear: the best single explanation leaves an observation of the case, or contradicts one
    left = set(observed) - set(best_single.ranked.explains)
    contradicted = any(c.note.startswith("excluding manifestation found") for c in best_single.ranked.contributions)
    unclear = bool(left) or contradicted
    pairs: dict[tuple[str, ...], _Hyp] = {}
    wave_list: list[frozenset[str]] = []
    if unclear and pool:
        accountable = sorted(set(observed) | {o.fact_id for o in obs if o.doubtful})

        def best_for(part: set[str], other: _Hyp | None = None) -> _Hyp | None:
            options = [h for h in pool if h is not other and h.explains(part)]
            return min(options, key=lambda h: (-len(h.explains(part)), h.energy, h.ids)) if options else None

        def pair(a: _Hyp, b: _Hyp, origin: str) -> None:
            ids = tuple(sorted(a.ids + b.ids))
            if len(set(ids)) < 2 or ids in pairs:
                return
            only_a = a.explains(set(by_id)) - b.explains(set(by_id))
            only_b = b.explains(set(by_id)) - a.explains(set(by_id))
            if only_a and only_b:
                pairs[ids] = score(a.diseases + b.diseases, origin)

        # the complement of each explanation worth having: who accounts for what it leaves
        for h in [best_single, *eligible]:
            if not h.diseases:
                continue
            partner = best_for((target | left) - set(h.ranked.explains), h)
            if partner is not None:
                pair(h, partner, "complement")
        # the waves: a part of the observations bound together, the rest inhibited
        if len(accountable) >= 2:
            wave_list = waves(accountable, n_waves)
            for t, active in enumerate(wave_list):
                rest = set(accountable) - active
                if not active or not rest:
                    continue
                a, b = best_for(set(active)), best_for(rest)
                if a is not None and b is not None:
                    pair(a, b, f"wave {t}")

    # 3. evaluation and 4. the guard
    def worth(h: _Hyp) -> bool:
        if not (h.energy < null.energy and h.explains(target)):
            return False
        if len(h.diseases) > 1:
            # the second disease must earn its prior: better than each of its diseases alone, and better than the
            # best single explanation of the case (a pair is the insight that simplifies, or it is not proposed)
            alone = [score((d,), "member") for d in h.diseases]
            return h.energy < min(x.energy for x in alone) and h.energy < best_single.energy
        return True

    candidates = [h for h in [*eligible, *pairs.values()] if worth(h)]
    front = []
    for h in candidates:
        # the front of two objectives, how many salient observations it accounts for and its energy: a proposal is
        # dominated by another that accounts for as many with a lower energy, or for more with no higher one
        mine = len(h.explains(target))
        dominated = any(
            other is not h and len(other.explains(target)) >= mine and other.energy <= h.energy
            and (len(other.explains(target)) > mine or other.energy < h.energy)
            for other in candidates)
        if not dominated:
            front.append(h)
    front.sort(key=lambda h: (h.energy, h.ids))
    before = [h.ranked for h in (null, *current)]
    return {
        "status": "ran",
        "observations": len(obs),
        "explained_beforehand": (
            [{"observation": f, "words": by_id[f].words, "by": "drug", **d} for f, ds in drugs.items()
             if f in by_id for d in ds]
            + [{"observation": f, "words": by_id[f].words, "by": "present disease", "disease": w}
               for f, w in by_present.items() if f in by_id]),
        "masked": [{"manifestation": k, "why": v} for k, v in sorted(masked.items())],
        "best_before": {"hypothesis": best0.ranked.words, "energy": round(best0.energy, 4)},
        "null_energy": round(null.energy, 4),
        "salient": [{"observation": f, "words": by_id[f].words if f in by_id else "", "reasons": why}
                    for f, why in sorted(salient.items())],
        "unclear": unclear,
        "left_by_the_best_single": sorted(left),
        "waves": len(wave_list),
        "evaluated": {"singles": len(singles), "pairs": len(pairs)},
        "entropy_bits": {"before": round(lr.entropy_bits(before), 4),
                         "after": round(lr.entropy_bits([*before, *[h.ranked for h in front]]), 4)},
        "accepted": [{
            "hypothesis": h.ranked.words,
            "diseases": list(h.ids),
            "kind": "single" if len(h.diseases) == 1 else "pair",
            "from": h.origin,
            "energy": round(h.energy, 4),
            "delta_energy": round(best0.energy - h.energy, 4),
            "insight": h.energy < best0.energy,
            "explains_salient": sorted(h.explains(target)),
            "status": "hypothesis_to_exclude",
        } for h in front],
        "constants": {"no_match": lr.NO_MATCH, "second_diagnosis": lr.SECOND_DIAGNOSIS, "waves": n_waves},
        # for the ranking of the diagnostic path, which weighs the case the same way
        "observed": observed,
        "masked_keys": masked,
    }


__all__ = ["LATERAL", "WAVES", "Observation", "drug_explanations", "lateral_cycle", "masked_by_drugs",
           "observations", "waves"]
