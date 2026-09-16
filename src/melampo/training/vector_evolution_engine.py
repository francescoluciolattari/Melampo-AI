"""Evolve hypothesis vectors continuously over time, and surface cross-case correlations by real overlap.

Built to replace a request for something else entirely: a "neuro-quantum"
layer using the Schrödinger equation and Dirac notation to evolve diagnostic
hypothesis vectors and measure their "quantum overlap". Verified and
rejected on two independent grounds, not merely a naming preference.

**Mathematically, it would not have added anything.** The quantum overlap
integral, `<psi_A|psi_B> = integral(psi_A*(x) psi_B(x) dx)`, is the inner
product of two functions. For real-valued vectors -- which hypothesis
embeddings are, here and everywhere else in this project -- that inner
product IS the dot product, and normalised, IS cosine similarity: the exact
operation `vector_memory.py`'s embedding infrastructure and the
normalisation cascade's tier 2 (SapBERT) already use. Writing it in bra-ket
notation would not compute anything different; it would only imply a
physical process, complex phase, and superposition that are not present in
what is actually a similarity score between two ordinary vectors.

**Physically, the constant does not apply.** The time-dependent Schrödinger
equation, `i*hbar * d|psi>/dt = H|psi>`, describes how a real physical
system's quantum state evolves under a Hamiltonian representing that
system's actual energy. hbar (~1.0546e-34 J*s) is a physical constant of the
universe, not a free parameter to repurpose. There is no Hamiltonian for "a
diagnostic hypothesis" and no physical energy being conserved when one
updates -- using the equation here would borrow a constant with a real
physical meaning to dress up an operation that has none.

**What is real, and built here instead**: a continuous-time evolution rule
for hypothesis vectors, drawn from an actual, citable, non-metaphorical
model of how biological systems integrate and forget input over time -- the
leaky integrator (Dayan & Abbott, "Theoretical Neuroscience", 2001, the
standard model for how a neuron's membrane potential accumulates synaptic
input while decaying toward baseline) -- plus Hebbian co-reinforcement
(Hebb, "The Organization of Behavior", 1949: cells that fire together wire
together) for the specific capability requested: strengthening a
correlation between two cases' hypothesis vectors once it is clinically
confirmed, so the space itself improves at surfacing similar correlations
later. Both are real, well-established, cited models; neither claims to be
more than linear algebra.
"""

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..memory.encrypted_store import EncryptedJsonlStore

Vector = tuple[float, ...]

EVENT_UPDATE = "update"
EVENT_REINFORCE = "reinforce"


def _dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def _norm(a: Vector) -> float:
    return math.sqrt(_dot(a, a))


def cosine_overlap(a: Vector, b: Vector) -> float:
    """The real-valued equivalent of the quantum overlap integral <psi_A|psi_B>.

    For real vectors, the inner product asked for in the original request
    and the cosine similarity already used elsewhere in this project
    (`vector_memory.py`, the normalisation cascade's SapBERT tier) are the
    same computation. Returns 0.0 for a zero vector rather than dividing by
    zero -- an unformed hypothesis has no defined direction to overlap with
    anything.

    The same reading applies as was described for quantum overlap:
    1.0 means the two vectors point the same way (the same hypothesis,
    in the same "phase" of reasoning); 0.0 means they share no direction
    at all (independent, unrelated hypotheses).
    """
    norm_a, norm_b = _norm(a), _norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return _dot(a, b) / (norm_a * norm_b)


def leaky_integrate(previous: Vector, evidence: Vector, *, elapsed_seconds: float, tau_seconds: float) -> Vector:
    """Evolve a vector toward new evidence, decaying old state exponentially with elapsed time.

    The closed-form solution of the leaky integrator's governing equation,
    dv/dt = -(v - input) / tau: state decays toward whatever input is held
    constant, at a rate set by the time constant `tau`. Used here exactly as
    it is used for a neuron's membrane potential -- old belief persists
    unless reinforced, and reinforcement pulls the vector toward new
    evidence rather than replacing it outright, so one new case does not
    erase everything the vector previously represented.

    `elapsed_seconds=0` returns `previous` unchanged: no time has passed, so
    there is nothing to decay. A very large `elapsed_seconds` relative to
    `tau` converges on `evidence` itself: old state that decayed away
    entirely, with the vector now fully whatever the new evidence says.
    """
    if elapsed_seconds <= 0:
        return previous
    decay = math.exp(-elapsed_seconds / tau_seconds)
    return tuple(old * decay + new * (1.0 - decay) for old, new in zip(previous, evidence, strict=True))


def hebbian_reinforce(a: Vector, b: Vector, *, learning_rate: float = 0.1) -> tuple[Vector, Vector]:
    """Pull two confirmed-correlated vectors toward each other, symmetrically.

    Applied only when a correlation this space surfaced is independently
    confirmed (Part VI's promotion cycle) -- not on every comparison, which
    would let noise reshape the space. `learning_rate` bounds how much one
    confirmation can move a vector, the same reason a promoted graph edge
    requires multiple independent confirmations before it is written: no
    single event should be able to overwrite what many cases have built.
    """
    midpoint = tuple((x + y) / 2.0 for x, y in zip(a, b, strict=True))
    new_a = tuple(old + learning_rate * (mid - old) for old, mid in zip(a, midpoint, strict=True))
    new_b = tuple(old + learning_rate * (mid - old) for old, mid in zip(b, midpoint, strict=True))
    return new_a, new_b


@dataclass(frozen=True)
class HypothesisVector:
    """One hypothesis's current position in the evolving vector space."""

    case_id: str
    hypothesis: str
    vector: Vector
    last_updated: float

    @property
    def key(self) -> str:
        return f"{self.case_id}::{self.hypothesis}"


@dataclass(frozen=True)
class CrossCaseCorrelation:
    """Two hypotheses, from different cases, whose vectors overlap above threshold."""

    case_a: str
    hypothesis_a: str
    case_b: str
    hypothesis_b: str
    overlap: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_a": self.case_a,
            "hypothesis_a": self.hypothesis_a,
            "case_b": self.case_b,
            "hypothesis_b": self.hypothesis_b,
            "overlap": round(self.overlap, 4),
        }


@dataclass
class HypothesisVectorSpace:
    """The evolving space of hypothesis vectors across every case, with an append-only event log.

    Persisted as events, not as current state: each update or reinforcement
    is appended, never overwritten, the same discipline `graph_store` and
    `term_history` already apply to the project's other irreplaceable data.
    Current state is *derived* by replaying the log, which keeps the store
    itself simple (append only) while the space genuinely evolves -- the
    two are not in tension once state is a fold over history rather than a
    field mutated in place.

    Backed by `EncryptedJsonlStore`: a hypothesis vector is derived from a
    real patient case, and in an MDR-regulated deployment that earns the
    same protection at rest UMLS-licensed data already has, not a lesser
    standard because the content is numeric rather than textual.
    """

    store: EncryptedJsonlStore
    tau_seconds: float = 86_400.0  # default: a case's influence roughly halves in life-min(0.5)*tau =~ 60000s ~ 16.7h; tune per deployment
    _vectors: dict[str, HypothesisVector] | None = field(default=None, repr=False)

    def _ensure_loaded(self) -> dict[str, HypothesisVector]:
        if self._vectors is None:
            vectors: dict[str, HypothesisVector] = {}
            for record in self.store.load():
                self._apply_event(vectors, record)
            self._vectors = vectors
        return self._vectors

    def _apply_event(self, vectors: dict[str, HypothesisVector], record: dict[str, Any]) -> None:
        key = f"{record['case_id']}::{record['hypothesis']}"
        timestamp = float(record["timestamp"])
        if record["event"] == EVENT_UPDATE:
            evidence = tuple(record["vector"])
            existing = vectors.get(key)
            if existing is None:
                vectors[key] = HypothesisVector(record["case_id"], record["hypothesis"], evidence, timestamp)
            else:
                elapsed = timestamp - existing.last_updated
                evolved = leaky_integrate(existing.vector, evidence, elapsed_seconds=elapsed, tau_seconds=self.tau_seconds)
                vectors[key] = HypothesisVector(record["case_id"], record["hypothesis"], evolved, timestamp)
        elif record["event"] == EVENT_REINFORCE:
            other_key = record["other_key"]
            if key in vectors and other_key in vectors:
                new_self, new_other = hebbian_reinforce(
                    vectors[key].vector, vectors[other_key].vector, learning_rate=record.get("learning_rate", 0.1)
                )
                vectors[key] = HypothesisVector(vectors[key].case_id, vectors[key].hypothesis, new_self, timestamp)
                vectors[other_key] = HypothesisVector(
                    vectors[other_key].case_id, vectors[other_key].hypothesis, new_other, timestamp
                )

    def update(self, case_id: str, hypothesis: str, vector: Sequence[float], *, now: float | None = None) -> HypothesisVector:
        """Record new evidence for a hypothesis, evolving its vector via the leaky integrator.

        The event is appended before the in-memory index is updated, so a
        crash between the two leaves the log as the single source of truth
        -- replaying it on the next load reaches the same state.
        """
        timestamp = now if now is not None else time.time()
        record = {"event": EVENT_UPDATE, "case_id": case_id, "hypothesis": hypothesis, "vector": list(vector), "timestamp": timestamp}
        self.store.append(record)
        vectors = self._ensure_loaded()
        self._apply_event(vectors, record)
        return vectors[f"{case_id}::{hypothesis}"]

    def reinforce(self, key_a: str, key_b: str, *, learning_rate: float = 0.1, now: float | None = None) -> None:
        """Confirm a cross-case correlation, pulling the two vectors toward each other.

        Called only from the promotion cycle (Part VI), once a correlation
        this space surfaced has been independently confirmed -- never on an
        unconfirmed comparison, for the reason `hebbian_reinforce` states.
        """
        timestamp = now if now is not None else time.time()
        case_a, hypothesis_a = key_a.split("::", 1)
        record = {
            "event": EVENT_REINFORCE,
            "case_id": case_a,
            "hypothesis": hypothesis_a,
            "other_key": key_b,
            "learning_rate": learning_rate,
            "timestamp": timestamp,
        }
        self.store.append(record)
        vectors = self._ensure_loaded()
        self._apply_event(vectors, record)

    def overlap(self, key_a: str, key_b: str) -> float | None:
        vectors = self._ensure_loaded()
        if key_a not in vectors or key_b not in vectors:
            return None
        return cosine_overlap(vectors[key_a].vector, vectors[key_b].vector)

    def find_cross_case_correlations(self, *, min_overlap: float = 0.85) -> list[CrossCaseCorrelation]:
        """Every pair of hypotheses from *different* cases whose vectors overlap above threshold.

        Restricted to different cases deliberately: two hypotheses within
        the same case overlapping is expected and uninteresting -- the
        capability this method exists for is surfacing a connection between
        cases nobody has reason to compare by hand, not confirming that a
        case's own differential is internally coherent.
        """
        vectors = list(self._ensure_loaded().values())
        found: list[CrossCaseCorrelation] = []
        for i, first in enumerate(vectors):
            for second in vectors[i + 1 :]:
                if first.case_id == second.case_id:
                    continue
                score = cosine_overlap(first.vector, second.vector)
                if score >= min_overlap:
                    found.append(
                        CrossCaseCorrelation(first.case_id, first.hypothesis, second.case_id, second.hypothesis, score)
                    )
        found.sort(key=lambda item: -item.overlap)
        return found

    def __len__(self) -> int:
        return len(self._ensure_loaded())
