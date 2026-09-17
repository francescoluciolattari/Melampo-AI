"""Real quantum-cognition formalism (Busemeyer & Bruza, 2012) for predicting order effects in a HUMAN reader's judgment.

Stage A of a three-stage validation plan discussed and agreed on directly:
verify the mathematical machinery is implemented correctly against the
real, published formalism before any claim about clinical usefulness.
Stage B (comparing predictions against existing published anchoring-bias
studies, e.g. the randomized vignette experiment on distracting-feature
position in diagnosis) and Stage C (a purpose-built prospective study) are
not attempted here and require resources -- clinical partnerships, human-
subjects review -- outside what a coding session can produce.

**This module predicts a human reader's potential anchoring bias. It must
never be read to describe, and must never influence, this system's own
diagnostic output.** `tests/test_order_invariance.py` (Part of a separate,
earlier verification) confirms the reasoning pipeline itself is
order-independent by construction, and stays that way -- this module has no
connection to `candidate_retrieval.py`, `differential_ranking.py`, or
`mechanism_enumeration.py`, and is never called from them. It exists purely
to flag, for the human reading a case, a hypothesis about how the order of
findings in the written report might bias their own judgment.

**Why this is not the same as `vector_evolution_engine.py`.** That module
evolves real-valued hypothesis vectors to find similarity between cases --
an ordinary linear-algebra problem, for which real numbers and cosine
similarity are the right and sufficient tool, verified earlier: the
quantum overlap integral reduces exactly to the real dot product for
real-valued vectors, so dressing it in bra-ket notation would have added
nothing. This module is different in kind: it targets order effects
specifically, a phenomenon that classical (commuting) probability provably
cannot produce at all, and quantum probability's non-commuting operators
can. The complex Hilbert space and non-commuting projectors here are not
decoration; they are the entire mechanism being tested.

**A known, honestly-stated limitation of the formalism itself, not of this
implementation.** The quantum-cognition literature itself has not resolved
how to model question-order effects and response replicability (getting
the same answer to a repeated question) simultaneously within standard
Hilbert-space quantum probability -- an open problem, with proposed
extensions ("beyond Hilbert space" models) still under development. This
module models order effects only; it makes no claim about, and should not
be used to predict, response replicability.

References:
- Busemeyer J.R., Bruza P.D. Quantum Models of Cognition and Decision.
  Cambridge University Press, 2012.
- Wang Z., Busemeyer J.R. A quantum question order model supported by
  empirical tests of an a priori and precise prediction. Topics in
  Cognitive Science, 2013.
- Huang J. et al. An overview of the quantum cognition research program.
  Psychonomic Bulletin & Review, 2025.
"""

import math
from dataclasses import dataclass

Complex = complex
Matrix = tuple[tuple[Complex, ...], ...]


def _matmul(a: Matrix, b: Matrix) -> Matrix:
    n = len(a)
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(n)) for j in range(n)) for i in range(n))


def _conjugate_transpose(a: Matrix) -> Matrix:
    n = len(a)
    return tuple(tuple(a[j][i].conjugate() for j in range(n)) for i in range(n))


def _apply(matrix: Matrix, vector: tuple[Complex, ...]) -> tuple[Complex, ...]:
    return tuple(sum(row[j] * vector[j] for j in range(len(vector))) for row in matrix)


def _inner(a: tuple[Complex, ...], b: tuple[Complex, ...]) -> Complex:
    return sum(x.conjugate() * y for x, y in zip(a, b, strict=True))


def _norm(a: tuple[Complex, ...]) -> float:
    return math.sqrt(_inner(a, a).real)


@dataclass(frozen=True)
class QuantumBeliefState:
    """A normalised complex state vector in a finite-dimensional Hilbert space.

    Represents an undetermined belief over competing diagnostic
    hypotheses -- the state has no definite "answer" to a question until a
    projector is applied, matching the Copenhagen-style measurement
    postulate the whole formalism rests on.
    """

    amplitudes: tuple[Complex, ...]

    def __post_init__(self) -> None:
        norm = _norm(self.amplitudes)
        if abs(norm - 1.0) > 1e-9:
            raise ValueError(f"state must be normalised (norm={norm:.6f}); construct via QuantumBeliefState.normalised()")

    @classmethod
    def normalised(cls, amplitudes: tuple[Complex, ...]) -> "QuantumBeliefState":
        norm = _norm(amplitudes)
        if norm == 0.0:
            raise ValueError("cannot normalise a zero vector")
        return cls(tuple(a / norm for a in amplitudes))

    @property
    def dimension(self) -> int:
        return len(self.amplitudes)


@dataclass(frozen=True)
class Projector:
    """A Hermitian, idempotent operator representing one possible answer to a question.

    Verified on construction, not assumed: a matrix that fails either
    property is not a valid quantum-mechanical projector, and using one
    that silently is not would make every probability computed from it
    meaningless without any error surfacing.
    """

    matrix: Matrix

    def __post_init__(self) -> None:
        n = len(self.matrix)
        conjugate_t = _conjugate_transpose(self.matrix)
        for i in range(n):
            for j in range(n):
                if abs(self.matrix[i][j] - conjugate_t[i][j]) > 1e-9:
                    raise ValueError("matrix is not Hermitian: P must equal its own conjugate transpose")
        squared = _matmul(self.matrix, self.matrix)
        for i in range(n):
            for j in range(n):
                if abs(squared[i][j] - self.matrix[i][j]) > 1e-9:
                    raise ValueError("matrix is not idempotent: P*P must equal P for a valid projector")

    def apply(self, state: QuantumBeliefState) -> tuple[Complex, ...]:
        return _apply(self.matrix, state.amplitudes)

    def probability(self, state: QuantumBeliefState) -> float:
        """Born rule: the probability of this outcome, measured directly from the given state."""
        projected = self.apply(state)
        return _inner(projected, projected).real

    def collapse(self, state: QuantumBeliefState) -> QuantumBeliefState:
        """The Lüders postulate: the state after this outcome is observed, renormalised.

        Raises if the outcome has zero probability under the given state --
        there is no state to collapse to for an event that cannot occur,
        and silently returning something would misrepresent that.
        """
        projected = self.apply(state)
        norm = math.sqrt(_inner(projected, projected).real)
        if norm < 1e-12:
            raise ValueError("cannot collapse to a zero-probability outcome")
        return QuantumBeliefState(tuple(a / norm for a in projected))

    def commutes_with(self, other: "Projector") -> bool:
        left = _matmul(self.matrix, other.matrix)
        right = _matmul(other.matrix, self.matrix)
        n = len(self.matrix)
        return all(abs(left[i][j] - right[i][j]) < 1e-9 for i in range(n) for j in range(n))


def sequential_probability(state: QuantumBeliefState, first: Projector, second: Projector) -> float:
    """P(first outcome, then second outcome), asked in that order.

    The quantum law of total probability for a sequence of two questions:
    measure `first`, collapse, then measure `second` on the collapsed
    state. Implemented via the closed form <psi|P_first P_second
    P_first|psi>, equivalent to (and verified against) actually performing
    the collapse step by step.
    """
    collapsed = first.collapse(state)
    p_first = first.probability(state)
    p_second_given_first = second.probability(collapsed)
    return p_first * p_second_given_first


def order_effect(state: QuantumBeliefState, question_a: Projector, question_b: Projector) -> float:
    """How much asking A before B changes the result, versus asking B before A.

    Zero exactly when the two questions' projectors commute -- classical
    (Kolmogorovian) probability is the special case where every question
    commutes with every other, which is why classical probability can never
    produce a genuine order effect. Non-zero precisely measures the
    non-commutativity the two questions exhibit, in the units of the
    probability itself.
    """
    return sequential_probability(state, question_a, question_b) - sequential_probability(state, question_b, question_a)


def rotation_projector_pair(theta: float) -> tuple[Projector, Projector]:
    """A textbook two-dimensional projector pair for a binary question, rotated by `theta`
    relative to the standard basis.

    theta=0 gives the standard {|0>, |1>} basis (commutes with anything
    diagonal in that basis); any other angle gives a genuinely different,
    non-commuting basis -- the simplest possible construction that lets two
    questions be "incompatible" in the quantum-cognition sense, matching
    the rotation-angle parameterisation Wang & Busemeyer (2013) use to fit
    real survey data.
    """
    c, s = math.cos(theta), math.sin(theta)
    yes = ((c * c, c * s), (c * s, s * s))
    no = ((s * s, -c * s), (-c * s, c * c))
    return Projector(yes), Projector(no)
