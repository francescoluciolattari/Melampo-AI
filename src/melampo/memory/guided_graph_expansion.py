"""Model-guided graph exploration -- an optional fallback, never a replacement.

`spreading_activation.spread()` is deterministic: same graph, same origins,
same result, every time, and the whole reason a relevance question routes to
the graph rather than to free text is that determinism. This module gives that
up, deliberately and only when asked: a root model chooses, turn by turn,
which neighbour to follow, when the constrained deterministic spread already
found nothing above threshold and there is nothing left to lose by trying
something less certain.

The trade was evaluated directly rather than assumed. What it might gain:
adapting the search to the specific question instead of applying the same
fixed decay and threshold to every one, and bringing whatever medical
plausibility a model carries to bear on which direction is worth following.
What it costs, and each of these is either already measured elsewhere in this
project or a known failure mode in clinical decision support: determinism
(two runs of the identical question can visit different concepts, especially
with the sub-model non-determinism already observed in this project's own
bench runs); confirmation bias (a model choosing where to look can stop at the
first thing that fits, where the constrained spread has no such preference and
explores every admitted direction until the threshold stops it); cost and
latency (one model call per hop, against a spread that is a pure function
costing milliseconds); and a much larger surface to test (spreading_activation
has 20 tests against one deterministic behaviour; a model-guided walk's
behaviour depends on whatever model is calling it, which can change without
this code changing).

Because of that cost, this module makes no decision about *when* it runs.
That decision belongs to the caller -- see `mechanism_verification.verify_mechanism`'s
`fallback_model` parameter -- and every result this module returns marks
itself as coming from a guided walk rather than the deterministic pass, so a
reader downstream is never left thinking a lucky guess was a proof.

What is preserved even here: the model can never invent an edge. Every move
is a choice among the graph's *actual* neighbours of the current concept,
offered through a closed, named action grammar -- `neighbor(concept)`,
`final(concept)`, `give_up()` -- never a code-execution surface. The
determinism given up is which real edges get visited, never whether an edge
is real.
"""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from .concept_paths import ConceptGraphView, normalise_concept
from .spreading_activation import DEFAULT_ALLOWED_RELATIONS

VERB_NEIGHBOR = "neighbor"
VERB_FINAL = "final"
VERB_GIVE_UP = "give_up"

STOP_FINAL = "final"
STOP_GAVE_UP = "gave_up"
STOP_ITERATIONS = "iteration_budget_exhausted"
STOP_NO_ACTION = "model_emitted_no_action"
STOP_DEAD_END = "no_neighbours_to_offer"

_ACTION_RE = re.compile(r"^\s*(neighbor|final|give_up)\s*\(([^)]*)\)\s*$", re.IGNORECASE)

DEFAULT_MAX_HOPS = 6


@dataclass(frozen=True)
class GuidedExpansionResult:
    """What a model-guided walk found, distinct in kind from a deterministic spread.

    `via_guided_expansion` is always True on this type -- it exists so a
    caller merging this with `SpreadingResult` output never has to remember to
    tag it separately; the type itself carries the provenance.
    """

    final_concept: str | None
    visited_path: tuple[str, ...]
    stop_reason: str
    hops: int
    via_guided_expansion: bool = field(default=True, init=False)

    @property
    def found_something(self) -> bool:
        return self.stop_reason == STOP_FINAL and self.final_concept is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "final_concept": self.final_concept,
            "visited_path": list(self.visited_path),
            "stop_reason": self.stop_reason,
            "hops": self.hops,
            "via_guided_expansion": self.via_guided_expansion,
            "found_something": self.found_something,
        }


def _neighbours(
    graph: ConceptGraphView, concept: str, allowed_relations: frozenset[str]
) -> list[tuple[str, str, float]]:
    """Real neighbours only, filtered to admitted relation types.

    The same relation-type constraint spread() applies -- a guided walk is
    still not free to wander through every edge type, only the ones a
    relevance question's mechanism is allowed to run along.
    """
    results = []
    for edge in graph.edges_from(concept):
        base_relation = edge.relation.removeprefix("inverse_")
        if base_relation in allowed_relations:
            results.append((normalise_concept(edge.target), edge.relation, edge.weight))
    return results


def _format_neighbours(neighbours: list[tuple[str, str, float]]) -> str:
    if not neighbours:
        return "(none)"
    return "\n".join(f"  - {relation} -> {target} (weight {weight:.2f})" for target, relation, weight in neighbours)


def _prompt(target: str, current: str, path: tuple[str, ...], neighbours: list[tuple[str, str, float]]) -> str:
    return (
        f"You are looking for a concept that connects to both '{target}' and the concept you started "
        f"from, by following real edges in a knowledge graph. You are currently at '{current}'.\n"
        f"Path so far: {' -> '.join(path)}\n"
        f"Neighbours of '{current}', reached by an allowed relation:\n{_format_neighbours(neighbours)}\n\n"
        f"Emit exactly one action, in this format, nothing else:\n"
        f"  neighbor(<one of the neighbours listed above>) -- move there\n"
        f"  final(<concept>) -- declare this concept as the mediating connection\n"
        f"  give_up() -- no direction here looks like it leads anywhere useful"
    )


def guided_expand(
    graph: ConceptGraphView,
    start: str,
    target: str,
    root_model: Callable[[str], str],
    *,
    allowed_relations: Iterable[str] | None = None,
    max_hops: int = DEFAULT_MAX_HOPS,
) -> GuidedExpansionResult:
    """Let a model choose, hop by hop, which real edge to follow from `start`.

    Not a general-purpose graph agent: bounded to `max_hops`, offered only the
    current concept's actual neighbours each turn, and the caller decides
    whether to invoke this at all -- see the module docstring for why that
    decision is deliberately kept external.
    """
    allowed = frozenset(allowed_relations) if allowed_relations is not None else DEFAULT_ALLOWED_RELATIONS
    current = normalise_concept(start)
    target_norm = normalise_concept(target)
    path: tuple[str, ...] = (current,)
    visited: set[str] = {current}

    for hop in range(max_hops):
        neighbours = [item for item in _neighbours(graph, current, allowed) if item[0] not in visited]
        if not neighbours:
            return GuidedExpansionResult(None, path, STOP_DEAD_END, hop)

        output = root_model(_prompt(target_norm, current, path, neighbours))
        match = _ACTION_RE.match(output or "")
        if not match:
            return GuidedExpansionResult(None, path, STOP_NO_ACTION, hop)

        verb, arg = match.group(1).lower(), match.group(2).strip()
        if verb == VERB_GIVE_UP:
            return GuidedExpansionResult(None, path, STOP_GAVE_UP, hop)
        if verb == VERB_FINAL:
            return GuidedExpansionResult(normalise_concept(arg) or current, path, STOP_FINAL, hop)
        if verb == VERB_NEIGHBOR:
            chosen = normalise_concept(arg)
            valid = {item[0] for item in neighbours}
            if chosen not in valid:
                # A move to a concept not actually offered is not an invented
                # edge taken -- it is an ill-formed move, treated the same as
                # any other action this walk cannot make sense of.
                return GuidedExpansionResult(None, path, STOP_NO_ACTION, hop)
            current = chosen
            visited.add(current)
            path = (*path, current)

    return GuidedExpansionResult(None, path, STOP_ITERATIONS, max_hops)
