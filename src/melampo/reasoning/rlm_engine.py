"""Recursive retrieval over the case environment, without executing code.

The recursive-language-model literature runs a root model that writes Python
in a REPL to navigate its context. This engine keeps the recursion and drops
the REPL. The root model emits **invocations of named primitives** — grep,
slice, search, expand, query, final — parsed from a strict line format and
dispatched against `ContextEnvironment`. No string the model produces is ever
evaluated as code.

That is the sandbox, and it is stronger than any sandbox around `exec`: there
is nothing to escape from. A dispatcher that knows six verbs and validates
their arguments has no filesystem to read, no network to reach, and no way to
be argued into acquiring either. The security review reduces to reading the
dispatcher, which is a page.

What the recursion keeps: the root decides *where to look* rather than
receiving a fixed top-k, iterates until it has what it needs, and may hand a
fragment to a sub-model for extraction. Coverage is measured by the
environment's ledger rather than assumed. Depth is capped at one and cannot be
raised: the literature reports fifteen points lost on simple retrieval at depth
one and thirty at depth two, with latency rising from seconds to minutes.

Three constraints are enforced in code rather than left to the operator:

- **Data class.** Every document must be marked synthetic or de-identified.
  The environment holds the raw case, and in phase one nothing real may enter
  it. A comment in a guide does not stop anyone; a refused constructor does.
- **Budget failure is explicit.** Exhausting iterations or wall clock ends the
  run with a named reason and no result, never a partial result presented as
  complete. A truncated dossier that looks whole is worse than none.
- **FINAL is required.** A run that stops without the model declaring
  completion is recorded as such, because "the model stopped emitting actions"
  and "the model finished" are different outcomes that look identical to a
  consumer downstream.

The trajectory — every action, every fragment returned with its offsets, every
sub-model call, the budget consumed and the termination reason — is a health
record when the case is real and is written to the audit store as one.
"""

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..memory.context_environment import (
    ContextEnvironment,
    EnvironmentDocument,
    Fragment,
)
from ..memory.retrieval_contract import RETRIEVAL_MODE_RLM

DATA_CLASS_SYNTHETIC = "synthetic"
DATA_CLASS_DEIDENTIFIED = "deidentified"
DATA_CLASS_REAL = "real"
PHASE_ONE_DATA_CLASSES = frozenset({DATA_CLASS_SYNTHETIC, DATA_CLASS_DEIDENTIFIED})

VERB_DESCRIBE = "describe"
VERB_GREP = "grep"
VERB_SLICE = "slice"
VERB_SEARCH = "search"
VERB_EXPAND = "expand"
VERB_QUERY = "query"
VERB_FINAL = "final"
VERBS = frozenset({VERB_DESCRIBE, VERB_GREP, VERB_SLICE, VERB_SEARCH, VERB_EXPAND, VERB_QUERY, VERB_FINAL})

MAX_DEPTH = 1

STOP_FINAL = "final"
STOP_ITERATIONS = "iteration_budget_exhausted"
STOP_WALL_CLOCK = "wall_clock_budget_exhausted"
STOP_NO_ACTION = "model_emitted_no_action"
STOP_ROOT_ERROR = "root_model_error"

_ACTION_LINE = re.compile(r"^\s*(describe|grep|slice|search|expand|query|final)\s*\((.*)\)\s*$", re.IGNORECASE)


class DataClassViolation(ValueError):
    """Raised when a document of the wrong data class is offered to the engine."""


@dataclass(frozen=True)
class Action:
    """One primitive invocation, parsed and validated before dispatch."""

    verb: str
    args: tuple[str, ...]
    raw: str

    def as_dict(self) -> dict[str, Any]:
        return {"verb": self.verb, "args": list(self.args)}


def parse_actions(text: str) -> tuple[list[Action], list[str]]:
    """Extract primitive invocations from the model's output, one per line.

    Anything that is not a well-formed invocation is returned separately as
    ignored, not silently dropped: a model that emits prose instead of actions
    is a diagnostic signal, and the trajectory records it.
    """
    actions: list[Action] = []
    ignored: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _ACTION_LINE.match(stripped)
        if match is None:
            ignored.append(stripped)
            continue
        verb = match.group(1).lower()
        args = tuple(_split_args(match.group(2)))
        actions.append(Action(verb=verb, args=args, raw=stripped))
    return actions, ignored


def _split_args(raw: str) -> list[str]:
    """Split comma-separated arguments, honouring quotes."""
    parts: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for character in raw:
        if quote:
            if character == quote:
                quote = None
            else:
                current.append(character)
        elif character in ("'", '"'):
            quote = character
        elif character == ",":
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    tail = "".join(current).strip()
    if tail or parts:
        parts.append(tail)
    return [part for part in parts if part != ""]


@dataclass
class Budget:
    """Iteration and wall-clock limits, with explicit exhaustion."""

    max_iterations: int = 12
    max_wall_clock_seconds: float = 60.0
    max_sub_model_calls: int = 8
    started_at: float = field(default_factory=time.monotonic)
    iterations: int = 0
    sub_model_calls: int = 0

    def exhausted(self) -> str | None:
        if self.iterations >= self.max_iterations:
            return STOP_ITERATIONS
        if time.monotonic() - self.started_at >= self.max_wall_clock_seconds:
            return STOP_WALL_CLOCK
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "max_iterations": self.max_iterations,
            "sub_model_calls": self.sub_model_calls,
            "max_sub_model_calls": self.max_sub_model_calls,
            "elapsed_seconds": round(time.monotonic() - self.started_at, 3),
            "max_wall_clock_seconds": self.max_wall_clock_seconds,
        }


@dataclass
class TrajectoryStep:
    iteration: int
    action: Action
    result_summary: str
    fragments: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "action": self.action.as_dict(),
            "result_summary": self.result_summary,
            "fragment_count": len(self.fragments),
            "fragments": self.fragments,
            "error": self.error,
        }


@dataclass
class Trajectory:
    """Everything that happened in a run. A health record when the case is real."""

    case_id: str
    data_class: str
    depth: int
    steps: list[TrajectoryStep] = field(default_factory=list)
    ignored_lines: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    final_answer: str | None = None
    coverage: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)

    @property
    def completed(self) -> bool:
        return self.stop_reason == STOP_FINAL

    def evidence(self) -> list[dict[str, Any]]:
        """Every fragment the run touched, deduplicated, in retrieval-contract shape."""
        seen: set[str] = set()
        items: list[dict[str, Any]] = []
        for step in self.steps:
            for fragment in step.fragments:
                key = fragment.get("record_id", "")
                if key and key not in seen:
                    seen.add(key)
                    items.append(fragment)
        for rank, item in enumerate(items, start=1):
            item["rank"] = rank
        return items

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "data_class": self.data_class,
            "depth": self.depth,
            "completed": self.completed,
            "stop_reason": self.stop_reason,
            "final_answer": self.final_answer,
            "steps": [step.as_dict() for step in self.steps],
            "ignored_lines": list(self.ignored_lines),
            "coverage": dict(self.coverage),
            "budget": dict(self.budget),
            "health_data": self.data_class == DATA_CLASS_REAL,
        }


RootModel = Callable[[str], str]
SubModel = Callable[[str, str], str]


@dataclass
class RlmEngine:
    """The recursive loop. Prompt → actions → dispatch → repeat, until FINAL or budget.

    ``root_model`` receives the prompt and returns text; ``sub_model`` receives a
    question and a fragment and returns an answer. Both are plain callables so
    the engine is testable with scripted stand-ins and wired to real clients
    without changing anything here.
    """

    root_model: RootModel
    sub_model: SubModel | None = None
    depth: int = 1
    allowed_data_classes: frozenset[str] = PHASE_ONE_DATA_CLASSES
    fragment_limit: int = 6

    def __post_init__(self) -> None:
        if self.depth < 0 or self.depth > MAX_DEPTH:
            raise ValueError(f"depth must be 0 or {MAX_DEPTH}; {self.depth} is not permitted")
        if self.depth == 0:
            self.sub_model = None

    def run(
        self,
        case_id: str,
        documents: Sequence[EnvironmentDocument],
        question: str,
        *,
        budget: Budget | None = None,
        search_fn: Any = None,
        graph_expand_fn: Any = None,
    ) -> Trajectory:
        data_class = self._check_data_class(documents)
        environment = ContextEnvironment.from_documents(
            documents, search_fn=search_fn, graph_expand_fn=graph_expand_fn
        )
        budget = budget or Budget()
        trajectory = Trajectory(case_id=case_id, data_class=data_class, depth=self.depth)
        history: list[str] = []

        while True:
            exhausted = budget.exhausted()
            if exhausted is not None:
                trajectory.stop_reason = exhausted
                break

            prompt = self._prompt(question, environment, history)
            try:
                output = self.root_model(prompt)
            except Exception as error:  # noqa: BLE001 - recorded as the run's outcome
                trajectory.stop_reason = STOP_ROOT_ERROR
                history.append(f"root model error: {error}")
                break

            actions, ignored = parse_actions(output)
            trajectory.ignored_lines.extend(ignored)
            if not actions:
                trajectory.stop_reason = STOP_NO_ACTION
                break

            finished = False
            for action in actions:
                budget.iterations += 1
                step = self._dispatch(action, environment, budget, trajectory)
                trajectory.steps.append(step)
                history.append(f"{action.raw} -> {step.result_summary}")
                if action.verb == VERB_FINAL:
                    trajectory.final_answer = action.args[0] if action.args else ""
                    trajectory.stop_reason = STOP_FINAL
                    finished = True
                    break
                if budget.exhausted():
                    break
            if finished:
                break

        trajectory.coverage = environment.coverage()
        trajectory.budget = budget.as_dict()
        return trajectory

    def to_retrieval_payload(self, trajectory: Trajectory, question: str) -> dict[str, Any]:
        """Render a completed trajectory in the shared retrieval contract.

        A run that did not complete yields an empty evidence list with the stop
        reason attached, never the partial fragments as if they were the answer.
        """
        evidence = trajectory.evidence() if trajectory.completed else []
        grounding = [float(item.get("grounding_score", 0.0) or 0.0) for item in evidence]
        return {
            "query": question,
            "focus": "recursive",
            "target_areas": [],
            "status": "grounded_retrieval_ready" if evidence else "insufficient_grounded_evidence",
            "retrieval_mode": RETRIEVAL_MODE_RLM,
            "evidence": evidence,
            "evidence_count": len(evidence),
            "retrieval_quality": {
                "memory_backed": True,
                "coverage": float(trajectory.coverage.get("coverage_ratio", 0.0)),
                "coverage_basis": "corpus_characters",
                "mean_grounding_score": round(sum(grounding) / len(grounding), 3) if grounding else 0.0,
                "fallback_used": False,
                "stop_reason": trajectory.stop_reason,
                "completed": trajectory.completed,
            },
            "case_context_keys": [],
        }

    def _check_data_class(self, documents: Sequence[EnvironmentDocument]) -> str:
        classes = {str(document.metadata.get("data_class", "")).lower() for document in documents}
        if not documents:
            raise DataClassViolation("no documents supplied")
        unmarked = "" in classes
        if unmarked:
            raise DataClassViolation(
                "every document must declare data_class; unmarked documents are refused because "
                "the environment holds the raw case and phase one admits only synthetic or de-identified text"
            )
        forbidden = classes - self.allowed_data_classes
        if forbidden:
            raise DataClassViolation(
                f"data_class {sorted(forbidden)} is not permitted for this engine; allowed: "
                f"{sorted(self.allowed_data_classes)}"
            )
        return next(iter(classes)) if len(classes) == 1 else DATA_CLASS_DEIDENTIFIED

    def _prompt(self, question: str, environment: ContextEnvironment, history: Sequence[str]) -> str:
        verbs = "describe(), grep(pattern), slice(document_id, start, end), search(query), expand(concept)"
        if self.depth >= 1 and self.sub_model is not None:
            verbs += ", query(question, document_id, start, end)"
        recent = "\n".join(history[-8:]) if history else "(none yet)"
        return (
            f"Question: {question}\n"
            f"Environment: {environment.describe()['document_count']} document(s), "
            f"{environment.describe()['total_characters']} characters.\n"
            f"Available actions, one per line: {verbs}, final(answer).\n"
            f"Emit final(answer) when done. Recent actions:\n{recent}\n"
        )

    def _dispatch(
        self, action: Action, environment: ContextEnvironment, budget: Budget, trajectory: Trajectory
    ) -> TrajectoryStep:
        iteration = budget.iterations
        try:
            if action.verb == VERB_DESCRIBE:
                described = environment.describe()
                return TrajectoryStep(iteration, action, f"{described['document_count']} document(s)")
            if action.verb == VERB_GREP:
                fragments = environment.grep(_arg(action, 0), limit=self.fragment_limit)
                return self._fragment_step(iteration, action, fragments, "grep")
            if action.verb == VERB_SLICE:
                fragment = environment.slice(_arg(action, 0), int(_arg(action, 1)), int(_arg(action, 2)))
                return self._fragment_step(iteration, action, [fragment], "slice")
            if action.verb == VERB_SEARCH:
                fragments = environment.search(_arg(action, 0), limit=self.fragment_limit)
                return self._fragment_step(iteration, action, fragments, "search")
            if action.verb == VERB_EXPAND:
                hits = environment.graph_expand(_arg(action, 0))
                return TrajectoryStep(iteration, action, f"{len(hits)} graph neighbour(s)")
            if action.verb == VERB_QUERY:
                return self._query_step(iteration, action, environment, budget)
            if action.verb == VERB_FINAL:
                return TrajectoryStep(iteration, action, "final")
        except (KeyError, ValueError, IndexError) as error:
            return TrajectoryStep(iteration, action, "error", error=str(error))
        return TrajectoryStep(iteration, action, "unknown verb", error=f"unknown verb {action.verb}")

    def _query_step(
        self, iteration: int, action: Action, environment: ContextEnvironment, budget: Budget
    ) -> TrajectoryStep:
        if self.depth < 1 or self.sub_model is None:
            return TrajectoryStep(iteration, action, "refused", error="query is not available at depth 0")
        if budget.sub_model_calls >= budget.max_sub_model_calls:
            return TrajectoryStep(iteration, action, "refused", error="sub-model call budget exhausted")
        fragment = environment.slice(_arg(action, 1), int(_arg(action, 2)), int(_arg(action, 3)))
        budget.sub_model_calls += 1
        environment.ledger.record_llm_call()
        answer = self.sub_model(_arg(action, 0), fragment.text)
        step = self._fragment_step(iteration, action, [fragment], "query")
        step.result_summary = f"query -> {answer[:120]}"
        return step

    def _fragment_step(
        self, iteration: int, action: Action, fragments: Sequence[Fragment], label: str
    ) -> TrajectoryStep:
        rendered = [fragment.as_evidence(rank=index, focus="recursive") for index, fragment in enumerate(fragments, 1)]
        return TrajectoryStep(iteration, action, f"{label}: {len(fragments)} fragment(s)", fragments=rendered)


def _arg(action: Action, index: int) -> str:
    if index >= len(action.args):
        raise ValueError(f"{action.verb} requires at least {index + 1} argument(s)")
    return action.args[index]
