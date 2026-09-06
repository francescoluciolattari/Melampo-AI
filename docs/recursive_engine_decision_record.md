# Recursive Engine Decision Record

Block C. How the recursive retrieval loop runs, what it may execute, and how
its trajectories are kept. Companion to `rlm_on_memory_decision_record.md`,
using the same three-tier status convention.

---

## 1. The decision that shapes everything else: no code execution

**Type: CONSTRAINT — accepted.**

The recursive-language-model literature runs a root model that writes Python in
a REPL. This engine keeps the recursion and drops the REPL. The root model emits
**invocations of named primitives** — `describe`, `grep`, `slice`, `search`,
`expand`, `query`, `final` — parsed from a strict one-per-line format and
dispatched against `ContextEnvironment`. No string the model produces is ever
evaluated.

### Why this is the sandbox, and stronger than one

A sandbox around `exec` is a boundary to defend: every capability the
interpreter has is a capability to remove, and the list of things to remove is
never provably complete. A dispatcher that knows six verbs and validates their
arguments has no filesystem to read, no network to reach, and no way to be
argued into acquiring either. **There is nothing to escape from.** The security
review reduces to reading the dispatcher, which is one page.

The cost is expressiveness. A model writing Python can compose arbitrary logic
between retrievals; a model emitting primitives can only sequence them. That
cost is accepted deliberately, and it is smaller than it looks: `ContextEnvironment`
was written to expose typed primitives rather than a REPL precisely so that
navigation would not require composition, and the loop's own iteration
supplies the sequencing.

### What is refused, by construction

```
exec(print(1))          -> not an action, recorded as ignored
eval('1+1')             -> not an action
open('/etc/passwd')     -> not an action
__import__('os')        -> not an action
```

None of these is *caught*. They are simply not in the grammar, so they cannot
be dispatched. Ignored lines are recorded in the trajectory rather than dropped:
a model that emits prose instead of actions is a diagnostic signal, and the
record keeps it.

---

## 2. Three constraints enforced in code

**Type: CONSTRAINT — accepted.**

Each of these was a sentence in a guide before. A sentence in a guide stops no
one; a refused call does.

### Data class

Every document must declare `data_class`. Unmarked documents are refused; real
data is refused in phase one. The environment holds the raw case, and phase one
admits only synthetic or de-identified text. Widening to real data requires
constructing the engine with an explicit allow-set — a visible, greppable act,
not a default that drifts.

### Budget exhaustion is explicit

Exhausting iterations or wall clock ends the run with a named stop reason and
**no result**, never a partial result presented as complete. A truncated
dossier that looks whole is worse than none, because downstream nothing
distinguishes it from a finished one.

### `final()` is required

A run that stops without the model declaring completion is recorded with
`stop_reason: model_emitted_no_action`, not as completed. "The model stopped
emitting actions" and "the model finished" are different outcomes that look
identical to a consumer, and the trajectory keeps them apart.

---

## 3. Depth is capped at one

**Type: CONSTRAINT — accepted.**

`depth` may be 0 or 1. Requesting 2 raises. The literature reports roughly
fifteen points lost on simple retrieval at depth one and thirty at depth two,
with latency rising from seconds to minutes. A depth-0 engine discards its
sub-model at construction, so `query()` cannot be reached by accident.

### Depth 0 is measured before depth 1 is trusted

**Type: PLAN — accepted.** Review trigger: `depth_comparison` verdict.

`evaluation/depth_comparison.py` runs every case at both depths with the same
models and budget, paired, and compares coverage, distinct fragments surfaced,
and cost. The verdict is a sentence: whether the recursive step earned its cost
on these cases. If depth 0 surfaces the same evidence, the recursion is not
justified, and that is discovered in three days rather than after three weeks
of building on it.

---

## 4. The environment inherits the quarantine

**Type: CONSTRAINT — accepted.** Verified by attempt.

`rlm_wiring.search_via_adapter` binds the environment's search primitive to
`WeaviateEnterpriseMemoryAdapter.hybrid_search`, which already refuses
quarantined classes. The engine therefore reaches memory through the same call
that keeps synthetic candidates out of the one-shot path, and inherits the
exclusion without repeating it — repeating a safety check in two places invites
the two to drift apart.

`documents_from_adapter_store` excludes quarantined records when materialising
the environment as well, so that even `describe()` does not reveal a
candidate's existence.

The test stores a candidate, has the engine search for it, and asserts it does
not surface. Reading the adapter proves how it is configured; the attempt
proves the candidate is unreachable.

---

## 5. Trajectories are health records

**Type: CONSTRAINT — accepted.**

A trajectory holds fragments of the case with their offsets, sub-model prompts
built from case text, and the model's own navigation. When the case is real
that is clinical data. `TrajectoryAuditWriter` appends it to the audit store
with `health_data: true` and `retention_class: clinical_record` — not to a log
that a rotation policy will discard or an access policy will treat as
operational telemetry.

For synthetic and de-identified runs the same writer marks `retention_class:
research`, so the distinction is carried by the record and not by which store
it happened to land in.

---

## 6. Retrieval contract

The engine renders a completed trajectory as `retrieval_mode = "rlm_environment"`
with `coverage_basis = "corpus_characters"`, satisfying the shared contract so
that downstream consumers — reconciliation, the pipeline — need no change. An
incomplete run renders with an empty evidence list and the stop reason
attached: the fragments it did retrieve are in the trajectory for inspection,
and nowhere else.

---

## 7. Modules

| Module | Role |
|---|---|
| `reasoning/rlm_engine.py` | Parser, dispatcher, budget, loop, trajectory |
| `reasoning/rlm_wiring.py` | Adapter binding, store materialisation, audit writer |
| `evaluation/depth_comparison.py` | Paired depth-0 vs depth-1 comparison |

---

## 8. Open items

1. **Root model binding.** The engine takes a callable; the tests use scripted
   stand-ins. Binding to a live model goes through `model_client`, whose
   `http_json` mode is gated by `enabled` and `allow_remote` — the phase-one
   API route.
2. **Prompt design.** The current prompt is minimal. A root model's ability to
   emit well-formed actions is the whole loop, and prompt work is where that is
   won or lost.
3. **Search hit offsets.** `search_via_adapter` reads `char_start` and
   `char_end` from hit metadata; hits without them fall back to the document
   head. Chunk-level offsets from ingestion would make search fragments as
   precise as grep fragments.
4. **A1 still gates D5.** The engine produces a `mean_grounding_score` from
   fragment scores, but those are still similarity-derived where present; the
   recursive equivalent remains the open prerequisite it was.
