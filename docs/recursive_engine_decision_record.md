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

## 7b. Choosing the root model

**Type: PLAN — accepted.** Review trigger: bench result.

The root model does not diagnose. It decides where to look and when to stop, so
the requirements are format adherence first, multi-step planning second, and
low cost per iteration third. **Medical knowledge is not required**, and a
medical fine-tune typically degrades format adherence — which is why the
diagnostic model is excluded from this role.

### Published benchmarks cannot settle it

IFEval saturates: models score far higher on its detectable-format subset than
on harder format benchmarks, and the gap between models there is narrow.
Failure modes are model-specific and telling them apart needs the trace rather
than the metric. A leaderboard position does not predict whether a model will
write `grep(prednisone)` rather than `grep prednisone` under this grammar.

### Candidates

| Model | Licence | EU commercial | Note |
|---|---|---|---|
| Mistral Small 3.1 | Apache 2.0 | Cleared | Reported as the most instruction-obedient of its class on exact output formats |
| Mistral Small 4 | Apache 2.0 | Cleared | Newer sparse MoE; the adherence figure is on 3.1, not this |
| Qwen 3.5 | Apache 2.0 | Cleared | Leads open-weight comparisons overall; same licence, cheap to include |
| Gemma 3 27B | Gemma Terms | **Needs review** | More restrictive than Apache 2.0 |
| Llama 3.3 70B | Llama Community | **Needs review** | Dense and text-only, so unaffected by the Llama 4 EU restriction; benched for comparison |
| Claude Sonnet 5 | Anthropic Commercial Terms | **Needs review** | Default Claude tier: mid-tier cost for a task that is format adherence, not depth of reasoning |
| Claude Opus 5 | Anthropic Commercial Terms | **Needs review** | Benched, not assumed unnecessary: a 25% premium is worth it only if it also raises adherence |
| Claude Fable 5.1 | Anthropic Commercial Terms | **Needs review** | Mythos-tier, 5x Sonnet's cost; benched for the same reason as Opus |
| GPT-6 Astra | OpenAI Commercial Terms | **Needs review** | Omitted from the first version of this registry with no reasoning given — an oversight, corrected here |

**Llama 4 is excluded.** Its Acceptable Use Policy withholds multimodal rights
from EU-based individuals and companies, which restricts the family here. Llama
3.3 70B is unaffected but means adopting a previous generation. Reports of a
"Llama 5" have not materialised on any first-party channel.

The candidate registry carries the licence status as data rather than in
someone's memory, so a model cannot be benched, liked and adopted before anyone
checks whether it can ship. Benching is not adopting: a model whose licence is
unresolved belongs on the bench, because comparison is how you learn what a
permissive licence costs in capability, and `BENCH_ONLY_UNTIL_LICENCE_REVIEW`
keeps the question attached to the result.

### Running the bench

The bench cannot be run from this repository's own execution environment: model
endpoints are outside its network allowlist, as `api.mistral.ai`,
`generativelanguage.googleapis.com` and the rest all refuse.

**Type: CONSTRAINT — accepted** for the credential discipline.

`.github/workflows/root-model-bench.yml` runs it on `workflow_dispatch` only —
deliberately excluded from the push and pull-request triggers `ci.yml` runs on,
because this workflow makes real, billed calls to external providers and must
never fire on every commit. Keys come from repository secrets (`MISTRAL_API_KEY`, `OPENROUTER_API_KEY`),
exposed only as environment variables to the step that needs them, never
written to a file or printed — the same discipline already established for
`NCBI_API_KEY`. No `ANTHROPIC_API_KEY` is used: the operator does not hold a
direct Anthropic key, and Claude is reached through OpenRouter instead, on the
same key as Qwen, Llama and Gemma.

### Why one proxy was rejected and OpenRouter was chosen instead

A third-party API gateway advertising Claude access, `oneprovider.dev`, was
considered and rejected before any code was written for it. A public review of
the service states the model actually served is not Claude — precisely the
failure this bench exists to prevent: a result that looks like a measurement of
one model while silently measuring another, indistinguishable from a genuine
result until something else contradicts it. The service's own advertised
payment model, cryptocurrency with no account requirement, is a further signal
that it is built for anonymity on the seller's side rather than accountability
on the buyer's.

OpenRouter is a named, established aggregator that proxies to the real
provider rather than an anonymous reseller, and it already carries Claude,
Qwen, Llama and Gemma in one catalogue. All four candidates therefore share
`OPENROUTER_API_KEY`, and Mistral remains on its own first-party endpoint. The
net effect is one fewer service to trust than four separate credentials would
have required, not a substitution of one uncertain proxy for another.

### Why three Claude tiers and OpenAI, not one Claude candidate

The first version of this registry benched a single Claude tier by default, on
the reasoning that the root model's task — format adherence, not depth of
reasoning — does not need a premium model's capability. That reasoning is
sound for *why a cheap tier is a reasonable default*; it is not a reason to
skip measuring whether a costlier tier does better, and defaulting to one
candidate without measuring the others would repeat the mistake this whole
bench exists to avoid: choosing on argument what should be decided by a
number.

So `claude-sonnet-5`, `claude-opus-5` and `claude-fable-5.1` are all benched.
Whichever tier's adherence and completion rate justify its cost is the
evidence-based answer; a 25% or 5x premium that does not raise adherence has
its own answer.

OpenAI was omitted from the first version of this registry alongside Mistral,
Qwen and Llama, with no reasoning recorded for the omission. That is corrected
here: `gpt-6-astra` is benched through the same `OPENROUTER_API_KEY`, and its
commercial terms need the same review as every other unresolved candidate's.

A candidate whose key is absent is skipped and reported as such rather than
causing the run to fail: partial coverage is still a usable comparison.
`scripts/run_format_adherence_bench.py` can also be run locally with the same
variables exported.

### A defect this testing found

The first run against a real endpoint — with a deliberately invalid key, to
exercise the failure path — returned a misdiagnosis: every call failed with
HTTP 403 and every model produced zero accepted and zero rejected lines, so
`near_miss_share` computed 0/0 as 0.0, and the verdict logic read that as "the
rejections are mostly near misses" and recommended prompt work. A connectivity
or authentication failure was about to be diagnosed as a syntax problem.

The fix separates "no output at all" from "output that failed to parse" before
computing a verdict: a model producing zero lines is not a formatting failure,
because there is no output to have a format, and the verdict now says
explicitly to check keys and connectivity first.

### What the bench measures

`evaluation/format_adherence_bench.py` produces two numbers per candidate:
**adherence** — the fraction of emitted lines the parser accepted, since a
rejected line is not a degraded action but no action — and **completion**, the
fraction of runs reaching `final()`, because a model can emit well-formed
actions forever and never declare it is finished.

Rejections are split into **near misses** and **prose**. A model writing
`grep prednisone` understood the task and missed the syntax: one line of parser
tolerance. A model writing prose did not receive the format: prompt work that no
model choice fixes. The raw adherence figure does not distinguish them, and the
verdict does.

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
