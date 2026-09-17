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
| Mistral Large 3 (via OpenRouter) | Apache 2.0 | Cleared | Second path to Mistral after the direct API's free-tier rate limit; separate limits, real redundancy |
| Mistral Small 4 (via OpenRouter) | Apache 2.0 | Cleared | Same rationale as Large 3, at the small tier |
| Qwen 3.5 | Apache 2.0 | Cleared | Slug corrected: the original was invented and never existed |
| Qwen 3.7 | Apache 2.0 | Cleared | Flagship of the 3.7 generation, benched alongside 3.5 and 3.8 |
| Qwen 3.8 | Apache 2.0 | Cleared | Current Qwen flagship as of Sept 2026; generic slug tracks Alibaba's own updates |
| Gemma 3 27B | Gemma Terms | **Needs review** | More restrictive than Apache 2.0 |
| Gemma 4 31B | Apache 2.0 | Cleared | Supersedes Gemma 3 in two ways at once: newer, and licence-cleared in the same release |
| Gemma 4 26B-A4B | Apache 2.0 | Cleared | Same generation and licence; MoE, cheaper per call |
| Llama 3.3 70B | Llama Community | **Needs review** | Dense and text-only, unaffected by the Llama 4 EU restriction |
| Llama 4 Maverick | Llama Community | **Needs review** | Bench-only, not adoption: EU AUP restriction stands regardless of result |
| Llama 4 Scout | Llama Community | **Needs review** | Same restriction and status as Maverick, at the smaller tier |
| GLM-5 | MIT | Cleared | No acceptable-use policy to review, unlike Llama or Gemma |
| GLM-5.3 | Unverified | **Needs review** | Reasoning always on and cannot be disabled — directly relevant to the completion-rate investigation below |
| Claude Sonnet 5 | Anthropic Commercial Terms | **Needs review** | Default Claude tier: mid-tier cost for a format-adherence task |
| Claude Opus 5 | Anthropic Commercial Terms | **Needs review** | A 25% premium is worth it only if it also raises adherence |
| Claude Fable 5.1 | Anthropic Commercial Terms | **Needs review** | Mythos-tier, 5x Sonnet's cost |
| GPT-6 Astra | OpenAI Commercial Terms | **Needs review** | Omitted from the first registry version with no reasoning given — corrected |
| Kimi K2.6 | Unverified | **Needs review** | Reported to sustain the longest correct open-weight tool-calling sequences available |
| DeepSeek V4 Flash | Unverified | **Needs review** | Cheapest capable candidate; predecessor's structured tool-calling reported unreliable, measured rather than assumed here |
| Grok 4 Fast | xAI Commercial Terms | **Needs review** | Verified slug for xAI's cost-efficient tier |

**Llama 4 is benched, not adopted.** Its Acceptable Use Policy withholds
multimodal rights from EU-based individuals and companies, which restricts the
family here regardless of how it performs on this bench. Maverick and Scout
are included so the comparison table states a measured gap rather than an
assumed one — the restriction stands either way. Llama
3.3 70B is unaffected but means adopting a previous generation. Reports of a
"Llama 5" have not materialised on any first-party channel.

The candidate registry carries the licence status as data rather than in
someone's memory, so a model cannot be benched, liked and adopted before anyone
checks whether it can ship. Benching is not adopting: a model whose licence is
unresolved belongs on the bench, because comparison is how you learn what a
permissive licence costs in capability, and `BENCH_ONLY_UNTIL_LICENCE_REVIEW`
keeps the question attached to the result.

### A defect this testing found: ten minutes to discover a bad key

The first live run against nine candidates took 10.5 minutes and ended in
failure. The cause was arithmetic the script never bounded: each candidate got
the full three-cases-times-up-to-twelve-iterations-times-60-seconds budget
before its unreachability became visible, and with most or all candidates
sharing the same broken credential, that cost was paid nine times over before
the run could report anything.

Two fixes, one in the workflow and one in the script. `timeout-minutes: 20` on
the job gives GitHub Actions an explicit ceiling instead of its default of six
hours, so a genuinely hung call fails the job visibly rather than occupying a
runner indefinitely. And `_preflight()` makes one short call per candidate,
timeout 15 seconds, before committing to the full bench: a bad key or an
unrecognised model slug almost always fails fast — an auth or not-found
response arrives in well under a second — so the same failure that took over
ten minutes to surface now surfaces in under one. The per-case budget for
survivors was also tightened (six iterations, 40-second wall clock) rather than
left at the engine's general-purpose defaults, since this bench asks three
one-fact questions about a two-sentence document and does not need the
allowance a harder task would.

### A third defect, found on the first real run: unhandled provider response shapes

The first execution against real endpoints failed with `actions/upload-artifact`
reporting "No files were found" — the previous fix (writing results even on
total preflight failure) had not fired at all, because the script crashed
*before* reaching that code path.

The cause: `_http_chat_completion` read `payload["choices"][0]["message"]["content"]`
unconditionally. OpenRouter, like most aggregators, returns HTTP 200 even when
a request could not be served — no endpoint available for the model, content
filtered, an upstream provider error — with the failure described inside the
JSON body rather than the status code. `"choices": []` is a real response
shape, and `payload["choices"][0]` on an empty list raises `IndexError`, which
was not in any `except` clause anywhere in the script. Reproduced with a mocked
`urlopen` returning exactly that body: the crash propagated through
`_preflight`, through `build_candidates`, out of `main`, with nothing written.

Two layers of fix, deliberately redundant.

`_extract_content` now validates the response shape explicitly — not a `dict`,
an `error` field, missing or empty `choices`, a choice with no text content —
and raises one exception type, `ProviderResponseError`, for all of them, which
`_bind` and `_preflight` already catch alongside `URLError` and `HTTPError`.

But enumerating every malformed shape a third-party API might someday return is
a losing game, so `_bind` and `_preflight` each also gained a final
`except Exception` clause: *any* unexpected failure from a provider call
degrades to a reported, catchable outcome, never a crash. And `main` was split
into `main`, a top-level safety net, and `_run`, the actual logic: if
something raises from anywhere `_bind`/`_preflight` do not cover — a bug in
this script, a change in a dependency — `main` catches it, writes a
`status: "crashed"` payload with the exception type, message and full
traceback, and still exits 1. Verified by injecting a `RuntimeError` nobody
anticipated directly into `build_candidates`: the results file is written
regardless.

### First real-provider results, and why completion was low

The first live run against nine candidates completed successfully (the
robustness work above held) and produced a genuine measurement: `gemma-3-27b`
won on adherence and completion, but **completion across all seven
candidates that produced output averaged 33%**, and two Claude tiers
(Sonnet, Fable) hit `iteration_budget_exhausted` on every single case with
100% adherence the whole time — well-formed actions throughout, never a
`final()`.

That pattern rules out confusion as the cause: a model emitting correct
actions for six iterations straight understood the grammar. Three
hypotheses were investigated instead of guessed at.

**Hidden reasoning consuming the token budget.** GLM-5.3's own OpenRouter
listing states reasoning "is always on and cannot be disabled." Mistral
Small 4 documents a configurable `reasoning_effort` parameter, implying
reasoning is on by default for at least some call shapes. If a model's
visible completion is preceded by reasoning tokens sharing the same
`max_tokens` budget, 256 tokens may not have left room for both the
reasoning and the action line — indistinguishable from "genuinely stuck"
without more room to see the whole output. Simulated with a mocked provider
that emits reasoning text before its action: at `max_tokens=256` the action
is truncated away; at `max_tokens=1024` the same model completes 100% of
cases. `max_tokens` raised accordingly, and OpenRouter's own
`reasoning: {enabled: false}` parameter (`openrouter.ai/docs/use-cases/reasoning-tokens`)
is now sent on every OpenRouter call as a best-effort hint — ignored by
models that cannot disable reasoning, harmless for models without one.

**Over-verification rather than a tight budget.** The system prompt asked
for one action per line and to call `final()` "once you can answer," which
does not rule out re-checking an already-sufficient answer before
committing to it. Rewritten with a worked example (`grep(dose)` →
`final(40 mg daily)`, nothing more) and an explicit instruction that one
clean lookup is enough and re-verification spends part of a small, fixed
budget.

**Two further, distinct artifacts, not budget-related at all.** Claude Opus
produced lines like `"Assistantslice(1, 0, 151)"` and `"human grep: 3
fragment(s)"` — the second nearly identical to this engine's own history
line format (`"{action.raw} -> {step.result_summary}"`), suggesting the
rendered history was being pattern-matched onto a chat transcript. GPT-6
Astra narrated tool output in prose (`"[doc 0] note.txt (152 chars)"`)
instead of emitting the next action. Neither is a completion problem a
wider budget fixes; the rewritten prompt explicitly prohibits role labels,
dialogue formatting, and narrating what an action returned.

**Opus's artifacts persisted into the second run, in a new form worth
recording rather than glossing over.** With the rewritten prompt in place,
Opus's adherence fell further, to 78.95%, and its rejected lines now include
visible chain-of-thought (`"Reasoning: grep(prednisone) matched, so I need
the surrounding text; the earlier bounded-context pattern failed, so try a
greedy wildcard..."`), a malformed run-together token (`"assistdescribe()"`,
plausibly a truncated "Assistant" fused to the next action), and a full
clinical sentence — `"Patient denies fever, chills, or night sweats. Afebrile
on exam."` — that does not appear in any bench document, which reads as
fabricated content rather than a quotation. `reasoning: {enabled: false}` is
sent for Opus as for every OpenRouter candidate, and it plausibly does not
suppress Claude's extended thinking the way it does for models whose
reasoning control OpenRouter's schema maps more directly — an open question,
not a fixed one, and not silently smoothed over: Opus remains the weakest
adherence in both runs, and the harder cases below give a further,
independent read on whether that persists.

The iteration budget was also raised, 6→10, and `budget_bound` — new on
`ModelResult` — reports per candidate whether every incomplete run used the
full ceiling it was given: `True` means the budget was plausibly the limit
and is worth revisiting with the next run's evidence; `False` means the
model stopped short for a different reason, which a wider budget will not
change. The case count went from 3 to 6 for the same reason precision
matters in a measurement: one case is 33 percentage points of
`completion_rate`, too coarse to tell a genuine pattern from noise.

### Registry corrections and additions found by the same investigation

`qwen-3.5` carried an invented slug, `qwen-3.5-72b-instruct` — there is no
72B-parameter Qwen 3.5 variant. Corrected to the verified
`qwen/qwen3.5-plus-02-15`. A user-requested "Mistral 3.6" was checked
against Mistral's full, dated release history (32 tracked releases, most
recent Medium 3.5) and does not exist; no slug was invented for it.

Verified additions: Qwen 3.7 and 3.8 (current flagships); Llama 4 Maverick
and Scout, benched for comparison under the same EU Acceptable Use Policy
restriction already documented for the family — benching is not adoption,
and the restriction stands regardless of this bench's result; GLM-5.3;
Gemma 4 in both sizes, which matters beyond being newer — **Gemma 4 shipped
under Apache 2.0**, resolving the licence-review flag Gemma 3 carried, in
the same release that made it more capable; Mistral Large 3 and Small 4 via
OpenRouter, added as a second path after the direct Mistral API's free-tier
rate limit (below) made the single path unreliable.

Three genuinely new families, from a survey of what exists as of September
2026 rather than only extending families already present: **Kimi K2.6**
(Moonshot), reported to sustain the longest correct open-weight tool-calling
sequences available — closer to this bench's actual task than a general
capability score; **DeepSeek V4 Flash**, the cheapest capable candidate by a
wide margin, included with a documented caveat rather than assumed reliable
— independent reports describe the predecessor generation's structured
tool-calling as unreliable, and this bench measures that question on the
specific six-verb grammar rather than inheriting the reputation; **Grok 4
Fast** (xAI), the verified slug for the cost-efficient tier — a costlier
"Grok 4.5" is referenced in press coverage but its exact OpenRouter slug was
not confirmed, so it was not guessed at. Licences not directly confirmed
from a primary source (GLM-5.3, Kimi, DeepSeek, Grok) carry a new
`LICENCE_UNVERIFIED` marker rather than being assumed to match a
permissively-licensed sibling.

**Data residency, noted for later.** Kimi and DeepSeek are Chinese-developed;
reached here through OpenRouter rather than a China-hosted endpoint
directly, and every document this bench sends is synthetic — enforced by
`RlmEngine`'s own data-class check, independent of this candidate list — so
there is no live exposure in this context. The consideration becomes live
the moment any candidate here is considered for production use on real case
content, where it would need review alongside the licence.

### Why the direct Mistral API failed, and the fix

`mistral-small-3.1`'s preflight returned `HTTP 429 Too Many Requests` on the
first live run. Verified against Mistral's own documentation: the free
evaluation tier carries conservative per-second limits, explicitly
described as intended for evaluation and prototyping, with rate-limit
responses that include a `Retry-After` header. `_http_chat_completion` now
retries once on 429, honouring that header (capped at 20 seconds, falling
back to 5 if absent or unparseable) rather than failing on the first
transient limit. Mistral is also now reachable through OpenRouter as a
second path, whose limits are separate from the direct API's — real
redundancy rather than hitting the same wall twice.

### Second live run: an 8-way tie, an HTTP 400 pattern explained, one slug corrected

The second live run, with the wider budget and rewritten prompt from the
completion-rate investigation above, produced eight candidates tied at
100% adherence and 100% completion out of fifteen that ran
(llama-4-maverick, llama-4-scout, gemma-3-27b, gemma-4-26b-a4b, both Mistral
OpenRouter candidates, kimi-k2.6, deepseek-v4-flash). The fixes worked; the
bench had become too easy to tell the survivors apart, exactly the outcome
that motivates a harder case set (below).

Four candidates failed preflight with **HTTP 400 Bad Request**:
claude-fable-5.1, gpt-6-astra, qwen-3.8, glm-5.3. A fifth failed with
**HTTP 404**: grok-4-fast.

**The 400 pattern.** All four share a property: each is reasoning-mandatory
or reasoning-heavy by design (GLM-5.3's own listing states this explicitly;
the other three are flagship-tier releases plausibly defaulting the same
way). The working hypothesis, consistent with external evidence — an
independent benchmark of Fable 5.1 on OpenRouter explicitly passes
`--thinking off` as a controlled parameter rather than omitting reasoning
control — is that OpenRouter returns 400 for some providers when
`reasoning: {enabled: false}` is sent to a model that cannot honour it,
rather than silently ignoring the field the way most unrecognised
OpenAI-compatible parameters are tolerated. `_http_chat_completion` now
retries once without the reasoning hint on a 400 that occurred with the hint
active, composing with the existing 429 retry rather than replacing it.
Verified with a mocked provider that rejects any request carrying the
`reasoning` field: the first call returns 400, the retry without the field
succeeds.

**The grok-4-fast 404.** A confirmed, reproducible example
(`simonwillison.net/tags/openrouter/`) uses `x-ai/grok-4-fast:free` rather
than the bare slug; a separate report describes some accounts or plans
lacking access to the unsuffixed route. The slug is corrected to the
`:free` variant.

### Model and case configuration moved to the top of the file

`CANDIDATE_MODELS` and `MISTRAL_DIRECT_MODEL`, declared immediately after the
module docstring, are now the only place a model name or slug appears in
this script; `build_candidates()` contains no model-specific logic and
iterates the table. Adding, removing or re-pointing a candidate at a newer
release is a one-line change at the top of the file, which was the explicit
purpose of the reorganisation: not needing to re-read `build_candidates()`'s
body to find where a slug lives.

### Harder cases: from six to eleven, and why "harder" means something specific

The original six cases were single-document, single-fact lookups, which the
8-way tie above shows a capable model can solve almost by construction. Five
new cases were added, each targeting a distinct navigation demand the
original six could not exercise, not merely longer or more technical
wording of the same lookup:

| Case | Tests |
|---|---|
| `negation_discrimination` | Distinguishing an affirmed finding from a negated one nearby — a plain keyword match hits both |
| `cross_document_correlation` | Discovering and reading a **second** document rather than answering from the first one looked at |
| `buried_fact_after_revision` | Locating a fact in the fourth of four paragraphs, after an earlier, superficially-plausible paragraph that is not the answer |
| `numeric_trend` | Comparing a first and last value across a series, with a middle value that goes the "wrong" direction |
| `confirmed_vs_candidate_diagnosis` | Picking the diagnosis explicitly confirmed, as distinct from other candidates named earlier in the same document |

Each is tested for the structural property it claims (the negation case
contains both an affirmed and negated finding; the cross-document case
supplies two distinct document IDs; the differential case states the
confirmation after the candidate list), and the cross-document case is
additionally verified navigable to completion within budget by a scripted
model that actually reads both documents.

**Scope boundary, stated explicitly because it is easy to elide:** none of
this bench's cases are graded against a correct answer. It measures whether
a candidate follows the action grammar and completes navigation within
budget, not whether its `final()` text is clinically right — a model that
confidently answers incorrectly scores identically to one that answers
correctly, provided both are well-formed. Evaluating diagnostic correctness
against a documented outcome is a different, already-built tool
(`evaluation/dream_capture_benchmark.py`, part of the B4 protocol). This
bench answers a prerequisite question — can the candidate navigate the loop
at all — not a substitute for that one.

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

### From one sequential job to a parallel matrix

A run against the full, now-21-candidate, 11-case roster took over 50
minutes in a single sequential job — plausible given the scope (more
candidates surviving preflight after the HTTP 400 fix, nearly double the
cases, several reasoning-heavy models with genuine per-call latency), but a
poor fit for a routine check regardless of whether it was "working as
designed." Raising the job timeout each time it was approached would have
treated the symptom: total time scales with the size of the roster as long
as every candidate runs in one job, one after another, and every model
added since the first run had made that worse.

The workflow is now three jobs. **`prepare`** reads the candidate roster
from the script itself (`--list-candidates`, backed by
`all_candidate_names()`) rather than duplicating it in YAML, so the matrix
always reflects whatever `CANDIDATE_MODELS` currently declares. **`bench`**
runs as a matrix, one job per candidate, each invoking
`run_format_adherence_bench.py --candidate NAME` — a new flag that restricts
`build_candidates()` to that single name via a filter parameter, recording
every other candidate as `"not requested"` rather than silently omitting it,
so one job's own output stays self-explanatory. Jobs run concurrently
(`fail-fast: false`, so one candidate's failure does not cancel the others),
each uploading its own small result artifact. **`combine`** downloads every
artifact and merges them with the new `scripts/merge_bench_results.py`.

Total time now scales with the slowest single candidate rather than the sum
of all of them — a single candidate's formal worst case (11 cases at the
60-second wall-clock ceiling plus one trailing call) is under 20 minutes,
which is also that job's timeout, down from the 90-minute roster-wide
ceiling the sequential version needed.

**One verdict function, two call sites.** `BenchReport.verdict()`'s logic
was extracted into a pure function, `compute_verdict(results: list[dict],
adherence_target)`, operating on the same plain-dict shape
`ModelResult.as_dict()` already produces. A live sequential run calls it via
`BenchReport.verdict()`; the merge script calls it directly on results
loaded back from JSON files written by separate processes. One place
decides what the numbers mean, so a parallel run and a sequential run reach
the same conclusion from the same underlying data — verified by asserting
the two call paths produce identical output on the same results.

**Malformed input degrades, it does not erase.** If one matrix job's
artifact is missing, corrupted, or JSON-but-not-an-object, `merge()` records
it in `merge_failures` and continues with whatever else is present, rather
than one broken candidate losing every result that arrived cleanly. An
empty input directory is itself a named failure rather than a silent empty
report.

### A focused companion workflow: four candidates, harder cases, wider budget

The full-roster run that motivated the harder eleven-case set also produced,
for the first time, a genuine gradient rather than a tie: llama-4-maverick
and gemma-3-27b reached 100% completion; gemma-4-31b and
mistral-large-openrouter reached 90.9%, each stopped on exactly one case by
`iteration_budget_exhausted` with `budget_bound: true` — evidence they were
still working, not stuck, when the shared 21-candidate budget ran out.

Two questions follow that the full-roster workflow is not built to answer
cheaply: would a wider budget let those two finish, and can four strong
candidates be told apart further on cases more subtle than the standard
eleven. Both cost more per candidate, which is affordable for four
candidates and would not be for twenty-one — hence a separate workflow
rather than changing the shared one.

**`--roster` and `--cases advanced` and `BENCH_MAX_ITERATIONS`/
`BENCH_WALL_CLOCK_SECONDS`, all additive.** `--list-candidates --roster
NAME,NAME,...` filters the printed roster to a named subset, validating
every name and failing loudly on one that does not exist rather than
silently printing a shorter list. `--cases advanced` runs `BENCH_CASES +
ADVANCED_CASES`; the default remains `BENCH_CASES` alone, so the full-roster
workflow's behaviour is unchanged unless the flag is passed. `_bench_budget()`
now reads `BENCH_MAX_ITERATIONS`/`BENCH_WALL_CLOCK_SECONDS` from the
environment with a fallback to the existing 10/60s default, so the
full-roster workflow — which does not set them — is unaffected.

**Eight further cases, each targeting a discrimination the standard eleven
do not exercise:**

| Case | Tests |
|---|---|
| `three_document_synthesis` | Synthesising across **three** documents, not two |
| `absence_of_requested_fact` | Concluding and reporting absence rather than confabulating an answer, or looping indefinitely searching for something that is not there |
| `confusable_terms` | Distinguishing two clinically opposite findings that share a word ("pulmonary embolism", excluded, vs "pulmonary oedema", confirmed) |
| `out_of_order_chronology` | Determining true temporal order when the document's paragraph order does not match it |
| `long_multi_section_lookup` | Connecting a fact in an early paragraph to one several paragraphs later in a genuinely long note |
| `conflicting_values_across_documents` | Noticing and citing two disagreeing values for the same measurement, rather than reporting only the first found |
| `weight_based_dose` | Locating two separate numbers (a rate and a weight) needed together, not graded on doing the arithmetic |
| `two_hop_family_history_inference` | Connecting two facts stated in different sentences into one relevant inference |

The absence case is the one worth dwelling on: every other case in this file,
baseline and advanced, has something to find. This one does not, and the
discriminating behaviour is recognising that and finalising anyway —
verified by a scripted model that greps, finds nothing, searches, finds
nothing, and still reaches `final()` within budget, which is exactly the
sequence a genuinely absent fact should produce rather than an endless
search.

`.github/workflows/four-model-comparison-bench.yml` wires these together:
`workflow_dispatch` with a `roster` input defaulting to the four candidates
above, the same three-job `prepare`/`bench`/`combine` structure as the main
workflow, `BENCH_MAX_ITERATIONS=16`/`BENCH_WALL_CLOCK_SECONDS=90` and
`--cases advanced` set on the bench step. `CANDIDATE_MODELS` in the script
remains the only place a model slug is declared; the roster here is a
snapshot of one comparison, changeable via the workflow's input without
touching the model registry.

### Four candidates become eight: a survey of the landscape, then four additions

Before adding anything, every family already in the registry was checked
against current sources rather than assumed still accurate, and a
deliberate search covered labs never considered: Cohere, NVIDIA, MiniMax,
AI21. Cohere's agentic offering is narrowly built for repository and
terminal coding tasks, a different shape of problem than document
navigation; MiniMax showed competitive numbers on an independent agentic
benchmark but no confirmed OpenRouter slug from available sources, so
nothing was guessed at; AI21 did not surface prominently in any of the
searches performed. Three families were found worth adding, plus one gap
in a family already present.

**Google Gemini had never been benched at all.** Every prior Google entry —
`gemma-3-27b`, `gemma-4-31b`, `gemma-4-26b-a4b` — is the open-weight Gemma
family; the proprietary Gemini line was never a candidate in either
workflow until now. `google/gemini-3-pro-preview` is added as the first:
Google's flagship, verified slug, 1M context, native tool-calling. Its
reasoning cannot be fully disabled (a "High"/"Low" effort choice only,
not off) — the same situation as `glm-5.3`, and the same handling applies:
the reasoning-disable hint is sent as a best-effort attempt, and the
existing HTTP-400 fallback (added when `glm-5.3` first demonstrated the
need) already covers a provider rejecting that hint outright. Verified
directly: a mocked provider that rejects any request carrying the
`reasoning` field behaves for `gemini-3-pro-preview` exactly as it does for
`glm-5.3` — one retry without the hint, then success. No new code was
needed for this case; the existing fallback generalised correctly to a
candidate that did not exist when it was written.

**`x-ai/grok-4.6`** joins `grok-4-fast` as the reasoning-capable tier next
to the cost-efficient one — the same cheap-plus-flagship pairing already
used for every other family in this registry. Verified as an August 2026
post-training refresh of the Grok 4.5 base (same $2/$6 per-million-token
pricing as 4.5, not a new foundation model) rather than assumed from the
unconfirmed "Grok 4.5" reference `grok-4-fast`'s own note previously
carried — that note is corrected here now that the flagship tier has an
actual, verified slug rather than an open question.

**`nvidia/nemotron-3-super-120b-a12b`.** NVIDIA's own description names
"cross-document reasoning" and "multi-step task planning" specifically —
language closer to this bench's actual demands than most candidates'
general capability marketing, which is why it was chosen over the larger
Ultra or smaller Nano tiers in the same family. 120B total / 12B active
MoE, verified native tool-calling support. Licence: NVIDIA's own terms,
not Apache or MIT — a new `LICENCE_NVIDIA_OPEN` constant carries this
distinction rather than assuming permissive because weights are published,
the same discipline already applied to Llama's and Gemma 3's licences.
`LICENCE_GOOGLE_COMMERCIAL` is added alongside it for Gemini, kept
distinct from `LICENCE_GEMMA_TERMS` since Gemini is a proprietary
commercial API, not an open-weight release with its own terms — conflating
the two would misrepresent what actually governs each candidate.

**`mistral-small-openrouter` joins the eight-candidate roster.** It already
existed in `CANDIDATE_MODELS` from an earlier addition; the
four-candidate comparison simply never included it, testing only its
Large sibling. Its slug, `mistralai/mistral-small-2603`, was explicitly
re-verified against OpenRouter for this change (four independent sources,
including OpenRouter's own listing, confirmed the same identifier) rather
than assumed unchanged from memory — it matches what was already in the
registry, so no correction was needed here, unlike `qwen-3.5`'s history.

The workflow file, its default `roster` input, its job/step names, and its
artifact name were all updated together (four-model-comparison-bench.yml
retitled "Eight-model comparison bench"), and a test reads the workflow
file directly and asserts every name in its default roster resolves to a
real candidate, so the workflow and the registry cannot silently drift
apart the way `qwen-3.5`'s slug once did undetected.

### The first real run of the eight-candidate roster: a deprecated slug and a candidate that needed more time than budgeted

The first live run against the expanded roster produced six usable results
out of eight. Two absences, two different causes.

**`gemini-3-pro-preview` failed preflight with HTTP 404.** Verified against
Google's own developer changelog: `gemini-3-pro-preview` (without the
`.1`) was deprecated and shut down on March 9, 2026 — after this candidate
was first added to the registry, when the slug was correct. OpenRouter's
own current Google provider listing confirms `gemini-3.1-pro-preview` as
the successor, alongside the 3.6/3.7/3.8 Flash tiers. Corrected with the
same discipline `qwen-3.5`'s slug correction used: a live failure checked
against a primary source before changing anything, not assumed. This is
also the second time in this registry a slug has gone stale between
verification and use (the first was `grok-4-fast`'s bare form needing the
`:free` suffix) — worth remembering as a standing risk of benching
`-preview` and other pre-GA model names, which providers deprecate on
their own schedule with no guarantee of notice reaching this codebase.

**`grok-4.6` never completed.** Its job ran 30.3 minutes against a
30-minute job timeout and was cancelled without writing a result — every
other candidate in the same run finished comfortably inside that ceiling.
This is not the general nineteen-case budget being too tight (the other
seven candidates, several also facing that same wider budget and case
count, all finished well within time); it is evidence that this specific
candidate's real-world latency is substantially higher than the rest of
this roster's. `timeout-minutes` was first raised 30→50 to give it room to
finish — the wrong lever, corrected below once the actual complaint was
understood: a wider timeout waits out slowness instead of recognising it,
and if a candidate needs minutes for a single case, that is not a budget
problem worth accommodating at all.

### A latency circuit breaker, and why the timeout went back down rather than up further

Widening the timeout to 50 minutes was reconsidered on a direct challenge:
a candidate that genuinely needs on the order of twenty minutes per
interaction is not a case to wait out with a longer ceiling, it is a
candidate to abandon early, and the diagnostics should say so with more
precision than "the job ran long."

`bench_model()` now tracks real elapsed wall-clock seconds per case
(`case_elapsed_seconds`) — distinct from iteration counts, which say how
many turns a model took but nothing about how long each one took; a
candidate needing many quick turns and one needing few slow ones can share
an iteration count and look identical without this. If the last
`LATENCY_CIRCUIT_BREAKER_WINDOW` (3) consecutive cases each consumed at
least `LATENCY_CIRCUIT_BREAKER_THRESHOLD_FRACTION` (1.0, i.e. the case's
entire nominal allowance) of that case's own configured wall-clock ceiling,
remaining cases for that candidate are not attempted: `abandoned_for_latency`
and `cases_skipped_for_latency` record what happened and why, and the
candidate's `adherence`/`completion_rate` are still computed correctly over
whatever cases did run.

**The threshold is a fraction of each case's own ceiling, not a fixed
number of seconds — this was reconsidered once, and the reconsideration
matters.** The two workflows here configure different per-case wall-clock
budgets (60s default, 90s for the eight-candidate comparison's wider
allowance). A fixed absolute threshold tuned to look reasonable against one
would be miscalibrated against the other: a value low enough to catch
genuine slowness against the 90s budget would flag candidates on the 60s
budget that are working productively, not stuck; a value high enough to
avoid that would rarely or never trip against the 60s budget at all,
silently disabling the mechanism there. Comparing each case's elapsed time
against that same case's own ceiling stays correctly calibrated for
whichever budget is actually configured, including one neither workflow
uses yet, without needing two hardcoded constants tuned to two specific
numbers.

**This is a practical reduction in risk, not a mathematical guarantee.** A
candidate whose slow cases are interleaved with fast ones (slow, slow,
fast, slow, slow, fast…) could avoid ever landing three genuinely
*consecutive* cases over threshold and still consume most of the case
budget before finishing normally. The breaker is sized for what was
actually observed — grok-4.6 was consistently slow throughout its run, not
intermittently — and `timeout-minutes` remains the true backstop for the
adversarial case the breaker does not cover, which is why both workflows'
job timeouts were tightened to 15 minutes (down from 20 and 50
respectively) rather than removed: with the breaker doing the routine work,
15 minutes is ample margin over its typical few-minutes case while staying
a firm ceiling rather than another number chosen to wait out whatever was
observed most recently.

Verified with a small real sleep (`elapsed_seconds` is rounded to
milliseconds by `Budget`, so an effectively-instant scripted model would
round to exactly 0.000 and never exceed any ceiling regardless of how
small it is set — an early version of this test used a near-zero ceiling
with an instant model and passed for the wrong reason, tripping nothing
because every ratio was `0.0/ceiling = 0.0`; caught before merging, not
after) paired with a ceiling smaller than that sleep: a candidate that is
slow on its last three cases is abandoned with the correct case count
skipped; a candidate slow on one isolated case among fast ones is not; a
uniformly fast candidate never approaches the threshold regardless of which
budget it is measured against.

### The very next live run found the gap the threshold's own documentation had predicted

With the Gemini slug corrected and the circuit breaker in place, the next
live run against the eight-candidate roster showed the breaker working
exactly as designed for one candidate and missing another entirely.

`grok-4.6` tripped after three cases, 5.8 minutes total, mean 109.4s per
case, `wall_clock_budget_exhausted` on all three — clean diagnostic data in
place of the 30-minute timeout kill the previous run ended in.
`gemini-3-pro-preview`, reached for the first time with the corrected
slug, ran the full 15-minute job timeout and was killed with no result at
all: not present in the merged output, not even a `status: "crashed"`
diagnostic, because a job timeout terminates the process itself, which no
`try`/`except` inside that process can intercept — unlike every other
failure mode this bench recovers from, this one kills the messenger before
it can write anything down.

The threshold at 1.0 only caught a candidate that reached its ceiling.
Gemini's job duration is consistent with reliably using most but not all
of its 90-second allowance, case after case — never hitting exactly 100%
on any three consecutive cases, so the breaker never fired, while the
cumulative time across nineteen cases still exceeded the job's own
ceiling. Lowered to 0.75: a candidate reliably spending three-quarters or
more of its budget, not only one exhausting it outright, is now recognised
and abandoned with real data before the external timeout has to be the
one to notice, with nothing to show for it.

**This is a tightening, not a closure.** A candidate reliably sitting just
under 0.75 would evade this threshold exactly as gemini-3-pro-preview
evaded 1.0, for the identical underlying reason. The deeper limitation —
that a job timeout's kill cannot be caught from inside the process it
kills, so whatever progress existed at that moment is lost regardless of
how the circuit breaker is tuned — remains open. A more complete fix would
write intermediate progress incrementally (after each case, not only at
the end), so a run terminated externally still leaves behind whatever was
completed up to that point; this was not built here, deliberately, since
it is a more invasive change to the write path than a threshold number and
deserves its own scoping rather than being bundled into a reactive fix.

Verified directly: a scripted model consistently using 80% of its
configured ceiling — reproducing the shape of what the real run showed,
not a guess at it — is confirmed to trip under the new 0.75 threshold and
confirmed to have been missed under the old 1.0, run side by side against
the same case sequence.

### An efficiency tiebreak, and a second field to resolve budget_bound's ambiguity

The same live run that validated the circuit breaker also produced a
three-way tie: `gemma-3-27b`, `mistral-large-openrouter`, and
`nemotron-3-super` all reached 100% adherence and 100% completion.
`mean_case_seconds` already distinguished them clearly — nemotron at 1.3s,
mistral-large at 4.3s, gemma-3 at 5.3s, a 3-4x spread — but nothing used
that number for ranking; the verdict named a winner among the tie without
saying why that one specifically, and the difference was visible only by
reading the table by hand.

`rank_result_dicts` and `BenchReport.ranked()` now sort on adherence,
completion, then mean seconds per completed case as a third key, and
`compute_verdict` names the tiebreak explicitly when more than one
candidate is tied on the first two: *"nemotron-3-super meets the adherence
target (100%) and is fastest among 3 candidates tied on adherence and
completion (1.3s mean per completed case)."* Verified directly against
the real numbers from that run. A result missing `mean_case_seconds` (an
older result predating the field, or a candidate with zero completed
cases) sorts last among its ties rather than raising or being treated as
fastest by a missing-value default.

**Scope, restated because a question raised it directly:** none of this
measures answer correctness. `budget_bound`, `completion_rate`, the new
efficiency ranking — all describe whether a candidate navigates the action
grammar and finishes within budget, not whether its `final()` text is
diagnostically right. A confidently wrong answer scores identically to a
correct one, provided both are well-formed. That evaluation is a
different, already-built tool (`dream_capture_benchmark.py`, part of B4),
grading against a documented outcome, not yet wired to this bench's
candidate list.

**`budget_bound`'s `False` was found to be ambiguous, from a real
comparison rather than by inspection.** `mistral-large-openrouter`
completed every case (`budget_bound` False because there was nothing
incomplete to be bound by — the best outcome) and
`mistral-small-openrouter` failed two cases by exhausting their iteration
ceiling (`budget_bound` True) — but a candidate that failed cases for a
*different* reason (a malformed action it could not recover from, giving
up outright) would also show `budget_bound` False, indistinguishable from
"completed everything" by that field alone. `all_cases_completed` — true
only when `completion_rate` is exactly 100% — is added as the field this
distinction actually needs; the two are meant to be read together, not
`budget_bound` alone.

### The workflow is renamed and the roster grows again: "eight-model" was never going to be the last count

`four-model-comparison-bench.yml`, retitled "Eight-model comparison bench"
in an earlier change, is renamed again — file and internal name both — to
`focused-comparison-bench.yml`, "Focused model comparison bench". Every
number this workflow has carried in its own name has gone stale within the
same conversation that gave it that name; "focused" describes what the
workflow is for without asserting a count that the next roster change would
immediately falsify again. References to "four-model-comparison-bench.yml"
and "the eight-candidate roster" earlier in this record describe the
workflow accurately as it existed at each point in this narrative — they
are not corrected retroactively, since doing so would misrepresent the
sequence of decisions rather than the file itself, which has moved.

The roster gained six candidates for the reason the efficiency-tiebreak
section above concluded with: `claude-sonnet-5`, `claude-opus-5`,
`claude-fable-5.1`, `qwen-3.5`, `qwen-3.7`, and `qwen-3.8` were all culled
early in `root-model-bench.yml` runs — completion as low as 0-18% — under
conditions substantially different from the current harness: before the
system prompt carried a worked example, before `max_tokens` was raised
256→1024, before the reasoning-disable hint existed, before the wider
budget or the latency circuit breaker. None of the six were ever
re-measured against what exists now. The low numbers on record may
describe a harness limitation from months of iteration ago rather than a
genuine model limitation — this run is what actually answers that question
rather than continuing to treat stale numbers as settled.

### Two root models instead of one: disagreement as a signal, not a tie to break

Fourteen candidates measured across two independent live runs left four tied
at 100% adherence and 100% completion, with `nemotron-3-super` and
`gemma-3-27b` the two most efficient by a consistent margin in both. The
question that follows is not which single one to adopt, but whether one is
the right shape of answer at all.

**The bench cannot answer the question that matters most.** It measures
whether a candidate follows the action grammar and finishes within budget;
it never grades whether `final()` is clinically right. Two models that both
navigate competently can reach different conclusions from the same
documents, and no single model can report that about itself — a confident
wrong answer and a confident right one are indistinguishable from the
inside.

`reasoning/root_model_cross_check.py` runs two root models over the
identical environment, independently, and compares what each produced. This
is not a new idea in this codebase: `retrieval_reconciliation` already
applies exactly this principle one level down, treating divergence between
the one-shot and recursive retrieval paths as an empirical uncertainty
estimate rather than noise. The same reasoning, applied to two root models,
reuses that module's vocabulary (agreement ratio, per-item disposition)
rather than inventing a parallel one.

**Non-adjudicating by construction.** No third model decides which of the
two is right, and neither is designated authoritative — `primary` and
`secondary` name run order for reproducibility, not precedence. A
disagreement records both answers and picks neither; silently preferring
one would discard the entire signal. Four dispositions are kept distinct
because they call for different responses: `agreed`, `disagreed`,
`single_answer_only` (one model finished, one did not — one answer is not a
second opinion), and `neither_completed`. Everything except `agreed` sets
`needs_review`, deliberately inclusive: one competent navigator failing a
case is itself a reason not to trust the other's answer unexamined.

**Answer comparison, and a measurement that changed the design.** The first
implementation compared answers by character-sequence ratio alone. Checked
against this bench's own vocabulary, that ranks a contradiction above a
paraphrase: "pulmonary embolism" vs "pulmonary oedema" — clinically
opposite, and a distinction one advanced case exists specifically to test —
scores 0.71, while "40 mg daily" vs "prednisone 40 mg daily" — the same
answer, one more verbose — scores 0.67. Exactly backwards. Containment
corrects it: when one normalised answer contains the other whole, the
shorter is a subset rather than a rival claim, which is the shape of "same
answer, different verbosity" and never the shape of two different findings.
Those score 1.0; everything else falls back to the ratio. Verified on both
pairs and on genuinely different answers.

**A limitation stated rather than hidden.** Two answers meaning the same
thing in entirely different words ("not documented" vs "the report does not
mention prednisone", 0.38) are reported as disagreement — a false alarm.
That is the direction the error is deliberately allowed to fall:
over-reporting sends a correct case to a human unnecessarily,
under-reporting lets a genuine divergence through unexamined. Closing it
properly needs semantic comparison, which needs a third model, which
reintroduces the adjudicating judgement this module exists to avoid.
`answer_similarity` is recorded alongside every verdict so a reviewer can
see whether a flagged disagreement scored near the threshold or far from
it.

**Pairing, and why Mistral stays available rather than dropped.**
`DEFAULT_PAIR` is nemotron-3-super plus gemma-3-27b, on the measured
merits: the two most efficient candidates in both live runs, near-identical
iteration counts, the least redundant cost for a doubled workload.
`ALTERNATIVE_PAIR` swaps the second for mistral-large-openrouter, which
matched on adherence and completion at roughly twice the iterations. It is
kept because of something the bench structurally cannot measure: Mistral is
the only one of the three from an EU-based lab, which may matter for a
system with MDR ambitions in a way no adherence figure will ever show. Both
are names only — the callables are supplied by the caller, so this module
holds no opinion about how a model is reached.

### Frame slots replace character comparison, applying the project's own theory to its own output

The character-similarity limitation documented above was raised as a
question — whether a false-alarm-prone comparison makes a second opinion an
obstacle rather than an advantage — together with a proposal: ask the models
to answer in a fixed logical order ("farmaco dose posologia"), and compare
field by field. That proposal is better than the containment patch it
replaces, and it is the same move the project already makes elsewhere.

**Why character comparison failed, mechanically.** `difflib`'s ratio is
`2 × shared_characters / total_length`. "pulmonary embolism" and "pulmonary
oedema" share 12 characters out of 34 — the word "pulmonary " plus "em"
appearing by coincidence in *em*bolism and oed*em*a — for 0.706. "40 mg
daily" and "prednisone 40 mg daily" share 11 out of 33 — the entire shorter
answer — for 0.667, penalised because "prednisone" inflates the
denominator. The comparison counts shared characters without knowing that
"pulmonary" discriminates nothing in a medical vocabulary (it prefixes
embolism, oedema, fibrosis, hypertension alike) while "embolism" versus
"oedema" discriminates everything.

**`reasoning/frame_answer.py` removes the need for that comparison.** A
model is asked to answer by filling named slots — `drug | dose | frequency
| polarity` for medication, `finding | site | polarity` for a finding — and
the two answers are compared slot against slot. The word that broke the
character comparison lands in `site`, where sharing it is correctly
uninformative, and the conflict localises to `finding`, which is where it
actually is.

**This is Frame Semantics and Mental Spaces applied to the comparison, not
imported into it.** The project already uses Fillmore for assertion
detection; a frame with roles to fill is exactly that apparatus, applied to
what a root model produces rather than to what a clinical document contains.
And `memory/assertion.py` already encodes Fauconnier's mental spaces as
`POLARITY_AFFIRMED` / `POLARITY_NEGATED`: "pulmonary embolism, confirmed"
and "pulmonary embolism, excluded" name the same finding in different
spaces, which a character comparison reads as near-identical and a polarity
slot reads as opposite. `frame_answer` imports those constants directly
rather than defining a parallel vocabulary that could drift from them, and
`polarity_conflict` is surfaced as its own flag because two models asserting
opposites about the same finding is categorically worse than naming two
different findings.

Three further distinctions the slot comparison makes that the ratio could
not. A slot one model left unstated is a **gap**, not a contradiction —
conflating them would report a partial answer as disagreement about
substance. Two answers with every slot unstated have no conflicts but agree
on nothing, so agreement requires at least one positively agreed slot rather
than merely an absence of conflict. And containment survives, but only
inside a slot, where "40 mg" inside "40 mg" is trivially the same value and
cannot silently forgive two different findings the way it would across a
whole answer.

`FRAME_FREE_TEXT` is a deliberate escape hatch: not every question this
bench asks decomposes into slots ("does the family history bear on today's
measurement, and why?"), and forcing an ill-fitting frame would produce a
worse answer rather than a better comparison. Those fall back to whole-answer
comparison and say so.

The frame path is optional on `cross_check` rather than mandatory, because
the caller has to ask its models for that format in the first place —
passing a frame without having given the models
`frame_prompt_instruction` would parse unstructured prose into slots that
were never filled, which is a worse failure than the one being fixed.

### Analysing what fell through to free text: two frames, and why only one of them is extraction

The two advanced-bench questions landing in `FRAME_FREE_TEXT` turned out to
be different in kind, and the difference is the whole point.

*"What laboratory abnormality supports the imaging impression, and which
document reports it?"* evokes two chained Fillmore frames: **Support**
(Support / Supported_claim) and **Statement** (Source / Message). Every slot
is locatable in the documents. This is extraction — careful reading, nothing
more. `FRAME_ATTRIBUTION` adds `source_document` as the slot that
distinguishes it from `FRAME_FINDING`.

*"Does the family history have any bearing on today's aortic measurement,
and why?"* evokes a **Relevance** frame — Factor, Target, a judgment, a
mechanism — but the operative difference is in Fauconnier's terms. "Marfan
in a sister" and "aortic root 4.8 cm" are both facts in the base space, the
document. The question opens a **third space**, one of clinical consequence,
and asks whether the two base-space facts map onto each other there. That
mapping is not in the text at any level of careful reading: it is in the
concept graph, or nowhere. `FRAME_RELEVANCE`'s `mechanism` slot is where a
graph path would be named.

**Recognition is Frame Semantics, not a heuristic standing in for it.** An
earlier framing in this conversation presented keyword routing as the weaker
alternative to declaring a type explicitly per case. That was imprecise:
identifying a frame from its **frame-evoking lexical units** is how Frame
Semantics works, and how FrameNet — the computational resource built on
Fillmore's theory — is organised. `FRAME_EVOKING_UNITS` catalogues which
predicates evoke which frame; `recognise_frame` consults no model, so the
classification is inspectable and cannot itself hallucinate. Mental Spaces
plays a different role and is deliberately absent from recognition: it
explains why a relevance question needs graph traversal *once identified*,
not how to identify one.

Ordering matters in one specific way: relevance is checked first, so a
question evoking both routes to relevance. The extraction-answerable half of
such a question would otherwise succeed silently while the other half failed
— the worse of the two failures.

`judgment_conflict` is the relevance frame's analogue of
`polarity_conflict`, and `contradicts` covers both: two models disagreeing
on whether a link exists at all are contradicting each other, not offering
two descriptions of one thing. Kept as separate flags because a reviewer
needs to know which kind of opposition they are looking at.

Two defects found while verifying this, both fixed before merging: the
instruction builder emitted a "for 'polarity' write exactly…" clause for
every frame including those with no polarity slot (inviting a model to
invent one, or to distrust the rest of the instruction), and yes/no finding
questions ("Is fever present according to the report?") fell through to free
text despite being exactly the finding frame with the answer carried in the
polarity slot.

`FRAME_FREE_TEXT` survives, and still should: *"How long has the dyspnoea
been present?"* is a duration, matching no frame here, and admitting that is
better than forcing one. The escape hatch narrowed rather than closed.

### Muse Glimmer 30B: the first Meta candidate without a licence restriction

Recommended in an earlier turn and, on audit, never actually added — a gap
found by checking the code rather than trusting the recollection that it had
been done. `meta/muse-glimmer-30b`, verified against OpenRouter's own Meta
provider listing: 30B dense, **Apache 2.0**, the first Meta entry in this
registry not under the Llama Community Licence and its EU acceptable-use
restriction. Described for long-horizon agentic workflows with multi-step
reasoning, tool use and failure recovery.

Benched with a caveat rather than on reputation: an independent reader of
its published scores flagged a high hallucination rate and advised against
critical tasks. This bench measures navigation and format adherence, not
answer correctness, so it can neither confirm nor refute that — the caveat
is recorded in the registry precisely because a good result here would not
address it.

Muse Spark, the model it is distilled from, is deliberately not benched:
closed-weight, and its cheap "contributor" tier states that prompts and
outputs may be used to improve Meta's products — not a habit worth forming
even on synthetic documents.

### The first real run of the parallel matrix lost three results out of four

The four-model comparison workflow's first live run reported "No candidate
produced a usable result" despite `List candidates` correctly naming all
four, all four `Bench <candidate>` jobs completing, and `Merge into one
report` exiting successfully — every step green, and yet zero results in
the merged output.

The cause was in the merge step's own shell, not in any candidate's run.
`actions/download-artifact@v4` with `merge-multiple: false` (the default)
downloads each artifact into its own subdirectory named after the artifact
— `all_results/candidate-result-0/result.json`,
`all_results/candidate-result-1/result.json`, and so on — because every
candidate job writes the same internal filename, `result.json`. The
workflow's merge step then flattened these with `find all_results -name
'*.json' -exec cp {} merged_input/ \;` before calling
`merge_bench_results.py`. Since every source file shares that one filename,
every copy into the same flat destination **overwrote the previous one**:
four files went in, one came out, and which one survived depended on
`find`'s traversal order rather than anything about the candidates
themselves.

This affected both workflows equally — `root-model-bench.yml` carried the
identical pattern, copied when `four-model-comparison-bench.yml` was built
from it — and had never been caught, because the merge script's own tests
exercised `merge()` against hand-built flat directories with distinctly
named files, never against the actual nested, identically-named layout
`download-artifact` produces. The logic was verified in isolation and
correct as far as it went; the gap was never testing the full pipeline
against GitHub's real artifact layout.

Fixed by removing the flattening step entirely rather than working around
the collision: `merge()` now searches `input_dir.rglob("*.json")` instead of
`input_dir.glob("*.json")`, so it can be pointed directly at
`all_results/` — the nested layout `download-artifact` already produces —
with nothing to flatten and nothing to collide. `source_files` and every
failure message now report each file's path relative to the input
directory rather than its bare name, since two files both literally named
`result.json` are indistinguishable by name alone once nesting is no
longer removed first.

Reproduced and verified directly: four files written into four
subdirectories exactly as `download-artifact` would produce them, merged
first through the old flattening step (one result survives) and then
through `merge()` pointed at the nested directory directly (all four
survive) — the same contrast is now a permanent regression test.

### The second run after that fix: an AttributeError the merge fix could not have caught

The very next live run, after the artifact-collision fix above, produced
the same symptom — "no candidate produced a usable result" — for a
different reason entirely. This time the merge worked correctly (no
collision), but the merged output was still empty because every one of the
four candidate jobs had individually crashed before writing a real result,
each caught by `main()`'s own safety net and reported as `status: "crashed"`
with a full traceback rather than silently vanishing — the safety net
worked exactly as designed. The traceback:

```
AttributeError: 'BenchReport' object has no attribute 'as_dict'
  File ".../run_format_adherence_bench.py", line 1049, in _run
    payload = report.as_dict()
```

`BenchReport.as_dict()` had existed since this tool's first commit. It was
lost during the parallel-matrix refactor that extracted `verdict()`'s logic
into the standalone `compute_verdict()` function — the class body was
rewritten in that change and `as_dict()` was apparently dropped along the
way, never re-added. Every test in the suite that called `.as_dict()`
called it on a `ModelResult` (which has its own, unaffected, `as_dict()`),
never on the `BenchReport` wrapping them — the one call site that mattered,
`_run()`'s `report.as_dict()`, was never exercised by anything except a
live run against a real, reachable candidate. Restored verbatim (rebuilding
the same payload shape the script has always expected: `models`,
`adherence_target`, `verdict`, ranked `results`), now delegating to
`compute_verdict` internally rather than duplicating that logic, and
verified to reproduce the exact reported traceback when reverted and to be
resolved by the fix.

**The gap this exposes, stated plainly:** two live-run failures in a row
were each caused by a code path no test ever reached, despite a large and
otherwise careful test suite. `BenchReport.as_dict()` is now covered
directly (`test_bench_report_as_dict_does_not_raise` and three siblings
checking its exact shape), and — the more load-bearing addition —
`test_the_full_script_does_not_crash_on_a_successful_candidate` drives
`main()` itself through a scripted, always-reachable candidate, the one
path (a candidate that succeeds, not one that is skipped or fails) neither
of the two real incidents' underlying bugs could have survived. Reverting
the fix and re-running that test reproduces the reported traceback exactly,
confirming it would have caught this before a live run did.

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

`glm-5` is added for the same reason as Qwen: MIT licensed, weights published,
no acceptable-use policy to review at all — the cleanest licence position of
any candidate here, Apache-2.0 included only because MIT and Apache-2.0 are
both unconditionally cleared while Llama, Gemma and the commercial APIs are
not.

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

### Three modules built from a research report, and a status check that found they were not wired together

`memory/information_content.py`, `memory/spreading_activation.py`, and
`reasoning/mechanism_verification.py` were built across three sequential
changes following `docs/semantic_comparison_research_report.md`'s proposed
sequence: Information Content weighting (concept specificity as `-log(p)`,
since IC-based measures outperform path-based ones on MSH-WSD), converging-path
rewards (ONTOSPREAD's principle that multiple independent routes to a
conclusion are worth more than their individual strengths), and constrained
spreading activation (relation, decay and threshold constraints, since
unconstrained spreading is a documented failure mode). Each is recorded in
full in the research report rather than repeated here.

A direct status question — "where are we" — prompted an audit rather than a
recollection, and the audit found two gaps neither of which had been caught
before being reported as done.

**The larger one:** `rlm_wiring.py`, the file that actually binds the
recursive engine to the live pipeline, wires the search environment and the
audit store and has never bound a root model at all. Every candidate
selected, every cross-check built, every frame and every graph verification
in this entire sequence of work exists in a bench harness and a set of tested
library modules that no real case passing through Melampo today would ever
reach. This remains open; it is a larger piece of wiring than the one fixed
here and deserves its own scoping rather than being folded into this fix.

**The smaller one, fixed in this change:** `mechanism_verification.py` was
presented as "closing the loop" on the `mechanism` slot — but `cross_check()`,
the actual entry point a caller would use, never called it. The mechanism
slot was still compared as a plain frame-answer string, exactly the
comparison the module existed to replace. The module itself was correct and
tested in isolation; the wiring that would have made it reachable was not
written. This is the same shape of gap already found twice earlier in this
work (Muse Glimmer recommended but never added to the registry;
`root_model_cross_check.py` built with no caller) — a module built, tested,
and reported complete, without verifying anything actually calls it.

`cross_check()` now takes `concept_graph`, and routes the `mechanism` slot
through `cross_check_mechanisms` when `frame == FRAME_RELEVANCE` and a graph
is supplied. `needs_review` is the union of the frame comparison's verdict
and the mechanism check's, not an override: a relevance case can pass plain
slot agreement (same words, or one containing the other) while the graph
finds neither model's claim grounded, and that combination -- two models
agreeing on an invented mechanism -- must not be masked by the slot
comparison having already said "agreed". Verified through the real entry
point rather than only the standalone module: the same agreed-but-invented
case used to test `mechanism_verification.py` in isolation was run again
through `cross_check()` itself, first without `concept_graph` (clean
agreement, invisible) and then with it (caught).

### Two more pieces wired: HypothesisYieldModel as a tool, and an optional guided fallback for the graph

Both closed a gap of the same shape found repeatedly across this work: a
well-designed module, tested in isolation, never connected to anything that
would call it.

**HypothesisYieldModel and ConfirmationRegistry existed independently.**
`training/hypothesis_yield.py`'s own docstring already states the design
choice that resolves a question raised directly: the estimator is "empirical
rates per feature bucket with Wilson intervals, not a fitted network" — data,
not weights, and for a stated reason (inspectable, and a thin bucket stays
visibly wide rather than pretending to knowledge). Because it is data, it
needs no weight-level integration with whatever model does the vetting —
`training/hypothesis_yield_wiring.py` exposes `estimate()` as a plain callable
plus a tool-spec dict in the same shape function-calling and MCP definitions
use, so it drops into whatever calling convention the vetting model's
integration already has, whether that model is a closed API or an
open-weight one hosted on-premise. `sync_from_registry()` connects
`ConfirmationRegistry.learning_set()` — confirmations already filtered for
independence, guarding against the automation-bias failure mode the registry's
own docstring names — through `outcomes_from_confirmations()` into the model.
Verified end to end: a non-independent confirmation is excluded from the
count; an independent one is observed; the tool call and a direct call to
`estimate()` return identical results, confirming the tool is a thin wrapper
and not a second implementation to drift from the first.

**The turn-by-turn graph exploration discussed and evaluated earlier is now
built, exactly as scoped: optional, and only as a fallback.**
`memory/guided_graph_expansion.py` gives a model a closed, named action
grammar over the graph — `neighbor(concept)`, `final(concept)`, `give_up()`
— never code execution, and every move is checked against the current
concept's *actual* edges before being accepted: a request to move to a
concept not on the offered list is refused as an ill-formed action, the same
treatment as any other output the walk cannot parse, not a followed
invented edge. The module's own docstring restates the trade evaluated
before building it — determinism, confirmation bias, cost, and test surface
against the mostly-hypothetical benefit of question-specific adaptivity —
and concludes by design that this module never decides *when* it runs; that
decision belongs to the caller.

Wired into exactly one caller, as scoped: `mechanism_verification.verify_mechanism`
gained a `fallback_model` parameter, tried only when the deterministic pass
returns `GROUNDING_NO_CONNECTION` — never when it found a connection but not
the claimed mechanism, since in that case the deterministic pass already has
an answer, just not the one asked about. A grounding reached this way sets a
distinct value, `GROUNDING_SUPPORTED_VIA_GUIDED_EXPANSION`, and
`via_guided_expansion = True` — never merged into the same state a fully
deterministic match produces, so a reader checking `grounding == "supported"`
specifically is not fooled by a result that came from a model-dependent walk.
`is_grounded` reads true for either, since both are genuinely grounded; which
one is which stays visible to anyone who needs to know. The matching between
a walk's found concept and the claimed mechanism now shares one function,
`_concept_names_match`, with the deterministic path's own matching — extracted
rather than duplicated, so the two comparisons cannot silently drift apart.

Verified directly: the case used throughout this investigation still resolves
to no connection without a fallback model; the same case with a scripted
guided model reaches the mechanism and is marked
`supported_via_guided_expansion`; and a case where the deterministic pass
already found *a* connection (just not the claimed one) never calls the
fallback at all, confirmed by a test that counts the calls.

### A bench for vetting, and a check on whether DPO's data already exists

Two decisions, taken with the same discipline the navigation bench established:
measure before choosing, and verify a claim rather than assume it.

**DoRA over GaLore, but nothing to change yet.** Checking the code rather than
recalling it found that no PEFT configuration exists in this project at all --
no LoRA, no fine-tuning code of any kind. The apparent matches were a
false positive ("exploratory" contains "lora"). The decision stands and is
recorded: DoRA's improvement over LoRA is measured and consistent (+0.84-0.88%
on GLUE, strongest in the low-rank regime, one flag in PEFT, no inference
overhead), while GaLore costs roughly four times as much and approaches full
fine-tuning in both capacity and risk -- a poor trade when confirmations
arrive as slowly as clinical ones do. But writing that configuration now,
before a base model is chosen and before the vetting bench says which
candidate to choose, would build another disconnected module.

**`evaluation/vetting_bench.py` measures the role nothing has measured.** The
project knows how candidates navigate documents; it knows nothing about how
they vet a hypothesis. The two reward different things -- navigation rewards
finding what is in the documents, vetting rewards proposing a mechanism that
is *not* in them and being right -- and this project has already been wrong
once by assuming a bench result transfers to a task it did not measure.

The bench reuses what exists rather than inventing scoring: a candidate is
asked for a `FRAME_RELEVANCE` answer, and `mechanism_verification.verify_mechanism`
scores it against a curated graph. **The graph is the judge, not another
model** -- the same refusal to appoint an adjudicator that governs
`root_model_cross_check`. Three rates are kept separate because they call for
different responses: `format_rate` (a formatting failure is prompt work),
`grounding_rate` (a grounding failure is a reason to prefer another
candidate), and `useful_rate` (naming the wrong mechanism between genuinely
connected concepts is wrong recoverably; claiming a connection the graph
does not know at all is not). `grounding_rate` divides by all runs, not by
well-formed ones -- three perfect answers among twenty malformed ones has not
earned 100%. Verified to separate a correct candidate, one inventing
mechanisms, and one producing prose, and a test asserts the fixture's own
expected mechanisms are graph-grounded, so the bench measures the candidate
rather than gaps in its own fixture.

**`training/preference_pairs.py` tests a claim made in discussion.** The claim
was that DPO's (prompt, chosen, rejected) shape already falls out of recorded
data -- the confirmed diagnosis as chosen, the alternatives raised for the
same case as rejected -- so no separate preference-collection pipeline would
be needed. It holds: the pairs extract cleanly, in the field names TRL and
Axolotl expect. But the same run quantified the qualification: of five cases,
one produced usable pairs. The rest were excluded for reasons each of which
is correct -- a non-independent confirmation (the automation-bias guard), a
case with no alternative to contrast against, and a case where the confirmed
diagnosis was never raised at all (a miss, counted as one, never turned into
a pair whose "preferred" answer the system never produced). That thinness is
itself an argument for DoRA over GaLore when the time comes.

### The vetting bench gets an execution path, and DoRA/DPO become code

A direct question -- "how do I run this, I don't see the action" -- found
the same gap as before, one level up: `vetting_bench.py` was a library with
scoring logic and no way to invoke itself against a real provider. No
script, no workflow.

`scripts/run_vetting_bench.py` closes it by reuse rather than duplication:
it imports `_http_chat_completion`, `_preflight`, and `CANDIDATE_MODELS`
directly from `run_format_adherence_bench.py` as a sibling module, so a slug
correction made there (the project has needed one more than once) is picked
up here automatically. One candidate per invocation, the same
`--candidate NAME` shape every other bench script uses, so
`.github/workflows/vetting-bench.yml` fans it into a parallel matrix the
same proven way rather than inventing a second orchestration pattern.
`gpt-oss-120b` — raised repeatedly in discussion, never added — is now in
`CANDIDATE_MODELS` itself, verified slug (`openai/gpt-oss-120b`, confirmed
against OpenRouter's own listing and OpenAI's own model page) rather than
carried as a special case in the new script.

**DoRA over GaLore and DPO over full RLVR are now code, not only a decision
recorded in a report.** `dora_config.py` and `dpo_config.py` are
configuration, deliberately not a training script: no PEFT configuration
existed anywhere in this project before this, no base model is chosen yet
(that is what the vetting bench exists to determine), and this sandboxed
environment has neither the disk space to install `peft`/`trl`/`torch` nor a
GPU to run them regardless -- confirmed directly rather than assumed, 2.9GB
free against dependencies that need far more. Both modules mirror the real
libraries' own parameter names exactly (`peft.LoraConfig`, `trl.DPOConfig`),
so `LoraConfig(**config.as_peft_kwargs())` and its DPO counterpart are a
straight drop-in wherever a real training environment exists, with no
translation layer to drift from the library's own evolving API.
`TrainingRunPlan` requires a base model and raises rather than defaulting to
a guess, since guessing it is exactly what the vetting bench exists to
avoid.

`dpo_config.build_training_dataset` does not reimplement
`preference_pairs.as_training_records` -- it calls it, verified identical
output on the same input, so the connection promised when DPO's data shape
was first verified is a real import, not a second copy that could drift.
`DpoReadiness` reports pair count against a recommended minimum without
enforcing it, the same posture `verify_mechanism`'s grounding states
take: visible, not silently gated.

### The vetting bench's first real run scored every candidate at zero, and the cause was in the bench, not the models

Claude Opus 5 and GPT-OSS-120B both scored `grounding_rate: 0.0` on the
first live run. One of GPT-OSS's answers for the CKD case stated
"secondary hyperparathyroidism" — the graph's own expected mechanism,
close to verbatim — and still scored `no_connection`. That is not a
plausible clinical failure for either candidate; it is a bench defect.

**The cause, found by reproducing the exact answers.** `verify_mechanism`
passed `factor`/`target` straight to `mediating_concepts`/`spread` as
literal strings. `spread`'s origin lookup requires an *exact* graph node —
`graph.edges_from(origin)` on a string that is not literally a node returns
nothing, silently. No real model spontaneously produces the graph's exact
node text: "chronic kidney disease (CKD)", a trailing full stop, "the
patient's chronic kidney disease" — each defeated resolution on its own,
confirmed directly by reproducing all four with the fix absent.

**The fix reuses code that already existed and was never connected.**
`concept_paths.mentioned_concepts(text, graph)` — built for finding which
graph concepts are named within a span of free text, longest match first —
already solves exactly this. `verify_mechanism` now resolves `factor` and
`target` through it before any traversal starts; a target that resolves to
no concept the graph recognises returns a new state,
`GROUNDING_NOT_CHECKABLE`, distinct from `GROUNDING_NO_CONNECTION` — the
graph genuinely searched and found nothing versus the graph never
successfully being asked at all. Verified against the real failing
answers: the CKD case, where the claim's text contains the graph's exact
mechanism, now resolves and grounds correctly.

**A second, separate finding surfaced once the first was fixed, and was not
treated as a second bug.** Claude's answers for the Marfan and sarcoidosis
cases remained ungrounded after the fix — but correctly reclassified as
`GROUNDING_CONNECTION_WITHOUT_THIS_MECHANISM` rather than `NO_CONNECTION`.
Checked directly: neither answer's text contains the graph's node name at
all ("granulomatous macrophages express 1-alpha-hydroxylase" versus the
graph's "granuloma formation" — different words for genuinely more
sophisticated, more upstream biology than the bench's small demonstration
graph represents as a single node). This is the already-documented
limitation of `_concept_names_match` — synonym resolution is deliberately
not attempted, a recorded miss rather than a silent guess — now visible
because the origin-resolution bug no longer masks it underneath a uniform
`no_connection`. Not fixed here: closing it would mean either a richer
graph fixture with intermediate biochemical nodes, or accepting the
limitation as already documented and scoped. Left as an open, separate
decision rather than folded into this fix.

### One shared concept-matching rule instead of two that happened to agree

A direct question -- could the same comparison technique already used to
compare Nemotron's and Gemma's answers to each other also be reused for
resolving a factor or target against the graph, instead of building a
separate mechanism -- was a sharper instinct than the first framing
suggested. It was not "use a model to judge equivalence" (that reintroduces
exactly the interpretive risk a deterministic graph exists to avoid); it was
"don't maintain two different deterministic comparison rules where one
would do."

Checked directly rather than assumed: `mentioned_concepts` (the fix in the
previous change) and `concept_names_match` (already used for mechanism
matching) gave identical results on every real failing case from the live
run. But a clean, isolated test -- the same three words, reordered
("kidney chronic disease" for "chronic kidney disease") -- showed them
diverge: `concept_names_match`'s word-set tier recognises it,
`mentioned_concepts`'s contiguous-substring search does not. They agreed on
the cases seen so far by coincidence, not by design.

**Not fixed by making `mentioned_concepts` more permissive.** It has a
second, existing consumer, `evaluation/grounding_judge.py`, on text that can
be much longer than a single factor or target -- loosening its matching
tolerance globally would change that consumer's behaviour without having
tested it for this change. Instead: `concept_names_match` moved to
`concept_paths.py`, made public, and a new `resolve_concept(text, graph)`
tries `mentioned_concepts` first (the stricter, already-relied-upon test)
and falls back to `concept_names_match`'s word-set tier only when the first
finds nothing. `mentioned_concepts` itself is untouched, confirmed by a test
that the reordering case it cannot handle alone is unchanged. Factor/target
resolution and mechanism matching now share one comparison philosophy,
reused, not two independent ones -- exactly what was asked for, achieved
without introducing a model into a step that has to stay deterministic to
mean what it claims to mean.

### A second live run found a hyphen the previous unification missed

`concept_names_match` used `normalise_concept` (lowercase, collapse
whitespace) rather than `mentioned_concepts`'s internal `_strip_punctuation`
(every non-alphanumeric character becomes a separator). A real GPT-OSS-120B
answer -- "connective\u2011tissue weakness", a non-breaking hyphen, not even
an ASCII one -- exposed the gap directly: `.split()` sees
"connective-tissue" as one token, never equal to the graph's two separate
words. Fixed by switching `concept_names_match` to `_strip_punctuation`
throughout, which also strengthens its exact/containment tier for free,
since a hyphenated and spaced-out form are now identical strings before any
containment or word-set logic runs at all. Verified against the exact
failing answer, and against the same red-line test applied throughout this
work: "pulmonary-embolism" still does not match "pulmonary oedema" merely
for sharing punctuation-stripped words.

The finding matters beyond the one bug: two functions this project committed
to keeping in sync (`concept_names_match` and `mentioned_concepts`) had
already drifted again, within the same investigation that unified them,
because each still normalised its own input independently. A shared
comparison rule is only as safe as the normalisation feeding it being shared
too.

### A bench for the differential, not for one claim at a time

The vetting bench measures a single factor/target/mechanism triple. That is
a real brick, but a direct observation in discussion identified what it does
not measure and what the system actually needs: given a case with several
findings, a spread of competing hypotheses in a sensible order, and the
judgement to decline ranking when there is not enough to conclude.

`MechanismEnumerator` already produces exactly that shape --
`EnumerationOutcome` carries either ranked `hypotheses` or `open_questions`
and picks the register itself from local graph density. Nothing measured
whether it picks well. `evaluation/enumeration_bench.py` does, scoring four
properties kept deliberately separate because they fail independently:
recall (is the confirmed condition in the spread at all), ranking (is it
near the top -- present-but-eleventh is a lesser failure than absent, and
one number would hide which happened), restraint (does it emit questions
rather than rank noise where the graph cannot support a conclusion), and
question quality (do those questions name findings and conditions actually
at issue, or filler a count-only metric could not distinguish).

**Building the restraint fixture took three attempts, and the two failures
are worth recording.** A weakly-attested edge still yields density 1.0, and
so does an unknown-strength `ConceptEdge.unknown()` edge: `local_density`
measures whether the *findings themselves* are mapped, not how strongly the
graph knows what it knows about them. "The graph knows this corner but knows
little" and "the graph has never heard of these findings" are different
conditions, and only the second is what restraint is for. The fixture now
uses findings genuinely absent from the graph, with a test asserting that
absence so the case cannot silently stop testing what it claims to.

First run: recall 100%, top-rank 100%, restraint 100%, question quality
100% -- the enumerator ranks correctly where the graph supports it and
abstains with four pertinent questions where it does not. A bench everything
passes discriminates nothing, so this is a floor to build harder cases on,
not a result to stop at.

### Graph persistence and candidate retrieval: the two pieces everything else waited on

Identified in discussion, in this order, and the order was right.

**Persistence was the real bottleneck.** `InMemoryConceptGraph` is a Python
list rebuilt from the HPO file on every start. An edge promoted by
`ConjectureLedger` after three independent confirmations would exist until
the process exited and then be silently lost -- which made the dream-replay
work, the confirmation ledger, and any weight adjustment from real outcomes
ceremonial rather than real.

`memory/graph_store.py` keeps learned edges in their own JSONL file, apart
from the imported layer, and that separation is the central design choice:
the imported layer is derivable (re-import a newer HPO release any time), the
learned layer is not (it is the accumulated product of confirmed cases).
One file would mean an HPO refresh either destroys what was learned or needs
a merge that has to get the distinction right anyway. Append-only, because an
edge justified by accumulated evidence should not be silently rewritten;
JSONL, because promotion appends one line rather than rewriting a store, and
a truncated write costs one line rather than everything. Intervals round-trip
exactly, verified -- an edge that loses its bounds becomes indistinguishable
from an unknown one, which is the distinction the whole interval design
exists to preserve.

**Candidate retrieval was the enumerator's missing half.**
`MechanismEnumerator.enumerate(findings, candidate_conditions)` answers "of
these conditions, which does the graph connect to these findings" -- not
"which conditions could explain these findings at all". Wiring it into the
pipeline without this would have connected a machine with nothing to chew on.

`memory/candidate_retrieval.py` walks outward from each resolved finding and
gathers what the graph links to it, ranked by how many of the case's findings
each condition touches -- breadth before strength, the same reasoning the
convergence reward already encodes: independent corroboration outweighs a
single strong link. Findings resolve through `resolve_concept` first, for the
reason `verify_mechanism` needed it, and an unresolvable finding is reported
rather than dropped, since a coverage gap is exactly what the density check
downstream reasons about.

**A defect caught in the first run, and how it was fixed.** The retrieval
ranked "night sweats" among the candidates -- a symptom proposed as a
diagnosis, reached two hops out through a disease it shares with the case's
own findings. The fix uses edge *direction*, not relation-name matching:
HPO imports as `disease -> has_phenotype -> finding`, and
`InMemoryConceptGraph` prefixes generated reverse edges with `inverse_`, so a
concept reached by traversing backwards is on the disease side. Matching
relation names instead would have hard-coded an assumption about HPO's
vocabulary that a second imported source (LOINC, ATC) would break.

Verified end to end: from three findings alone, with no caller supplying
candidates, the chain produces sarcoidosis, lymphoma and tuberculosis ranked
in that order -- and still abstains with knowledge-gap questions on findings
the graph has never heard of.

### The bridge between what the RLM reads and what the graph knows

Identified directly in discussion as the missing link, and it was: the RLM
read a patient's documents and never touched the concept graph; the
enumerator walked the graph and never read a document. Each was a complete
half of a diagnostic reasoner with no path between them.

`reasoning/rlm_graph_bridge.py` runs in both directions. **Documents to
graph**: findings the RLM located become entry points for
`retrieve_candidates` and then `MechanismEnumerator`, so the differential is
grounded in what this patient's documents say rather than a candidate list
someone supplied. Findings are read from the trajectory's evidence
fragments, not only its final answer -- the answer is one sentence, the
fragments are everything the run touched.

**Graph back to documents**: a ranked hypothesis predicts findings that
should be present if it is right, and `predicted_findings` returns those,
excluding what the case already shows. A ranking nobody can act on is not
yet useful; this is what turns it into something checkable.

**And the direction raised in the same discussion -- the RLM's own
conjectures.** An RLM reading documents may notice a correlation the graph
has no edge for. `vet_rlm_claims` puts each through the *same*
`verify_mechanism` the cross-check and the vetting bench use, deliberately
not a gentler path for the system's own ideas, since a separate lenient
route is how a system starts trusting its own output. Three outcomes are
kept distinct: grounded (the graph confirms), candidate conjecture (the
graph knows both concepts but has no connection -- new material for
`ConjectureLedger` to hold pending confirmation), and not checkable (the
graph cannot resolve the concepts at all, so there is nothing to hold).

Every part of the output declares its origin -- read from a document,
supplied by the graph, or merely proposed by the RLM -- so a reader
downstream never has to guess which is which.

Verified end to end on a realistic trajectory: three raw document fragments
produce sarcoidosis/lymphoma/tuberculosis ranked, "night sweats" returned as
a finding to look for, and an RLM conjecture correctly held for the ledger
rather than accepted or discarded.

### The assembly point, and the learning loop finally closing

This project built a correct, tested module and then never called it seven
times over: `MechanismEnumerator`, `ConjectureLedger`, `HypothesisYield`, the
guided expansion fallback, the RLM-graph bridge, and before them Muse Glimmer
and `root_model_cross_check`. `reasoning/diagnostic_assembly.py` is the
answer to the pattern rather than to any one instance -- one file where the
wiring lives, so "is it connected?" has a place to look rather than a search
across the codebase.

Nothing was modified to fit. The assembly composes components that keep their
own behaviour: it builds the graph from imported plus learned layers, binds
an enumerator to that graph, supplies the candidate retrieval the enumerator
was missing, and routes what the RLM read through the bridge. The judgement
calls -- which model, whether the cross-check is always on -- stay with the
caller, because they are decisions about the system rather than about how its
parts fit.

**`promote_confirmed` is deliberately a separate call, not something
`run_case` does.** Promotion changes the shared knowledge base; it should
happen when someone runs it, not as a side effect of answering one patient's
case. Verified by a test asserting that running a case alone writes nothing.

**The loop now closes, verified across a simulated restart.** A case runs,
its hypotheses' leaps are recorded as conjectures, three independent
confirmations arrive, `promote_confirmed` writes an edge with its Wilson
interval and a `learned:` provenance into the persistent store -- and a
freshly assembled process sees it. `[0.44, 1.00]` on three confirmations,
visibly not certainty, which is what the interval is for.

**And the DreamTrainer hook is finally usable.** `_enumerated` returns None
unless `case_context` carries `candidate_conditions`, and nothing in the
pipeline ever put them there -- which is why the trainer fell through to
placeholder rehearsal labels on every real case. `dream_context_for` builds
that context with the candidates retrieved from the graph, so the hook has
something to find.

### Docling removed; Nemotron-Parse the default, LlamaParse an optional cloud-only cross-check

Decided after a direct comparison found no evidence Docling was ever the
strongest choice specifically for clinical documents: one 2026 assessment
calls it "weaker on complex layouts" against current leaders, a
clinical-document-specific comparison names a different tool "the benchmark
solution" for exactly this domain. Keeping it wired in as an unused default
would have left a reference to a decision this project no longer holds.

Two real code sites carried it, not only documentation:
`data/document_processing.py` (`_docling_available`, `load_with_docling`,
`docling_integration_plan`, `prefer_docling`) and one entry in
`orchestration/model_capability_registry.py`. Both replaced, not patched
around -- the graceful-degradation contract (report unavailable rather than
raise, fall back to plain text) preserved exactly, so no caller handling
that status needs to change.

**Nemotron-Parse is the default because it is the only genuinely on-premise
option.** Checking further on the LlamaParse claim from an earlier report
found it did not hold: a more specific source states LlamaParse "does not
offer true on-premise deployment -- VPC is the closest equivalent." Open
weights with no vendor API dependency is what a hard on-premise requirement
actually needs, and Nemotron-Parse is that; a hosted parser with a VPC option
is not, regardless of accuracy.

**LlamaParse is kept as an optional cross-check, off by default, cloud-only.**
`also_cross_check_with_llamaparse` defaults to `False` specifically so an
on-premise-only deployment does not report a spurious "unavailable"
cross-check on every call. Where cloud deployment is already accepted, its
documented strength on clinical tables and mixed formatting is worth testing
against Nemotron-Parse's own output -- the same double-reading-at-ingestion
principle already agreed for this project, since an ingestion error is more
costly than a navigation error: it propagates silently into everything read
afterwards.

Both parsers are called over HTTP against a configured endpoint, the same
pattern the bench scripts already use for every model candidate in this
project, rather than a local pip import -- neither is a simple installable
library the way Docling nominally was. The actual transport call is left
unimplemented (`NotImplementedError`) deliberately: the exact request shape
depends on how the endpoint is deployed, a configuration decision for
whoever operates a given installation, not something to hard-code here.

### The RLM-conjecture origin, refined: a citable qualifier, not a fourth branch

A proposal to add `proposed_from_literature` as a fourth origin alongside
`read_from_document`, `supplied_by_graph`, and `proposed_by_rlm` was
reconsidered after a direct objection: a claim citing retrieved literature is
still produced by the RLM (Claude or GPT-OSS-120B), not a peer source
competing with the graph or the patient's chart -- correct, and the original
proposal conflated "who produced this claim" with "how verifiable is it".

The distinction actually worth preserving is the second one. An RLM
conjecture with no citation has nothing to check it against, which is why it
is held in `ConjectureLedger` pending independent confirmation. An RLM
conjecture citing a specific retrieved paper already carries an
independently checkable reference -- a reviewer can open that one paper and
verify it today, without waiting for confirmations to accumulate. That
distinction belongs on `VettedClaim` as a qualifier (which citations support
this conjecture, if any), not as a separate top-level origin -- the claim
stays in the `proposed_by_rlm` branch either way, since that is genuinely
what it is. Not yet implemented; recorded here so the eventual literature
retrieval work builds the field in the right place from the start.

### Literature as retrieval, and the citation qualifier the origin design actually needed

Two outstanding items from the same discussion, implemented together because
the second depends on the first.

**`memory/literature_index.py`: retrieval, never training.** Four arguments
decided this rather than one preference. Literature ages faster than a model
can be retrained -- the mirror of the reasoning that chose DoRA over GaLore
for slowly-arriving clinical confirmations. Training would work on an
open-weight engine and never on Claude, so two engines enriched unequally
would stop being comparable, which is the whole premise of running both. A
weight has no citation, while a retrieved passage carries its own reference
-- the same reasoning that chose DPO over full RLVR for auditability. And
the risk is already documented here: Meditron was excluded as a navigation
model because medical fine-tuning erodes format adherence.

Concept-matched rather than embedding-matched, deliberately. Not because
embeddings are wrong, but because this project already measured what latent
similarity costs clinically -- a RAG system asked about heart failure
retrieving "acute coronary syndrome" because it is close in latent space,
not because it answers the question. Matching on concepts the graph already
names keeps retrieval in the same vocabulary as the rest of the reasoning,
and leaves an embedding layer as something a deployment adds rather than an
assumption baked in.

`is_independently_checkable` gates retrieval by default: a source a reviewer
cannot open is an assertion wearing a bibliography, which is worse than no
citation because it looks like one. Passages are stored regardless -- hiding
them would conceal that the index holds unverifiable material -- but never
retrieved into a vetting context unless asked for explicitly. And
`as_vetting_context` puts each citation *before* its passage, since a model
attributes what it reads to whatever is nearest and a trailing reference is
easy to lose; truncation drops whole passages rather than cutting text, so a
citation never covers something its source did not finish saying.

**The citation qualifier, correcting an earlier proposal.** A fourth origin,
`proposed_from_literature`, was proposed alongside `read_from_document`,
`supplied_by_graph` and `proposed_by_rlm`. A direct objection rejected it and
was right: a claim citing retrieved literature is still produced by the RLM,
not by a source standing peer to the graph or the patient's chart. The
proposal had conflated "who produced this" with "how checkable is it".

`VettedClaim.citations` and `is_citation_supported` implement the second
question where it belongs -- as a qualifier inside the existing branch. Two
conjectures the graph cannot confirm are not equally checkable: one with no
citation has nothing outside its own assertion, which is why
`ConjectureLedger` holds it for confirmations measured in cases and months;
one citing a specific paper carries a reference anyone can open today. Both
remain conjectures, and a test asserts that citations never make an
ungrounded claim grounded -- promoting on a citation alone would be trusting
the model's own reading of a paper it selected itself.

### Two literature connectors, populating the index built for exactly this

`literature_index.py` shipped with storage and retrieval but no way to fill
it. Two connectors close that, chosen for being genuinely complementary
rather than redundant.

**`connectors/europe_pmc.py` is the primary source, not raw PubMed
E-utilities.** PubMed does not index full text -- Europe PMC aggregates
PubMed's citations with PMC full text and life-science preprints (bioRxiv,
medRxiv) in one search, no API key required. `connectors/pmc_case_reports.py`
already wraps E-utilities for a different job entirely (fetching case-report
full text for evaluation cases via the OAI endpoint) and was left untouched
-- this is a separate connector for a separate purpose, not a replacement.

Every result becomes a `LiteraturePassage` only when it has both a real
title and abstract; a record with neither is skipped rather than stored
empty, since an empty passage can never match a search and would only
inflate a count past what is actually retrievable. Identifier preference is
PMID, then PMCID, then DOI -- PMID is the most universally resolvable of the
three -- and a record with none still constructs, since
`is_independently_checkable` already exists to flag that case; dropping it
here would duplicate that check in a second place.

**`connectors/clinical_trials.py` is a different kind of source, not a second
copy of the first.** A trial record is not a finding about established
mechanism the way a published paper is -- it is a registration of ongoing or
completed investigation. Where Europe PMC answers "does the literature
support this mechanism", ClinicalTrials.gov can answer "is there a study
this case could be referred to, or an outcome already reported here" --
closer to the further-investigation role `OpenQuestion` already serves for
the graph, applied to the wider research landscape rather than to what the
graph itself contains. Terminated and withdrawn trials are excluded by
default: a stopped trial is not a place to refer a case or a source of an
outcome to cite, and excluding it explicitly rather than silently makes that
a choice, not an oversight.

Kept in `LiteraturePassage`'s own shape rather than a new type: an NCT
number is exactly as openly resolvable as a PMID (at
clinicaltrials.gov/study/NCT...), added to `CHECKABLE_ID_PREFIXES` --
introducing a second passage type would have fragmented
`LiteratureIndex.search` into two code paths for what is, from the
retrieval side, one operation.

Both connectors share one `RateLimiter` (defined in `europe_pmc.py`, plain
N-requests-per-second pacing with no source-specific fields) rather than
importing `pmc_case_reports.RateLimiter`, which sits beside NCBI-specific
configuration neither of these needs -- shared between connectors with the
same actual contract, not coupled to one with a different one. Verified end
to end: both connectors populate the same `LiteratureIndex`, and a search
for the same concepts returns results from both sources together, each with
its own distinct, checkable citation.

### The vetting bench, widened for a decision that actually holds: 16 cases, restraint scoring, confidence intervals

Asked directly for a bench precise and broad enough to decide between
candidates, not just measure them once. Four things changed, and each closes
a specific gap the two live runs had already exposed.

**The graph gained the biochemistry real answers actually cited.** Both live
runs showed Claude and GPT-OSS describing genuine intermediate steps --
1-alpha-hydroxylase activity, calcitriol excess, a fibrillin-1 mutation --
that the nine-edge v1 graph collapsed into a single link, scoring a correct,
more detailed answer as unfounded. The graph now carries both the coarse
chain and the detailed one for sarcoidosis, and the upstream genetic cause
for Marfan, so either level of detail grounds correctly.

**Case count went from four to sixteen, across ten distinct mechanisms and
organ systems** (renal, endocrine, electrolyte, haematologic, cardiovascular,
neurologic, metabolic, rheumatologic, hepatic, and connective tissue) rather
than four cases repeated. Breadth is asserted by a test, not just claimed --
`test_the_case_set_spans_multiple_organ_systems` checks the distinct-mechanism
count directly.

**Restraint is now a measured property, not assumed.** A candidate that
invents a plausible connection where the graph has none is a more dangerous
failure than one that gets a real mechanism wrong, and grounding_rate alone
cannot see it -- a candidate could ground every real connection perfectly
while confidently fabricating ones that do not exist, and the rate would
never reflect it. Three restraint cases pair a real factor and target the
graph genuinely does not connect (verified directly:
`test_every_restraint_case_genuinely_has_no_path` asserts no shared neighbour
exists), and correct behaviour is `bears_on: no`, or a proposed mechanism the
graph itself then finds unsupported. A demonstration case
(`_grounded_but_reckless_model`) scores identical grounding_rate to a fully
correct candidate while failing every restraint case, and ranks below it --
proof the property catches what grounding_rate alone would miss.

**`grounding_wilson_lower`, not the raw point estimate, is what ranking
actually sorts on.** 3/4 and 12/16 are both 75% grounded, but the first could
plausibly be anywhere from roughly 30% to 95% while the second is pinned
much tighter -- the same Wilson interval this project already uses for HPO
frequency parsing, HypothesisYield, and ConjectureLedger promotion, reused
here rather than reinvented, for the same reason: a proportion from a small
sample should say so rather than being compared as if it were exact.

**`trials_per_case` repeats every case and folds the trials into one
result**, since a single trial reports what a candidate did once, not what
it reliably does -- exposed via `--trials` on the runner script (default 2,
trading a wider confidence interval against real API cost) and reflected in
the workflow's summary table and merge-ranking logic, both updated to sort on
restraint first, then the Wilson lower bound, matching `rank_vetting_results`
exactly rather than drifting from it in a second implementation.

### A real vetting-bench run revealed a frame-parsing bug: excess pipes silently discarded

Two uploaded runs both showed a recurring, specific artifact:
`claimed_mechanism` values reading as bare fragments like "channel" --
clearly not a complete clinical explanation. Traced to
`parse_frame_answer`'s positional split: `raw.split(SLOT_SEPARATOR)` followed
by taking `parts[index]` for each declared slot silently discarded
everything past the fourth segment whenever a model's own free-text
mechanism used the pipe character itself -- arrows between biochemical
steps, alternatives, a chain of clauses. Whatever segment happened to land
at the mechanism index survived; the rest, including whatever contained the
graph's own matching term, was lost with no signal that it had been.

Verified directly: a constructed six-segment answer where the real,
graph-matching term ("secondary hyperparathyroidism") sat in the fifth
segment extracted only the fourth ("reduced calcitriol synthesis") before
the fix, and the term the graph would have recognised was silently gone.

Fixed by having only the *last declared slot* absorb any excess segments,
rejoined with the same separator rather than discarded -- free text by
construction in every frame that has one, and the only slot where a model's
own pipe usage is plausible in the first place. `factor`, `target`, and
`bears_on` keep their exact positional extraction unchanged, since widening
the fix to every slot would risk a different bug for no benefit. Verified
end to end: the same six-segment case now grounds correctly against the
recovered term, and a normal four-segment answer is byte-identical to what
it produced before this fix.

This was caught by comparing two live vetting-bench runs against `main`
directly, the same discipline applied throughout this project: a
suspicious, repeated pattern in real output is a reason to reproduce it
against current code before accepting the numbers at face value, not after.

### A third live run's leaked reasoning exposed the real cause behind three runs of bad numbers: the wrong system prompt

A third uploaded vetting-bench run contained something the first two had
only hinted at: one malformed answer's raw text was Claude's own leaked
reasoning, agonising over "the document environment", "what grep returned",
and "budget for one more read" -- for a question that was never about a
document at all. GPT-OSS's own malformed answers showed the same pattern
from the other side: `search(...)`, `grep(...)` sequences, and in one case
the model wrapping its actual answer inside `final(hereditary
haemochromatosis|cirrhosis|yes|iron overload)` -- the navigation engine's own
completion syntax, with a stray `<|return|>` control token leaking through.

Both candidates were producing document-navigation actions because they were
being told, explicitly, that they were in a document-navigation task.
`run_vetting_bench.py` reused `run_format_adherence_bench.py`'s
`_http_chat_completion` -- sound reuse of real HTTP-calling infrastructure,
as documented when that script was built -- but that function hard-codes
`_SYSTEM_PROMPT`, the navigation bench's own instruction: *"You navigate a
document environment by emitting exactly one action per line, chosen from:
[grep, slice, search, describe, expand, query, final]"*. Reusing the
function meant reusing that prompt too, unchanged, for a task with no
document environment at all. This was likely the largest single contributor
to the low format rates and malformed answers across all three uploaded
runs -- not a candidate quality signal, a setup bug shared by both
candidates identically.

Fixed by parameterising `_http_chat_completion` and `_bind` with an optional
`system_prompt`, defaulting to the navigation prompt so every existing
navigation-bench caller is unaffected -- verified directly, with a mocked
`urlopen` confirming the navigation bench still receives its original prompt
byte for byte. `run_vetting_bench.py` now supplies its own
`VETTING_SYSTEM_PROMPT`, stating plainly that there is no document to
navigate and naming every navigation verb it must not emit, since the
concrete failure a live run exposed was a model believing `final(...)` was
still the right way to close its answer.

### Restraint gets a confidence interval, and the number it replaces was badly overstated

Grounding had a Wilson interval from the moment the bench was widened;
restraint did not, despite resting on the thinner sample of the two -- a
handful of restraint cases against a couple of dozen conclusive ones. That
asymmetry made the weaker number look like the stronger one.

The correction is not cosmetic. The most recent live run reported 66.7%
restraint for both candidates, which reads as a settled finding. Its actual
95% interval is **[30.0%, 90.3%]** -- four correct declines out of six
establishes almost nothing. The same 66.7% observed over sixty cases would
be [54.1%, 77.3%], a genuinely usable number. Reporting the point estimate
alone had been implying a confidence the evidence never supported.

`rank_vetting_results` now sorts on `restraint_wilson_lower` rather than the
raw rate, matching what grounding already did and for the same reason: two
candidates both declining every restraint case are not equally established
if one faced two cases and the other twenty. The workflow's summary table
and merge-ranking were updated together, so the CI view and
`rank_vetting_results` stay one implementation rather than drifting apart.

### The bench was scoring against a 33-edge fixture, and the normalisation cascade that followed

An audit prompted by a direct question -- why does the graph not know
sarcoidosis can cause villous atrophy -- found something larger than the
missing edge. Two graphs exist in this codebase and had been conflated: the
285,598-edge HPO import, and a 33-edge hand-written fixture with
`provenance=None` on every edge, written to make the vetting bench runnable.
**The bench used the fixture.** Three live runs had been measuring how
closely a candidate's phrasing matched thirty-three hand-written lines, then
reporting it as grounding against the concept graph.

Worth separating from the scale problem: the sarcoidosis-villous-atrophy link
is real medicine -- a review of 305 gastrointestinal sarcoidosis cases finds
the duodenum among the most affected sites, and documented cases pair
sarcoidosis with duodenal villous atrophy and malabsorption. Claude's
"restraint failure" on that case was clinically correct and the fixture was
wrong. That reframes the 100%-restraint goal entirely: measured against a
graph, restraint asks a model to be as ignorant as the graph is, and a model
that knows more medicine will always "fail" it.

`memory/graph_sources.py` closes the conflation. Every load returns a
`GraphSource` naming its source and size, and `require_real_data=True`
raises rather than falling back -- the setting a selection-deciding bench run
should use. Falling back is allowed; falling back silently is what made the
original error undetectable. A structural limit is recorded there too and
remains open: HPO imports a single relation type, `has_phenotype`, and
encodes no mechanistic causal chains, which is exactly what a vetting
question asks about. Connecting to the real graph fixes scale, not kind.

**The cascade, built in the order proposed: tiers by determinism, not just
cost.** Tier 1 is the existing lexical rule, reused rather than
reimplemented. Tier 2 is SapBERT-style embedding similarity -- a bi-encoder
self-aligned on UMLS synonym pairs, deterministic at runtime though learned.
Tier 3 extracts structure from both the claim and a concept's cached
literature description and compares those structures arithmetically. The
ordering means the least reproducible tier only ever sees what the
reproducible ones could not resolve, and `NormalisationResult.is_deterministic`
plus `usage_report()` make visible how much of any result rested on a
generated step.

Two guards on tier 2, not one. A test with a deliberately degenerate embedder
-- same vector for every input -- exposed that a threshold alone judges the
winner in isolation: every concept scored 1.0, passed any threshold, and the
first one encountered won by accident. The runner-up margin judges whether
there was a winner at all.

Tier 3's design was proposed in discussion and is better than the obvious
alternative it replaces. Asking a model "do these two phrases mean the same
thing?" is adjudication that cannot be checked, on precisely the question
the graph exists to answer deterministically. Having a model *extract*
entities and relations, and deciding overlap arithmetically, keeps the
judgement deterministic while letting a model do what it is good at. The
concept side is cached because it is re-derivable and reused across every
case; relations are weighted above entities in the overlap score because two
texts about the same clinical area share entities easily, and only a shared
relation suggests the same mechanism rather than the same subject.

Wired into `verify_mechanism` via an optional `cascade`, tried only after
lexical matching fails and only against the candidates the graph already
surfaced -- letting looser tiers roam the whole concept set would admit a
distant concept that happens to embed closely. `normalisation_tier` on the
result records which tier grounded it, so an exact match and an
embedding-neighbourhood match are never reported as the same thing.

### HPO's layperson synonyms: parsed, then found being silently discarded, then bridged into the cascade

A direct question -- should we use HPO's own layperson synonym translations
-- led to a real defect, not just an opportunity. `concept_resolution.py`
already parsed OBO synonym lines (`_parse_synonym`), built months earlier for
work that was never connected to production. It captured the OBO scope
(EXACT/BROAD/NARROW/RELATED) but silently discarded the type tag that
follows it -- "layperson" in `synonym: "Big head" BROAD layperson [...]` --
because only `remainder[0]` was ever kept. And `SAFE_SCOPES` admits EXACT
only by default, so even with the tag captured, every layperson synonym
(always BROAD-scoped, never EXACT) would have been excluded regardless.

Fixed in two parts, kept deliberately separate. `_parse_synonym` now returns
`(text, scope, type_tag)` rather than discarding the third piece.
`surface_forms(include_layperson=True)` is a new, explicitly opt-in
parameter -- not a wider `SAFE_SCOPES` -- because a layperson synonym is safe
to treat as the same concept specifically for being layperson-tagged, not
because BROAD synonyms became safe generally. Widening `SAFE_SCOPES` itself
would have quietly admitted every other BROAD synonym too: genuinely
broader, different concepts, which is exactly the imprecision this scope
distinction exists to keep out. A test constructs the case directly: a
generic `BROAD` synonym with no layperson tag stays excluded even when
`include_layperson=True`.

**The bridge into `concept_normalisation`'s tier 1, not a new tier.**
`NormalisationCascade` gained an optional `synonym_index`; the lexical tier
now checks a claim against a concept's curated synonyms (via `TermIndex`,
looked up by surface form rather than assuming the graph's concept name is
itself a term id) in addition to its bare graph label. This stays tier 1 in
spirit and in determinism: every synonym checked came from a curated
ontology release, not a guess, so matching against one is exactly as safe as
matching against the node's own name -- what changed is how many
known-correct strings a concept has, not how the comparison works.

**`graph_sources.py` gained the missing loader.** The HPOA loader from the
previous change had no counterpart for hp.obo, so there was no way to
actually populate a `TermIndex` from a real file without doing it by hand.
`find_hp_obo_file`/`load_synonym_index` mirror the HPOA loader's honesty:
`load_synonym_index` returns `None` when no file is found, not an empty
index, so "no synonym data available" and "a file parsed into nothing" stay
distinguishable rather than looking identical from the outside.

### Two new annotation files, a new relation type the graph never carried

Asked directly whether the project should use HPO's other annotation
files -- `genes_to_phenotype.txt`, `phenotype_to_genes.txt`,
`genes_to_disease.txt` -- alongside `phenotype.hpoa`. These add gene
involvement, an edge type the graph has never modelled at all (HPO's
annotation file gives only `has_phenotype`).

`gene_annotations.py` parses both, header-driven like `parse_hpoa` rather
than assuming a fixed column order: `genes_to_phenotype.txt`'s header was
confirmed from a working parse (`entrez_gene_id`, `entrez_gene_symbol`,
`hpo_term_id`, `hpo_term_name`, `frequency_raw`, `frequency_hpo`), and an
early version of this parser used the wrong field names for it -- a
self-caught error, fixed by making both parsers alias-tolerant across the
column-naming variants different HPO release generations have used, rather
than committing to one and failing silently on the other. An unrecognised
header raises rather than silently producing zero associations with no
explanation. `genes_to_disease.txt`'s exact header was not independently
confirmed to the same standard; the alias mechanism and an explicit code
comment flag this as worth checking against whatever release is actually in
use.

Edge weight is uniform (1.0) rather than invented: neither file carries a
per-association strength the way `phenotype.hpoa`'s frequency column does,
and assigning one would fabricate precision the source data does not have.

### The real HPO data arrived, and it exposed four defects the fixture had hidden

With `phenotype.hpoa`, `hp.obo`, and the gene annotation files placed in
`data/`, the graph could finally be loaded for real. It did not load. Each
problem below was invisible on the 33-edge hand-written fixture and obvious
within minutes of running against 573,302 real edges.

**1. A quadratic lookup.** `InMemoryConceptGraph.edges_from` scanned the
whole edge list *twice* per call, re-normalising both endpoints of every
edge and allocating a fresh inverse ConceptEdge for each match. Loading the
real graph did not finish inside five minutes. Indexed by normalised concept
on construction, with inverse edges pre-built once: **over 300s to 2.8s**. A
regression test asserts 200 lookups on a 20,000-edge graph complete in under
half a second, so the fixture can never hide this again. A second defect
surfaced while writing that index -- the pre-built inverse dropped
`lower`/`upper`, silently turning an interval-valued edge into a point
estimate the moment it was traversed backwards -- caught by its own test.

**2. Phenotypes were bare HPO ids.** `phenotype.hpoa` identifies phenotypes
only as `HP:0001166`, and `annotation_to_edge` already had a `label_for`
parameter for exactly this, which nothing passed. Worse than unreadable: every
comparison downstream works on clinical text, and `HP:0001166` is not text
anyone writes, so an id-shaped graph would have matched nothing while
appearing to work. `hp.obo` supplies the map, and `GraphSource.detail` now
records which case applied.

**3. `genes_to_disease.txt` has no disease name column at all.** The real
header is `ncbi_gene_id`, `gene_symbol`, `association_type`, `disease_id`,
`source`. The parser required a `disease_name` field and silently dropped
every row -- `FBN1 -> causes_disease` returned nothing. Names are recovered
from `phenotype.hpoa`, which indexes the same identifiers and does carry
them; a row with no name available still yields an edge targeting the bare
id, which is at least traversable and visibly an id rather than silently
absent.

**4. Genes were being proposed as diagnoses.** Gene edges point
gene -> phenotype, so traversing one backwards from a finding lands on a
gene, which passes `candidate_retrieval`'s reached-by-reverse test. Retrieval
for an aortic finding returned AEBP1, ALG9 and B3GALT6 alongside the actual
syndromes. Excluding them exposed a second, subtler error in the first fix:
it turned every gene into a dead end, because a disease reached *forward*
from a gene via `causes_disease` is not reached by a reverse edge. The
reverse-only rule encodes "diseases sit on the source side of their edges",
which holds for `has_phenotype` and breaks for `causes_disease`. Both are now
admissible, and two tests pin the distinction: a gene is never a candidate,
and a gene is always a waypoint.

The graph now loads in **6.1 seconds: 1,273,466 edges across 29,053
concepts**, including 333,983 gene-annotation edges. `FBN1 -> causes_disease`
returns Marfan syndrome, MASS syndrome and familial ectopia lentis by name,
and retrieval from two connective-tissue findings returns connective-tissue
syndromes rather than gene symbols.

### MAxO: measured before wiring, and it does not answer the question it was proposed for

Asked directly whether MAxO is sufficient to discriminate between conditions
and reach a diagnosis. Measured against the shipped file, the answer is no,
and the earlier suggestion that MAxO could answer `OpenQuestion`'s "which
test would settle this" was wrong.

**The annotations carry no diagnostic relation at all.** All 438 rows are
401 TREATS, 34 PREVENTS, 2 NO_OBSERVED_BENEFIT, 1 CONTRAINDICATED. Not one
says a procedure distinguishes A from B. The file answers "what do you do
once you know it is A" -- a different question from the one a differential
asks.

**Coverage is 1.6%**: 202 diseases out of the 12,880 in `phenotype.hpoa`.

It is still worth parsing. A CONTRAINDICATED or NO_OBSERVED_BENEFIT row is
clinical information nothing else in this project has, curated with a PMID
attached -- the shipped file documents, for instance, sodium channel
inhibitor therapy as contraindicated in Dravet syndrome (PMID:9596203).
`cautions_for` surfaces exactly those rows, separated from treatments
because they are the asymmetric case: at 1.6% coverage a missing TREATS row
is uninformative, while a present CONTRAINDICATED row is a positive
statement that applies whenever that disease is under consideration.
`coverage_note` travels with any output derived from the index, because a
reader who does not know the coverage figure cannot tell "nothing is
recommended" from "nobody has annotated this".

**These edges are deliberately kept out of the concept graph.**
`MedicalActionIndex` is not a `ConceptGraphView` and a test asserts it has no
`edges_from`. Treatment relations alongside `has_phenotype` would let a
differential traverse "disease -> treated by -> physical therapy -> treats ->
other disease" and surface two conditions as related because they share a
therapy, which is not a diagnostic connection. The cleanest prevention is to
give those edges nowhere to be traversed from.

### A weekly update workflow that proposes rather than applies

`.github/workflows/data-and-dependency-updates.yml` checks the HPO release
and the Python dependencies weekly, or on demand.

The central choice: it opens a pull request, never commits to main. These
files are a clinical knowledge base, and an HPO release can rename and
obsolete terms in bulk -- the 2026-02-16 release renamed 75 and obsoleted 29.
A renamed term is one the concept graph no longer matches under its old
label, so a silently applied refresh would move every downstream number with
nobody having seen the diff. The full test suite runs against the new data
before the PR opens, so a reviewer sees either a green run or precisely what
broke.

The current release is read from `phenotype.hpoa`'s own header rather than a
version file someone must remember to update. Dependencies are reported but
never bumped in the same change: a dependency that alters tokenisation or
numerical behaviour would move bench results, and that belongs in its own
reviewed change rather than arriving alongside a data refresh.

### Term history: nothing renamed is ever lost, corrected from a design this project got wrong

A direct correction to the previous change: the update workflow had treated
a rename as a risk to flag for review, and stopped there. That is not
sufficient -- a term must never stop being recognisable under a name it used
to have, and renaming is not removal.

`term_history.py` records every rename and obsoletion permanently, append-only,
the same discipline as `graph_store`'s learned layer: a term's history is
discovered once, when first detected between two consecutive releases, and
cannot be rederived afterward, since the old release is gone by then. Bridged
into `NormalisationCascade`'s tier 1 through `TermHistoryStore.synonyms_by_term_id`,
looked up by term id the same way curated OBO synonyms already are. Verified
across two successive renames of the same term: both historical names survive,
and `load_synonym_index` includes history automatically (`include_history=True`
by default) so no caller has to remember to attach it separately.

The update workflow now diffs the previous `hp.obo` against the new one
before overwriting it -- the only point at which the previous release's
names are still readable -- and appends every detected rename and
obsoletion to `data/term_renames.jsonl` / `data/term_obsoletions.jsonl`
before the pull request opens.

### Differential ranking: the inverse-selection approach, confirmed as sufficient and built

Asked directly whether discriminating between conditions requires a separate
diagnostic data source, or whether observing findings against what HPO
already associates is enough. The answer, confirmed rather than assumed: the
inverse approach is not a workaround, it is the established method in
clinical bioinformatics -- information-content-weighted phenotype semantic
similarity, the approach behind tools such as Phenomizer -- and it is
exactly what the IC table and the has_phenotype graph were already built to
support. There is no separate "diagnostic differential" dataset to look for;
HPO's own disease-phenotype associations are that source, used correctly.

`differential_ranking.py` ranks candidates by the summed IC of findings they
actually share with the case, not by raw overlap count (`candidate_retrieval`'s
job, which only needs to gather candidates cheaply). Deliberately not
normalised by a candidate's own profile size, which would reward a
sparsely-annotated rare disease purely for being under-curated; `profile_size`
is reported instead of folded silently into the score. Verified on the real
graph: Marfan syndrome, congenital contractural arachnodactyly, and familial
ectopia lentis tie for first place on three classic Marfan findings --
correct, not a defect, since these are genuinely difficult differentials to
separate with only those three findings in real clinical practice.

A test-writing mistake caught a real design gap worth recording: the first
version of a tie-break test used a fixture (`enumeration_bench`'s
`differential_graph`) built with relation name `manifests_as`, while the
ranker defaulted to HPO's own `has_phenotype` -- every candidate silently
scored zero, and the alphabetical tie-break decided an order that looked
like a real result. Fixed by making `relation` a parameter rather than a
hard-coded constant, since other graphs in this project legitimately use
different vocabulary for the same kind of edge.

### Level 3, completed: a real extractor, persistent descriptions, automatic population on promotion

Three pieces requested together, and built together.

**A real extractor**, not a mock. `structural_extraction.py` follows the
same pattern as every other external model call in this project: isolated
HTTP call, transport left unimplemented since the exact request shape
depends on deployment (the same choice `document_processing.py` made for
Nemotron-Parse and LlamaParse), graceful degradation throughout -- an
unconfigured extractor, a failing call, and a malformed response all yield
an empty structure rather than an exception reaching the cascade. Markdown
fences around the JSON response are stripped before parsing, since models
asked for "only JSON" reliably wrap it in fences anyway, and treating that
as a parse failure would degrade tier 3 on the most common well-formed
response shape.

**Persistent descriptions.** `ConceptDescriptionStore` gained the same
append-only JSONL persistence `graph_store` already established: a
description that took a model call to extract is exactly as expensive to
lose as a promoted graph edge. Verified round-tripping through a restart.

**Automatic population on promotion**, the concrete behaviour requested:
every edge `DiagnosticAssembly.promote_confirmed` promotes -- whether the
underlying conjecture came from the Dream Engine's offline exploration or a
newly confirmed RLM hypothesis -- now builds and persists a description for
each of its concepts, if one does not already exist. Built from the edge's
own justification (what was confirmed, by how many cases) rather than left
unexplained. A concept with an existing, curated description is never
overwritten -- verified directly. The feature is additive: a caller not
passing the new optional arguments gets promotion exactly as it worked
before.

### The update workflow gained a second, daily schedule for literature

Asked to check not only HPO data weekly but literature and other sources on
a faster cadence. HPO releases every 6-10 weeks; PubMed and
ClinicalTrials.gov content appears daily, a genuinely different rate that
gets a genuinely different schedule rather than one compromise cadence
serving both -- a second cron entry (`0 5 * * *`) alongside the existing
weekly one, with `github.event.schedule` distinguishing which fired so each
job runs only on its own cadence (or on manual dispatch, gated by its own
`workflow_dispatch` input).

`refresh-literature` refreshes concepts the literature index already tracks,
read from `data/literature_index_concepts.json`, rather than crawling the
full HPO term set daily -- the index grows around concepts this project has
actually reasoned about, and a blind daily crawl of thousands of terms would
mostly fetch literature nothing has asked about. Not committed automatically:
the literature index is runtime state, not a versioned data file, unlike the
HPO release update, which does open a reviewable pull request.

### The description store's source was already on disk, and one format now governs both sides of tier 3

Two pieces built together, because the second is what makes the first safe
to run at scale.

**17,441 curated definitions, downloaded and never parsed.** `hp.obo` carries
a prose definition for most of its terms, each with a PMID attached -- exactly
the source text tier 3's concept side needs, already licensed and versioned
with the release. `parse_obo` captured names, synonyms, parents and
obsoletion flags, and dropped the `def:` line entirely. Now parsed, with the
bracketed reference stripped: it is provenance, not clinical text, and
leaving "pmid:19125436" in would have put it in front of every extraction.
Verified on the real file: 456 descriptions from the first 500 terms, with
skips counted separately for "no definition to work from" and "extraction
yielded nothing" -- two different situations a single count would blur.

**The canonical format, raised directly in discussion, and the reason it
matters more than it first appears.** The proposal was to have a vetting
model emit structure alongside its prose rather than have the extractor
re-read the answer, with the format derived from the extractor so changes
cannot desynchronise the two sides. That last clause is load-bearing:
`compare_structures` matches entity and relation strings literally, so if the
cached concept descriptions were extracted under one convention and claims
arrive under another -- "required_for" against "requires", "1-alpha-hydroxylase"
against "1α-hydroxylase" -- the arithmetic returns a low score for two
structures describing the same mechanism, and the failure reads as
disagreement rather than as drift. `canonical_format_example` generates the
prompt fragment from the same `ExtractedStructure` the extractor produces,
and a test round-trips the generated example back through
`parse_model_emitted_structure` to assert the two halves still agree.

`StructuralResolver` now prefers a model-emitted structure and falls back to
the extractor otherwise, recording which route ran in
`last_structure_source`. Verified: with a `STRUCTURE:` line present the
extractor is called zero times; without one, the previous behaviour is
unchanged.

**What emitting its own structure does not let a model do.** It does not let
it grade itself. The comparison stays arithmetic, against a cached
description the model never sees. A structure shaped to look agreeable has
no target to shape toward -- a test asserts a flattering, unrelated structure
resolves to nothing. What the design buys is fidelity: the model that formed
the claim reports its structure better than a second model parsing the
sentence afterwards, and one model call is saved per resolution.

### LiteratureIndex finally persists, using infrastructure this project already built and never connected

An audit confirmed what was suspected: `LiteratureIndex` had no persistence
at all, and this should not have needed building from scratch.
`vector_memory.py` is a complete, provider-neutral vector store --
`PersistentJsonlVectorStore` with durable JSONL persistence, Weaviate named
as the recommended production backend, a documented object-property schema
-- never instantiated anywhere in production. The same pattern this project
has now found and fixed eight times over: built, tested, never connected.

**What this integration deliberately does not do: change how relevance is
judged.** `PersistentJsonlVectorStore.search` ranks by embedding cosine
similarity; `literature_index.py`'s own docstring already states why that
was rejected for this purpose -- a RAG system asked about heart failure
retrieves "acute coronary syndrome" because it sits close in latent space,
not because it answers the question. Routing literature retrieval through
vector search now would have undone that decision by accident, through a
persistence change nobody meant as a relevance change. The vector store here
is storage only: passages persist and survive a restart, and
`LiteratureIndex.search` still does its own concept matching over whatever
was loaded. The embedding each record carries exists only because the
store's API requires one; nothing reads it back for ranking. Verified
directly: a passage persisted by one store instance and loaded by a second,
independent instance (the actual restart scenario) is found through
`LiteratureIndex.search`'s ordinary concept matching, not vector similarity.

One store for both sources: `EuropePmcConnector.populate` and
`ClinicalTrialsConnector.populate` both gained an optional `store` parameter
persisting to the same `PersistentJsonlVectorStore`, deduplicating by the
passage's own id so a passage rediscovered by a later refresh updates in
place rather than duplicating.

### The tracked-concept queue, and the daily workflow wired to the real architecture

Confirmed directly: "29,053" was the graph's total concept count, cited
earlier only to illustrate why an unbounded crawl is the wrong shape --
never a literal scope for literature refresh. The chosen strategy is
Dream-Engine style: bounded, nightly, working through concepts this project
has actually reasoned about, growing as real use grows it.

`tracked_concepts.py` is a queue, not a bare list -- a bare set of names
cannot answer "which ones are overdue", and a bounded batch keeps each
nightly run short regardless of how many concepts are tracked. Every entry
carries when it was added, why, and when it was last refreshed;
`next_batch` prioritises never-refreshed concepts first, then the ones
refreshed longest ago. `seed_from_vetting_bench` gives the queue ~40
concepts to start from with zero curation effort, since the vetting bench's
cases already establish that they matter.

*Correction, recorded rather than silently edited away:* this section
originally justified the batch limit by citing "PubMed's unauthenticated
rate limit (3 requests/second)". That figure is real but belongs to NCBI
E-utilities, a service this project's literature connectors do not call --
`pmc_case_reports.py` uses it for a different job entirely. Verified
directly: Europe PMC's actual limit is 10 requests/second (500/minute,
confirmed by EBI staff), and ClinicalTrials.gov publishes no single
documented ceiling at all. Neither connector requires or accepts an API
key; an earlier statement that the literature workflow was "inert until a
key arrives" was also wrong and is corrected here rather than only in the
code. The batch size (20) exists to keep each run short and its summary
readable, not because either service would be strained by more.

The daily workflow job was rewritten to the real architecture rather than
reading a file that never existed: seeds from the vetting bench on first
run, pulls a rate-limit-bounded batch (20 concepts) from the queue, persists
every result through `literature_persistence.py` into
`data/literature_vectors.jsonl`, and commits the refreshed queue and store
back to the repository.

**That commit step reverses an earlier design note, stated directly rather
than silently overwritten.** An earlier version of this workflow
deliberately did not commit its output, reasoning that "the literature
index is runtime state, not a versioned data file" -- true only because no
real persistence existed then. A GitHub Actions runner is itself discarded
after every run; not committing now would mean every night starts from an
empty store again, defeating the point of today's work. Growth is modest at
this batch size (tens of KB per night) and committing daily is reasonable
for now, flagged directly as worth revisiting if the tracked-concept list or
batch size grows enough to change that.

### DailyMed: the direct complement to MAxO's measured 1.6% contraindication gap

Built after verifying the real API directly against NIH's own documentation
rather than any third-party scraper description: base URL
`dailymed.nlm.nih.gov/dailymed/services/v2/`, JSON by file extension, GET
only, no key required or accepted.

Chosen specifically because it targets a gap this project already measured
and could not fill: MAxO's contraindication annotations cover 202 of 12,880
diseases (1.6%), with a single genuine `CONTRAINDICATED` row in the entire
file. DailyMed carries the Structured Product Label for every FDA-approved
drug -- the entire formulary, not a curator's incidental annotations.

**A documented limitation, not a silent gap.** `/spls.json` and
`/spls/{SETID}/packaging.json` return structured metadata -- title, active
ingredients, packaging -- not the prose contraindications and warnings
sections, which live inside the full SPL document (an HL7-standard XML
structure, downloadable as ZIP) and would need its own LOINC-coded-section
parser, a distinct and larger piece of work not built here. This connector
gives citable, checkable drug identification -- enough to recognise a drug
a vetting model names and resolve it to its official label -- not yet the
contraindication text itself. Verified against the real, documented example
from NIH's own API help pages (ZOCOR/simvastatin).

### AIFA and SNOMED CT for Italy: researched, neither built, for different honest reasons

Two follow-up questions -- does Italy have an equivalent drug database, and
what is Italy's SNOMED CT status -- were researched rather than assumed or
built past.

**AIFA's Banca Dati Farmaci is real and exactly on-target** (RCP and Foglio
Illustrativo for every drug authorised in Italy, including contraindications
and interactions per the AIFA Medicinali app's own description) but every
source found describes a searchable web portal and a mobile app, never a
documented REST API. Building a connector against an undocumented HTML
portal would break the discipline every other connector in this project
follows -- confirmed, official, documented APIs only. Not built; flagged as
worth a direct inquiry to AIFA about institutional data-sharing access,
the same posture UMLS licensing already requires.

**Italy's SNOMED International membership could not be confirmed present**
across multiple sources checked (Wikipedia's member enumeration, SNOMED
International's own members page) -- both list comparable European
countries (Spain, Belgium, and France as a recent addition) without Italy
appearing. Not a certain finding, stated as such, and worth verifying
directly at snomed.org/members before any cost planning depends on it: if
Italy is not a member, SNOMED CT licensing falls under the fee-based
non-member path (World Bank Territory Band pricing) rather than the free
Member-country path UMLS's own SNOMED CT bundling would otherwise imply.

### UMLS connected: search, HPO-to-anything crosswalk, and genuine encryption at rest

Built after confirming the real UTS REST API directly against NLM's own
documentation (base `https://uts-ws.nlm.nih.gov/rest`, `apiKey` query
parameter). The connector's central capability is `/crosswalk`, and NLM's
own documentation uses this project's exact scenario as its worked example:
crosswalking an HPO code to SNOMED CT. Every concept in this project's graph
is already an HPO code, so crosswalk needs no free-text search first -- it
takes what the graph already has and returns whatever other vocabulary
(RxNorm, MeSH, LOINC, SNOMED CT when available) shares its CUI. Verified
directly against the documented example (`HP:0001947` -> SNOMEDCT_US
`233604007`, sharing CUI `C0022099`).

**A test's first version passed for the wrong reason, caught before commit.**
A first attempt at verifying "a UMLS-crosswalked synonym resolves through the
cascade" used the phrase "distal renal tubular acidosis" against the node
"Renal tubular acidosis" -- which turned out to already match via the
existing word-set tier, containment alone, with no UMLS involvement at all.
Rewritten with "RTA", a genuine abbreviation sharing no words with the node,
with an explicit assertion that resolution fails without UMLS configured --
the only way to confirm the test isolates what it claims to.

**Genuine encryption at rest, not obfuscation.** `encrypted_store.py` uses
Fernet (AES-128-CBC with an HMAC, from the `cryptography` library, now a
core dependency) with a key derived from `DB_PASSWORD` via PBKDF2HMAC
(600,000 iterations, OWASP's 2023 minimum) rather than using the password
directly as a key. A per-store random salt is generated once and reused on
every reopen -- a fresh salt on each open would derive a different key each
time and nothing previously written would ever decrypt again, a mistake
caught by a test that reopens a store and confirms the salt is unchanged.
Verified directly: the raw bytes on disk contain neither the plaintext nor
any recognisable structure, and a wrong password raises `WrongPasswordError`
explicitly rather than silently returning nothing.

A file-based store rather than a database server, for the same reason every
other persistent store in this project is one: no existing database
infrastructure to build on, and a single encrypted file costs far less
operationally than standing up a server this project has no other use for.

**A caching bug caught by the restart test, not by the happy path.** The
first version of `UmlsCache._ensure_loaded` appended each cached record's
result list as a single nested element (`.append(record["result"])`),
producing a list of lists rather than a flat list -- invisible until a
second process reopened the cache and tried to reconstruct
`CrosswalkResult` objects from what was actually a list, not a mapping.
Fixed to overwrite by key rather than accumulate, matching normal cache
semantics: the most recently written entry for a key wins.

`pyproject.toml` also corrected in the same change: `docling>=2.0` was still
declared as an optional dependency months after Docling's removal from the
codebase -- a stale declaration nobody had reason to notice until adding a
genuinely new dependency required looking at the file directly.

### EMA's PMS Public API: complement to DailyMed, not a replacement, built on the confirmed live beta

DailyMed carries the FDA's US formulary; PMS carries the EU's centrally
authorised one. The two overlap substantially for major international drugs
but are not identical, and for an Italian clinical context PMS is the more
directly authoritative source -- a drug centrally authorised in the EU and
prescribed in Italy may have no DailyMed entry at all. Both are kept,
feeding the same literature index, because neither alone covers what the
other does.

Built on the confirmed-live beta (verified against EMA's own July 2026 FAQ
document): base `https://api.pms.ema.europa.eu/public/v1`,
`MedicinalProductDefinition` resources in FHIR R5, `PMS_EMA_API_KEY` required
since registration is mandatory -- unlike the three prior connectors, this
one is not open by default. The request shape is isolated into one method
specifically because EMA's own documentation calls this a beta: when the
public contract stabilises into something different, one method needs
updating, not every caller.

### The secrets were named but never read: UMLS_API_KEY, DB_PASSWORD, PMS_EMA_API_KEY wired to real code

An audit prompted by a direct question -- confirm these three secrets are
used correctly -- found they were not used at all. `UMLS_API_KEY` and
`PMS_EMA_API_KEY` appeared only inside error messages ("UMLS_API_KEY not
configured") and documentation; `DB_PASSWORD` only inside
`encrypted_store.py`'s own docstrings. Nothing anywhere called
`os.environ.get` for any of the three, and the daily workflow never
imported `UmlsConnector` or `PmsEmaConnector` at all. Adding the three
secrets to the repository would have changed nothing.

Fixed with `UmlsConfig.from_env()` / `PmsEmaConfig.from_env()`, and
`build_umls_for_cascade()` -- the assembly point that reads both
`UMLS_API_KEY` and `DB_PASSWORD` and returns a ready `CachedUmlsConnector`,
or `None` when no key is configured, matching the graceful-degradation
contract every other tier already has. Three configuration states verified
directly: no key (`None`, not a connector that silently does nothing); key
without password (a working connector making live calls, never falling back
to caching UMLS content in plaintext); key and password together (connector
plus encrypted cache). `CachedUmlsConnector` is the adapter that lets a
connector-plus-cache pair present itself as the single
`crosswalk_from_hpo(hpo_id)` method `NormalisationCascade` actually calls.

A bug introduced while writing this fix, caught before commit: an edit
meant to insert `CachedUmlsConnector` and `build_umls_for_cascade` ahead of
`crosswalk_with_cache` matched only that function's signature line,
orphaning its entire body as unreachable code stuffed after a `return`
statement in a different function -- syntactically valid Python (dead code
raises no `SyntaxError`), so `ast.parse` reported success while
`crosswalk_with_cache` had silently stopped existing. Caught by the test
suite, not by the syntax check, which is the reason this project runs both.

The daily workflow's literature-refresh job now declares all three as `env:`
via `secrets.*`, and actually uses them: `PmsEmaConnector` populates
alongside Europe PMC, ClinicalTrials.gov and DailyMed for each batch
concept (the same audit found DailyMed itself had never been wired into the
automated workflow either, despite needing no key at all -- added in the
same pass); `build_umls_for_cascade()` pre-warms the encrypted crosswalk
cache for the batch's concepts, resolved to their HPO ids through the same
`hp.obo` synonym index the normalisation cascade itself uses, so a live
cascade run later finds this batch's results already cached rather than
making its own first-use network call.

### Il controllo di indipendenza dall'ordine, e una scoperta collaterale sulle prestazioni

Richiesto direttamente, dopo aver discusso gli effetti d'ordine della
cognizione quantistica: il sistema stesso potrebbe cambiare conclusione a
seconda dell'ordine in cui i reperti arrivano? Per uno strumento diagnostico
regolato questo sarebbe un difetto, non una caratteristica -- distinto
esplicitamente dalla domanda, diversa e legittima, se un **lettore umano**
possa mostrare bias di ancoraggio in base all'ordine di presentazione.

Verificato empiricamente, non presunto dalla lettura del codice: 30
permutazioni casuali contro il grafo HPO reale (1.273.466 archi) su
`retrieve_candidates` e `rank_differential`, 50 su `MechanismEnumerator`
contro una fixture -- **zero risultati dipendenti dall'ordine** in tutte le
esecuzioni. La struttura del codice spiega perché: `breadth` si calcola per
unione di insiemi, `nearest_hops` per minimo progressivo -- entrambe
operazioni indipendenti dall'ordine per costruzione -- e l'ordinamento
finale dei candidati usa il nome come spareggio esplicito, mai l'ordine di
iterazione del dizionario.

Una suite di regressione permanente e veloce (`tests/test_order_invariance.py`)
ripete lo stesso controllo su grafi fixture piccoli -- 800 permutazioni
totali in 0,23 secondi -- così la proprietà resta verificata ad ogni
esecuzione della suite, senza il costo del grafo reale.

**Scoperta collaterale, non richiesta ma rilevante**: una singola chiamata
a `retrieve_candidates` contro il grafo reale impiega **3,4 secondi**. Le 30
permutazioni di verifica hanno richiesto 102 secondi totali. Per un
contesto genuinamente in tempo reale questo merita un'indagine separata
sulle prestazioni della ricerca in ampiezza su un grafo di quella
dimensione -- non affrontata qui, segnalata per lavoro futuro.

4 nuovi test (800 permutazioni), 1281 totali passanti.

### La cognizione quantistica come campo reale: Tappa A della verifica

Una domanda diretta sul confine fra la cognizione quantistica come campo
accademico legittimo (Huang et al. 2025, Fuyama et al. 2025, Busemeyer &
Bruza 2012 -- verificati come pubblicazioni reali) e l'errore concreto già
corretto in questo progetto (l'equazione di Schrödinger letterale, con
$\hbar$ fisico, per confrontare vettori reali) ha portato a un piano di
validazione in tre tappe, concordato esplicitamente. Tappa A: verificare
che il formalismo matematico sia implementato correttamente, prima di
qualunque affermazione sull'utilità clinica.

`training/quantum_cognition_order_model.py`: spazio di Hilbert complesso
reale, proiettori verificati come Hermitiani e idempotenti alla
costruzione (non assunti), collasso sequenziale secondo il postulato di
Lüders. La proprietà centrale, dimostrabile e verificata direttamente:
**l'effetto d'ordine è esattamente zero quando i proiettori commutano, e
diverso da zero quando non commutano** -- verificato con un esempio in
stile Clinton-Gore (due domande binarie, base ruotata di 30 gradi):
P(prima A poi B) = 0.270, P(prima B poi A) = 0.634, una differenza reale,
non un artefatto. La conservazione della probabilità totale
(P(sì)+P(no)=1, esatta a 10 cifre) e il rifiuto di matrici non valide sono
verificati altrettanto direttamente.

**Dichiarato esplicitamente, non taciuto**: questo modulo non è mai
collegato al ragionamento del sistema stesso, che resta indipendente
dall'ordine per costruzione (verificato separatamente). Predice
un'eventuale distorsione nel giudizio del *lettore umano*, non modifica
mai la conclusione del sistema. E un limite noto del campo stesso, non di
questa implementazione: gli effetti d'ordine e la ripetibilità della
risposta non si modellano ancora simultaneamente nel formalismo
standard in spazio di Hilbert -- un problema aperto in letteratura, non
affrontato qui.

Le Tappe B (confronto con studi clinici già pubblicati sull'ancoraggio
diagnostico -- individuato un candidato concreto: uno studio randomizzato
su medici in formazione con la posizione del reperto fuorviante
manipolata) e C (uno studio prospettico proprio) restano non tentate,
richiedono risorse fuori dalla portata di una sessione di scrittura di
codice.

13 nuovi test, 1294 totali passanti, lint pulito.
