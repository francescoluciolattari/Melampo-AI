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
