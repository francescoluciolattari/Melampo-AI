#!/usr/bin/env python3
"""Run the root-model format-adherence bench against real endpoints.

Reads model API keys from environment variables — populated from GitHub
Actions secrets in CI, exported locally otherwise — and never accepts a key as
a command-line argument or literal, for the same reason the PMC connector
does not: a key is a value that must never appear in a diff, a log, or a shell
history file.

A model whose secret is absent is skipped rather than causing the run to fail.
Partial results are useful — knowing that three of four candidates were
reachable is better than no result because the fourth key was never set — and
the report says explicitly which were skipped and why.

Usage:
    python scripts/run_format_adherence_bench.py [--out results.json]

Environment variables consulted, all optional:
    MISTRAL_API_KEY, OPENROUTER_API_KEY

Candidates that need a key with no variable set are reported as skipped, not
silently dropped.

WHAT THIS BENCH DOES AND DOES NOT MEASURE. It measures format adherence and
navigation persistence: does a candidate emit the action grammar correctly,
and does it complete a multi-step lookup within budget. It does NOT grade
whether a candidate's final(answer) is clinically correct — there is no
answer key here, only a parser and a completion flag. A model that
confidently emits final(wrong answer) scores identically to one that gets it
right, provided both are well-formed. Choosing which model reasons best about
a differential diagnosis is a different, already-built evaluation
(evaluation/dream_capture_benchmark.py, part of the B4 protocol), which grades
against a documented outcome rather than a format grammar. This bench answers
"can this model navigate and follow the loop", which is the prerequisite for
that other question, not a substitute for it.

============================================================================
MODEL AND ENGINE CONFIGURATION -- edit here when newer versions ship
============================================================================
Every model this bench can call is declared in the two constants below,
CANDIDATE_MODELS and MISTRAL_DIRECT_MODEL, so that adding, removing or
re-pointing a model at a newer release is a one-line change at the top of the
file rather than a hunt through build_candidates()'s body. Nothing past this
block should need to change when a provider ships a new version -- only the
tuples themselves.

Each CANDIDATE_MODELS entry is (bench_name, openrouter_slug, disable_reasoning).
``disable_reasoning`` requests OpenRouter's reasoning:{enabled:false} hint
(openrouter.ai/docs/use-cases/reasoning-tokens); providers that reject the
override outright fall back to a plain call automatically (see the HTTP 400
handling in _http_chat_completion), so setting it to True is always safe to
try and never fatal to a candidate that cannot honour it.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from melampo.evaluation.format_adherence_bench import (
    BenchCase,
    bench_models,
)
from melampo.memory.context_environment import EnvironmentDocument
from melampo.reasoning.rlm_engine import Budget

# The one candidate called at its own first-party endpoint rather than through
# OpenRouter. "latest" is a Mistral-hosted alias that follows their own
# updates, so unlike a dated slug it does not need editing here as new Small
# releases ship -- verify occasionally that "latest" still points where you
# expect, since an alias can also jump to a size/price tier you did not choose.
MISTRAL_DIRECT_MODEL = "mistral-small-latest"

# name, OpenRouter slug, disable_reasoning. Every slug here was checked
# against a primary or independently-verified source before being added --
# qwen-3.5 previously carried an invented slug that never existed, which is
# the reason for that discipline, not a formality. Grouped by family for
# readability; order has no effect on the bench, which runs every reachable
# candidate against every case.
CANDIDATE_MODELS: tuple[tuple[str, str, bool], ...] = (
    # Anthropic. Called via OpenRouter (see recursive_engine_decision_record.md
    # for why a third-party gateway offering direct Claude access was
    # evaluated and rejected). Commercial terms need review before shipping.
    ("claude-sonnet-5", "anthropic/claude-sonnet-5", True),
    ("claude-opus-5", "anthropic/claude-opus-5", True),
    ("claude-fable-5.1", "anthropic/claude-fable-5.1", True),
    # OpenAI.
    ("gpt-6-astra", "openai/gpt-6-astra", True),
    # Qwen (Alibaba), Apache 2.0.
    ("qwen-3.5", "qwen/qwen3.5-plus-02-15", True),
    ("qwen-3.7", "qwen/qwen3.7-max", True),
    ("qwen-3.8", "qwen/qwen3.8-max", True),
    # Z.ai, GLM. glm-5 is MIT; glm-5.3's terms were not directly confirmed at
    # the time it was added (see LICENCE_UNVERIFIED in rlm_model_adapter.py).
    # glm-5.3's own listing states reasoning "is always on and cannot be
    # disabled" -- disable_reasoning is still requested here because the 400
    # fallback makes the attempt free, not because it is expected to succeed.
    ("glm-5", "z-ai/glm-5", True),
    ("glm-5.3", "z-ai/glm-5.3", True),
    # Meta, Llama. 3.3 is unaffected by the Llama 4 EU Acceptable Use Policy
    # restriction; 4-Maverick and 4-Scout are benched for comparison only --
    # that restriction governs adoption regardless of this bench's result.
    ("llama-3.3-70b", "meta-llama/llama-3.3-70b-instruct", True),
    ("llama-4-maverick", "meta-llama/llama-4-maverick", True),
    ("llama-4-scout", "meta-llama/llama-4-scout", True),
    # Google, Gemma. Gemma 4 shipped under Apache 2.0, resolving Gemma 3's
    # licence-review flag in the same release that made it newer.
    ("gemma-3-27b", "google/gemma-3-27b-it", True),
    ("gemma-4-31b", "google/gemma-4-31b-it", True),
    ("gemma-4-26b-a4b", "google/gemma-4-26b-a4b-it", True),
    # Mistral, via OpenRouter as a second path: the direct API's free tier
    # rate-limited on the first live run (HTTP 429), and OpenRouter's
    # pass-through has separate limits, so this is redundancy, not duplication.
    ("mistral-large-openrouter", "mistralai/mistral-large-2512", True),
    ("mistral-small-openrouter", "mistralai/mistral-small-2603", True),
    # Three families outside those already covered, from a survey of what
    # exists as of September 2026. Licences not directly confirmed for any of
    # the three; see LICENCE_UNVERIFIED.
    ("kimi-k2.6", "moonshotai/kimi-k2.6", True),
    ("deepseek-v4-flash", "deepseek/deepseek-v4-flash", True),
    # xAI. ":free" confirmed working via an independent, reproducible example
    # (Simon Willison, simonwillison.net/tags/openrouter/) after the bare
    # "x-ai/grok-4-fast" slug returned HTTP 404 on the first live run of this
    # bench -- some accounts or plans appear to need the explicit free-tier
    # route for this model rather than the unsuffixed slug.
    ("grok-4-fast", "x-ai/grok-4-fast:free", True),
)

ACTION_GRAMMAR = (
    "describe() | grep(pattern) | slice(document_id, start, end) | "
    "search(query) | expand(concept) | final(answer)"
)

# ============================================================================
# BENCH DOCUMENTS AND CASES
# ============================================================================
# All synthetic, phase-one data-class discipline (see rlm_engine.py) applies
# here as everywhere else the environment is populated. A first live run
# produced an 8-way tie at 100% adherence / 100% completion on the original
# six single-document, single-fact cases -- too easy to discriminate among
# strong candidates. The additions below are built around specific navigation
# demands the easy tier cannot exercise, not just "harder wording" of the
# same lookup: distinguishing a stated finding from a nearby negated one,
# discovering and reading a second document, locating a fact buried in a
# longer note, comparing values across a series, and picking the confirmed
# diagnosis out of a differential rather than the first name mentioned.
#
# Reminder from the module docstring: completing a harder case still is not
# graded against a correct answer. What these add is discrimination on
# whether a candidate does the additional navigation at all, which is a
# necessary condition for a correct answer even though it is not sufficient
# for one.

BASELINE_DOCUMENT = EnvironmentDocument(
    document_id="report_1",
    text=(
        "Chest radiograph shows bibasilar opacities. Prednisone 40 mg daily was "
        "started. The patient reports progressive dyspnoea over three weeks with "
        "no fever."
    ),
    source="synthetic_bench_fixture",
    metadata={"data_class": "synthetic"},
)

# Tests assertion status, not just keyword presence: three related findings
# sit close together, one affirmed, two negated, and the affirmed one is not
# the first mentioned -- a plain keyword grep on "effusion" hits all three
# lines, and only reading the qualifier on each distinguishes them.
NEGATION_DOCUMENT = EnvironmentDocument(
    document_id="report_2",
    text=(
        "Cardiac silhouette mildly enlarged. No pleural effusion. Trace "
        "pericardial effusion noted, unchanged from six months prior. "
        "Pneumothorax is absent. Small apical pleural bulla, stable."
    ),
    source="synthetic_bench_fixture",
    metadata={"data_class": "synthetic"},
)

# Two documents, deliberately cross-referencing: the imaging report names a
# finding and defers the explanation to "the accompanying laboratory
# report," which is a separate document. Answering requires discovering that
# a second document exists (describe() or search(), not assumed from one
# grep) and reading it specifically, rather than answering from document 1
# alone because it was the only one looked at.
CROSS_REF_IMAGING = EnvironmentDocument(
    document_id="report_3a",
    text=(
        "Right upper lobe consolidation with air bronchograms, most consistent "
        "with infective process. Correlate clinically with inflammatory markers "
        "in the accompanying laboratory report."
    ),
    source="synthetic_bench_fixture",
    metadata={"data_class": "synthetic"},
)
CROSS_REF_LABS = EnvironmentDocument(
    document_id="report_3b",
    text=(
        "White cell count 16.4 (elevated). C-reactive protein 142 (markedly "
        "elevated). Procalcitonin 3.8, supportive of a bacterial source."
    ),
    source="synthetic_bench_fixture",
    metadata={"data_class": "synthetic"},
)

# The fact the question asks about sits in the fourth of four paragraphs,
# each plausible enough that a model might stop at the first or second and
# answer from the wrong one -- tests whether grep-then-slice for surrounding
# context happens, or whether the model settles for the first superficially
# relevant hit in a longer note.
LONG_DOCUMENT = EnvironmentDocument(
    document_id="report_4",
    text=(
        "History: 68-year-old with prior myocardial infarction, admitted with "
        "acute dyspnoea. Initial impression favoured decompensated heart failure.\n\n"
        "Examination: Bibasilar crackles, elevated jugular venous pressure, "
        "peripheral oedema to the mid-shin bilaterally.\n\n"
        "Initial management: Intravenous furosemide 40 mg was given, with "
        "improvement in oxygen saturation over the following two hours.\n\n"
        "Revised assessment after echocardiography: Ejection fraction 58%, "
        "preserved. Findings are now judged more consistent with a pulmonary "
        "embolism than primary cardiac decompensation; CT pulmonary angiogram "
        "requested."
    ),
    source="synthetic_bench_fixture",
    metadata={"data_class": "synthetic"},
)

# A series of values across time; the question requires comparing the first
# and last rather than reading either in isolation, and the middle value is a
# distractor that goes the "wrong" direction before the trend resolves.
TREND_DOCUMENT = EnvironmentDocument(
    document_id="report_5",
    text=(
        "Serial inflammatory markers. Day 1: white cell count 14.2. Day 3: "
        "white cell count 16.8, mild interval rise. Day 5: white cell count "
        "9.1, following initiation of antibiotics on day 2."
    ),
    source="synthetic_bench_fixture",
    metadata={"data_class": "synthetic"},
)

# A differential list names three candidates before the confirmed diagnosis
# is stated separately -- the correct answer is not the first name in the
# document, and a model that stops at the first plausible-looking diagnostic
# term will report a candidate that was explicitly ruled out.
DIFFERENTIAL_DOCUMENT = EnvironmentDocument(
    document_id="report_6",
    text=(
        "Differential considered pneumonia, pulmonary oedema, and pulmonary "
        "embolism. Pneumonia was considered less likely given the afebrile "
        "course. Oedema was excluded on the basis of a normal echocardiogram. "
        "CT pulmonary angiogram confirmed a segmental pulmonary embolism as "
        "the cause of the presentation."
    ),
    source="synthetic_bench_fixture",
    metadata={"data_class": "synthetic"},
)

BENCH_CASES = (
    # Baseline tier: single document, single fact. Kept as a floor -- a
    # candidate that fails here has a problem the harder tier cannot help
    # diagnose any further.
    BenchCase("dose", (BASELINE_DOCUMENT,), "What steroid dose was started, and from which document?"),
    BenchCase("finding", (BASELINE_DOCUMENT,), "What imaging finding is documented?"),
    BenchCase("symptom_duration", (BASELINE_DOCUMENT,), "How long has the dyspnoea been present?"),
    BenchCase("fever_status", (BASELINE_DOCUMENT,), "Is fever present according to the report?"),
    BenchCase("onset_pattern", (BASELINE_DOCUMENT,), "Is the dyspnoea described as progressive or sudden?"),
    BenchCase("treatment_frequency", (BASELINE_DOCUMENT,), "How often is the prednisone dose taken?"),
    # Harder tier: each case below exercises a distinct navigation demand the
    # baseline tier cannot, per the comment on each document above.
    BenchCase(
        "negation_discrimination",
        (NEGATION_DOCUMENT,),
        "Is a pericardial effusion present, and how does that differ from the pleural finding?",
    ),
    BenchCase(
        "cross_document_correlation",
        (CROSS_REF_IMAGING, CROSS_REF_LABS),
        "What laboratory abnormality supports the imaging impression, and which document reports it?",
    ),
    BenchCase(
        "buried_fact_after_revision",
        (LONG_DOCUMENT,),
        "What is the current leading diagnosis after echocardiography, and how does it differ from the initial impression?",
    ),
    BenchCase(
        "numeric_trend",
        (TREND_DOCUMENT,),
        "Did the white cell count rise or fall from day 1 to day 5, and what happened in between?",
    ),
    BenchCase(
        "confirmed_vs_candidate_diagnosis",
        (DIFFERENTIAL_DOCUMENT,),
        "Which diagnosis was ultimately confirmed, as distinct from the other candidates considered?",
    ),
)

# ============================================================================
# ADVANCED DOCUMENTS AND CASES -- opt-in via --cases advanced
# ============================================================================
# Built for a focused comparison among a small number of strong candidates
# (see docs/four_model_comparison.md and four-model-comparison-bench.yml)
# where the 21-candidate roster's cost and time constraints do not apply, so
# a larger, more subtle set is affordable. Each case targets a discrimination
# the eleven cases above do not: three-document synthesis rather than two,
# confirming the *absence* of a requested fact rather than always having one
# to find, distinguishing two similarly-worded but clinically opposite terms,
# chronology presented out of reading order, a genuinely long document
# requiring several slices, two documents that disagree on the same value,
# a dose expressed as a rate requiring two located numbers rather than one,
# and a two-hop inference connecting facts stated far apart. As with every
# other case in this file: none of these are graded against a correct
# answer. They discriminate navigation persistence and format adherence
# under harder demands, which is what this bench measures throughout.

CONSULT_IMAGING = EnvironmentDocument(
    document_id="report_7a",
    text=(
        "Echocardiogram: small circumferential pericardial effusion, no "
        "tamponade physiology. Normal left ventricular wall motion."
    ),
    source="synthetic_bench_fixture",
    metadata={"data_class": "synthetic"},
)
CONSULT_LABS = EnvironmentDocument(
    document_id="report_7b",
    text="Troponin mildly elevated at 0.08, trending down on repeat testing six hours later.",
    source="synthetic_bench_fixture",
    metadata={"data_class": "synthetic"},
)
CONSULT_NOTE = EnvironmentDocument(
    document_id="report_7c",
    text=(
        "Cardiology consult: taking the effusion, the down-trending troponin, "
        "and the preserved wall motion together, the picture is most "
        "consistent with pericarditis rather than myocardial infarction. A "
        "true infarction would be expected to show a rising troponin and a "
        "regional wall motion abnormality, neither of which is present."
    ),
    source="synthetic_bench_fixture",
    metadata={"data_class": "synthetic"},
)

# The question asks about a drug the document never mentions -- it discusses
# a real, different treatment instead. Tests whether a candidate concludes
# and reports absence, or keeps searching (and burns its budget) for
# something that was never there. Unlike every case above, there is nothing
# to find; correct navigation ends in a documented "not found," not a longer
# search.
ABSENCE_DOCUMENT = EnvironmentDocument(
    document_id="report_8",
    text=(
        "Community-acquired pneumonia was treated with azithromycin 500 mg "
        "on day one, then 250 mg daily for four further days. Symptoms "
        "improved over the course of treatment."
    ),
    source="synthetic_bench_fixture",
    metadata={"data_class": "synthetic"},
)

# Two clinically opposite conditions share the word "pulmonary" and sit one
# sentence apart, one ruled out and one confirmed. A candidate matching on
# the shared word rather than reading the whole term will confuse them.
CONFUSABLE_TERMS_DOCUMENT = EnvironmentDocument(
    document_id="report_9",
    text=(
        "CT pulmonary angiogram performed to assess for pulmonary embolism; "
        "no filling defect identified, embolism excluded. Findings instead "
        "show bilateral interstitial pulmonary oedema, judged cardiogenic in "
        "origin given the associated cardiomegaly."
    ),
    source="synthetic_bench_fixture",
    metadata={"data_class": "synthetic"},
)

# The discharge diagnosis is stated first; the symptom that actually came
# first in time is described later, in the "history" section. A candidate
# that answers from the first paragraph it reads, rather than locating the
# section that actually addresses onset, reports the wrong symptom as first.
OUT_OF_ORDER_DOCUMENT = EnvironmentDocument(
    document_id="report_10",
    text=(
        "Discharge diagnosis: decompensated heart failure with acute kidney "
        "injury.\n\n"
        "Hospital course: Diuresis was initiated on admission with good "
        "response; renal function improved over the following four days.\n\n"
        "History of presenting complaint: The illness began eight days prior "
        "to admission with ankle swelling, which preceded the breathlessness "
        "that ultimately prompted presentation three days later."
    ),
    source="synthetic_bench_fixture",
    metadata={"data_class": "synthetic"},
)

# Six paragraphs; the answer requires connecting a fact in the second
# paragraph to a fact in the fifth, forcing at least two located reads
# rather than one grep-and-answer.
LONG_MULTI_SECTION_DOCUMENT = EnvironmentDocument(
    document_id="report_11",
    text=(
        "Admission note: 74-year-old admitted with fever and productive "
        "cough of three days' duration.\n\n"
        "A sputum culture was sent on admission and empirical antibiotics "
        "were started pending results.\n\n"
        "Day two: Patient remained febrile; no growth on blood cultures at "
        "24 hours.\n\n"
        "Day three: Repeat observations show improving oxygen requirement.\n\n"
        "Microbiology, final report: Sputum culture from admission grew "
        "Streptococcus pneumoniae, sensitive to the antibiotic already in "
        "use.\n\n"
        "Day five: Afebrile for 48 hours, planned for discharge tomorrow "
        "to complete a seven-day antibiotic course."
    ),
    source="synthetic_bench_fixture",
    metadata={"data_class": "synthetic"},
)

# Two documents report a value for what is presented as the same blood draw,
# and the numbers disagree. There is no single correct value to report --
# the discriminating behaviour is noticing and citing both rather than
# reporting only the first one found.
CONFLICT_NURSING_NOTE = EnvironmentDocument(
    document_id="report_12a",
    text="Morning bloods: potassium 5.8, flagged to the covering doctor.",
    source="synthetic_bench_fixture",
    metadata={"data_class": "synthetic"},
)
CONFLICT_LAB_REPORT = EnvironmentDocument(
    document_id="report_12b",
    text="Potassium 4.2, sample collected 07:10, processed without delay.",
    source="synthetic_bench_fixture",
    metadata={"data_class": "synthetic"},
)

# The dose is expressed per kilogram; the weight is stated in a different
# sentence. Locating both is the discriminating behaviour -- whether a
# candidate computes the absolute dose is not graded, but a candidate that
# never finds the weight cannot have used it.
WEIGHT_BASED_DOSE_DOCUMENT = EnvironmentDocument(
    document_id="report_13",
    text=(
        "Gentamicin was dosed at 5 mg/kg for presumed Gram-negative sepsis. "
        "The patient's admission weight was recorded as 68 kg. Renal "
        "function was monitored daily given the nephrotoxicity risk."
    ),
    source="synthetic_bench_fixture",
    metadata={"data_class": "synthetic"},
)

# Two facts stated in different sentences must be connected: a family
# history and a measurement that only becomes significant in light of it.
# Both facts are explicit; the inference is in relating them, not finding
# either alone.
TWO_HOP_DOCUMENT = EnvironmentDocument(
    document_id="report_14",
    text=(
        "Family history is notable for a sister with confirmed Marfan "
        "syndrome. Echocardiography today shows an aortic root diameter of "
        "4.8 cm, above the threshold at which surgical referral is "
        "considered in patients with a connective tissue disorder."
    ),
    source="synthetic_bench_fixture",
    metadata={"data_class": "synthetic"},
)

ADVANCED_CASES = (
    BenchCase(
        "three_document_synthesis",
        (CONSULT_IMAGING, CONSULT_LABS, CONSULT_NOTE),
        "What is the working diagnosis once all three notes are read together, and why was myocardial infarction excluded?",
    ),
    BenchCase(
        "absence_of_requested_fact",
        (ABSENCE_DOCUMENT,),
        "What dose of prednisone was prescribed?",
    ),
    BenchCase(
        "confusable_terms",
        (CONFUSABLE_TERMS_DOCUMENT,),
        "Was pulmonary embolism or pulmonary oedema the actual finding, and what happened to the other?",
    ),
    BenchCase(
        "out_of_order_chronology",
        (OUT_OF_ORDER_DOCUMENT,),
        "Which symptom appeared first in time: the ankle swelling or the breathlessness?",
    ),
    BenchCase(
        "long_multi_section_lookup",
        (LONG_MULTI_SECTION_DOCUMENT,),
        "What organism grew on the sputum culture sent at admission, and was it sensitive to the antibiotic already being given?",
    ),
    BenchCase(
        "conflicting_values_across_documents",
        (CONFLICT_NURSING_NOTE, CONFLICT_LAB_REPORT),
        "What potassium value is reported for this blood draw, and is there any discrepancy between sources?",
    ),
    BenchCase(
        "weight_based_dose",
        (WEIGHT_BASED_DOSE_DOCUMENT,),
        "What is the per-kilogram gentamicin dose, and what is the patient's weight needed to compute the total?",
    ),
    BenchCase(
        "two_hop_family_history_inference",
        (TWO_HOP_DOCUMENT,),
        "Does the family history have any bearing on today's aortic measurement, and why?",
    ),
)

# Rewritten after a first live run showed three distinct failure patterns that
# a short instruction left room for: two Claude tiers exhausted their full
# iteration budget without ever calling final() despite well-formed actions
# throughout (over-verification, not confusion); Claude Opus produced
# dialogue-style artifacts ("human error", "Assistantslice(...)") suggesting
# it was pattern-matching the rendered history onto a chat transcript; GPT-6
# Astra narrated tool output in prose ("[doc 0] note.txt (152 chars)") instead
# of emitting the next action. A worked example and explicit prohibitions
# target each pattern directly rather than hoping a longer budget alone fixes
# behaviour a longer budget cannot address.
_SYSTEM_PROMPT = (
    "You navigate a document environment by emitting exactly one action per line, "
    f"chosen from: {ACTION_GRAMMAR}. Emit nothing else: no prose, no explanation, no "
    "commentary on what an action returned, no role labels or dialogue formatting "
    "(never write \"human\" or \"Assistant\" or similar).\n\n"
    "One clean lookup that answers the question is enough. Call final(answer) as soon "
    "as you can answer -- do not re-verify with a second lookup if the first one already "
    "gave you the answer; every extra action spends part of a small, fixed budget.\n\n"
    "Example of a complete, correct exchange for a question like "
    "\"What dose was prescribed?\":\n"
    "grep(dose)\n"
    "final(40 mg daily)\n\n"
    "That is the whole exchange: one lookup, then final() on the next turn. Longer "
    "exchanges are for questions one lookup cannot answer, not the default."
)

# The run that motivated these constants took 10.5 minutes and ended in
# failure because most or all candidates were unreachable (bad key, wrong
# model slug) -- and every one of them was still given the full 3 cases x up
# to 12 iterations x 60s-per-call budget before the failure became visible.
# A single bad candidate should fail in seconds, not minutes.
PREFLIGHT_TIMEOUT_SECONDS = 15
CALL_TIMEOUT_SECONDS = 30
# A first live run showed two model families (Claude Sonnet and Fable) hitting
# max_iterations=6 on every single case without ever calling final() -- 100%
# adherence, 0% completion, always stopped by the ceiling rather than by
# choice. That is the signature of a budget too tight for those models'
# exploration style, not a comprehension failure: they were emitting
# well-formed actions the whole time. Raised to 10/60s; report.budget_bound
# on each result now says explicitly whether the ceiling was the limiting
# factor, so this number can be revisited with evidence instead of guessing
# again. A factory, not a shared instance: Budget carries mutable per-run
# state (iteration count, start time), and reusing one instance across cases
# would corrupt both.
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_WALL_CLOCK_SECONDS = 60.0


def _bench_budget() -> Budget:
    """The per-case budget, overridable via environment variables.

    Two candidates on the second live run (gemma-4-31b, mistral-large-openrouter)
    hit exactly the 10-iteration ceiling on their one incomplete case each,
    with budget_bound=true -- evidence they were still working, not stuck, when
    the budget ran out. BENCH_MAX_ITERATIONS and BENCH_WALL_CLOCK_SECONDS let a
    focused, smaller-roster run (see four-model-comparison-bench.yml) give
    those candidates the room to actually finish, without changing the default
    for the full 21-candidate roster, where a wider budget multiplied across
    every candidate would cost meaningfully more time and money for a question
    this run does not need answered.
    """
    max_iterations = _env_int("BENCH_MAX_ITERATIONS", DEFAULT_MAX_ITERATIONS)
    wall_clock = _env_float("BENCH_WALL_CLOCK_SECONDS", DEFAULT_WALL_CLOCK_SECONDS)
    return Budget(max_iterations=max_iterations, max_wall_clock_seconds=wall_clock)


def _env_int(name: str, default: int) -> int:
    """Read an environment variable as int, falling back on absence or a malformed value.

    A typo in a workflow's env block should degrade to the documented default,
    not crash the bench before it does anything useful.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"  [config] {name}={raw!r} is not an integer, using default {default}", file=sys.stderr)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"  [config] {name}={raw!r} is not a number, using default {default}", file=sys.stderr)
        return default


RATE_LIMIT_MAX_RETRIES = 1
RATE_LIMIT_DEFAULT_WAIT_SECONDS = 5.0
RATE_LIMIT_MAX_WAIT_SECONDS = 20.0


def _http_chat_completion(
    endpoint: str, api_key: str, model: str, prompt: str, *, timeout: int, disable_reasoning: bool = False
) -> str:
    """Minimal OpenAI-compatible chat completion call, stdlib only.

    Mistral, OpenRouter and most inference gateways implement this shape.

    ``disable_reasoning`` sends OpenRouter's own ``reasoning: {enabled: false}``
    parameter (documented at openrouter.ai/docs/use-cases/reasoning-tokens),
    best-effort: models that always reason (GLM-5.3's listing states this
    explicitly) will ignore it, and models without a reasoning mode at all are
    unaffected either way. Scoped to OpenRouter calls only -- the direct
    Mistral endpoint's tolerance for unrecognised top-level fields is not
    verified from here, so the hint is not sent there.

    A 429 is retried once, honouring the provider's own ``Retry-After`` header
    when present rather than guessing a wait. Mistral's free evaluation tier in
    particular has conservative per-second limits and documents that it
    returns this header on every 429; ignoring it and failing immediately
    turns a transient, self-resolving condition into a permanently skipped
    candidate. One retry, not a backoff loop: this is a preflight-scale
    utility, not a production client, and a candidate that is rate-limited
    twice in a row is more informatively reported as such than retried
    indefinitely.
    """
    def _build_body(with_reasoning_hint: bool) -> bytes:
        return json.dumps(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                # Raised from 256 after a first live run showed 0% completion on
                # two model families that used well-formed actions the entire
                # time (100% adherence). GLM-5.3's own listing states its
                # reasoning "is always on and cannot be disabled"; several other
                # candidates here are reasoning-capable by default. If hidden
                # reasoning tokens are consuming the completion budget before the
                # visible action line is ever written, 256 tokens may simply not
                # have left room for both -- and there is no way to tell that
                # apart from "genuinely stuck" without more room to see the whole
                # output. 1024 gives that room while staying a small fraction of
                # a cent per call at every candidate's pricing.
                "max_tokens": 1024,
                **({"reasoning": {"enabled": False}} if with_reasoning_hint else {}),
            }
        ).encode("utf-8")

    def _request(with_reasoning_hint: bool) -> urllib.request.Request:
        return urllib.request.Request(
            endpoint,
            data=_build_body(with_reasoning_hint),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )

    rate_limit_attempt = 0
    reasoning_hint_active = disable_reasoning
    reasoning_fallback_used = False
    while True:
        try:
            with urllib.request.urlopen(_request(reasoning_hint_active), timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return _extract_content(payload)
        except urllib.error.HTTPError as error:
            if error.code == 429 and rate_limit_attempt < RATE_LIMIT_MAX_RETRIES:
                wait = _parse_retry_after(error.headers.get("Retry-After") if error.headers else None)
                print(f"  [rate limit] {model}: HTTP 429, waiting {wait:.0f}s before one retry", file=sys.stderr)
                time.sleep(wait)
                rate_limit_attempt += 1
                continue
            if error.code == 400 and reasoning_hint_active and not reasoning_fallback_used:
                # Some providers reject the reasoning-disable hint outright for
                # models that cannot honour it, rather than ignoring it as most
                # OpenAI-compatible fields are ignored when unrecognised. GLM-5.3
                # documents reasoning as permanently on; the first live run
                # returned exactly this HTTP 400 for it and three other
                # candidates. One retry without the hint distinguishes "this
                # model rejects the override" from "this model or key is
                # actually broken" instead of losing the candidate to the first
                # explanation without checking the second.
                print(f"  [400] {model}: retrying once without the reasoning-disable hint", file=sys.stderr)
                reasoning_hint_active = False
                reasoning_fallback_used = True
                continue
            raise


def _parse_retry_after(header_value: str | None) -> float:
    """Read Retry-After as seconds, capped, falling back when absent or malformed.

    RFC 7231 also allows an HTTP-date in this header; that form is not parsed
    here -- on the small set of providers this script calls, a delta-seconds
    value is what has been documented, and an unparseable value falls back to
    the default wait rather than raising, since a malformed header should not
    prevent the retry it announces.
    """
    if header_value is None:
        return RATE_LIMIT_DEFAULT_WAIT_SECONDS
    try:
        seconds = float(header_value)
    except ValueError:
        return RATE_LIMIT_DEFAULT_WAIT_SECONDS
    return max(0.0, min(seconds, RATE_LIMIT_MAX_WAIT_SECONDS))


class ProviderResponseError(Exception):
    """The provider answered (no network or HTTP-status failure) but the body
    was not a usable completion. Distinct from a connectivity failure because
    the two need different fixes: this one usually means the model is
    unavailable or was rejected, not that the key or the network is broken.
    """


def _extract_content(payload: object) -> str:
    """Pull the completion text out of a chat-completion response body.

    OpenRouter and similar aggregators return HTTP 200 even when the request
    could not be served -- no endpoint available for the model, content
    filtered, upstream provider error -- with the failure described inside
    the JSON body instead of the status code. ``payload["choices"][0]`` on an
    empty list is a real response shape, not a hypothetical one, and it must
    raise something the callers already catch rather than an IndexError or
    TypeError that was never in their except clause. Every failure mode here
    raises ProviderResponseError, so one exception type covers all of them.
    """
    if not isinstance(payload, dict):
        raise ProviderResponseError(f"response body is not a JSON object: {type(payload).__name__}")
    if error := payload.get("error"):
        raise ProviderResponseError(f"provider returned an error: {error}")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderResponseError(f"no choices in response: {json.dumps(payload)[:300]}")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ProviderResponseError(f"no text content in first choice: {json.dumps(choices[0])[:300]}")
    return content


def all_candidate_names() -> list[str]:
    """Every candidate name this bench could run, in the order they are declared.

    Used by the workflow's matrix-preparation step to fan out one job per
    candidate without duplicating the roster in YAML: the list lives once,
    in CANDIDATE_MODELS and the Mistral direct entry above, and this reads it
    back rather than keeping a second copy in sync by hand.
    """
    return ["mistral-small-3.1", *(name for name, _, _ in CANDIDATE_MODELS)]


def build_candidates(only: str | None = None) -> tuple[dict[str, "callable"], list[str], dict[str, str]]:
    """Preflight every candidate whose key is present, then wrap the survivors.

    Iterates CANDIDATE_MODELS and MISTRAL_DIRECT_MODEL, declared at the top of
    this file -- this function contains no model names or slugs of its own,
    so a version bump never means editing logic, only the table above it.

    ``only``, when given, restricts preflight and the bench to that single
    candidate name -- what a parallel matrix job needs to run its one
    assigned candidate without also preflighting the other twenty, most of
    which it will never use. Every other candidate still appears in
    ``skipped``/``preflight_detail`` with a reason naming the restriction, so
    a single-candidate run's own output stays self-explanatory rather than
    silently reporting on only a fraction of the roster with no note of why.

    Returns (candidates, skipped, preflight_detail) rather than raising on a
    missing key or a failed preflight, so a partial run still produces a
    usable comparison and a failed one still explains itself.
    """
    candidates: dict[str, object] = {}
    skipped: list[str] = []
    preflight_detail: dict[str, str] = {}
    to_check: list[tuple[str, str, str, str, bool]] = []  # (name, endpoint, key, model, disable_reasoning)

    def _wanted(name: str) -> bool:
        return only is None or name == only

    mistral_key = os.environ.get("MISTRAL_API_KEY")
    if not _wanted("mistral-small-3.1"):
        preflight_detail["mistral-small-3.1"] = f"not requested (running only {only!r})"
    elif mistral_key:
        to_check.append(
            ("mistral-small-3.1", "https://api.mistral.ai/v1/chat/completions", mistral_key, MISTRAL_DIRECT_MODEL, False)
        )
    else:
        skipped.append("mistral-small-3.1 (MISTRAL_API_KEY not set)")
        preflight_detail["mistral-small-3.1"] = "MISTRAL_API_KEY not set"

    wanted_openrouter = [entry for entry in CANDIDATE_MODELS if _wanted(entry[0])]
    for name, _, _ in CANDIDATE_MODELS:
        if not _wanted(name):
            preflight_detail[name] = f"not requested (running only {only!r})"

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        # Every candidate here is reached through OpenRouter's own catalogue
        # rather than a first-party endpoint: one key covers all of them
        # instead of requiring a separate credential per provider. OpenRouter
        # is a named, established aggregator that proxies to the real
        # provider -- unlike an unverified gateway once considered and
        # rejected for this bench (see recursive_engine_decision_record.md).
        #
        # Kimi and DeepSeek are Chinese-developed; reached here through
        # OpenRouter rather than a China-hosted endpoint directly, and every
        # document this bench sends is synthetic (enforced by RlmEngine's
        # data_class check, independent of this list), so there is no live
        # data-residency exposure here. The consideration becomes relevant
        # the moment any candidate here is considered for production use on
        # real case content.
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        for name, model, disable_reasoning in wanted_openrouter:
            to_check.append((name, endpoint, openrouter_key, model, disable_reasoning))
    elif wanted_openrouter:
        skipped.append(f"{', '.join(name for name, _, _ in wanted_openrouter)} (OPENROUTER_API_KEY not set)")
        for name, _, _ in wanted_openrouter:
            preflight_detail[name] = "OPENROUTER_API_KEY not set"

    if to_check:
        print(f"Preflighting {len(to_check)} candidate(s) (timeout {PREFLIGHT_TIMEOUT_SECONDS}s each)...")
    for name, endpoint, key, model, disable_reasoning in to_check:
        reachable, reason = _preflight(name, endpoint, key, model, disable_reasoning=disable_reasoning)
        preflight_detail[name] = reason
        if reachable:
            candidates[name] = _bind(_http_chat_completion, endpoint, key, model, disable_reasoning=disable_reasoning)
        else:
            skipped.append(f"{name} (preflight failed: {reason})")

    return candidates, skipped, preflight_detail



def _bind(fn, endpoint, key, model, *, disable_reasoning=False):
    def _call(prompt: str) -> str:
        try:
            return fn(endpoint, key, model, prompt, timeout=CALL_TIMEOUT_SECONDS, disable_reasoning=disable_reasoning)
        except urllib.error.HTTPError as error:
            print(f"  [warn] {model}: HTTP {error.code} {error.reason}", file=sys.stderr)
            return ""
        except (urllib.error.URLError, ProviderResponseError, KeyError, json.JSONDecodeError, TimeoutError) as error:
            # A provider error becomes empty text, which the engine already
            # treats as model_emitted_no_action -- consistent with how the
            # adapter treats a refused SafeModelClient call.
            print(f"  [warn] {model}: {type(error).__name__}: {error}", file=sys.stderr)
            return ""
        except Exception as error:  # noqa: BLE001 - see module docstring: any failure here
            # degrades to an empty response, it never crashes the script. A
            # provider integration talks to code we do not control, and its
            # failure modes are open-ended -- the case that motivated this
            # clause was OpenRouter returning HTTP 200 with an empty choices
            # list, which is neither a network error nor a KeyError.
            print(f"  [warn] {model}: unexpected {type(error).__name__}: {error}", file=sys.stderr)
            return ""

    return _call


def _preflight(name: str, endpoint: str, key: str, model: str, *, disable_reasoning: bool = False) -> tuple[bool, str]:
    """One short, short-timeout call per candidate before committing to the full bench.

    The run that motivated this function spent 10.5 minutes discovering that
    most candidates were unreachable, because each one was given the full
    per-case budget before its failure became visible. A bad key or an
    unrecognised model slug almost always fails fast (an auth or not-found
    response arrives in well under a second); a preflight call with a short
    timeout catches that in seconds per candidate instead of minutes.

    Returns (reachable, reason) so the reason survives into the results file
    rather than existing only as a line on stderr -- when every candidate
    fails, that reason is the entire useful output of the run.
    """
    try:
        response = _http_chat_completion(
            endpoint, key, model, "final(preflight check -- respond with exactly this action)",
            timeout=PREFLIGHT_TIMEOUT_SECONDS, disable_reasoning=disable_reasoning,
        )
        if not response.strip():
            print(f"  [preflight] {model}: reachable but returned empty text", file=sys.stderr)
            return True, "reachable, empty response to preflight"
        return True, "reachable"
    except urllib.error.HTTPError as error:
        # Distinguished from a generic URLError because the status code is the
        # single most useful diagnostic: 401/403 means the key, 404 means the
        # model slug, 429 means rate limiting. Guessing between those from a
        # generic message is what makes a failed run hard to act on.
        detail = f"HTTP {error.code} {error.reason}"
        print(f"  [preflight] {model}: unreachable ({detail}), skipping the full bench for it", file=sys.stderr)
        return False, detail
    except (urllib.error.URLError, ProviderResponseError, KeyError, json.JSONDecodeError, TimeoutError) as error:
        detail = f"{type(error).__name__}: {error}"
        print(f"  [preflight] {model}: unreachable ({detail}), skipping the full bench for it", file=sys.stderr)
        return False, detail
    except Exception as error:  # noqa: BLE001 - see _bind: any unexpected provider
        # behaviour must degrade to a reported, catchable outcome, never a
        # script crash. This is the same defensive posture as _bind's final
        # clause, kept here too because _preflight has its own except chain
        # rather than calling through _bind.
        detail = f"unexpected {type(error).__name__}: {error}"
        print(f"  [preflight] {model}: unreachable ({detail}), skipping the full bench for it", file=sys.stderr)
        return False, detail


def main() -> int:
    """Top-level safety net: every path below writes a results file before returning.

    _bind and _preflight already convert provider failures into reported
    outcomes, but that only covers calls made through them. Anything else
    unexpected -- a bug in this script, a change in a dependency's behaviour,
    a JSON payload shaped in a way nothing here anticipated -- must still
    leave a diagnostic file behind rather than exiting via an uncaught
    traceback with nothing written. The run that motivated this function did
    exactly that: an IndexError from an empty ``choices`` list, raised two
    calls below _bind's except clause at the time, crashed the whole script
    before a single byte was written.
    """
    out_path = Path("bench_results.json")
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--out", type=Path, default=out_path)
        parser.add_argument(
            "--candidate",
            default=None,
            help=(
                "Run only this one candidate (by bench name, e.g. 'claude-sonnet-5') instead of "
                "the full roster. What a parallel matrix job passes so 21 candidates run as 21 "
                "concurrent single-candidate jobs rather than one long sequential job -- see "
                "all_candidate_names() and the workflow's 'prepare' job."
            ),
        )
        parser.add_argument(
            "--list-candidates",
            action="store_true",
            help="Print every candidate name as a JSON array and exit, for the workflow's matrix step.",
        )
        parser.add_argument(
            "--roster",
            default=None,
            help=(
                "Comma-separated candidate names to restrict --list-candidates to (e.g. "
                "'llama-4-maverick,gemma-4-31b'), for a focused workflow comparing a handful of "
                "candidates rather than the full roster. Unknown names are reported and exit "
                "non-zero rather than silently producing a shorter-than-expected list. Has no "
                "effect without --list-candidates."
            ),
        )
        parser.add_argument(
            "--cases",
            choices=("baseline", "advanced"),
            default="baseline",
            help=(
                "'baseline' (default) runs the eleven-case set every candidate is measured "
                "against. 'advanced' adds ADVANCED_CASES -- eight further, more subtle cases "
                "(three-document synthesis, absence of a requested fact, confusable terms, "
                "out-of-order chronology, a longer multi-section document, conflicting values "
                "across documents, a weight-based dose, a two-hop inference) built for a "
                "smaller, focused comparison where the added cost and time are affordable. "
                "Never changes the default for the full-roster workflow."
            ),
        )
        args = parser.parse_args()
        out_path = args.out

        if args.list_candidates:
            names = all_candidate_names()
            if args.roster:
                requested = [item.strip() for item in args.roster.split(",") if item.strip()]
                unknown = [name for name in requested if name not in names]
                if unknown:
                    print(f"Unknown candidate name(s) in --roster: {unknown}", file=sys.stderr)
                    print(f"Valid names: {names}", file=sys.stderr)
                    return 1
                names = requested
            print(json.dumps(names))
            return 0

        cases = BENCH_CASES + ADVANCED_CASES if args.cases == "advanced" else BENCH_CASES
        return _run(args.out, only=args.candidate, cases=cases)
    except Exception as error:  # noqa: BLE001 - last-resort net, see docstring
        import traceback

        trace = traceback.format_exc()
        print(f"\nUnexpected error: {type(error).__name__}: {error}", file=sys.stderr)
        print(trace, file=sys.stderr)
        try:
            _write_results(
                out_path,
                {
                    "status": "crashed",
                    "verdict": f"the bench script crashed with an unhandled {type(error).__name__}; see traceback",
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "traceback": trace,
                    "results": [],
                },
            )
            print(f"Crash diagnostics written to {out_path}", file=sys.stderr)
        except Exception as write_error:  # noqa: BLE001 - do not let the handler itself crash
            print(f"Could not write crash diagnostics either: {write_error}", file=sys.stderr)
        return 1


def _run(out: Path, *, only: str | None = None, cases: tuple = BENCH_CASES) -> int:
    candidates, skipped, preflight_detail = build_candidates(only=only)

    if not candidates:
        # Write the results file even here. This is the case where the
        # artifact matters most: the run failed, and the per-candidate reason
        # is the only thing that tells the operator whether to fix a key, a
        # model slug, or nothing at all. Exiting without writing leaves them
        # with an empty artifact and a red X.
        payload = {
            "status": "no_candidates",
            "verdict": (
                "no candidate survived preflight; check the per-candidate reasons below -- "
                "401/403 indicates the API key, 404 indicates the model slug, 429 indicates rate limiting"
            ),
            "models": 0,
            "results": [],
            "skipped": skipped,
            "preflight": preflight_detail,
        }
        _write_results(out, payload)
        print("\nNo candidate survived preflight; nothing to bench.", file=sys.stderr)
        for name, detail in sorted(preflight_detail.items()):
            print(f"  {name}: {detail}", file=sys.stderr)
        print(f"\nDiagnostics written to {out}", file=sys.stderr)
        return 1

    print(f"\nBenching: {', '.join(sorted(candidates))}")
    if skipped:
        print(f"Skipped: {'; '.join(skipped)}")

    report = bench_models(candidates, cases, adherence_target=0.95, budget_factory=_bench_budget)
    payload = report.as_dict()
    payload["status"] = "completed"
    payload["skipped"] = skipped
    payload["preflight"] = preflight_detail

    _write_results(out, payload)

    print(f"\nVerdict: {payload['verdict']}\n")
    print(f"{'model':<20}{'adherence':>11}{'completion':>12}{'near-miss share':>18}")
    for row in payload["results"]:
        print(f"{row['model_name']:<20}{row['adherence']:>10.0%}{row['completion_rate']:>12.0%}{row['near_miss_share']:>17.0%}")
    print(f"\nFull report written to {out}")
    return 0


def _write_results(path: Path, payload: dict) -> None:
    """Write results, creating the parent directory if the caller named one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    sys.exit(main())
