#!/usr/bin/env python
"""Point 1 (12 October 2026): does a chat model read the role of an anatomical word better when it is given
what the Kintsch reader built? Measured on the roles of the development ("silver") gold set.

Arms, all on the same items:

* ``lattice``     the chunk lattice alone (blocks, head, frames): link / role / not a site / underspecified.
* ``reader``      the construction-integration reader alone (six readings, integration); ``underspecified``
                  when the best reading does not beat the second by the margin.
* ``llm``         each chat model with the question of ``melampo.evaluation.llm_reading`` (one mark for the
                  word, rules, roles, structured answer, one follow-up), no evidence.
* ``llm+reader``  the same question plus what the lattice and the reader built (readings with their share,
                  evidence nodes, the lattice's block and why). The model may disagree with the reader.

Every arm answers a role of our schema or ``uncertain``. Measures, per arm and per stratum: exact role
accuracy on the items the arm answered, the share it left uncertain, **critical errors** (it says the
word is a body site, structure or a site role, where the gold says it is not one, or the opposite), the
three-way grouping (structure / site role / not a site) and a bootstrap interval grouped by report.

Where the text goes. The chat models are reached through an OpenAI-compatible endpoint: a local server
(vLLM, ``--endpoint http://localhost:8000/v1/chat/completions``, the default) or OpenRouter. A
non-local endpoint is refused unless ``--public-data`` says the texts are public (E3C, PARROT, IU X-ray
are; a patient's report is not and never goes to an external service). Answers are cached by
(model, prompt): a rerun asks nothing twice.

    python scripts/role_probe.py --gold silver/gold.jsonl --reports silver/reports.jsonl \
        --reader-dir ci_resources --arms lattice,reader,llm,llm+reader \
        --models openai/gpt-oss-120b,mistralai/mistral-small-2603 \
        --endpoint https://openrouter.ai/api/v1/chat/completions --public-data \
        --out role_probe.json --markdown role_probe.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melampo.evaluation import llm_reading as lr  # noqa: E402
from melampo.evaluation.selective_calibration import grouped_bootstrap  # noqa: E402

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})
SITE_ROLES = lr.SITE_ROLES - {"structure"}
NOT_SITE = frozenset({"inside_a_name", "not_a_body_site", "other"})


def group(role: str) -> str:
    """structure / site role / not a site / uncertain."""
    if role == "structure":
        return "structure"
    if role in SITE_ROLES:
        return "site_role"
    if role in NOT_SITE:
        return "not_a_site"
    return "uncertain"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text("utf-8").split("\n") if x.strip()]


def place(row: dict, reports: dict[str, str]) -> int | None:
    """Offset of the mention in its sentence (the gold offsets are in the report)."""
    s, m = row["sentence"], row["mention"]
    text = reports.get(row["report_id"], "")
    off = text.find(s) if text else -1
    if off >= 0:
        at = row["start"] - off
        if 0 <= at and s[at: at + len(m)] == m:
            return at
    at = s.find(m)
    return at if at >= 0 else None


def items_of(gold: list[dict], reports: dict[str, str]) -> list[dict]:
    out = []
    for g in gold:
        if not g.get("role"):  # AMBIGUOUS rows carry no role: nothing to compare
            continue
        at = place(g, reports)
        if at is None:
            continue
        out.append({**g, "at": at})
    return out


# -- the lattice and the reader -----------------------------------------------------------------


def build_reading(reader_dir: Path | None):
    from melampo.memory import chunk_lattice as cl
    from melampo.memory.longer_names import LongerNames

    lattice = cl.ChunkLattice(cl.BlockMemory.load(cl.DEFAULT_PATH, LongerNames.load().names, cl.load_anatomy_names()))
    reader = None
    if reader_dir is not None:
        from melampo.memory import ci_reader as ci

        reader = ci.load_reader(reader_dir, lattice)
    return lattice, reader


def lattice_role(reading) -> str:
    if reading.outcome == "link":
        return "structure"
    if reading.outcome == "role":
        return reading.role if reading.role in lr.ROLES else "other"
    if reading.outcome == "not_a_site":
        return reading.role if reading.role in NOT_SITE else "not_a_body_site"
    if reading.outcome == "unread":
        return "structure"  # nothing in the lattice: the word is read as itself
    return "uncertain"


def evidence(item: dict, lattice, reader, docs: dict) -> tuple[dict, list[str]]:
    """The two readings and the lines given to the model in the ``llm+reader`` arm."""
    seen = lattice.read(item["mention"], item["sentence"], item["at"])
    got = {"lattice": lattice_role(seen), "lattice_detail": {k: getattr(seen, k) for k in
                                                            ("outcome", "role", "kind", "block", "why", "alternative", "exact")}}
    lines = [f"- chunk lattice ({'a name the vocabulary has as written' if seen.exact else 'composed from the kind of one head word'}): outcome {seen.outcome}"
             + (f", role {seen.role}" if seen.role else "")
             + (f", block «{seen.block}»" if seen.block else "")
             + (f", head kind {seen.kind}" if seen.kind else "")
             + (f" ({seen.why})" if seen.why else "")]
    if reader is not None:
        rid = item["report_id"]
        if rid not in docs:
            text = item.get("_report", "")
            vector, key = reader.document(text) if text else (None, "")
            docs[rid] = (vector, key, reader.discourse(text) if text else None)
        vector, key, discourse = docs[rid]
        r = reader.read(item["mention"], item["sentence"], item["at"], doc=None, doc_vector=vector,
                        doc_key=key, discourse=discourse, doc_at=item.get("start"))
        got["reader"] = "uncertain" if r.decision == "underspecified" else r.decision
        got["reader_detail"] = {"shares": {k: round(v, 3) for k, v in r.shares.items()}, "margin": round(r.margin, 3),
                                "nodes": r.nodes[:12]}
        ranked = sorted(r.shares.items(), key=lambda kv: -kv[1])
        lines.append("- reader, readings after integration: " + ", ".join(f"{k} {v:.2f}" for k, v in ranked)
                     + f" (decision: {r.decision}; margin {r.margin:.2f})")
        lines.append("- reader, evidence in the network: " + ", ".join(r.nodes[:12]))
    return got, lines


# -- chat models --------------------------------------------------------------------------------


class Cache:
    def __init__(self, path: Path | None):
        self.path, self.lock, self.data = path, threading.Lock(), {}
        if path and path.exists():
            for line in path.read_text("utf-8").split("\n"):
                if line.strip():
                    x = json.loads(line)
                    self.data[x["key"]] = x["text"]

    @staticmethod
    def key(model: str, prompt: str) -> str:
        return hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()

    def wrap(self, model: str, chat):
        def call(prompt: str) -> str:
            k = self.key(model, prompt)
            if k in self.data:
                return self.data[k]
            text = chat(prompt)
            with self.lock:
                self.data[k] = text
                if self.path:
                    with self.path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"key": k, "model": model, "text": text}, ensure_ascii=False) + "\n")
            return text
        return call


def check_endpoint(endpoint: str, public_data: bool) -> None:
    host = urlparse(endpoint).hostname or ""
    if host not in LOCAL_HOSTS and not public_data:
        raise SystemExit(
            f"refused: {host} is not a local server. Only public texts may go to an external service; "
            "pass --public-data if they are (E3C, PARROT, IU X-ray), never for patients' reports."
        )


# Precision of the weights, per model: part of what is measured. gpt-oss-120b runs in FP8 (decided by
# Frank, 12 October 2026); the official checkpoint is MXFP4, so FP8 means a provider that declares fp8
# (OpenRouter) or a pinned FP8 checkpoint on the local server.
DEFAULT_QUANTIZATION = "openai/gpt-oss-120b=fp8"
QUANTIZATIONS = frozenset({"int4", "int8", "fp4", "fp6", "fp8", "fp16", "bf16", "fp32"})


def parse_quantization(spec: str) -> dict[str, str]:
    """``model=fp8,other=bf16`` -> {model: fp8, other: bf16}."""
    out = {}
    for part in (p for p in spec.split(",") if p.strip()):
        model, _, q = part.strip().partition("=")
        if q not in QUANTIZATIONS:
            raise SystemExit(f"quantization {q!r} for {model}: one of {sorted(QUANTIZATIONS)}")
        out[model] = q
    return out


def make_chats(models: list[str], endpoint: str, cache: Cache, max_tokens: int,
               quantization: dict[str, str] | None = None) -> dict:
    from melampo.evaluation import linking_bench as lb

    router = "openrouter.ai" in endpoint
    key = os.environ.get("OPENROUTER_API_KEY", "") if router else os.environ.get("LLM_API_KEY", "local")
    if router and not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    chats = {}
    for m in models:
        q = (quantization or {}).get(m)
        # On OpenRouter the precision is a hard filter (no fallback to another one); a local server serves
        # the checkpoint it was started with, and the run records which one was asked for.
        provider = {"quantizations": [q], "allow_fallbacks": False} if (router and q) else None
        name = f"{m}@{q}" if q else m
        chats[name] = cache.wrap(name, lb.OpenRouterChat(m, key, endpoint=endpoint, max_tokens=max_tokens, timeout=180.0,
                                                         retries=4, min_interval=0.2, provider=provider))
    return chats


def probe_models(chats: dict) -> None:
    """One short question per model before the run: a model no provider serves at the asked precision
    stops the run here, instead of turning every answer into an invalid one."""
    for name, chat in chats.items():
        try:
            chat('Answer with JSON only: {"ok": "yes"}')
        except Exception as error:  # noqa: BLE001
            raise SystemExit(f"{name}: not reachable as asked ({str(error)[:200]})") from error


def examples_for(item: dict, pool: list[dict], k: int) -> list[dict]:
    """Annotated items of *other reports* most similar to this one (shared words), roles balanced so that
    one non-structure role is among them when the pool has one."""
    if k <= 0:
        return []
    words = set(item["sentence"].lower().split())
    scored = sorted((-len(words & set(p["sentence"].lower().split())), p["item_id"], p) for p in pool
                    if p["report_id"] != item["report_id"] and p["language"] == item["language"])
    chosen = [p for _, _, p in scored[:k]]
    if chosen and all(p["role"] == "structure" for p in chosen):
        other = next((p for _, _, p in scored if p["role"] != "structure"), None)
        if other is not None:
            chosen[-1] = other
    out = []
    for p in chosen:
        phrase = p.get("span_text") or p["mention"]
        out.append({"marked": lr.mark(p["sentence"], p["at"], p["at"] + len(p["mention"])), "phrase": phrase,
                    "type": "anatomical structure" if p["role"] == "structure" else "other", "role": p["role"]})
    return out


LANG = {"it": "Italian", "en": "English"}


def ask_item(item: dict, chats: dict, arms: list[str], reader_lines: list[str], examples: list[dict]) -> dict:
    out = {}
    s, a, e = item["sentence"], item["at"], item["at"] + len(item["mention"])
    for arm in arms:
        if not arm.startswith("llm"):
            continue
        prompt = lr.prompt(s, a, e, reader=reader_lines if arm == "llm+reader" else None, examples=examples,
                           language=LANG.get(item["language"], ""))
        for name, chat in chats.items():
            asked = lr.ask(chat, prompt, s, a, e)
            out[f"{arm}:{name}"] = asked.to_json()
    return out


UNREACHABLE = "unreachable"   # the model could not be asked (HTTP error): not an answer, not "uncertain"
STOP_AFTER = 12               # consecutive refusals (401/402/403/429-style key or credit limits) that stop the run


def cascade(items: list[dict]) -> None:
    """Arms composed from the others (12 October 2026, Frank: deterministic only where the vocabulary has
    the correspondence, the reader and the models reading the phrase otherwise):

    * ``exact->reader``             the lattice's reading when it is exact, else the reader's;
    * ``exact->llm+reader:<model>`` the lattice's reading when exact, else the model given the reader;
    * ``exact->agree:<model>``      the lattice's reading when exact, else the reader's and the model's
                                    when they agree, ``uncertain`` when they do not (to a person)."""
    for it in items:
        a = it["answers"]
        exact = bool((it.get("evidence") or {}).get("lattice_detail", {}).get("exact")) and "lattice" in a \
            and a["lattice"] != "structure"
        if "reader" in a:
            a["exact->reader"] = a["lattice"] if exact else a["reader"]
        for key in [k for k in a if k.startswith("llm+reader:")]:
            model = key.split(":", 1)[1]
            a[f"exact->llm+reader:{model}"] = a["lattice"] if exact else a[key]
            if "reader" in a:
                agree = a[key] if a[key] == a["reader"] else "uncertain"
                if a[key] == UNREACHABLE:
                    agree = UNREACHABLE
                a[f"exact->agree:{model}"] = a["lattice"] if exact else agree


# -- scoring ------------------------------------------------------------------------------------


def score(items: list[dict], arm: str) -> dict:
    said = [(it, it["answers"].get(arm)) for it in items if it["answers"].get(arm) not in (None, UNREACHABLE)]
    if not said:
        return {}
    answered = [(it, r) for it, r in said if r != "uncertain"]
    right = [r == it["role"] for it, r in answered]
    grouped = [group(r) == group(it["role"]) for it, r in answered]
    critical = [(group(r) == "not_a_site") != (group(it["role"]) == "not_a_site") for it, r in answered]
    wrong_structure = sum(r == "structure" and it["role"] != "structure" for it, r in answered)
    by_gold = defaultdict(Counter)
    for it, r in said:
        by_gold[it["role"]][r] += 1
    strata = defaultdict(lambda: [0, 0])
    for (it, r), ok in zip(answered, right, strict=True):
        strata[it["stratum"]][0] += ok
        strata[it["stratum"]][1] += 1
    return {
        "items": len(said),
        "uncertain": round(1 - len(answered) / len(said), 4),
        "accuracy": round(sum(right) / len(answered), 4) if answered else None,
        "accuracy_3way": round(sum(grouped) / len(answered), 4) if answered else None,
        "critical": sum(critical),
        "said_structure_wrongly": wrong_structure,
        "error_by_report": grouped_bootstrap([it["report_id"] for it, _ in answered], right) if answered else None,
        "by_stratum": {k: f"{v[0]}/{v[1]}" for k, v in sorted(strata.items())},
        "confusion": {g: dict(c.most_common()) for g, c in sorted(by_gold.items())},
    }


def markdown(report: dict) -> str:
    lines = ["# Role probe: lattice, reader, chat models with and without the reader", "",
             f"{report['n']} items with a role (roles compared on the anatomy items) ({json.dumps(report['gold_roles'])}); endpoint {report['endpoint']}; "
             f"models {report['models']}; examples {report['examples']}.", "",
             "| arm | items | uncertain | accuracy (answered) | 3-way | critical | said structure wrongly | error by report [95%] |",
             "|---|---|---|---|---|---|---|---|"]
    for arm, s in report["arms"].items():
        if not s:
            continue
        b = s.get("error_by_report") or {}
        lines.append(f"| {arm} | {s['items']} | {s['uncertain']:.3f} | {s['accuracy']} | {s['accuracy_3way']} | "
                     f"{s['critical']} | {s['said_structure_wrongly']} | {b.get('error')} [{b.get('low')}, {b.get('high')}] |")
    if report.get("common_items") not in (None, report["n"]):
        lines += ["", f"The same arms on the {report['common_items']} items every arm could answer "
                  "(some calls were refused: HTTP error, not counted as uncertain):", "",
                  "| arm | items | uncertain | accuracy (answered) | 3-way | critical | said structure wrongly |",
                  "|---|---|---|---|---|---|---|"]
        for arm, s in sorted(report["common"].items(), key=lambda kv: -(kv[1] or {}).get("accuracy") if kv[1] and kv[1].get("accuracy") is not None else 0):
            if s:
                lines.append(f"| {arm} | {s['items']} | {s['uncertain']:.3f} | {s['accuracy']} | {s['accuracy_3way']} | "
                             f"{s['critical']} | {s['said_structure_wrongly']} |")
    lines += ["", "Mentions the gold set calls NOT_ANATOMY (lesions, fluids, cells), scored apart: "
              "how many each arm reads as the structure itself.", "", "| arm | items | said structure |", "|---|---|---|"]
    for arm, s in report.get("not_anatomy", {}).items():
        if s:
            lines.append(f"| {arm} | {s['items']} | {s['said_structure_wrongly']} |")
    lines += ["", "Answers of the chat models (invalid, asked again, changed by the follow-up, bare word):", "",
              "```", json.dumps(report["llm_answers"], indent=1), "```", "", "Per stratum and confusion:", "", "```",
              json.dumps({a: {"by_stratum": s.get("by_stratum"), "confusion": s.get("confusion")}
                          for a, s in report["arms"].items() if s}, indent=1, ensure_ascii=False), "```"]
    return "\n".join(lines) + "\n"


def run(args) -> dict:
    gold = [g for path in args.gold for g in _jsonl(path)]
    reports = {r["report_id"]: r["text"] for path in (args.reports or []) for r in _jsonl(path)}
    items = items_of(gold, reports)
    if args.limit:
        items = items[: args.limit]
    for it in items:
        it["_report"] = reports.get(it["report_id"], "")
        it["answers"] = {}
    arms = [a for a in args.arms.split(",") if a]
    models = [m for m in args.models.split(",") if m]
    lattice = reader = None
    if any(a in arms for a in ("lattice", "reader", "llm+reader")):
        lattice, reader = build_reading(args.reader_dir)
    docs: dict = {}
    lines_of = {}
    for it in items:
        if lattice is None:
            break
        got, lines = evidence(it, lattice, reader, docs)
        it["evidence"] = {k: v for k, v in got.items() if k.endswith("detail")}
        if "lattice" in arms:
            it["answers"]["lattice"] = got["lattice"]
        if "reader" in arms and "reader" in got:
            it["answers"]["reader"] = got["reader"]
        lines_of[it["item_id"]] = lines
    llm_arms = [a for a in arms if a.startswith("llm")]
    raw: dict[str, dict] = {}
    if llm_arms and models:
        check_endpoint(args.endpoint, args.public_data)
        chats = make_chats(models, args.endpoint, Cache(args.cache), args.max_tokens,
                           parse_quantization(args.quantization))
        probe_models(chats)
        started = time.monotonic()

        refused = {"streak": 0, "stopped": False}
        lock = threading.Lock()

        def one(it):
            if refused["stopped"]:
                return it["item_id"], {}
            got = ask_item(it, chats, llm_arms, lines_of.get(it["item_id"], []),
                           examples_for(it, items, args.examples))
            with lock:
                if got and all(a.get("error") for a in got.values()):
                    refused["streak"] += 1
                    if refused["streak"] >= STOP_AFTER and not refused["stopped"]:
                        refused["stopped"] = True
                        print(f"::error::every model refused the last {STOP_AFTER} items "
                              f"({next(iter(got.values()))['error']}): key, credit or rate limit? "
                              "Stopping; the answers so far stay in the cache.", file=sys.stderr, flush=True)
                elif got:
                    refused["streak"] = 0
            return it["item_id"], got

        with ThreadPoolExecutor(args.workers) as pool:
            for n, (iid, answers) in enumerate(pool.map(one, items), 1):
                raw[iid] = answers
                if n % 50 == 0:
                    print(f"[{time.monotonic() - started:6.0f}s] {n}/{len(items)}", file=sys.stderr, flush=True)
        args.stopped = refused["stopped"]
        for it in items:
            for key, a in raw.get(it["item_id"], {}).items():
                it["answers"][key] = (UNREACHABLE if a.get("error") else
                                      a["answer"].get("role") if a["valid"] else "uncertain")
            it["llm"] = raw.get(it["item_id"], {})
    cascade(items)
    keys = sorted({k for it in items for k in it["answers"]})
    stats: dict[str, Counter] = defaultdict(Counter)
    for it in items:
        for key, a in (it.get("llm") or {}).items():
            n = stats[key]
            n["answers"] += 1
            n["invalid"] += int(not a["valid"])
            n["asked_again"] += int(a["asked_again"])
            n["changed"] += int(a["changed"])
            n["bare_word"] += int(lr._norm(a["answer"].get("phrase", "")) == lr._norm(it["mention"]))
    common = [it for it in items if all(it["answers"].get(k) not in (None, UNREACHABLE) for k in keys)]
    report = {
        "n": len(items), "gold_roles": dict(Counter(it["role"] for it in items).most_common()),
        "endpoint": urlparse(args.endpoint).hostname if llm_arms and models else None,
        "models": models, "examples": args.examples,
        "quantization": parse_quantization(args.quantization),
        # Roles are compared where the gold says the mention is anatomy (a class or none of our classes);
        # NOT_ANATOMY mentions (lesions, fluids, cells: a convention of the gold set) are scored apart.
        "arms": {k: score([it for it in items if it["structure"] != "NOT_ANATOMY"], k) for k in keys},
        "not_anatomy": {k: score([it for it in items if it["structure"] == "NOT_ANATOMY"], k) for k in keys},
        # Same items for every arm: those every arm could answer (a model that could not be reached on some
        # items would otherwise be compared with the others on a different, easier or harder, set).
        "common_items": len(common),
        "common": {k: score([it for it in common if it["structure"] != "NOT_ANATOMY"], k) for k in keys},
        "llm_answers": {k: dict(v) for k, v in stats.items()},
        "items": [{k: it.get(k) for k in ("item_id", "report_id", "stratum", "language", "mention", "sentence",
                                          "role", "structure", "answers", "evidence", "llm")} for it in items],
    }
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gold", type=Path, nargs="+", required=True,
                    help="gold rows (silver gold set; Rad-SpRL landmarks from scripts/rad_sprl.py, role structure)")
    ap.add_argument("--reports", type=Path, nargs="+")
    ap.add_argument("--reader-dir", type=Path, help="resources of the reader (scripts/build_ci_resources.py); "
                                                     "without it the reader arm is off and llm+reader sees the lattice only")
    ap.add_argument("--arms", default="lattice,reader,llm,llm+reader")
    ap.add_argument("--models", default="openai/gpt-oss-120b,google/gemma-4-31b-it,mistralai/mistral-small-2603")
    ap.add_argument("--endpoint", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--quantization", default=DEFAULT_QUANTIZATION,
                    help="precision of the weights per model, model=fp8,... (OpenRouter: hard filter, no fallback)")
    ap.add_argument("--public-data", action="store_true", help="the texts are public: an external endpoint is allowed")
    ap.add_argument("--examples", type=int, default=0, help="annotated items of other reports shown per question")
    ap.add_argument("--cache", type=Path, default=Path("role_probe_answers.jsonl"))
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("role_probe.json"))
    ap.add_argument("--markdown", type=Path, default=Path("role_probe.md"))
    args = ap.parse_args(argv)
    report = run(args)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), "utf-8")
    args.markdown.write_text(markdown(report), "utf-8")
    print(markdown(report).split("\n\nAnswers")[0])
    return 3 if getattr(args, "stopped", False) else 0


if __name__ == "__main__":
    raise SystemExit(main())
