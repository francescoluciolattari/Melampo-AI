#!/usr/bin/env python
"""The readers' adapters, every two or three months (11 November 2026): the third way after the manual and the
memory of examples, connected to them.

    python scripts/reader_training.py status                     # the memory since the last training
    python scripts/reader_training.py export --run owner_probe.json [--run ...] --items 'owner_items_*.jsonl' \\
        --out reader_train.jsonl                                 # the verified examples, as conversations
    python scripts/reader_training.py train --data reader_train.jsonl --base google/gemma-4-31b-it \\
        --out adapters/reader-2027-01                            # LoRA, on the local GPUs, never on GitHub
    python scripts/reader_training.py decide --base base_probe.json --adapter adapter_probe.json \\
        --set ctrate9 --name reader-2027-01                       # the rule fixed before

What the adapter learns is only the memory: examples on which the label written before any reading and every
model's reading agree (``reading_memory.verified``), labelled with the convention in force. The target of each
example is the answer of a model that read it as the label says, so the adapter learns the whole answer, not
only the label.

The rule, fixed on 11 November 2026 before any adapter exists: on a new sample, labelled before any reading and
never in the memory, the probe runs twice with the same manual and memory, without and with the adapter. The
adapter enters if every criterion passes with it and its errors (criteria 1, 2, 3, 4, 6 with the closed doubts wrong, 7, 8, as counts) are
fewer than without it; otherwise it stays out, and the reason is written in data/reading/training_log.yml.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from melampo.memory import reading_memory as rm  # noqa: E402

LOG = ROOT / "data/reading/training_log.yml"
CONFIG = ROOT / "data/reading/training.yml"
# every two or three months (decided 11 November 2026)
PERIOD_DAYS = 60


def _log() -> list[dict]:
    import yaml

    return (yaml.safe_load(LOG.read_text("utf-8")) or {}).get("trainings", []) if LOG.exists() else []


def _memory() -> list[dict]:
    return [json.loads(x) for x in rm.MEMORY.read_text("utf-8").splitlines() if x.strip()] if rm.MEMORY.exists() \
        else []


def cmd_status(args: argparse.Namespace) -> int:
    memory, log = _memory(), _log()
    last = max((t.get("date", "") for t in log), default="")
    new = [m for m in memory if m.get("added", "") > last]
    due = not last or (dt.date.today() - dt.date.fromisoformat(last)).days >= PERIOD_DAYS
    print(json.dumps({"examples": len(memory), "since_last_training": len(new), "last_training": last or None,
                      "due": due, "period_days": PERIOD_DAYS}, indent=1))
    return 0


def target(answer: dict) -> dict:
    """The answer a model gave, in the question's form, from its parsed reading."""
    out = {"marked": answer.get("marked", ""), "same": answer.get("same", "yes"), "named": answer.get("named", ""),
           "observed": [{k: o.get(k, "") for k in ("words", "aspect", "usual", "differs", "said_to_be", "of",
                                                    "relation", "reaches", "refers_to")}
                        for o in answer.get("observed", [])]}
    for key in ("normal", "past", "device", "not_assessable", "indication", "procedure", "other_person",
                "reported", "undetermined", "limits"):
        out[key] = list(answer.get(key, []))
    refs = {r["words"]: r for r in answer.get("measure_refs", [])}
    out["measure"] = [{"words": w, "reference": refs.get(w, {}).get("reference", ""),
                       "against": refs.get(w, {}).get("against", "")} for w in answer.get("measure", [])]
    out["uncertainty"] = [{"words": u["words"], "about": u.get("about", ""), "level": u.get("level", "possible")}
                          for u in answer.get("uncertain_about", [])]
    out["conclusion"] = answer.get("conclusion", "none")
    out["conclusion_of"] = answer.get("conclusion_of", "")
    return out


def cmd_export(args: argparse.Namespace) -> int:
    from melampo.evaluation.llm_reading import claims_prompt

    memory = {m["item_id"]: m for m in _memory()}
    items = {}
    for f in sorted(glob.glob(args.items)):
        for x in Path(f).read_text("utf-8").splitlines():
            if x.strip():
                it = json.loads(x)
                items[it["item_id"]] = it
    rows = {}
    for run in args.run:
        for r in json.loads(Path(run).read_text("utf-8")).get("rows", []):
            rows.setdefault(r["item_id"], r)
    manual = rm.manual_text()
    out, missing = [], 0
    for item_id, m in memory.items():
        it, row = items.get(item_id), rows.get(item_id)
        answers = ((row or {}).get("claims") or {}).get("models") or {}
        if not it or not answers:
            missing += 1
            continue
        model, answer = sorted(answers.items())[0]
        at = it["clause"].find(it["mention"])
        prompt = claims_prompt(it["clause"], at, at + len(it["mention"]), it["structure"].replace("_", " "),
                               manual=manual)
        out.append({"messages": [{"role": "user", "content": prompt},
                                 {"role": "assistant", "content": json.dumps(target(answer), ensure_ascii=False)}],
                    "item_id": item_id, "from_model": model, "memory_note": m.get("note", "")})
    Path(args.out).write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out), "utf-8")
    print(f"{len(out)} conversations -> {args.out}; {missing} examples without their sentence or answer at hand")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """LoRA on the local GPUs (transformers, peft, trl); never on GitHub's runners. The settings are in
    data/reading/training.yml, recorded with the adapter."""
    import yaml

    try:
        import torch  # noqa: F401
        from datasets import load_dataset
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        raise SystemExit(f"training needs torch, datasets, peft and trl on a machine with GPUs ({error})") from error
    cfg = yaml.safe_load(CONFIG.read_text("utf-8"))
    data = load_dataset("json", data_files=args.data, split="train")
    lora = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"],
                      target_modules=cfg["lora"]["target_modules"], task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=args.base, train_dataset=data, peft_config=lora,
                         args=SFTConfig(output_dir=args.out, num_train_epochs=cfg["epochs"],
                                        learning_rate=cfg["learning_rate"], seed=cfg["seed"],
                                        per_device_train_batch_size=cfg["batch_size"],
                                        gradient_accumulation_steps=cfg["gradient_accumulation"]))
    trainer.train()
    trainer.save_model(args.out)
    manifest = {"base": args.base, "data": args.data, "examples": len(data), "config": cfg,
                "manual_version": rm.manual_version(), "date": dt.date.today().isoformat()}
    Path(args.out, "melampo_manifest.json").write_text(json.dumps(manifest, indent=1), "utf-8")
    print(f"adapter -> {args.out}")
    return 0


def _errors(crit: dict) -> dict[str, int]:
    counts = crit.get("counts") or {}
    return {"1_aside_attached_as_certain": crit.get("1_aside_attached_as_certain", 0),
            "2_findings_set_aside": crit.get("2_findings_set_aside", 0),
            "3_to_a_person": counts.get("3_to_a_person", 0),
            "4_attached_otherwise": counts.get("4_attached_otherwise", 0),
            "6_doubt_without": counts.get("6_doubt_without_the_right_reading", 0),
            "6_closed_doubts_wrong": counts.get("6_closed_doubts_wrong", 0),
            "7_uncertainty_lost": crit.get("7_uncertainty_lost", 0),
            "8_uncertainty_given": crit.get("8_uncertainty_given", 0)}


def _passes(crit: dict) -> bool:
    shares = [crit.get(k, 0) for k in ("3_to_a_person_share", "4_certain_attached_otherwise_share",
                                       "6_doubt_without_the_right_reading_share", "7_uncertainty_lost_share",
                                       "8_uncertainty_given_share")]
    n = crit.get("items", 0) or 1
    return crit.get("1_aside_attached_as_certain", 0) / n <= 0.01 and crit.get("2_findings_set_aside", 0) == 0 \
        and all(s <= 0.02 for s in shares)


def cmd_decide(args: argparse.Namespace) -> int:
    import yaml

    def crit(path: str) -> dict:
        report = json.loads(Path(path).read_text("utf-8"))
        return report["by_set"][args.set]["criteria"]["without_doubt"]

    base, adapter = crit(args.base), crit(args.adapter)
    eb, ea = _errors(base), _errors(adapter)
    if not _passes(adapter):
        enters, why = False, "a criterion does not pass with the adapter"
    elif sum(ea.values()) >= sum(eb.values()):
        enters, why = False, f"not fewer errors than without it ({sum(ea.values())} against {sum(eb.values())})"
    else:
        enters, why = True, f"every criterion passes and fewer errors ({sum(ea.values())} against {sum(eb.values())})"
    log = _log() + [{"name": args.name, "date": dt.date.today().isoformat(), "sample": args.set,
                     "enters": enters, "why": why, "errors_without": eb, "errors_with": ea,
                     "examples": len(_memory()), "manual_version": rm.manual_version()}]
    LOG.write_text(yaml.safe_dump({"trainings": log}, sort_keys=False, allow_unicode=True), "utf-8")
    print(f"{args.name}: {'enters' if enters else 'stays out'}: {why}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    e = sub.add_parser("export")
    e.add_argument("--run", action="append", required=True)
    e.add_argument("--items", default="owner_items_*.jsonl")
    e.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--base", required=True)
    t.add_argument("--out", required=True)
    d = sub.add_parser("decide")
    d.add_argument("--base", required=True)
    d.add_argument("--adapter", required=True)
    d.add_argument("--set", required=True)
    d.add_argument("--name", required=True)
    args = ap.parse_args(argv)
    return {"status": cmd_status, "export": cmd_export, "train": cmd_train, "decide": cmd_decide}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
