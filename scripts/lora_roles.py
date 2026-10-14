#!/usr/bin/env python
"""Point 2 (12 October 2026): teach an open model the role of an anatomical word with LoRA, and measure it
on reports it has not seen.

Three steps, one script:

* ``prepare``  the items of the gold set with a role -> the role-only question of
               ``melampo.evaluation.llm_reading`` (optionally with what the Kintsch reader built) and the
               answer ``{"role": ...}``; five folds **grouped by report** (an item never trains the model
               that is tested on another item of the same report). No GPU; runs anywhere.
* ``train``    QLoRA (4-bit base, LoRA adapters r=16 on the attention and MLP projections) for one fold:
               trains on the four other folds. Needs a GPU (default base ``openai/gpt-oss-20b``,
               Apache 2.0, same family as the stage-6 model; another base with ``--base``).
* ``evaluate`` greedy answers of the base model **with** the fold's adapter and **without** it (the same
               question zero-shot) on the held-out fold; the predictions in the format of
               ``scripts/role_probe.py`` and the same scores (critical errors, uncertain, bootstrap by report).

What it can show today, and what not. The gold set is 663 items with a role from one labeller (Claude),
70% "structure": a model trained on it learns *these* labels, with their conventions. The test answers
"does the role become learnable from a few hundred examples, and does the reader's evidence help?";
it does not certify anything. A certifying run needs the radiologists' gold set (and RadGraph-XL, if
the PhysioNet credentials arrive, trains roles on 2,300 reports: on the credentialed person's machine).

    python scripts/lora_roles.py prepare --gold silver/gold.jsonl --reports silver/reports.jsonl \
        [--reader-dir ci_resources] --out lora_data
    python scripts/lora_roles.py train --data lora_data --fold 0 --out adapters/fold0
    python scripts/lora_roles.py evaluate --data lora_data --fold 0 --adapter adapters/fold0 --out eval_fold0.json

One procedure for every model (13 October 2026; ``melampo.evaluation.model_profiles``): ``export`` writes
the fold as chat JSONL for any trainer (NeMo AutoModel recipes for Gemma 4 31B and Nemotron 3 Super /
Ultra, TRL here), ``plan --profile <name>`` prints the steps for a model, and ``evaluate --endpoint``
scores any model served behind an OpenAI-compatible endpoint, base and base + adapter, the same way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import role_probe as rp  # noqa: E402

from melampo.evaluation import llm_reading as lr  # noqa: E402
from melampo.evaluation import model_profiles as mp  # noqa: E402

FOLDS = 5
DEFAULT_BASE = "openai/gpt-oss-20b"


def fold_of(report_id: str, folds: int = FOLDS) -> int:
    """The fold of a report: a fixed hash, so the split does not depend on the order of the items."""
    return int(hashlib.sha256(report_id.encode()).hexdigest(), 16) % folds


def prepare(args) -> int:
    gold = [g for path in args.gold for g in rp._jsonl(path)]
    reports = {r["report_id"]: r["text"] for path in (args.reports or []) for r in rp._jsonl(path)}
    items = rp.items_of(gold, reports)
    lines_of: dict[str, list[str]] = {}
    if args.reader_dir is not None:
        lattice, reader = rp.build_reading(args.reader_dir)
        docs: dict = {}
        for it in items:
            it["_report"] = reports.get(it["report_id"], "")
            _, lines_of[it["item_id"]] = rp.evidence(it, lattice, reader, docs)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for it in items:
        e = it["at"] + len(it["mention"])
        question = lr.role_prompt(it["sentence"], it["at"], e, reader=lines_of.get(it["item_id"]),
                                  language=rp.LANG.get(it["language"], ""))
        rows.append({
            "item_id": it["item_id"], "report_id": it["report_id"], "stratum": it["stratum"],
            "language": it["language"], "mention": it["mention"], "sentence": it["sentence"],
            "structure": it["structure"], "role": it["role"], "fold": fold_of(it["report_id"]),
            "prompt": [{"role": "user", "content": question}],
            "completion": [{"role": "assistant", "content": json.dumps({"role": it["role"]})}],
        })
    (args.out / "items.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", "utf-8")
    summary = {
        "items": len(rows), "with_reader": args.reader_dir is not None,
        "folds": {f: dict(Counter(r["role"] for r in rows if r["fold"] == f)) for f in range(FOLDS)},
        "reports_per_fold": {f: len({r["report_id"] for r in rows if r["fold"] == f}) for f in range(FOLDS)},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), "utf-8")
    print(json.dumps(summary, indent=1))
    return 0


def _split(data: Path, fold: int) -> tuple[list[dict], list[dict]]:
    rows = rp._jsonl(data / "items.jsonl")
    return [r for r in rows if r["fold"] != fold], [r for r in rows if r["fold"] == fold]


def _need_gpu():
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("train/evaluate need a GPU (CUDA); 'prepare' runs anywhere")
    return torch


def export(args) -> int:
    """The fold as chat-format JSONL (one ``messages`` list per item: the question and the answer), the
    format NeMo AutoModel, TRL and most trainers read: the same files for every model."""
    train_rows, test_rows = _split(args.data, args.fold)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train_rows), ("test", test_rows)):
        lines = [json.dumps({"item_id": r["item_id"], "messages": r["prompt"] + r["completion"]}, ensure_ascii=False)
                 for r in rows]
        (args.out / f"{name}.jsonl").write_text("\n".join(lines) + "\n", "utf-8")
    print(f"fold {args.fold}: {len(train_rows)} train, {len(test_rows)} test -> {args.out}")
    return 0


def plan(args) -> int:
    """The steps of the one procedure for a model profile (nothing is run)."""
    p = mp.profile(args.profile)
    fold = f"lora_data/fold{args.fold}"
    steps = [
        f"# {p.name}: {p.hf_id} ({p.licence}; {p.params})",
        f"# to serve: {p.run_hardware}",
        f"# to train: {p.train_hardware}",
        "python scripts/lora_roles.py prepare --gold silver/gold.jsonl --reports silver/reports.jsonl --out lora_data",
        f"python scripts/lora_roles.py export --data lora_data --fold {args.fold} --out {fold}",
    ]
    if p.train_backend == "trl":
        steps.append(f"python scripts/lora_roles.py train --data lora_data --fold {args.fold} --profile {p.name} "
                     f"--out adapters/{p.name}-fold{args.fold}")
    else:
        steps += [
            f"# NeMo AutoModel, recipe: {p.train_recipe}",
            f"#   its dataset -> {fold}/train.jsonl (chat 'messages'), validation -> {fold}/test.jsonl;",
            "#   LoRA rank 16, alpha 32, 2 epochs, lr 2e-4 as in the TRL path; checkpoint -> "
            f"adapters/{p.name}-fold{args.fold}",
            f"#   sources: {', '.join(p.sources)}",
        ]
    steps += [
        "# serve the base and the base + adapter behind an OpenAI-compatible endpoint (e.g. vLLM --enable-lora), then:",
        f"python scripts/lora_roles.py evaluate --data lora_data --fold {args.fold} --endpoint http://localhost:8000/v1/chat/completions "
        f"--model {p.hf_id} --adapter-model {p.name}-fold{args.fold} --out eval_{p.name}_fold{args.fold}.json",
    ]
    print("\n".join(steps))
    return 0


def train(args) -> int:
    if args.profile:
        args.base = mp.profile(args.profile).hf_id
    torch = _need_gpu()
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    train_rows, _ = _split(args.data, args.fold)
    tokenizer = AutoTokenizer.from_pretrained(args.base, revision=args.revision)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.base, revision=args.revision, quantization_config=quant,
                                                 torch_dtype=torch.bfloat16, device_map="auto")
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    data = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in train_rows])
    config = SFTConfig(output_dir=str(args.out), num_train_epochs=args.epochs, learning_rate=args.lr,
                       per_device_train_batch_size=1, gradient_accumulation_steps=8, bf16=True,
                       logging_steps=10, save_strategy="no", seed=20261012, report_to=[],
                       completion_only_loss=True, max_length=4096)
    trainer = SFTTrainer(model=model, args=config, train_dataset=data, processing_class=tokenizer, peft_config=lora)
    trainer.train()
    trainer.model.save_pretrained(str(args.out))
    (Path(args.out) / "melampo_training.json").write_text(json.dumps({
        "base": args.base, "revision": args.revision, "fold": args.fold, "items": len(train_rows),
        "epochs": args.epochs, "lr": args.lr, "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
    }, indent=1), "utf-8")
    return 0


def _score(args, items, arms, base) -> int:
    report = {"fold": args.fold, "base": base, "adapter": str(getattr(args, "adapter", "") or getattr(args, "adapter_model", "") or ""),
              "arms": {a: rp.score([i for i in items if i["structure"] != "NOT_ANATOMY"], a) for a in arms},
              "not_anatomy": {a: rp.score([i for i in items if i["structure"] == "NOT_ANATOMY"], a) for a in arms},
              "items": [{k: i[k] for k in ("item_id", "report_id", "role", "answers")} for i in items]}
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), "utf-8")
    print(json.dumps(report["arms"], indent=1))
    return 0


def evaluate_endpoint(args) -> int:
    """Any model behind an OpenAI-compatible endpoint: the base (``--model``) and, if given, the base with
    the adapter (``--adapter-model``, the name the server gives the LoRA). Same question, same scores."""
    from melampo.evaluation import linking_bench as lb

    rp.check_endpoint(args.endpoint, args.public_data)
    _, test_rows = _split(args.data, args.fold)
    arms = {"base (zero-shot)": args.model}
    if args.adapter_model:
        arms["base + LoRA"] = args.adapter_model
    key = os.environ.get("OPENAI_API_KEY", "local")
    items = [{**r, "answers": {}} for r in test_rows]
    for arm, model in arms.items():
        chat = lb.OpenRouterChat(model, key, endpoint=args.endpoint, max_tokens=args.max_new_tokens, timeout=180.0)
        for it in items:
            text = chat(it["prompt"][0]["content"])
            it["answers"][arm] = lr.read_role(text)
    return _score(args, items, arms, args.model)


def evaluate(args) -> int:
    if args.endpoint:
        return evaluate_endpoint(args)
    if args.profile:
        args.base = mp.profile(args.profile).hf_id
    torch = _need_gpu()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _, test_rows = _split(args.data, args.fold)
    tokenizer = AutoTokenizer.from_pretrained(args.base, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.base, revision=args.revision, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    arms = {"base (zero-shot)": None}
    if args.adapter:
        arms["base + LoRA"] = args.adapter
    items = [{**r, "answers": {}} for r in test_rows]
    for arm, adapter in arms.items():
        use = model
        if adapter:
            from peft import PeftModel

            use = PeftModel.from_pretrained(model, adapter)
        use.eval()
        for it in items:
            ids = tokenizer.apply_chat_template(it["prompt"], add_generation_prompt=True, return_tensors="pt").to(use.device)
            with torch.no_grad():
                out = use.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
            text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            it["answers"][arm] = lr.read_role(text)
            it.setdefault("raw", {})[arm] = text[-400:]
    return _score(args, items, arms, args.base)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--gold", type=Path, nargs="+", required=True)
    p.add_argument("--reports", type=Path, nargs="+")
    p.add_argument("--reader-dir", type=Path)
    p.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("export")
    q.add_argument("--data", type=Path, required=True)
    q.add_argument("--fold", type=int, required=True, choices=range(FOLDS))
    q.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("plan")
    q.add_argument("--profile", default=mp.DEFAULT_PROFILE, choices=sorted(mp.PROFILES))
    q.add_argument("--fold", type=int, default=0, choices=range(FOLDS))
    for name in ("train", "evaluate"):
        q = sub.add_parser(name)
        q.add_argument("--data", type=Path, required=True)
        q.add_argument("--fold", type=int, required=True, choices=range(FOLDS))
        q.add_argument("--profile", choices=sorted(mp.PROFILES), help="the base model of a profile (model_profiles.py)")
        q.add_argument("--base", default=DEFAULT_BASE)
        q.add_argument("--revision", default=None, help="pin the base model's commit")
        q.add_argument("--out", type=Path, required=True)
        if name == "train":
            q.add_argument("--epochs", type=float, default=2.0)
            q.add_argument("--lr", type=float, default=2e-4)
        else:
            q.add_argument("--adapter", type=Path)
            q.add_argument("--max-new-tokens", type=int, default=512)
            q.add_argument("--endpoint", help="an OpenAI-compatible server instead of loading the weights here")
            q.add_argument("--model", help="with --endpoint: the base model's name on the server")
            q.add_argument("--adapter-model", help="with --endpoint: the adapter's name on the server")
            q.add_argument("--public-data", action="store_true", help="the texts are public: a remote endpoint is allowed")
    args = ap.parse_args(argv)
    return {"prepare": prepare, "export": export, "plan": plan, "train": train, "evaluate": evaluate}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
