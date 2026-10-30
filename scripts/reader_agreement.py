#!/usr/bin/env python
"""How much two radiologists differ on the same nodule, and how many measures a tolerance sends to a person
(LIDC-IDRI, 30 October 2026).

Reads the output of ``scripts/lidc_probe.py`` (each nodule with its four readers' longest axial diameters) and
writes, by size band (the readers' mean: <10, 10-20, >=20 mm):

* the **limits of agreement between two readers** (Bland-Altman, 95%): 1.96 x sqrt(2) x the within-nodule standard
  deviation, with a bootstrap 95% interval over nodules (fixed seed), and the 90th/95th percentile of the absolute
  differences between pairs of readers;
* **who goes to a person** under each tolerance of ``melampo.imaging.size_tolerance``: a reader against the
  mean of the other three, and our readings (VoxTell's region; the rule frozen on 29 October: solid f=0.1,
  part-solid f=0.3, ground glass VoxTell), the report being one reader's diameter rounded to the millimetre.

The limits written here are measured, never tuned on our errors. Run on the set "misura" they give the table of
``size_tolerance.READER_LIMITS_MM``; run on "verifica" they say whether that table holds on nodules never seen.

    python scripts/reader_agreement.py lidc.json [--out reader_agreement.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from melampo.imaging.size_tolerance import POLICIES, tolerance_mm  # noqa: E402

BANDS = ("<10", "10-20", ">=20")
FROZEN_LEVEL = {"solido": "0.1", "parzialmente_solido": "0.3"}   # ground glass: VoxTell's region


def band(mm: float) -> str:
    return BANDS[0] if mm < 10 else (BANDS[1] if mm < 20 else BANDS[2])


def _long(reading: dict | None) -> float | None:
    return (reading or {}).get("axial_long_mm")


def frozen_rule(prompt: dict, texture: str) -> float | None:
    """The diameter of the reading fixed on 29 October for this kind of nodule."""
    level = FROZEN_LEVEL.get(texture)
    return _long((prompt.get("sweep") or {}).get(level)) if level else _long(prompt.get("segmenter"))


def nodules(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if "error" in r:
            continue
        diam = [x["axial_long_mm"] for x in (r.get("readers") or {}).get("readers", []) if x.get("axial_long_mm")]
        if len(diam) < 2:
            continue
        prompt = r.get("prompt") or {}
        found = (prompt.get("found") or {}).get("found")
        texture = (r.get("stratum") or "").split(" ")[0]
        out.append({"id": r["id"], "readers": diam, "band": band(st.mean(diam)), "texture": texture,
                    "segmenter": _long(prompt.get("segmenter")) if found else None,
                    "frozen_rule": frozen_rule(prompt, texture) if found else None})
    return out


def limits_of_agreement(group: list[dict]) -> float:
    sw = math.sqrt(st.mean(st.variance(n["readers"]) for n in group))
    return 1.96 * math.sqrt(2) * sw


def agreement(group: list[dict], boot: int = 2000, seed: int = 7) -> dict:
    pairs = sorted(abs(a - b) for n in group for a, b in itertools.combinations(n["readers"], 2))
    rng = random.Random(seed)
    sample = sorted(limits_of_agreement([rng.choice(group) for _ in group]) for _ in range(boot))
    return {"nodules": len(group), "pairs": len(pairs),
            "limits_of_agreement_mm": round(limits_of_agreement(group), 2),
            "limits_bootstrap_95_mm": [round(sample[int(0.025 * boot)], 2), round(sample[int(0.975 * boot) - 1], 2)],
            "pair_abs_median_mm": round(st.median(pairs), 2),
            "pair_abs_p90_mm": round(pairs[int(0.9 * len(pairs))], 2),
            "pair_abs_p95_mm": round(pairs[min(len(pairs) - 1, int(0.95 * len(pairs)))], 2)}


def to_a_person(group: list[dict], measure, policy: str) -> dict:
    """Each reader in turn writes the report (rounded to the millimetre); how often the measure falls outside."""
    sent = total = 0
    for n in group:
        for i, ref in enumerate(n["readers"]):
            value = measure(n, i)
            if value is None:
                continue
            report = round(ref)
            total += 1
            sent += abs(value - report) > tolerance_mm(report, policy)
    return {"to_a_person": sent, "of": total, "share": round(sent / total, 3) if total else None}


MEASURES = {
    "another_reader": lambda n, i: st.mean(v for j, v in enumerate(n["readers"]) if j != i),
    "segmenter": lambda n, i: n["segmenter"],
    "frozen_rule": lambda n, i: n["frozen_rule"],
}


def report(rows: list[dict]) -> dict:
    ns = nodules(rows)
    out = {"nodules": len(ns), "agreement": {"all": agreement(ns)}, "to_a_person": {}}
    for b in BANDS:
        group = [n for n in ns if n["band"] == b]
        if len(group) >= 2:
            out["agreement"][b] = agreement(group)
    for name, measure in MEASURES.items():
        out["to_a_person"][name] = {p: {"all": to_a_person(ns, measure, p)} |
                                    {b: to_a_person([n for n in ns if n["band"] == b], measure, p) for b in BANDS}
                                    for p in POLICIES}
    out["not_found"] = {b: sum(1 for n in ns if n["band"] == b and n["segmenter"] is None) for b in BANDS}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("probe", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    res = report(json.loads(args.probe.read_text()))
    print(f"{res['nodules']} noduli\n\nDue radiologi sullo stesso nodulo (diametro assiale maggiore, mm)")
    for k, a in res["agreement"].items():
        print(f"  {k:6} n={a['nodules']:2}  limiti di accordo 95% {a['limits_of_agreement_mm']:5.1f} "
              f"[{a['limits_bootstrap_95_mm'][0]:.1f}-{a['limits_bootstrap_95_mm'][1]:.1f}]  "
              f"coppie: mediana {a['pair_abs_median_mm']:.1f}  p90 {a['pair_abs_p90_mm']:.1f}  "
              f"p95 {a['pair_abs_p95_mm']:.1f}")
    print("\nA una persona (il referto è un radiologo, arrotondato al mm)")
    for name, by_policy in res["to_a_person"].items():
        for p, by_band in by_policy.items():
            cells = "  ".join(f"{b} {v['to_a_person']}/{v['of']}" for b, v in by_band.items())
            print(f"  {name:15} {p:8} {cells}")
    print("\nnon trovati:", res["not_found"])
    if args.out:
        args.out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
