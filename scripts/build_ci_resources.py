#!/usr/bin/env python
"""Write the resources the construction-integration reader needs to run without the corpora.

  build_ci_resources.py --medmentions <dir> --craft <dir> --ncit ncit.obo --out ci_resources

Reads the corpora once, builds the semantic space (frozen test documents are left out), the prototypes
of the readings (NCIt definitions) and the trace memory (MedMentions training documents), and saves
``space.npz``, ``protos.npz`` and ``traces.json.gz`` in ``--out``. Load them with
``ci_reader.load_reader(out, lattice)`` and give the reader to ``AnatomyLinker(ci_reader=...)``: trace
only, it decides nothing.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import ci_probe as cp  # noqa: E402

from melampo.memory import chunk_lattice as cl  # noqa: E402
from melampo.memory import ci_reader as ci  # noqa: E402
from melampo.memory.longer_names import LongerNames  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--craft")
    ap.add_argument("--medmentions", required=True)
    ap.add_argument("--ncit", required=True)
    ap.add_argument("--space", help="semantic space file to reuse or write")
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--memory", type=Path, default=cl.DEFAULT_PATH)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    start = time.monotonic()
    frozen = cp.load_frozen()
    mm_docs, craft_docs, splits, space, protos = cp.build_resources(args, start, frozen)
    lattice = cl.ChunkLattice(cl.BlockMemory.load(args.memory, LongerNames.load().names))
    keys = {**cp.doc_keys(mm_docs, "medmentions", lattice), **cp.doc_keys(craft_docs, "craft", lattice)}
    traces = cp.write_traces(mm_docs, splits, space, lattice, keys, frozen)
    ci.save_resources(args.out, space, protos, traces)
    cp.say(start, f"saved {args.out}: {len(space.words)} words, {traces.size} annotations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
