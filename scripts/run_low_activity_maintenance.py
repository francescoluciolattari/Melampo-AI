#!/usr/bin/env python3
"""Run NexusScheduler.run_once(), sweep_expired_pending_cases(), and training_extraction.extract_and_purge() -- the periodic trigger none of them has today.

All three exist and work, verified directly against the real
pipeline, but nothing calls any of them on a schedule -- this script is that
call. It is meant to be invoked periodically (a cron entry, a Kubernetes
CronJob, any scheduler a real deployment already runs) during a window
when the live service is genuinely quiet, matching what
LowActivityPolicy checks for.

**Where this must run, stated directly rather than left to be discovered
the hard way.** Unlike this project's literature-refresh workflow
(.github/workflows/data-and-dependency-updates.yml), which commits its
output back to git because a GitHub Actions runner's filesystem is
thrown away after every job, this script is NOT a GitHub Actions
candidate. NexusCandidateStore's persisted file
(data/nexus_candidates.jsonl, or wherever DB_PASSWORD-backed storage
points) needs to be the SAME file the live, request-serving process
writes to -- a throwaway CI runner starting from a fresh git checkout
would see an empty or stale store, and any expiry/processing it did
would vanish the moment the job ends rather than reaching the live
service at all. Literature is versioned reference data, refreshed
occasionally, for which "commit the result to git" is a reasonable
persistence strategy; this store is live application state, updated by
every case the live service processes, for which it is not. Run this
on the same host or container as the live service, or wherever mounts
the same persistent volume it writes to -- not in CI.

Usage:
    DB_PASSWORD=... python scripts/run_low_activity_maintenance.py
    DB_PASSWORD=... python scripts/run_low_activity_maintenance.py --idle-seconds 600
    DB_PASSWORD=... python scripts/run_low_activity_maintenance.py --active-requests 0

Reads DB_PASSWORD from the environment, never as a command-line
argument -- the same posture every secret-reading script in this project
already takes. Without it, NexusCandidateStore falls back to pure
in-memory storage (see clinical_pipeline.py's _build_nexus_candidate_store),
which makes running this script pointless -- a fresh, empty store has
nothing to process or expire -- so this script exits early with a clear
message rather than silently doing nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from melampo.governance.confirmation_registry import (
    DEFAULT_CONFIRMATION_REGISTRY_PATH,
    ConfirmationRegistry,
)
from melampo.reasoning.clinical_pipeline import (
    DEFAULT_NEXUS_CANDIDATE_STORE_PATH,
    DEFAULT_NEXUS_QUEUE_PATH,
    _build_nexus_candidate_store,
)
from melampo.training.confirmed_case_store import (
    DEFAULT_CONFIRMED_CASE_STORE_PATH,
    ConfirmedCaseStore,
)
from melampo.training.nexus_scheduler import NexusScheduler
from melampo.training.pending_case_router import (
    DEFAULT_RETENTION_SECONDS,
    sweep_expired_pending_cases,
)
from melampo.training.training_extraction import extract_and_purge


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--idle-seconds", type=float, default=600.0, help="reported idle duration (default: 600)")
    parser.add_argument("--active-requests", type=int, default=0, help="reported active request count (default: 0)")
    parser.add_argument(
        "--retention-seconds", type=float, default=DEFAULT_RETENTION_SECONDS,
        help="expire pending cases older than this with no confirmation (default: one year)",
    )
    args = parser.parse_args()

    password = os.environ.get("DB_PASSWORD")
    if not password:
        print(
            "DB_PASSWORD is not set -- NexusCandidateStore would be pure in-memory here, "
            "so there is nothing this script's own process could have written for a live "
            "service to have populated. Set DB_PASSWORD to the same value the live service "
            "uses, and run this where it can see the same store file "
            f"({DEFAULT_NEXUS_CANDIDATE_STORE_PATH}).",
            file=sys.stderr,
        )
        return 1

    candidate_store = _build_nexus_candidate_store()
    scheduler = NexusScheduler(candidate_store=candidate_store, password=password, path=Path(DEFAULT_NEXUS_QUEUE_PATH))
    confirmed_store = ConfirmedCaseStore(password=password, path=Path(DEFAULT_CONFIRMED_CASE_STORE_PATH))
    registry = ConfirmationRegistry(password=password, path=Path(DEFAULT_CONFIRMATION_REGISTRY_PATH))

    activity = {"active_requests": args.active_requests, "idle_seconds": args.idle_seconds}
    run_result = scheduler.run_once(activity=activity)
    expired_ids = sweep_expired_pending_cases(candidate_store, retention_seconds=args.retention_seconds)
    # extract_and_purge only deletes a confirmed case once DPO extraction
    # has genuinely produced a pair from it -- see training_extraction.py's
    # own docstring for why a case with nothing usable stays retained
    # rather than being purged as if training had consumed it.
    extraction_result = extract_and_purge(confirmed_store, registry)

    report = {
        "run_once": run_result,
        "expired_candidate_ids": expired_ids,
        "expired_count": len(expired_ids),
        "training_extraction": extraction_result.as_dict(),
    }
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
