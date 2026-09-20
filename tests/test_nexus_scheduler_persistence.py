"""Tests for NexusScheduler's queue persistence -- the gap found running
scripts/run_low_activity_maintenance.py for real: making
NexusCandidateStore persistent was not enough on its own, since a job
sits in NexusScheduler's own queue *before* candidate_store ever gains a
record for it. A genuinely separate periodic-trigger process needs to see
that queue too, not just the candidate store.
"""

from melampo.training.nexus_scheduler import NexusScheduler


def _area_dynamics():
    return {"neuro_dynamic_metrics": {"pi_score": 0.6, "convergence_index": 0.5}}


def test_a_scheduler_with_no_password_or_path_is_pure_in_memory_as_before():
    scheduler = NexusScheduler()
    job = scheduler.enqueue(case_context={"case_id": "case-1"}, area_dynamics=_area_dynamics())
    assert job in scheduler.queue


def test_an_enqueued_job_survives_a_new_scheduler_instance(tmp_path):
    path = tmp_path / "queue.jsonl"
    first = NexusScheduler(password="secret", path=path)
    job = first.enqueue(case_context={"case_id": "case-1"}, area_dynamics=_area_dynamics())

    second = NexusScheduler(password="secret", path=path)

    assert len(second.queue) == 1
    assert second.queue[0].job_id == job.job_id
    assert second.queue[0].status == "queued"


def test_a_genuinely_separate_process_can_process_a_queue_it_never_enqueued_to(tmp_path):
    """The property that mattered in practice: process A enqueues, process
    B (a fresh NexusScheduler instance, same password/path) processes."""
    path = tmp_path / "queue.jsonl"
    producer = NexusScheduler(password="secret", path=path)
    producer.enqueue(case_context={"case_id": "case-1"}, area_dynamics=_area_dynamics())

    consumer = NexusScheduler(password="secret", path=path)
    result = consumer.run_once(activity={"active_requests": 0, "idle_seconds": 600})

    assert result["status"] == "completed"
    assert result["processed_jobs"] == 1


def test_a_processed_job_status_is_persisted_as_completed(tmp_path):
    path = tmp_path / "queue.jsonl"
    scheduler = NexusScheduler(password="secret", path=path)
    scheduler.enqueue(case_context={"case_id": "case-1"}, area_dynamics=_area_dynamics())
    scheduler.run_once(activity={"active_requests": 0, "idle_seconds": 600})

    reloaded = NexusScheduler(password="secret", path=path)

    assert reloaded.queue[0].status == "completed"


def test_only_the_latest_event_per_job_wins_on_reload(tmp_path):
    path = tmp_path / "queue.jsonl"
    scheduler = NexusScheduler(password="secret", path=path)
    scheduler.enqueue(case_context={"case_id": "case-1"}, area_dynamics=_area_dynamics())
    scheduler.run_once(activity={"active_requests": 0, "idle_seconds": 600})

    reloaded = NexusScheduler(password="secret", path=path)

    assert len(reloaded.queue) == 1


def test_multiple_enqueued_jobs_all_survive_reload(tmp_path):
    path = tmp_path / "queue.jsonl"
    first = NexusScheduler(password="secret", path=path)
    first.enqueue(case_context={"case_id": "case-1"}, area_dynamics=_area_dynamics())
    first.enqueue(case_context={"case_id": "case-2"}, area_dynamics=_area_dynamics())

    second = NexusScheduler(password="secret", path=path)

    assert len(second.queue) == 2
