import json
from pathlib import Path

from melampo.datasets.open_catalog import datasets_by_modality, list_open_clinical_datasets
from melampo.prototype import run_prototype_case


def test_open_clinical_dataset_catalog_lists_known_sources():
    datasets = list_open_clinical_datasets()
    names = {item["name"] for item in datasets}
    assert "NIH ChestX-ray14" in names
    assert "Open-i / Indiana University Chest X-rays" in names
    assert "MIMIC-IV" in names
    assert datasets_by_modality("chest_xray_labels")


def test_synthetic_open_clinical_case_runs_through_prototype():
    example_path = Path("examples/open_clinical_case_chest_xray_synthetic.json")
    payload = json.loads(example_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["contains_real_patient_data"] is False

    result = run_prototype_case(payload, runtime_profile="local_research")
    assert result["status"] == "completed"
    assert result["case_id"] == "open-synthetic-cxr-001"
    assert result["top_hypothesis"]
    assert result["recommended_actions"]
    assert result["runtime"]["config"]["runtime_profile"] == "local_research"
