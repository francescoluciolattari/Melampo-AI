from pathlib import Path

from melampo.datasets.chestxray14_loader import ChestXray14CsvLoader
from melampo.prototype import run_prototype_case


def test_chestxray14_loader_converts_csv_rows_to_prototype_payloads():
    loader = ChestXray14CsvLoader()
    payloads = loader.load_csv(Path("examples/chestxray14_metadata_sample.csv"), limit=1)
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["case_id"] == "cxr14-synthetic_000001"
    assert payload["imaging"][0]["metadata"]["finding_labels"] == "Nodule|Infiltration"
    assert payload["provenance"]["contains_real_patient_data"] is False

    result = run_prototype_case(payload, runtime_profile="local_research")
    assert result["status"] == "completed"
    assert result["case_id"] == "cxr14-synthetic_000001"
    assert result["recommended_actions"]
