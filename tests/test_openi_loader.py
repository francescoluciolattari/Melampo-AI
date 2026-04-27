from pathlib import Path

from melampo.datasets.openi_loader import OpenIReportCsvLoader
from melampo.prototype import run_prototype_case


def test_openi_loader_converts_report_metadata_to_prototype_payloads():
    loader = OpenIReportCsvLoader()
    payloads = loader.load_csv(Path("examples/openi_metadata_sample.csv"), limit=1)
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["case_id"] == "openi-synthetic-openi-001"
    assert "Possible pneumonia" in payload["report_text"]
    assert payload["imaging"][0]["metadata"]["problem_terms"] == "pneumonia|opacity"
    assert payload["provenance"]["contains_real_patient_data"] is False

    result = run_prototype_case(payload, runtime_profile="local_research")
    assert result["status"] == "completed"
    assert result["case_id"] == "openi-synthetic-openi-001"
    assert result["recommended_actions"]
