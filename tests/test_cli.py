import json

from melampo.cli import main


def test_cli_runs_prototype_case_without_raw_output(tmp_path, capsys):
    payload_path = tmp_path / "case.json"
    payload_path.write_text(
        json.dumps(
            {
                "case_id": "case-cli-001",
                "report_text": "possible pulmonary lesion",
                "patient_complaints": "persistent cough",
            }
        ),
        encoding="utf-8",
    )

    exit_code = main([str(payload_path), "--runtime-profile", "local_research"])
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert exit_code == 0
    assert output["status"] == "completed"
    assert output["case_id"] == "case-cli-001"
    assert "raw_result" not in output
    assert output["runtime"]["config"]["runtime_profile"] == "local_research"


def test_cli_returns_nonzero_for_invalid_payload(tmp_path, capsys):
    payload_path = tmp_path / "invalid.json"
    payload_path.write_text(json.dumps({"report_text": "missing id"}), encoding="utf-8")

    exit_code = main([str(payload_path)])
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert exit_code == 2
    assert output["status"] == "invalid_payload"
    assert output["validation"]["valid"] is False
