from melampo.orchestration.contracts import ServiceContract


def test_service_contract_describe():
    contract = ServiceContract(name="rag", provider="api_for_service_document_rag", protocol="service")
    payload = contract.describe()
    assert payload["name"] == "rag"
    assert payload["protocol"] == "service"
