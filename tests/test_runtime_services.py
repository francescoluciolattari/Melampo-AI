from melampo.config import build_default_config
from melampo.orchestration.runtime_services import RuntimeServices
from melampo.utils.logging_utils import build_logger


def test_runtime_services_resolve_known_task():
    services = RuntimeServices.build(config=build_default_config(), logger=build_logger("test"))
    result = services.resolve("volume_encoder")
    assert result["service"]["provider"] == "api_for_service_volume_encoder"
