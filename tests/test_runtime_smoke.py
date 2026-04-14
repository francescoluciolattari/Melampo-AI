from melampo.app import build_default_runtime


def test_build_default_runtime():
    runtime = build_default_runtime()
    assert runtime.config.project_name == "melampo"
    assert runtime.pipeline is not None
    assert runtime.validator is not None
