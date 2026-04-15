from melampo.evaluation.metrics import MetricsCatalog


def test_metrics_catalog_lists_core_metrics():
    metrics = MetricsCatalog().list_metrics()
    assert "ece" in metrics
    assert "diagnostic_accuracy" in metrics
