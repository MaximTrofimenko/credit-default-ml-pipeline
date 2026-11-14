import pytest

from src.models.train import calculate_metrics


def test_perfect_predictions():
    metrics = calculate_metrics([0, 1], [0, 1], [0.1, 0.9])
    assert metrics["roc_auc"] == pytest.approx(1)
    assert metrics["precision"] == pytest.approx(1)
    assert metrics["recall"] == pytest.approx(1)
    assert metrics["f1"] == pytest.approx(1)
