"""Pruebas de las métricas de evaluación."""

import numpy as np

from src.evaluation.evaluate import compute_metrics, per_class_report, top_k_accuracy


def test_metricas_con_prediccion_perfecta():
    metrics = compute_metrics([0, 1, 2, 2], [0, 1, 2, 2])
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert all(isinstance(value, float) for value in metrics.values())


def test_f1_macro_penaliza_especies_minoritarias():
    y_true = [0] * 9 + [1]
    y_pred = [0] * 10  # nunca predice la especie minoritaria
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.9
    assert metrics["f1_macro"] < 0.5


def test_exactitud_top_k():
    probabilities = np.array([[0.6, 0.3, 0.1], [0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    y_true = [1, 2, 2]
    assert top_k_accuracy(y_true, probabilities, k=1) == 1 / 3
    assert top_k_accuracy(y_true, probabilities, k=2) == 2 / 3
    assert top_k_accuracy(y_true, probabilities, k=3) == 1.0


def test_reporte_por_especie():
    report = per_class_report([0, 1, 1], [0, 1, 0], ["Especie_a", "Especie_b"])
    assert {"Especie_a", "Especie_b"}.issubset(report.index)
    assert "f1-score" in report.columns
