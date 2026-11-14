"""Pruebas del cálculo de F1 por grabación e intervalos de confianza."""

import numpy as np
import pytest

from src.evaluation.uncertainty import aggregate_by_recording, bootstrap_f1_macro


def test_aggregate_by_recording_promedia_probabilidades():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array(
        [[0.9, 0.1], [0.4, 0.6], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]]
    )
    ids = ["a", "a", "a", "b", "b"]
    labels, proba, names = aggregate_by_recording(y, p, ids)

    assert list(names) == ["a", "b"]
    assert list(labels) == [0, 1]
    assert proba[0] == pytest.approx([0.7, 0.3])
    assert proba.argmax(axis=1).tolist() == [0, 1]


def test_aggregate_by_recording_rechaza_etiquetas_mezcladas():
    with pytest.raises(ValueError, match="más de una etiqueta"):
        aggregate_by_recording([0, 1], np.eye(2), ["a", "a"])


def test_bootstrap_perfecto_tiene_intervalo_degenerado():
    y = np.array([0, 1, 2] * 10)
    result = bootstrap_f1_macro(y, y, n_boot=50, threshold=0.9)

    assert result["f1_macro"] == 1.0
    assert result["ic_inferior"] == 1.0 and result["ic_superior"] == 1.0
    assert result["fraccion_remuestras_sobre_umbral"] == 1.0


def test_bootstrap_intervalo_contiene_la_estimacion_y_es_reproducible():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 120)
    pred = np.where(rng.random(120) < 0.7, y, rng.integers(0, 3, 120))
    groups = np.repeat(np.arange(30), 4)

    a = bootstrap_f1_macro(y, pred, groups=groups, n_boot=300, seed=1)
    b = bootstrap_f1_macro(y, pred, groups=groups, n_boot=300, seed=1)

    assert a == b
    assert a["ic_inferior"] <= a["f1_macro"] <= a["ic_superior"]
    assert a["ic_superior"] - a["ic_inferior"] > 0


def test_bootstrap_por_grupos_es_mas_ancho_que_por_ventanas():
    """Si las ventanas de una grabación son idénticas, tratarlas como independientes subestima la incertidumbre."""
    rng = np.random.default_rng(3)
    per_recording_label = rng.integers(0, 2, 40)
    per_recording_ok = rng.random(40) < 0.7
    y = np.repeat(per_recording_label, 5)
    pred = np.repeat(np.where(per_recording_ok, per_recording_label, 1 - per_recording_label), 5)
    groups = np.repeat(np.arange(40), 5)

    por_grupo = bootstrap_f1_macro(y, pred, groups=groups, n_boot=400, seed=0)
    por_ventana = bootstrap_f1_macro(y, pred, n_boot=400, seed=0)

    ancho = lambda r: r["ic_superior"] - r["ic_inferior"]
    assert ancho(por_grupo) > ancho(por_ventana)
