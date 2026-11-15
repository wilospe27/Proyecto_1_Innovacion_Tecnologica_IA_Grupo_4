"""Pruebas de la predicción con Perch 2.0 (con un modelo falso, sin TensorFlow)."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import AudioConfig
from src.inference.perch import load_perch_predictor, predict_signal_perch
from src.utils.helpers import save_json

SR = 32_000


class FakeEmbedder:
    """Devuelve como vector la energía media y el máximo de cada ventana."""

    def __call__(self, windows):
        windows = np.asarray(windows)
        return np.stack([np.abs(windows).mean(axis=1), np.abs(windows).max(axis=1)], axis=1)


def write_processed(processed):
    """Embeddings separables con la escala de ``FakeEmbedder`` para ruido uniforme de amplitud A: (A/2, A)."""
    processed.mkdir()
    save_json(AudioConfig(max_seconds=60).to_dict(), processed / "audio_config.json")
    rng = np.random.default_rng(0)
    labels = ["Especie_a"] * 20 + ["Especie_b"] * 20
    energy = np.r_[rng.uniform(0.05, 0.15, 20), rng.uniform(0.4, 0.6, 20)]
    pd.DataFrame(
        {
            "label": labels,
            "recording_id": [f"{label}/XC{i}" for i, label in enumerate(labels)],
            "split": (["train"] * 16 + ["test"] * 4) * 2,
        }
    ).to_csv(processed / "index.csv", index=False)
    np.save(processed / "embeddings_perch_v2.npy", np.stack([energy / 2, energy], axis=1).astype(np.float32))


def test_load_perch_predictor_entrena_y_guarda_el_clasificador(tmp_path):
    processed, probe_path = tmp_path / "processed", tmp_path / "ckpt" / "probe.joblib"
    write_processed(processed)

    predictor = load_perch_predictor(probe_path, processed, embedder=FakeEmbedder())

    assert probe_path.exists()
    assert predictor.labels == ["Especie_a", "Especie_b"]
    assert predictor.audio_config.max_seconds == 60
    # La segunda vez carga el archivo guardado, aunque ya no estén los datos.
    (processed / "embeddings_perch_v2.npy").unlink()
    again = load_perch_predictor(probe_path, processed, embedder=FakeEmbedder())
    assert again.labels == predictor.labels


def test_predict_signal_perch_da_probabilidades_por_ventana(tmp_path):
    processed = tmp_path / "processed"
    write_processed(processed)
    predictor = load_perch_predictor(tmp_path / "probe.joblib", processed, embedder=FakeEmbedder())
    rng = np.random.default_rng(1)
    signal = np.r_[0.1 * rng.uniform(-1, 1, SR * 5), 0.5 * rng.uniform(-1, 1, SR * 5)].astype(np.float32)

    probabilities = predict_signal_perch(predictor, signal, batch_size=2)

    assert probabilities.shape == (3, 2)  # ventanas 0-5, 2,5-7,5 y 5-10 s
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(3))
    assert probabilities[0].argmax() == 0  # la primera ventana es suave: especie a
    assert probabilities[2].argmax() == 1  # la última es fuerte: especie b
