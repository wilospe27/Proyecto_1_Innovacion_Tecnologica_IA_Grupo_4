"""Pruebas de la extracción de embeddings (con un modelo falso, sin TensorFlow)."""

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.data.preprocess import AudioConfig
from src.models.embeddings import extract_embeddings, pool_embeddings, recording_windows
from src.utils.helpers import save_json

SR = 32_000


class FakeEmbedder:
    """Devuelve como vector la energía media y el máximo de cada ventana."""

    def __call__(self, windows):
        windows = np.asarray(windows)
        return np.stack([np.abs(windows).mean(axis=1), np.abs(windows).max(axis=1)], axis=1)


def test_pool_embeddings_promedia_ejes_intermedios():
    raw = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    pooled = pool_embeddings(raw, 2)
    assert pooled.shape == (2, 4)
    assert pooled[0] == pytest.approx(raw[0].mean(axis=0))


def test_pool_embeddings_rechaza_numero_incorrecto():
    with pytest.raises(ValueError):
        pool_embeddings(np.zeros((3, 8)), 2)


def test_recording_windows_respeta_tope_de_duracion():
    cfg = AudioConfig(max_seconds=10)
    windows = recording_windows(np.ones(SR * 30, dtype=np.float32), cfg)
    assert windows.shape == (3, SR * 5)  # 0-5, 2,5-7,5 y 5-10 s


def test_extract_embeddings_alinea_con_el_indice(tmp_path):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    (raw / "Especie_a").mkdir(parents=True)
    processed.mkdir()
    rng = np.random.default_rng(0)
    sf.write(raw / "Especie_a" / "XC1.wav", 0.1 * rng.standard_normal(SR * 10), SR)  # 3 ventanas
    sf.write(raw / "Especie_a" / "XC2.wav", 0.5 * rng.standard_normal(SR * 5), SR)   # 1 ventana
    save_json(AudioConfig(max_seconds=60).to_dict(), processed / "audio_config.json")
    index = pd.DataFrame(
        {
            "path": ["a", "b", "c", "d"],
            "label": ["Especie_a"] * 4,
            "recording_id": ["Especie_a/XC2", "Especie_a/XC1", "Especie_a/XC1", "Especie_a/XC1"],
            "segment": [0, 0, 1, 2],
        }
    )
    index.to_csv(processed / "index.csv", index=False)

    path = extract_embeddings(processed_dir=processed, raw_dir=raw, embedder=FakeEmbedder(), batch_size=2)
    vectors = np.load(path)

    assert vectors.shape == (4, 2)
    assert vectors[0, 0] > vectors[1, 0]  # la fila 0 es XC2, más fuerte que XC1


def test_extract_embeddings_detecta_indice_desalineado(tmp_path):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    (raw / "Especie_a").mkdir(parents=True)
    processed.mkdir()
    sf.write(raw / "Especie_a" / "XC1.wav", np.zeros(SR * 5), SR)
    save_json(AudioConfig().to_dict(), processed / "audio_config.json")
    pd.DataFrame(
        {"path": ["a", "b"], "label": ["Especie_a"] * 2, "recording_id": ["Especie_a/XC1"] * 2, "segment": [0, 1]}
    ).to_csv(processed / "index.csv", index=False)

    with pytest.raises(ValueError, match="ventanas"):
        extract_embeddings(processed_dir=processed, raw_dir=raw, embedder=FakeEmbedder())
