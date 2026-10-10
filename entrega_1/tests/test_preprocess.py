"""Pruebas del preprocesamiento de audio (sin datos externos: señales sintéticas)."""

import numpy as np
import pytest
import soundfile as sf

from src.data.preprocess import (
    AudioConfig,
    audio_to_features,
    mel_spectrogram,
    preprocess_directory,
    segment_signal,
    standardize,
)


def synthetic_call(seconds: float, sr: int = 32_000, freq: float = 3_000.0) -> np.ndarray:
    """Tono con modulación de amplitud más ruido leve (sustituto de un canto)."""
    t = np.arange(int(seconds * sr)) / sr
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 2 * t))
    rng = np.random.default_rng(0)
    return (0.5 * envelope * np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(len(t))).astype(
        np.float32
    )


def test_segmentacion_con_cola_rellenada():
    signal = np.ones(12_000, dtype=np.float32)
    segments = segment_signal(signal, segment_samples=5_000, hop_samples=2_500)
    # Ventanas en 0, 2500 y 5000 + una ventana final que inicia en 7500 (4500 muestras reales)
    assert segments.shape == (4, 5_000)
    assert np.all(segments[-1, :4_500] == 1.0)
    assert np.all(segments[-1, 4_500:] == 0.0)


def test_senal_corta_se_rellena_con_ceros():
    segments = segment_signal(np.ones(3_000, dtype=np.float32), 5_000, 2_500)
    assert segments.shape == (1, 5_000)
    assert segments[0, 3_000:].sum() == 0.0


def test_senal_de_longitud_exacta():
    segments = segment_signal(np.ones(5_000, dtype=np.float32), 5_000, 2_500)
    assert segments.shape == (1, 5_000)


def test_segmentacion_rechaza_senal_2d():
    with pytest.raises(ValueError):
        segment_signal(np.ones((2, 100), dtype=np.float32), 50, 25)


def test_forma_del_espectrograma_mel():
    cfg = AudioConfig()
    spectrogram = mel_spectrogram(synthetic_call(cfg.segment_seconds), cfg)
    assert spectrogram.shape == (cfg.n_mels, cfg.n_frames)
    assert np.isfinite(spectrogram).all()
    assert spectrogram.max() <= 0.0 + 1e-4  # referencia = máximo -> 0 dB
    assert spectrogram.min() >= -cfg.top_db - 1e-4


def test_estandarizacion():
    values = np.random.default_rng(1).normal(5.0, 3.0, size=(16, 20)).astype(np.float32)
    standardized = standardize(values)
    assert abs(float(standardized.mean())) < 1e-4
    assert abs(float(standardized.std()) - 1.0) < 1e-3
    assert np.all(standardize(np.zeros((4, 4), dtype=np.float32)) == 0.0)  # sin NaN


def test_audio_a_caracteristicas():
    cfg = AudioConfig()
    features = audio_to_features(synthetic_call(12.0), cfg)
    assert features.shape == (4, cfg.n_mels, cfg.n_frames)
    assert features.dtype == np.float32


def test_preprocesar_directorio(tmp_path):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    for label, freq in [("Especie_a", 2_000.0), ("Especie_b", 5_000.0)]:
        (raw / label).mkdir(parents=True)
        sf.write(raw / label / "XC1.wav", synthetic_call(7.0, freq=freq), 32_000)

    index = preprocess_directory(raw, processed)

    assert set(index["label"]) == {"Especie_a", "Especie_b"}
    assert index["recording_id"].nunique() == 2
    assert all((processed / path).exists() for path in index["path"])
    assert (processed / "index.csv").exists()
    assert (processed / "audio_config.json").exists()
