"""Predicción con Perch 2.0: *embeddings* por ventana + clasificador lineal.

Es el mejor modelo del proyecto (F1 macro de 0,97 por grabación en prueba).
Cada ventana de 5 s pasa por Perch 2.0, que la resume en un vector de 1.536
números, y una regresión logística entrenada con nuestras grabaciones decide la
especie. El clasificador se entrena una sola vez con los *embeddings* de
``python -m src.main embeddings`` y se guarda en
``experiments/checkpoints/perch_v2_regresion_logistica.joblib``, junto con las
especies y la configuración de audio, para no depender después de los datos.

Necesita TensorFlow y ``perch-hoplite`` (ver ``docs/instalacion.md``); este
módulo no importa PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.data.preprocess import AudioConfig
from src.models.embeddings import PerchEmbedder, embeddings_file, recording_windows
from src.utils.helpers import CHECKPOINTS_DIR, PROCESSED_DIR, ensure_dir, get_logger, load_json

PROBE_FILE = CHECKPOINTS_DIR / "perch_v2_regresion_logistica.joblib"

logger = get_logger("perch")


@dataclass
class PerchPredictor:
    """Perch 2.0 cargado con el clasificador lineal, sus especies y la configuración de audio."""

    embedder: Callable[[np.ndarray], np.ndarray]
    probe: Any
    labels: list[str]
    audio_config: AudioConfig
    model_name: str = "perch_v2"


def train_probe(
    processed_dir: str | Path = PROCESSED_DIR,
    model_name: str = "perch_v2",
    seed: int = 42,
) -> dict[str, Any]:
    """Entrena la regresión logística sobre los *embeddings* de la partición de entrenamiento.

    Usa el mismo modelo y la misma partición por grabación que ``baselines``,
    así que reproduce el resultado reportado.
    """
    from src.models.classical import build_classical_models, load_feature_dataset

    processed_dir = Path(processed_dir)
    index, features = load_feature_dataset(processed_dir, features_file=embeddings_file(model_name))
    labels = sorted(str(label) for label in index["label"].unique())
    y = index["label"].map({label: i for i, label in enumerate(labels)}).to_numpy()
    train = (index["split"] == "train").to_numpy()
    probe = build_classical_models(seed)["regresion_logistica"].fit(features[train], y[train])
    return {
        "probe": probe,
        "labels": labels,
        "audio_config": load_json(processed_dir / "audio_config.json"),
        "embedding_model": model_name,
    }


def load_perch_predictor(
    probe_path: str | Path = PROBE_FILE,
    processed_dir: str | Path = PROCESSED_DIR,
    embedder: Callable[[np.ndarray], np.ndarray] | None = None,
) -> PerchPredictor:
    """Carga el clasificador guardado (o lo entrena y lo guarda la primera vez) y Perch 2.0."""
    import joblib

    probe_path = Path(probe_path)
    if probe_path.exists():
        bundle = joblib.load(probe_path)
    else:
        logger.info("Entrenando el clasificador sobre los embeddings de Perch 2.0 (solo la primera vez)")
        bundle = train_probe(processed_dir)
        ensure_dir(probe_path.parent)
        joblib.dump(bundle, probe_path)
    return PerchPredictor(
        embedder=embedder or PerchEmbedder(bundle["embedding_model"]),
        probe=bundle["probe"],
        labels=list(bundle["labels"]),
        audio_config=AudioConfig(**bundle["audio_config"]),
        model_name=bundle["embedding_model"],
    )


def predict_signal_perch(predictor: PerchPredictor, signal: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """Probabilidades por ventana, de forma ``(n_ventanas, n_especies)``, como ``predict_signal``."""
    windows = recording_windows(np.asarray(signal, dtype=np.float32), predictor.audio_config)
    if len(windows) == 0:
        raise ValueError("El audio no contiene ventanas válidas")
    vectors = np.concatenate([predictor.embedder(windows[i : i + batch_size]) for i in range(0, len(windows), batch_size)])
    return predictor.probe.predict_proba(vectors)
