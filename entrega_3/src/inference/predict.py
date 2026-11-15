"""Predicción de la especie que vocaliza en un archivo de audio.

El audio se divide en ventanas con los mismos parámetros usados en el
entrenamiento (guardados en el checkpoint); se promedian las probabilidades
de todas las ventanas y se retornan las ``top_k`` especies más probables.

Uso (desde ``entrega_3/``)::

    python -m src.main predict --audio grabacion.wav --checkpoint experiments/checkpoints/<corrida>.pt

La interfaz de demostración (``app/streamlit_app.py``) usa ``load_predictor``
y ``predict_signal`` para cargar el modelo una sola vez.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.data.preprocess import AudioConfig, audio_to_features, load_audio
from src.evaluation.evaluate import load_checkpoint
from src.utils.helpers import get_device

if TYPE_CHECKING:
    from torch import nn


@dataclass
class Predictor:
    """Modelo cargado con sus especies y la configuración de audio del entrenamiento."""

    model: nn.Module
    labels: list[str]
    audio_config: AudioConfig
    device: str
    model_name: str


def load_predictor(checkpoint: str | Path, device: str | None = None) -> Predictor:
    """Carga un checkpoint de ``train`` listo para predecir."""
    device = device or get_device()
    model, ckpt = load_checkpoint(checkpoint, device)
    cfg = AudioConfig(**ckpt["audio_config"]) if ckpt.get("audio_config") else AudioConfig()
    return Predictor(model, list(ckpt["labels"]), cfg, device, ckpt["model_name"])


def predict_signal(predictor: Predictor, signal: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """Probabilidades por ventana, de forma ``(n_ventanas, n_especies)``.

    ``signal`` es audio mono a la frecuencia de muestreo del entrenamiento.
    """
    import torch

    features = audio_to_features(signal, predictor.audio_config)
    if len(features) == 0:
        raise ValueError("El audio no contiene ventanas válidas")
    batches = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).unsqueeze(1).to(predictor.device)
            batches.append(torch.softmax(predictor.model(batch), dim=1).cpu().numpy())
    return np.concatenate(batches)


def top_species(probabilities: np.ndarray, labels: list[str], top_k: int = 5) -> pd.DataFrame:
    """Promedia las probabilidades de las ventanas y retorna las ``top_k`` especies."""
    mean_probabilities = np.asarray(probabilities).mean(axis=0)
    order = np.argsort(mean_probabilities)[::-1][: min(top_k, len(labels))]
    return pd.DataFrame(
        {
            "especie": [labels[i].replace("_", " ") for i in order],
            "probabilidad": mean_probabilities[order].round(4),
        }
    )


def predict_file(
    audio_path: str | Path,
    checkpoint: str | Path,
    top_k: int = 5,
    device: str | None = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Retorna una tabla con las ``top_k`` especies y su probabilidad media."""
    predictor = load_predictor(checkpoint, device)
    signal = load_audio(audio_path, predictor.audio_config.sample_rate)
    try:
        probabilities = predict_signal(predictor, signal, batch_size)
    except ValueError as exc:
        raise ValueError(f"El audio {audio_path} no contiene ventanas válidas") from exc
    return top_species(probabilities, predictor.labels, top_k)
