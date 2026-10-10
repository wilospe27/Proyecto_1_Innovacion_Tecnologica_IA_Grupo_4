"""Predicción de la especie que vocaliza en un archivo de audio.

El audio se divide en ventanas con los mismos parámetros usados en el
entrenamiento (guardados en el checkpoint); se promedian las probabilidades
de todas las ventanas y se retornan las ``top_k`` especies más probables.

Uso (desde ``entrega_1/``)::

    python -m src.main predict --audio grabacion.wav --checkpoint experiments/checkpoints/<corrida>.pt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.preprocess import AudioConfig, audio_to_features, load_audio
from src.evaluation.evaluate import load_checkpoint
from src.utils.helpers import get_device


def predict_file(
    audio_path: str | Path,
    checkpoint: str | Path,
    top_k: int = 5,
    device: str | None = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Retorna una tabla con las ``top_k`` especies y su probabilidad media."""
    device = device or get_device()
    model, ckpt = load_checkpoint(checkpoint, device)
    labels: list[str] = list(ckpt["labels"])
    cfg = AudioConfig(**ckpt["audio_config"]) if ckpt.get("audio_config") else AudioConfig()

    features = audio_to_features(load_audio(audio_path, cfg.sample_rate), cfg)
    if len(features) == 0:
        raise ValueError(f"El audio {audio_path} no contiene ventanas válidas")

    batches = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).unsqueeze(1).to(device)
            batches.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    mean_probabilities = np.concatenate(batches).mean(axis=0)

    order = np.argsort(mean_probabilities)[::-1][: min(top_k, len(labels))]
    return pd.DataFrame(
        {
            "especie": [labels[i].replace("_", " ") for i in order],
            "probabilidad": mean_probabilities[order].round(4),
        }
    )
