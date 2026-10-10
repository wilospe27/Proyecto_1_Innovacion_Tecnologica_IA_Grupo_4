"""Dataset de PyTorch, aumentación tipo SpecAugment y particiones por grabación."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset


def spec_augment(
    spectrogram: torch.Tensor,
    max_freq_mask: int = 16,
    max_time_mask: int = 40,
    n_masks: int = 2,
) -> torch.Tensor:
    """Enmascara bandas aleatorias de frecuencia y de tiempo (estilo SpecAugment).

    Se espera un tensor ``(canales, n_mels, n_frames)`` estandarizado; el valor de
    relleno es 0, que corresponde a la media del espectrograma.
    """
    augmented = spectrogram.clone()
    _, n_freq, n_time = augmented.shape
    for _ in range(n_masks):
        width = np.random.randint(0, max_freq_mask + 1)
        start = np.random.randint(0, max(1, n_freq - width))
        augmented[:, start : start + width, :] = 0.0
        width = np.random.randint(0, max_time_mask + 1)
        start = np.random.randint(0, max(1, n_time - width))
        augmented[:, :, start : start + width] = 0.0
    return augmented


class SpectrogramDataset(Dataset):
    """Lee las ventanas ``.npy`` listadas en el índice generado por el preprocesamiento."""

    def __init__(
        self,
        index: pd.DataFrame,
        root: str | Path,
        label_to_idx: dict[str, int],
        augment: bool = False,
    ) -> None:
        self.index = index.reset_index(drop=True)
        self.root = Path(root)
        self.label_to_idx = label_to_idx
        self.augment = augment

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        row = self.index.iloc[i]
        spectrogram = np.load(self.root / row["path"])
        tensor = torch.from_numpy(spectrogram).float().unsqueeze(0)  # (1, n_mels, n_frames)
        if self.augment:
            tensor = spec_augment(tensor)
        return tensor, self.label_to_idx[row["label"]]


def _held_out_rows(frame: pd.DataFrame, fraction: float, seed: int) -> pd.Index:
    """Selecciona ~``fraction`` de las filas sin partir ninguna grabación."""
    n_splits = max(2, int(round(1.0 / fraction)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, held = next(splitter.split(frame, frame["label"], groups=frame["recording_id"]))
    return frame.index[held]


def assign_splits(
    index: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Agrega la columna ``split`` (train/val/test) agrupando por grabación.

    Todas las ventanas de una misma grabación quedan en la misma partición, lo
    que evita que el modelo se evalúe con fragmentos de audios que ya vio.
    La partición se estratifica por especie en la medida de lo posible.
    """
    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise ValueError("val_size y test_size deben estar en (0, 1) y sumar menos de 1")
    result = index.copy()
    result["split"] = "train"
    result.loc[_held_out_rows(result, test_size, seed), "split"] = "test"
    remaining = result[result["split"] == "train"]
    result.loc[_held_out_rows(remaining, val_size / (1 - test_size), seed), "split"] = "val"
    return result
