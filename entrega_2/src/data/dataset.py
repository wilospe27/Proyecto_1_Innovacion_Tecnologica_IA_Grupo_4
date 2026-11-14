"""Dataset de PyTorch, aumentación tipo SpecAugment y particiones por grabación."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
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


WINDOW_SPLIT_COLUMN = "split_ventana"  # partición al azar por ventana (solo para H3)


def ensure_splits(
    index: pd.DataFrame,
    column: str = "split",
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, bool]:
    """Garantiza que el índice tenga la partición ``column``.

    Retorna ``(índice, creada)``. La columna ``split_ventana`` se crea al azar
    por ventana; cualquier otra, agrupando por grabación.
    """
    if column in index.columns:
        return index, False
    by_recording = column != WINDOW_SPLIT_COLUMN
    return assign_splits(index, val_size, test_size, seed, by_recording=by_recording, column=column), True


def _held_out_rows(frame: pd.DataFrame, fraction: float, seed: int) -> pd.Index:
    """Selecciona ~``fraction`` de las filas sin partir ninguna grabación."""
    n_splits = max(2, int(round(1.0 / fraction)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, held = next(splitter.split(frame, frame["label"], groups=frame["recording_id"]))
    return frame.index[held]


def _held_out_windows(frame: pd.DataFrame, fraction: float, seed: int) -> pd.Index:
    """Selecciona ~``fraction`` de las ventanas al azar, estratificando solo por especie."""
    n_splits = max(2, int(round(1.0 / fraction)))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, held = next(splitter.split(frame, frame["label"]))
    return frame.index[held]


def assign_splits(
    index: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    by_recording: bool = True,
    column: str = "split",
) -> pd.DataFrame:
    """Agrega la columna ``column`` (train/val/test), estratificada por especie.

    Con ``by_recording=True`` (lo correcto) todas las ventanas de una misma
    grabación quedan en la misma partición, lo que evita que el modelo se
    evalúe con fragmentos de audios que ya vio. Con ``by_recording=False`` las
    ventanas se reparten al azar sin importar su grabación: solo lo usamos para
    medir cuánto sobrestima el desempeño esa fuga de información (hipótesis H3).
    """
    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise ValueError("val_size y test_size deben estar en (0, 1) y sumar menos de 1")
    held_out = _held_out_rows if by_recording else _held_out_windows
    result = index.copy()
    result[column] = "train"
    result.loc[held_out(result, test_size, seed), column] = "test"
    remaining = result[result[column] == "train"]
    result.loc[held_out(remaining, val_size / (1 - test_size), seed), column] = "val"
    return result
