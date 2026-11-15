"""*Embeddings* de un modelo bioacústico preentrenado (Perch 2.0) para cada ventana.

Pone a prueba la hipótesis H1: una representación aprendida con millones de
grabaciones de aves supera a una CNN entrenada desde cero con nuestros datos.
El flujo tiene dos pasos:

1. ``python -m src.main embeddings``: recorta cada grabación en las **mismas
   ventanas** del preprocesamiento (5 s, salto de 2,5 s, primeros 60 s) y guarda
   un vector por ventana en ``data/processed/embeddings_<modelo>.npy``, alineado
   fila a fila con ``index.csv``.
2. ``python -m src.main baselines --features embeddings_<modelo>.npy ...``:
   entrena un clasificador lineal sobre esos vectores con la misma partición
   por grabación que el resto de modelos.

Perch 2.0 recibe audio mono a 32 kHz en ventanas de 5 s, igual que nuestro
preprocesamiento. Necesita TensorFlow y ``perch-hoplite``, que instalamos en un
entorno aparte (ver ``docs/instalacion.md``); por eso este módulo no importa
PyTorch.

**Advertencia:** Perch 2.0 se entrenó con grabaciones de xeno-canto, entre otras
fuentes, así que es posible que algunas de nuestras grabaciones de prueba
estuvieran en su entrenamiento. Sus resultados pueden ser optimistas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import AUDIO_EXTENSIONS, AudioConfig, load_audio, segment_signal
from src.utils.helpers import PROCESSED_DIR, RAW_DIR, get_logger, load_json

EMBEDDING_MODELS = ("perch_v2",)

logger = get_logger("embeddings")


def embeddings_file(model_name: str) -> str:
    """Nombre del archivo de *embeddings* dentro de ``data/processed``."""
    return f"embeddings_{model_name}.npy"


def recording_windows(signal: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    """Ventanas de audio crudo idénticas a las del preprocesamiento (sin espectrograma)."""
    if cfg.max_seconds is not None:
        signal = signal[: int(round(cfg.max_seconds * cfg.sample_rate))]
    return segment_signal(signal, cfg.segment_samples, cfg.hop_samples)


def find_audio(raw_dir: Path, recording_id: str) -> Path:
    """Archivo de audio de una grabación (``recording_id`` = ``Especie/XC123`` sin extensión)."""
    candidates = [
        path for path in (raw_dir / recording_id).parent.glob(Path(recording_id).name + ".*")
        if path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    if not candidates:
        raise FileNotFoundError(f"No se encontró el audio de {recording_id} en {raw_dir}")
    return candidates[0]


def pool_embeddings(raw: np.ndarray, n_windows: int) -> np.ndarray:
    """Reduce la salida del modelo a un vector por ventana promediando ejes intermedios.

    Perch devuelve un arreglo de forma ``(ventanas, [cuadros], [canales], dim)``;
    promediamos todo lo que hay entre el primer y el último eje.
    """
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape[0] != n_windows:
        raise ValueError(f"Se esperaban {n_windows} vectores y el modelo devolvió {raw.shape[0]}")
    if raw.ndim > 2:
        raw = raw.reshape(n_windows, -1, raw.shape[-1]).mean(axis=1)
    return raw


class PerchEmbedder:
    """Envoltura mínima de Perch 2.0 (``perch-hoplite``)."""

    def __init__(self, model_name: str = "perch_v2") -> None:
        from perch_hoplite.zoo import model_configs

        self.model = model_configs.load_model_by_name(model_name)

    def __call__(self, windows: np.ndarray) -> np.ndarray:
        outputs = self.model.batch_embed(np.asarray(windows, dtype=np.float32))
        return pool_embeddings(outputs.embeddings, len(windows))


def extract_embeddings(
    model_name: str = "perch_v2",
    processed_dir: str | Path = PROCESSED_DIR,
    raw_dir: str | Path = RAW_DIR,
    index_file: str = "index.csv",
    batch_size: int = 32,
    embedder=None,
) -> Path:
    """Calcula un vector por ventana del índice y lo guarda alineado con ``index.csv``.

    Verifica que cada grabación produzca exactamente las ventanas registradas en
    el índice, para que las filas correspondan a las mismas ventanas que usan
    los demás modelos.
    """
    from tqdm import tqdm

    if model_name not in EMBEDDING_MODELS:
        raise ValueError(f"Modelo desconocido: {model_name}; opciones: {EMBEDDING_MODELS}")
    processed_dir, raw_dir = Path(processed_dir), Path(raw_dir)
    index = pd.read_csv(processed_dir / index_file)
    cfg = AudioConfig(**load_json(processed_dir / "audio_config.json"))
    embedder = embedder or PerchEmbedder(model_name)

    vectors: np.ndarray | None = None
    for recording_id, rows in tqdm(index.groupby("recording_id", sort=False), desc="Embeddings", unit="audio"):
        windows = recording_windows(load_audio(find_audio(raw_dir, recording_id), cfg.sample_rate), cfg)
        segments = rows["segment"].to_numpy()
        if len(windows) != len(rows) or segments.max() >= len(windows):
            raise ValueError(
                f"{recording_id}: {len(windows)} ventanas de audio y {len(rows)} en el índice; "
                "el preprocesamiento usó otra configuración"
            )
        chunks = [embedder(windows[i : i + batch_size]) for i in range(0, len(windows), batch_size)]
        recording_vectors = np.concatenate(chunks)[segments]
        if vectors is None:
            vectors = np.zeros((len(index), recording_vectors.shape[1]), dtype=np.float32)
        vectors[rows.index.to_numpy()] = recording_vectors

    path = processed_dir / embeddings_file(model_name)
    np.save(path, vectors)
    logger.info("%s: %d vectores de dimensión %d en %s", model_name, *vectors.shape, path)
    return path
