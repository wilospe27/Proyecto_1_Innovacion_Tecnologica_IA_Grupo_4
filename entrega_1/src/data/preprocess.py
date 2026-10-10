"""Preprocesamiento de audio: carga, segmentación y espectrogramas Mel.

Flujo por grabación::

    audio (mp3/wav) -> mono + remuestreo -> [reducción de ruido opcional]
        -> ventanas de duración fija con solapamiento -> log-Mel (dB) -> estandarización

Estructura esperada de los datos crudos (la carpeta es la etiqueta)::

    data/raw/<Genero_especie>/<id_grabacion>.mp3

Cada ventana procesada se guarda como ``.npy`` y se registra en un índice CSV
con las columnas ``path, label, recording_id, segment``. La columna
``recording_id`` permite hacer particiones por grabación y evitar fuga de
información entre entrenamiento y validación.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.helpers import get_logger, save_json

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
INDEX_COLUMNS = ["path", "label", "recording_id", "segment"]

logger = get_logger("preprocess")


@dataclass
class AudioConfig:
    """Parámetros del preprocesamiento (justificación en ``docs/arquitectura.md``)."""

    sample_rate: int = 32_000  # Hz; permite analizar hasta 16 kHz
    segment_seconds: float = 5.0  # duración de cada ventana de análisis
    hop_seconds: float = 2.5  # desplazamiento entre ventanas (50 % de solapamiento)
    n_fft: int = 1024
    hop_length: int = 320  # 10 ms a 32 kHz
    n_mels: int = 128
    fmin: float = 50.0
    fmax: float = 14_000.0
    top_db: float = 80.0
    reduce_noise: bool = False  # reducción espectral de ruido con `noisereduce`
    min_rms_db: float | None = None  # descarta ventanas casi silenciosas si se define

    @property
    def segment_samples(self) -> int:
        return int(round(self.segment_seconds * self.sample_rate))

    @property
    def hop_samples(self) -> int:
        return int(round(self.hop_seconds * self.sample_rate))

    @property
    def n_frames(self) -> int:
        """Cuadros temporales por ventana (librosa usa ``center=True``)."""
        return 1 + self.segment_samples // self.hop_length

    def to_dict(self) -> dict:
        return asdict(self)


def load_audio(path: str | Path, sample_rate: int = 32_000) -> np.ndarray:
    """Carga un archivo de audio en mono y lo remuestrea a ``sample_rate``."""
    import librosa

    signal, _ = librosa.load(path, sr=sample_rate, mono=True)
    return signal.astype(np.float32)


def reduce_noise(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Reducción espectral de ruido estacionario (viento, ruido de fondo)."""
    import noisereduce as nr

    return nr.reduce_noise(y=signal, sr=sample_rate).astype(np.float32)


def segment_signal(
    signal: np.ndarray,
    segment_samples: int,
    hop_samples: int,
    pad_last: bool = True,
) -> np.ndarray:
    """Divide una señal mono en ventanas de igual longitud.

    - Si la señal es más corta que una ventana, se completa con ceros.
    - Si ``pad_last`` es ``True`` y queda audio sin cubrir al final, se agrega
      una ventana adicional completada con ceros (contiene al menos la mitad de
      audio real cuando el solapamiento es del 50 %).

    Retorna un arreglo de forma ``(n_ventanas, segment_samples)``.
    """
    if segment_samples <= 0 or hop_samples <= 0:
        raise ValueError("segment_samples y hop_samples deben ser positivos")
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 1:
        raise ValueError("Se espera una señal mono (arreglo 1-D)")

    if len(signal) <= segment_samples:
        padded = np.pad(signal, (0, segment_samples - len(signal)))
        return padded[np.newaxis, :]

    starts = list(range(0, len(signal) - segment_samples + 1, hop_samples))
    segments = [signal[s : s + segment_samples] for s in starts]
    if pad_last and starts[-1] + segment_samples < len(signal):
        tail = signal[starts[-1] + hop_samples :]
        segments.append(np.pad(tail, (0, segment_samples - len(tail))))
    return np.stack(segments)


def mel_spectrogram(segment: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    """Espectrograma Mel en decibeles, de forma ``(n_mels, n_frames)``."""
    import librosa

    power = librosa.feature.melspectrogram(
        y=segment,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0,
    )
    return librosa.power_to_db(power, ref=np.max, top_db=cfg.top_db).astype(np.float32)


def standardize(spectrogram: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Estandariza un espectrograma a media 0 y desviación 1 (por ventana)."""
    mean, std = float(spectrogram.mean()), float(spectrogram.std())
    return ((spectrogram - mean) / (std + eps)).astype(np.float32)


def rms_db(segment: np.ndarray) -> float:
    """Nivel RMS de una ventana en dBFS."""
    rms = np.sqrt(np.mean(np.square(segment, dtype=np.float64)))
    return float(20.0 * np.log10(rms + 1e-12))


def audio_to_features(signal: np.ndarray, cfg: AudioConfig | None = None) -> np.ndarray:
    """Convierte una señal completa en ``(n_ventanas, n_mels, n_frames)``."""
    cfg = cfg or AudioConfig()
    if cfg.reduce_noise:
        signal = reduce_noise(signal, cfg.sample_rate)
    features = [
        standardize(mel_spectrogram(segment, cfg))
        for segment in segment_signal(signal, cfg.segment_samples, cfg.hop_samples)
        if cfg.min_rms_db is None or rms_db(segment) >= cfg.min_rms_db
    ]
    if not features:
        return np.empty((0, cfg.n_mels, cfg.n_frames), dtype=np.float32)
    return np.stack(features)


def list_audio_files(raw_dir: str | Path) -> list[Path]:
    """Lista recursivamente los archivos de audio soportados."""
    return sorted(
        path for path in Path(raw_dir).rglob("*") if path.suffix.lower() in AUDIO_EXTENSIONS
    )


def preprocess_directory(
    raw_dir: str | Path,
    output_dir: str | Path,
    cfg: AudioConfig | None = None,
    index_name: str = "index.csv",
) -> pd.DataFrame:
    """Procesa todas las grabaciones de ``raw_dir`` y guarda cada ventana como ``.npy``.

    Retorna (y guarda en ``output_dir/index_name``) el índice de ventanas.
    También guarda ``audio_config.json`` para reutilizar los mismos parámetros
    en la inferencia.
    """
    from tqdm import tqdm

    cfg = cfg or AudioConfig()
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    files = list_audio_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"No se encontraron audios en {raw_dir}")

    rows: list[dict] = []
    for audio_path in tqdm(files, desc="Preprocesando", unit="audio"):
        label = audio_path.parent.name
        recording_id = audio_path.relative_to(raw_dir).with_suffix("").as_posix()
        try:
            signal = load_audio(audio_path, cfg.sample_rate)
        except Exception as exc:  # archivo corrupto o formato no soportado
            logger.warning("No se pudo leer %s: %s", audio_path, exc)
            continue

        features = audio_to_features(signal, cfg)
        (output_dir / label).mkdir(parents=True, exist_ok=True)
        for i, spectrogram in enumerate(features):
            relative = Path(label) / f"{audio_path.stem}_{i:03d}.npy"
            np.save(output_dir / relative, spectrogram)
            rows.append(
                {
                    "path": relative.as_posix(),
                    "label": label,
                    "recording_id": recording_id,
                    "segment": i,
                }
            )

    index = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    index.to_csv(output_dir / index_name, index=False)
    save_json(cfg.to_dict(), output_dir / "audio_config.json")
    logger.info(
        "%d ventanas de %d grabaciones (%d especies) guardadas en %s",
        len(index),
        index["recording_id"].nunique(),
        index["label"].nunique(),
        output_dir,
    )
    return index
