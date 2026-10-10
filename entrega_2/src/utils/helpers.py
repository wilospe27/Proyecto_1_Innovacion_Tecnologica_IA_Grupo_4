"""Funciones auxiliares compartidas por todo el proyecto.

Incluye las rutas estándar de la entrega, la fijación de semillas para
reproducibilidad, la selección de dispositivo (CPU/GPU), utilidades JSON
y la configuración de *logging*.
"""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Raíz de la entrega: carpeta que contiene src/, docs/, data/, experiments/...
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
LOGS_DIR = EXPERIMENTS_DIR / "logs"
CHECKPOINTS_DIR = EXPERIMENTS_DIR / "checkpoints"
RESULTS_DIR = EXPERIMENTS_DIR / "results"

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def ensure_dir(path: str | Path) -> Path:
    """Crea el directorio (y sus padres) si no existe y lo retorna como ``Path``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int = 42) -> None:
    """Fija las semillas de Python, NumPy y PyTorch para resultados reproducibles."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # PyTorch no es necesario para las utilidades de datos
        pass


def get_device() -> str:
    """Retorna ``"cuda"``, ``"mps"`` (Apple) o ``"cpu"`` según disponibilidad."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def timestamp() -> str:
    """Marca de tiempo compacta para nombrar corridas: ``AAAAMMDD-HHMMSS``."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    """Guarda un diccionario como JSON (UTF-8, indentado)."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    """Carga un archivo JSON como diccionario."""
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def get_logger(
    name: str = "cantos",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Logger con salida a consola y, opcionalmente, a un archivo de texto."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    if not any(type(h) is logging.StreamHandler for h in logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file).resolve()
        ensure_dir(log_path.parent)
        already = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path
            for h in logger.handlers
        )
        if not already:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger


def close_file_handlers(logger: logging.Logger) -> None:
    """Cierra y retira los manejadores de archivo de un logger (al final de una corrida)."""
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()
