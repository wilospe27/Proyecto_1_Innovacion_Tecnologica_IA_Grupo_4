"""Modelos del proyecto: redes sobre espectrogramas, clasificadores clásicos y embeddings.

Las redes (``cnn_baseline``) dependen de PyTorch y los embeddings de Perch, de
TensorFlow; por eso los importamos solo cuando se usan, y cada submódulo
funciona en el entorno que tenga su biblioteca.
"""

from __future__ import annotations

__all__ = ["BirdCNN", "build_model", "count_parameters"]


def __getattr__(name: str):
    if name in __all__:
        from src.models import cnn_baseline

        return getattr(cnn_baseline, name)
    raise AttributeError(f"module 'src.models' has no attribute {name!r}")
