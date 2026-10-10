"""Código fuente del proyecto «Identificación de especies por su canto» (Grupo 4).

Paquetes:
    data        Descarga, preprocesamiento y conjuntos de datos (PyTorch).
    models      Arquitecturas de clasificación sobre espectrogramas Mel.
    training    Bucle de entrenamiento con parada temprana.
    evaluation  Métricas, matriz de confusión y evaluación de checkpoints.
    inference   Predicción de especies para un archivo de audio.
    utils       Rutas del proyecto, semillas, logging y utilidades JSON.

Punto de entrada: ``python -m src.main --help`` (desde la carpeta ``entrega_2/``).
"""

__version__ = "0.1.0"
