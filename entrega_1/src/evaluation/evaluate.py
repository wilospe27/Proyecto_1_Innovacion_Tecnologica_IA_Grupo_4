"""Métricas de clasificación y evaluación de checkpoints.

Métricas principales del proyecto (ver ``docs/reporte_entrega_1.md``):

- **F1 macro**: métrica principal; promedia todas las especies por igual, por
  lo que no se deja inflar por las especies con más grabaciones.
- Exactitud, exactitud balanceada, precisión y exhaustividad (macro).
- Exactitud top-3: la especie correcta está entre las tres más probables.
- Matriz de confusión normalizada y reporte por especie.

Uso (desde ``entrega_1/``)::

    python -m src.main evaluate --checkpoint experiments/checkpoints/<corrida>.pt --split test
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.utils.helpers import PROCESSED_DIR, RESULTS_DIR, ensure_dir, get_device, save_json


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Métricas agregadas (valores ``float`` de Python, serializables)."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def top_k_accuracy(y_true, probabilities, k: int = 3) -> float:
    """Proporción de muestras cuya clase real está entre las ``k`` más probables."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    k = min(k, probabilities.shape[1])
    top_k = np.argsort(probabilities, axis=1)[:, -k:]
    return float(np.mean([label in row for label, row in zip(y_true, top_k)]))


def per_class_report(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    """Precisión, exhaustividad, F1 y soporte por especie."""
    from sklearn.metrics import classification_report

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).T


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list[str],
    path: str | Path,
    normalize: bool = True,
) -> Path:
    """Guarda la matriz de confusión (normalizada por fila) como imagen PNG."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        normalize="true" if normalize else None,
    )
    size = max(6.0, 0.45 * len(labels) + 3.0)
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("Especie predicha")
    ax.set_ylabel("Especie real")
    ax.set_title("Matriz de confusión" + (" (normalizada por fila)" if normalize else ""))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


@torch.no_grad()
def predict_loader(model: nn.Module, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (etiquetas reales, probabilidades) para todas las muestras del loader."""
    model.eval()
    y_true, probabilities = [], []
    for inputs, targets in loader:
        logits = model(inputs.to(device))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        y_true.append(np.asarray(targets))
    return np.concatenate(y_true), np.concatenate(probabilities)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Reconstruye el modelo guardado por ``train`` y lo deja en modo evaluación."""
    from src.models.cnn_baseline import build_model

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    name = checkpoint["model_name"]
    extra = {"pretrained": False} if name == "efficientnet_b0" else {}
    model = build_model(name, n_classes=len(checkpoint["labels"]), **extra)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint


def evaluate_checkpoint(
    checkpoint: str | Path,
    processed_dir: str | Path = PROCESSED_DIR,
    index_file: str = "index.csv",
    split: str = "test",
    batch_size: int = 64,
    output_dir: str | Path | None = None,
) -> dict[str, float]:
    """Evalúa un checkpoint sobre una partición y guarda métricas, reporte y matriz."""
    from torch.utils.data import DataLoader

    from src.data.dataset import SpectrogramDataset

    device = get_device()
    model, ckpt = load_checkpoint(checkpoint, device)
    labels: list[str] = list(ckpt["labels"])
    label_to_idx = {label: i for i, label in enumerate(labels)}

    processed_dir = Path(processed_dir)
    index = pd.read_csv(processed_dir / index_file)
    if "split" in index.columns:
        index = index[index["split"] == split]
    index = index[index["label"].isin(labels)]
    if index.empty:
        raise ValueError(f"No hay ventanas para evaluar en la partición '{split}'")

    loader = DataLoader(
        SpectrogramDataset(index, processed_dir, label_to_idx), batch_size=batch_size
    )
    y_true, probabilities = predict_loader(model, loader, device)
    y_pred = probabilities.argmax(axis=1)

    metrics = compute_metrics(y_true, y_pred)
    metrics["top3_accuracy"] = top_k_accuracy(y_true, probabilities, k=3)

    out = ensure_dir(output_dir or RESULTS_DIR)
    stem = f"{Path(checkpoint).stem}_{split}"
    save_json(
        {"checkpoint": str(checkpoint), "split": split, "n_windows": int(len(y_true)), **metrics},
        out / f"{stem}_metricas.json",
    )
    per_class_report(y_true, y_pred, labels).to_csv(out / f"{stem}_por_especie.csv")
    plot_confusion_matrix(y_true, y_pred, labels, out / f"{stem}_matriz_confusion.png")
    return metrics
