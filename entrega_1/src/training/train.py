"""Entrenamiento de un clasificador de especies sobre ventanas de espectrograma Mel.

Uso (desde ``entrega_1/``)::

    python -m src.main train --model cnn_baseline --epochs 30

Si el índice no tiene la columna ``split``, se crea una partición
entrenamiento/validación/prueba agrupada por grabación y se guarda en el mismo
índice para que todas las corridas usen exactamente la misma partición.

Salidas (por defecto en ``experiments/``)::

    checkpoints/<corrida>.pt              mejor modelo según F1 macro de validación
    logs/<corrida>.log                    bitácora de la corrida
    logs/<corrida>_history.csv            pérdidas y métricas por época
    results/<corrida>_train_summary.json  resumen y configuración usada
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import SpectrogramDataset, assign_splits
from src.evaluation.evaluate import compute_metrics
from src.models.cnn_baseline import build_model, count_parameters
from src.utils.helpers import (
    EXPERIMENTS_DIR,
    PROCESSED_DIR,
    close_file_handlers,
    ensure_dir,
    get_device,
    get_logger,
    load_json,
    save_json,
    set_seed,
    timestamp,
)


@dataclass
class TrainConfig:
    """Hiperparámetros y rutas de una corrida de entrenamiento."""

    processed_dir: str = str(PROCESSED_DIR)
    index_file: str = "index.csv"
    output_dir: str = str(EXPERIMENTS_DIR)
    model_name: str = "cnn_baseline"
    pretrained: bool = True  # solo aplica a modelos con pesos preentrenados
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_size: float = 0.15
    test_size: float = 0.15
    patience: int = 5  # épocas sin mejora antes de detener
    augment: bool = True
    class_weights: bool = True  # compensa el desbalance entre especies
    num_workers: int = 0
    seed: int = 42
    run_name: str | None = None


def model_kwargs(model_name: str, pretrained: bool) -> dict:
    """Argumentos extra según el modelo (solo EfficientNet usa ``pretrained``)."""
    return {"pretrained": pretrained} if model_name == "efficientnet_b0" else {}


def class_weight_tensor(train_labels: pd.Series, labels: list[str], device: str) -> torch.Tensor:
    """Pesos inversamente proporcionales a la frecuencia de cada especie."""
    counts = (
        pd.Series(train_labels)
        .value_counts()
        .reindex(labels, fill_value=0)
        .to_numpy(dtype=np.float64)
    )
    weights = counts.sum() / (len(labels) * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """Una pasada completa sobre el conjunto de entrenamiento; retorna la pérdida media."""
    model.train()
    total_loss, n_samples = 0.0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
        n_samples += len(targets)
    return total_loss / max(n_samples, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Retorna (pérdida media, etiquetas reales, probabilidades por clase)."""
    model.eval()
    total_loss, n_samples = 0.0, 0
    y_true, probabilities = [], []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        total_loss += criterion(logits, targets).item() * len(targets)
        n_samples += len(targets)
        y_true.append(targets.cpu().numpy())
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return total_loss / max(n_samples, 1), np.concatenate(y_true), np.concatenate(probabilities)


def train(cfg: TrainConfig | None = None) -> dict:
    """Entrena el modelo indicado en ``cfg`` y guarda el mejor checkpoint."""
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)
    run_name = cfg.run_name or f"{cfg.model_name}_{timestamp()}"
    output_dir = Path(cfg.output_dir)
    logs_dir = ensure_dir(output_dir / "logs")
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    results_dir = ensure_dir(output_dir / "results")
    logger = get_logger("train", logs_dir / f"{run_name}.log")

    try:
        processed_dir = Path(cfg.processed_dir)
        index_path = processed_dir / cfg.index_file
        index = pd.read_csv(index_path)
        if "split" not in index.columns:
            index = assign_splits(index, cfg.val_size, cfg.test_size, cfg.seed)
            index.to_csv(index_path, index=False)
            logger.info("Partición por grabación guardada en %s", index_path)

        labels = sorted(str(label) for label in index["label"].unique())
        label_to_idx = {label: i for i, label in enumerate(labels)}
        train_df = index[index["split"] == "train"]
        val_df = index[index["split"] == "val"]
        if train_df.empty or val_df.empty:
            raise ValueError("Las particiones de entrenamiento y validación no pueden estar vacías")

        train_loader = DataLoader(
            SpectrogramDataset(train_df, processed_dir, label_to_idx, augment=cfg.augment),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
        )
        val_loader = DataLoader(
            SpectrogramDataset(val_df, processed_dir, label_to_idx),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )

        device = get_device()
        model = build_model(
            cfg.model_name, n_classes=len(labels), **model_kwargs(cfg.model_name, cfg.pretrained)
        ).to(device)
        logger.info(
            "Modelo %s | %s parámetros | %d especies | dispositivo %s",
            cfg.model_name,
            f"{count_parameters(model):,}",
            len(labels),
            device,
        )
        logger.info("Ventanas: %d entrenamiento | %d validación", len(train_df), len(val_df))

        weights = class_weight_tensor(train_df["label"], labels, device) if cfg.class_weights else None
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )

        audio_config_path = processed_dir / "audio_config.json"
        audio_config = load_json(audio_config_path) if audio_config_path.exists() else None
        checkpoint_path = checkpoints_dir / f"{run_name}.pt"

        history: list[dict] = []
        best_f1, best_epoch, epochs_without_improvement = -1.0, 0, 0
        for epoch in range(1, cfg.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, y_true, probabilities = validate(model, val_loader, criterion, device)
            metrics = compute_metrics(y_true, probabilities.argmax(axis=1))
            scheduler.step(metrics["f1_macro"])
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **metrics,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            logger.info(
                "Época %02d | pérdida %.4f | pérdida val %.4f | exactitud %.3f | F1 macro %.3f",
                epoch,
                train_loss,
                val_loss,
                metrics["accuracy"],
                metrics["f1_macro"],
            )

            if metrics["f1_macro"] > best_f1:
                best_f1, best_epoch, epochs_without_improvement = metrics["f1_macro"], epoch, 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_name": cfg.model_name,
                        "labels": labels,
                        "epoch": epoch,
                        "val_metrics": metrics,
                        "train_config": asdict(cfg),
                        "audio_config": audio_config,
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= cfg.patience:
                    logger.info("Parada temprana en la época %d", epoch)
                    break

        pd.DataFrame(history).to_csv(logs_dir / f"{run_name}_history.csv", index=False)
        summary = {
            "run_name": run_name,
            "checkpoint": str(checkpoint_path),
            "best_epoch": best_epoch,
            "best_val_f1_macro": best_f1,
            "epochs_run": len(history),
            "n_classes": len(labels),
            "n_train_windows": int(len(train_df)),
            "n_val_windows": int(len(val_df)),
            "config": asdict(cfg),
        }
        save_json(summary, results_dir / f"{run_name}_train_summary.json")
        logger.info("Mejor F1 macro de validación: %.3f (época %d)", best_f1, best_epoch)
        return summary
    finally:
        close_file_handlers(logger)
