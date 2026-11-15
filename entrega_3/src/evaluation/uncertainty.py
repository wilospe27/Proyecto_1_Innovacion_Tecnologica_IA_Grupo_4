"""Incertidumbre de las métricas: F1 por grabación e intervalos de confianza.

Con pocas grabaciones de prueba (119 en este proyecto) una sola cifra de F1
macro es aproximada. Este módulo la acompaña de un intervalo de confianza por
*bootstrap*, remuestreando **grabaciones** (no ventanas), porque las ventanas de
una misma grabación no son independientes.

Dos niveles de reporte:

- **Por ventana**: cada ventana de 5 s cuenta como una muestra.
- **Por grabación**: se promedian las probabilidades de las ventanas de cada
  grabación y se predice una especie por grabación, que es como se usaría el
  modelo con un audio nuevo.

Uso (desde ``entrega_2/``)::

    python -m src.main intervalos --checkpoint experiments/checkpoints/<corrida>.pt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.helpers import PROCESSED_DIR, RESULTS_DIR, ensure_dir, save_json


def aggregate_by_recording(
    y_true, probabilities, recording_ids
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Promedia las probabilidades de las ventanas de cada grabación.

    Retorna (etiqueta real por grabación, probabilidad media por grabación,
    identificadores de las grabaciones). Cada grabación debe tener una sola
    etiqueta real.
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    recording_ids = np.asarray(recording_ids).astype(str)

    frame = pd.DataFrame(probabilities)
    frame["recording"] = recording_ids
    frame["label"] = y_true
    grouped = frame.groupby("recording", sort=True)
    if (grouped["label"].nunique() > 1).any():
        raise ValueError("Hay grabaciones con más de una etiqueta real")
    mean_probabilities = grouped[list(range(probabilities.shape[1]))].mean()
    labels = grouped["label"].first()
    return labels.to_numpy(), mean_probabilities.to_numpy(), mean_probabilities.index.to_numpy()


def bootstrap_f1_macro(
    y_true,
    y_pred,
    groups=None,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
    threshold: float | None = None,
) -> dict[str, float]:
    """F1 macro con intervalo de confianza por *bootstrap*.

    ``groups`` agrupa las muestras (por ejemplo, la grabación de cada ventana):
    se remuestrean grupos completos con reemplazo. Sin ``groups`` se remuestrean
    las muestras. ``threshold`` agrega la fracción de remuestras con F1 macro
    mayor o igual a ese valor (una probabilidad aproximada de alcanzarlo).
    """
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)

    if groups is None:
        members = [np.array([i]) for i in range(len(y_true))]
    else:
        groups = np.asarray(groups).astype(str)
        members = [np.where(groups == g)[0] for g in np.unique(groups)]

    scores = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.integers(0, len(members), len(members))
        rows = np.concatenate([members[i] for i in chosen])
        scores[b] = f1_score(y_true[rows], y_pred[rows], average="macro", zero_division=0)

    low, high = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    result = {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "ic_inferior": float(low),
        "ic_superior": float(high),
        "confianza": 1 - alpha,
    }
    if threshold is not None:
        result["umbral"] = float(threshold)
        result["fraccion_remuestras_sobre_umbral"] = float(np.mean(scores >= threshold))
    return result


def intervals_from_scores(
    y_true,
    scores,
    recording_ids,
    n_boot: int = 2000,
    seed: int = 42,
    threshold: float = 0.70,
) -> dict:
    """F1 macro por ventana y por grabación, con intervalos, a partir de puntajes por clase.

    ``scores`` puede ser una matriz de probabilidades o de puntajes de decisión
    (por ejemplo, de un SVM): por grabación se promedian y se elige el mayor.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    recordings = np.asarray(recording_ids).astype(str)
    rec_true, rec_scores, _ = aggregate_by_recording(y_true, scores, recordings)
    rec_pred = rec_scores.argmax(axis=1)
    return {
        "n_ventanas": int(len(y_true)),
        "n_grabaciones": int(len(rec_true)),
        "n_remuestras": n_boot,
        "por_ventana": bootstrap_f1_macro(
            y_true, scores.argmax(axis=1), groups=recordings, n_boot=n_boot, seed=seed,
            threshold=threshold,
        ),
        "por_grabacion": {
            **bootstrap_f1_macro(rec_true, rec_pred, n_boot=n_boot, seed=seed, threshold=threshold),
            "exactitud": float(np.mean(rec_true == rec_pred)),
        },
    }


def evaluate_with_intervals(
    checkpoint: str | Path,
    processed_dir: str | Path = PROCESSED_DIR,
    split: str = "test",
    split_column: str | None = None,
    n_boot: int = 2000,
    seed: int = 42,
    threshold: float = 0.70,
    output_dir: str | Path | None = None,
) -> dict:
    """Calcula F1 macro por ventana y por grabación, con intervalos de confianza."""
    from src.evaluation.evaluate import predict_split

    y_true, probabilities, index, _ = predict_split(
        checkpoint, processed_dir, split=split, split_column=split_column
    )
    recordings = index["recording_id"].astype(str).to_numpy()
    result = {
        "checkpoint": Path(checkpoint).as_posix(),
        "split": split,
        **intervals_from_scores(y_true, probabilities, recordings, n_boot, seed, threshold),
    }
    out = ensure_dir(output_dir or RESULTS_DIR)
    save_json(result, out / f"{Path(checkpoint).stem}_{split}_intervalos.json")
    return result
