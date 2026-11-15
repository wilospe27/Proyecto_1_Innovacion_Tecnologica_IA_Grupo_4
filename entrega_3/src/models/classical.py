"""Modelos de referencia clásicos sobre estadísticas de MFCC (tarea 2.7).

Cada ventana de 5 s se resume en un vector de estadísticas de MFCC
(``data/processed/mfcc_stats.npy``, ver ``src/data/preprocess.py``) y se
clasifica con modelos de ``scikit-learn``:

- ``mayoritaria``: predice siempre la especie más frecuente (piso de desempeño).
- ``regresion_logistica``, ``svm`` (núcleo RBF), ``bosque_aleatorio`` y
  ``potenciacion_gradiente``.

Todos se ajustan con la partición de entrenamiento y se evalúan en validación
y prueba con las mismas métricas de la CNN, para que sean comparables.

Uso (desde ``entrega_2/``)::

    python -m src.main baselines
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import ensure_splits
from src.data.preprocess import MFCC_FILE
from src.evaluation.evaluate import compute_metrics, per_class_report, plot_confusion_matrix, top_k_accuracy
from src.utils.helpers import PROCESSED_DIR, RESULTS_DIR, ensure_dir, get_logger, save_json

MODEL_TITLES = {
    "mayoritaria": "Clase mayoritaria",
    "regresion_logistica": "Regresión logística (MFCC)",
    "svm": "SVM RBF (MFCC)",
    "bosque_aleatorio": "Bosque aleatorio (MFCC)",
    "potenciacion_gradiente": "Potenciación de gradiente (MFCC)",
}

logger = get_logger("classical")


def build_classical_models(seed: int = 42) -> dict:
    """Modelos de referencia con sus hiperparámetros (sin ajuste: son pruebas iniciales)."""
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return {
        "mayoritaria": DummyClassifier(strategy="most_frequent"),
        "regresion_logistica": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=seed),
        ),
        "svm": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", decision_function_shape="ovr", random_state=seed),
        ),
        "bosque_aleatorio": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", n_jobs=-1, random_state=seed
        ),
        "potenciacion_gradiente": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, class_weight="balanced", random_state=seed
        ),
    }


def class_scores(model, features: np.ndarray) -> np.ndarray:
    """Puntaje por clase: probabilidades si el modelo las ofrece; si no, la función de decisión."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)
    return model.decision_function(features)


def load_feature_dataset(
    processed_dir: str | Path = PROCESSED_DIR,
    index_file: str = "index.csv",
    features_file: str = MFCC_FILE,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Carga el índice de ventanas y una matriz de características alineada por fila.

    ``features_file`` es ``mfcc_stats.npy`` (preprocesamiento) o un archivo de
    *embeddings* (``embeddings_<modelo>.npy``, comando ``embeddings``).
    """
    processed_dir = Path(processed_dir)
    index = pd.read_csv(processed_dir / index_file)
    features = np.load(processed_dir / features_file)
    if len(index) != len(features):
        raise ValueError(
            f"El índice ({len(index)} filas) y {features_file} ({len(features)} filas) no coinciden: "
            "vuelva a generar las características"
        )
    return index, features


def load_mfcc_dataset(
    processed_dir: str | Path = PROCESSED_DIR,
    index_file: str = "index.csv",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Carga el índice de ventanas y la matriz de estadísticas de MFCC (alineadas por fila)."""
    return load_feature_dataset(processed_dir, index_file, MFCC_FILE)


def evaluate_classical(
    features: np.ndarray,
    index: pd.DataFrame,
    split_column: str = "split",
    models: dict | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Ajusta cada modelo con ``train`` y lo evalúa en ``val`` y ``test``.

    Retorna ``(tabla de métricas, predicciones)``; ``predicciones[modelo]`` tiene
    ``y_true``, ``y_pred``, ``scores`` (puntaje por clase), ``recording_ids`` y
    ``labels`` de la partición de prueba.
    """
    labels = sorted(str(label) for label in index["label"].unique())
    label_to_idx = {label: i for i, label in enumerate(labels)}
    y = index["label"].map(label_to_idx).to_numpy()
    masks = {name: (index[split_column] == name).to_numpy() for name in ("train", "val", "test")}
    if not all(mask.any() for mask in masks.values()):
        raise ValueError(f"La partición '{split_column}' debe tener train, val y test")

    rows, predictions = [], {}
    for name, model in (models or build_classical_models(seed)).items():
        model.fit(features[masks["train"]], y[masks["train"]])
        for split in ("val", "test"):
            probabilities = np.full((int(masks[split].sum()), len(labels)), -np.inf)
            probabilities[:, model.classes_] = class_scores(model, features[masks[split]])
            y_pred = probabilities.argmax(axis=1)
            rows.append(
                {
                    "modelo": name,
                    "particion": split_column,
                    "conjunto": split,
                    "n_ventanas": int(masks[split].sum()),
                    **compute_metrics(y[masks[split]], y_pred),
                    "top3_accuracy": top_k_accuracy(y[masks[split]], probabilities, k=3),
                }
            )
            if split == "test":
                predictions[name] = {
                    "y_true": y[masks[split]],
                    "y_pred": y_pred,
                    "scores": probabilities,
                    "recording_ids": index.loc[masks[split], "recording_id"].astype(str).to_numpy(),
                    "labels": labels,
                }
        logger.info("%s | F1 macro en prueba: %.3f", name, rows[-1]["f1_macro"])
    return pd.DataFrame(rows), predictions


def run_baselines(
    processed_dir: str | Path = PROCESSED_DIR,
    index_file: str = "index.csv",
    split_column: str = "split",
    output_dir: str | Path | None = None,
    seed: int = 42,
    features_file: str = MFCC_FILE,
    model_names: list[str] | None = None,
    prefix: str = "baseline",
    intervals: bool = True,
) -> pd.DataFrame:
    """Evalúa modelos clásicos y guarda métricas, reportes por especie, matrices e intervalos.

    Por defecto usa las estadísticas de MFCC y todos los modelos; con
    ``features_file`` y ``model_names`` sirve también para entrenar clasificadores
    sobre *embeddings* (``prefix`` distingue los archivos de salida). Si falta la
    columna de partición la crea y la guarda en el índice, de modo que todos los
    modelos usen exactamente la misma partición.
    """
    from src.evaluation.uncertainty import intervals_from_scores

    processed_dir = Path(processed_dir)
    index, features = load_feature_dataset(processed_dir, index_file, features_file)
    index, created = ensure_splits(index, split_column, seed=seed)
    if created:
        index.to_csv(processed_dir / index_file, index=False)
        logger.info("Partición '%s' guardada en el índice", split_column)

    models = build_classical_models(seed)
    if model_names is not None:
        unknown = set(model_names) - set(models)
        if unknown:
            raise ValueError(f"Modelos desconocidos: {sorted(unknown)}; opciones: {sorted(models)}")
        models = {name: models[name] for name in model_names}

    metrics, predictions = evaluate_classical(features, index, split_column, models=models, seed=seed)
    out = ensure_dir(output_dir or RESULTS_DIR)
    suffix = "" if split_column == "split" else f"_{split_column}"
    metrics_name = "baselines_metricas" if prefix == "baseline" else f"{prefix}_metricas"
    metrics.to_csv(out / f"{metrics_name}{suffix}.csv", index=False)
    for name, result in predictions.items():
        stem = f"{prefix}_{name}{suffix}_test"
        per_class_report(result["y_true"], result["y_pred"], result["labels"]).to_csv(
            out / f"{stem}_por_especie.csv"
        )
        if intervals and name != "mayoritaria":
            save_json(
                {
                    "modelo": name,
                    "caracteristicas": features_file,
                    "particion": split_column,
                    **intervals_from_scores(result["y_true"], result["scores"], result["recording_ids"], seed=seed),
                },
                out / f"{stem}_intervalos.json",
            )
    best = metrics[metrics["conjunto"] == "val"].sort_values("f1_macro", ascending=False).iloc[0]["modelo"]
    plot_confusion_matrix(
        predictions[best]["y_true"],
        predictions[best]["y_pred"],
        predictions[best]["labels"],
        out / f"{prefix}_{best}{suffix}_test_matriz_confusion.png",
    )
    return metrics
