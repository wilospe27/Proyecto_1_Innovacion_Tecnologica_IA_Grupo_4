"""Verificación de fuga en modelos preentrenados con datos de xeno-canto.

Perch 2.0 se entrenó con grabaciones de xeno-canto, así que algunas de nuestras
grabaciones de prueba pudieron estar en su entrenamiento. Si su ventaja viniera
de "recordarlas", desaparecería en las grabaciones subidas **después** de su
entrenamiento, que no pudo ver. Comparamos el desempeño antes y después de una
fecha de corte, junto con un modelo que no tiene ese riesgo (la CNN base).

Uso (desde ``entrega_3/``)::

    python -m src.main fuga --corte 2025-08-01
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.uncertainty import aggregate_by_recording
from src.utils.helpers import METADATA_DIR, PROCESSED_DIR, RESULTS_DIR, ensure_dir


def upload_dates(metadata_csv: str | Path, recording_ids) -> np.ndarray:
    """Fecha de subida a xeno-canto de cada ``recording_id`` (``Especie/XC123``); ``NaT`` si falta."""
    meta = pd.read_csv(metadata_csv, usecols=["id", "uploaded"], dtype=str)
    uploaded = pd.Series(pd.to_datetime(meta["uploaded"], errors="coerce").to_numpy(), index="XC" + meta["id"])
    uploaded = uploaded[~uploaded.index.duplicated()]
    names = [Path(str(r)).name for r in recording_ids]
    return uploaded.reindex(names).to_numpy(dtype="datetime64[ns]")


def compare_by_upload_date(
    y_true, scores_by_model: dict[str, np.ndarray], recording_ids, dates, cutoff: str
) -> pd.DataFrame:
    """Exactitud por ventana y por grabación, antes y después de ``cutoff``, para cada modelo."""
    y_true = np.asarray(y_true)
    recording_ids = np.asarray(recording_ids).astype(str)
    after = np.asarray(dates) >= np.datetime64(pd.Timestamp(cutoff))
    rows = []
    for model, scores in scores_by_model.items():
        scores = np.asarray(scores)
        for group, mask in (("antes del corte", ~after), ("desde el corte", after)):
            if not mask.any():
                continue
            rec_true, rec_scores, _ = aggregate_by_recording(y_true[mask], scores[mask], recording_ids[mask])
            rows.append(
                {
                    "modelo": model,
                    "grupo": group,
                    "corte": cutoff,
                    "grabaciones": int(len(rec_true)),
                    "ventanas": int(mask.sum()),
                    "exactitud_ventana": float(np.mean(scores[mask].argmax(axis=1) == y_true[mask])),
                    "exactitud_grabacion": float(np.mean(rec_scores.argmax(axis=1) == rec_true)),
                }
            )
    return pd.DataFrame(rows)


def run_leakage_check(
    cutoff: str = "2025-08-01",
    checkpoint: str | Path = "experiments/checkpoints/cnn_base.pt",
    embeddings: str = "embeddings_perch_v2.npy",
    processed_dir: str | Path = PROCESSED_DIR,
    metadata_csv: str | Path = METADATA_DIR / "xeno_canto.csv",
    seed: int = 42,
) -> pd.DataFrame:
    """Compara Perch 2.0 + regresión logística y la CNN base antes y después de ``cutoff``."""
    from src.evaluation.evaluate import predict_split
    from src.models.classical import build_classical_models, load_feature_dataset

    index, features = load_feature_dataset(processed_dir, features_file=embeddings)
    labels = sorted(str(label) for label in index["label"].unique())
    y = index["label"].map({label: i for i, label in enumerate(labels)}).to_numpy()
    train, test = (index["split"] == "train").to_numpy(), (index["split"] == "test").to_numpy()
    probe = build_classical_models(seed)["regresion_logistica"].fit(features[train], y[train])

    y_cnn, p_cnn, cnn_index, cnn_labels = predict_split(checkpoint, processed_dir)
    test_ids = index.loc[test, "recording_id"].astype(str).to_numpy()
    if cnn_labels != labels or not np.array_equal(cnn_index["recording_id"].astype(str).to_numpy(), test_ids):
        raise ValueError("La CNN y los embeddings no se evaluaron sobre las mismas ventanas")

    table = compare_by_upload_date(
        y[test],
        {"Perch 2.0 + regresión logística": probe.predict_proba(features[test]), "CNN base": p_cnn},
        test_ids,
        upload_dates(metadata_csv, test_ids),
        cutoff,
    )
    table.to_csv(ensure_dir(RESULTS_DIR) / f"fuga_perch_corte_{cutoff}.csv", index=False)
    return table
