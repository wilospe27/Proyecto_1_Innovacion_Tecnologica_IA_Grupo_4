"""Pruebas de los modelos clásicos de referencia (características sintéticas separables)."""

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import WINDOW_SPLIT_COLUMN, assign_splits, ensure_splits
from src.data.preprocess import MFCC_FILE
from src.models.classical import MODEL_TITLES, build_classical_models, load_mfcc_dataset, run_baselines


@pytest.fixture()
def processed_dir(tmp_path):
    """Tres especies con 12 grabaciones de 4 ventanas; cada especie tiene su propio centro."""
    rng = np.random.default_rng(0)
    rows, features = [], []
    for k, label in enumerate(["Especie_a", "Especie_b", "Especie_c"]):
        for recording in range(12):
            for segment in range(4):
                rows.append(
                    {
                        "path": f"{label}/rec{recording}_{segment:03d}.npy",
                        "label": label,
                        "recording_id": f"{label}/rec{recording}",
                        "segment": segment,
                    }
                )
                features.append(rng.normal(loc=4.0 * k, scale=1.0, size=8))
    pd.DataFrame(rows).to_csv(tmp_path / "index.csv", index=False)
    np.save(tmp_path / MFCC_FILE, np.asarray(features, dtype=np.float32))
    return tmp_path


def test_modelos_disponibles():
    assert set(build_classical_models()) == set(MODEL_TITLES)


def test_baselines_superan_a_la_clase_mayoritaria(processed_dir, tmp_path):
    out = tmp_path / "resultados"
    metrics = run_baselines(processed_dir, output_dir=out)
    test = metrics[metrics["conjunto"] == "test"].set_index("modelo")

    assert set(test.index) == set(MODEL_TITLES)
    assert test.loc["mayoritaria", "f1_macro"] < 0.3
    for name in ("regresion_logistica", "svm", "bosque_aleatorio", "potenciacion_gradiente"):
        assert test.loc[name, "f1_macro"] > 0.9
    assert (out / "baselines_metricas.csv").exists()
    assert (out / "baseline_svm_test_por_especie.csv").exists()
    assert list(out.glob("baseline_*_test_matriz_confusion.png"))
    assert "split" in pd.read_csv(processed_dir / "index.csv").columns  # la partición queda guardada


def test_indice_y_caracteristicas_desalineados(processed_dir):
    np.save(processed_dir / MFCC_FILE, np.zeros((3, 8), dtype=np.float32))
    with pytest.raises(ValueError):
        load_mfcc_dataset(processed_dir)


def test_particion_por_ventana_mezcla_grabaciones(processed_dir):
    index = pd.read_csv(processed_dir / "index.csv")
    by_window = assign_splits(index, by_recording=False, column=WINDOW_SPLIT_COLUMN)
    assert set(by_window[WINDOW_SPLIT_COLUMN]) == {"train", "val", "test"}
    assert (by_window.groupby("recording_id")[WINDOW_SPLIT_COLUMN].nunique() > 1).any()

    both, created = ensure_splits(ensure_splits(index)[0], WINDOW_SPLIT_COLUMN)
    assert created and {"split", WINDOW_SPLIT_COLUMN} <= set(both.columns)
    assert (both.groupby("recording_id")["split"].nunique() == 1).all()
    assert ensure_splits(both)[1] is False  # no la vuelve a crear
