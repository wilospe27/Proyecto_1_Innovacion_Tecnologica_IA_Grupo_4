"""Prueba de humo del flujo completo: entrenamiento -> evaluación -> predicción.

Usa espectrogramas aleatorios pequeños (no mide desempeño, solo que el
pipeline funcione de extremo a extremo y genere sus archivos de salida).
"""

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.data.dataset import assign_splits
from src.evaluation.evaluate import evaluate_checkpoint
from src.inference.predict import predict_file
from src.training.train import TrainConfig, train


@pytest.fixture()
def processed_dir(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for label in ["Especie_a", "Especie_b", "Especie_c"]:
        (tmp_path / label).mkdir()
        for recording in range(7):
            for segment in range(2):
                path = f"{label}/rec{recording}_{segment:03d}.npy"
                np.save(tmp_path / path, rng.standard_normal((32, 64)).astype(np.float32))
                rows.append(
                    {
                        "path": path,
                        "label": label,
                        "recording_id": f"{label}/rec{recording}",
                        "segment": segment,
                    }
                )
    pd.DataFrame(rows).to_csv(tmp_path / "index.csv", index=False)
    return tmp_path


def test_particion_no_mezcla_grabaciones(processed_dir):
    index = assign_splits(pd.read_csv(processed_dir / "index.csv"))
    assert set(index["split"]) == {"train", "val", "test"}
    assert (index.groupby("recording_id")["split"].nunique() == 1).all()


def test_entrenar_evaluar_y_predecir(processed_dir, tmp_path):
    output_dir = tmp_path / "experimentos"
    summary = train(
        TrainConfig(
            processed_dir=str(processed_dir),
            output_dir=str(output_dir),
            epochs=2,
            batch_size=8,
            run_name="prueba",
        )
    )
    checkpoint = output_dir / "checkpoints" / "prueba.pt"
    assert checkpoint.exists()
    assert (output_dir / "logs" / "prueba_history.csv").exists()
    assert summary["epochs_run"] >= 1

    metrics = evaluate_checkpoint(checkpoint, processed_dir=processed_dir, output_dir=output_dir)
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert (output_dir / "prueba_test_matriz_confusion.png").exists()

    audio = tmp_path / "audio.wav"
    t = np.arange(32_000 * 6) / 32_000
    sf.write(audio, (0.3 * np.sin(2 * np.pi * 3_000 * t)).astype(np.float32), 32_000)
    prediction = predict_file(audio, checkpoint, top_k=2, device="cpu")
    assert len(prediction) == 2
    assert prediction["probabilidad"].between(0, 1).all()
