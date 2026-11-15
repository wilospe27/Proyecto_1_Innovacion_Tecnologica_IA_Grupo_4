"""Pruebas de la depuración de grabaciones (archivos sintéticos, sin red)."""

import numpy as np
import pandas as pd
import soundfile as sf

from src.data.clean import (
    build_manifest,
    clean_dataset,
    mentions_voice,
    parse_also,
    summarize_manifest,
    xc_id_from_path,
)

SR = 32_000


def write_tone(path, seconds: float, freq: float = 3_000.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * SR)) / SR
    sf.write(path, (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), SR)


def make_raw(tmp_path):
    raw = tmp_path / "raw"
    write_tone(raw / "Especie_a" / "XC1.wav", 3.0)
    write_tone(raw / "Especie_a" / "XC2.wav", 3.0)  # mismo contenido que XC1 -> duplicado
    write_tone(raw / "Especie_a" / "XC3.wav", 0.2, freq=2_000.0)  # muy corto
    write_tone(raw / "Especie_a" / "XC4.wav", 3.0, freq=4_000.0)  # fondo con otra especie objetivo
    write_tone(raw / "Especie_b" / "XC5.wav", 3.0, freq=5_000.0)  # se conserva (voz solo se marca)
    (raw / "Especie_b" / "XC6.wav").write_bytes(b"esto no es audio")  # ilegible
    metadata = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "also": ["[]", "[]", "[]", "['Especie b', 'Otra especie']", "['Especie b']", "[]"],
            "rmk": ["", "", "", "", "People talking in the background", ""],
        }
    )
    return raw, metadata


def test_utilidades():
    assert xc_id_from_path("data/raw/Especie_a/XC123.mp3") == "123"
    assert parse_also("['Turdus fuscater', ' Zonotrichia  capensis ']") == ["Turdus fuscater", "Zonotrichia capensis"]
    assert parse_also(float("nan")) == [] and parse_also("[]") == [] and parse_also(None) == []
    assert parse_also(["Turdus fuscater"]) == ["Turdus fuscater"]
    assert mentions_voice("Voces de personas al fondo") and not mentions_voice("Canto al amanecer", None)


def test_reglas_del_manifiesto(tmp_path):
    raw, metadata = make_raw(tmp_path)
    manifest = build_manifest(raw, metadata, target_species=["Especie a", "Especie_b"], min_seconds=1.0)
    reasons = manifest.set_index("xc_id")["motivo"].to_dict()

    assert reasons == {"1": "", "2": "duplicado", "3": "muy_corto", "4": "fondo_objetivo", "5": "", "6": "ilegible"}
    assert manifest["conservar"].sum() == 2
    kept_b = manifest[manifest["xc_id"] == "5"].iloc[0]
    assert kept_b["posible_voz"]  # se marca para revisión, pero no se descarta
    assert abs(kept_b["duracion_s"] - 3.0) < 1e-3 and kept_b["frecuencia_muestreo"] == SR


def test_resumen_y_archivos(tmp_path):
    raw, metadata = make_raw(tmp_path)
    metadata_csv = tmp_path / "metadatos.csv"
    metadata.to_csv(metadata_csv, index=False)

    manifest = clean_dataset(
        raw,
        metadata_csv,
        manifest_csv=tmp_path / "depuracion.csv",
        summary_csv=tmp_path / "resumen.csv",
        target_species=["Especie a", "Especie b"],
    )
    summary = summarize_manifest(manifest).set_index("especie")

    assert (tmp_path / "depuracion.csv").exists() and (tmp_path / "resumen.csv").exists()
    assert summary.loc["Especie_a", ["descargadas", "duplicado", "muy_corto", "fondo_objetivo", "conservadas"]].tolist() == [4, 1, 1, 1, 1]
    assert summary.loc["Especie_b", ["descargadas", "ilegible", "conservadas", "posible_voz"]].tolist() == [2, 1, 1, 1]
    assert len(list(raw.rglob("*.wav"))) == 6  # la depuración no borra archivos
