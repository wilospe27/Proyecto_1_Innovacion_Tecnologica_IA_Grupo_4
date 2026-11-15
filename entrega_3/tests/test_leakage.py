"""Pruebas de la comparación por fecha de subida."""

import numpy as np
import pandas as pd

from src.evaluation.leakage import compare_by_upload_date, upload_dates


def test_upload_dates_usa_el_id_de_xeno_canto(tmp_path):
    csv = tmp_path / "meta.csv"
    pd.DataFrame({"id": [1, 2], "uploaded": ["2024-01-01", "2026-03-01"]}).to_csv(csv, index=False)
    dates = upload_dates(csv, ["Especie_a/XC2", "Especie_b/XC1", "Especie_b/XC9"])
    assert str(dates[0])[:10] == "2026-03-01"
    assert str(dates[1])[:10] == "2024-01-01"
    assert np.isnat(dates[2])


def test_compare_by_upload_date_separa_grupos():
    y = np.array([0, 0, 1, 1])
    perfect = np.eye(2)[y]
    wrong_after = perfect.copy()
    wrong_after[2:] = np.eye(2)[1 - y[2:]]
    ids = ["a/XC1", "a/XC1", "b/XC2", "b/XC2"]
    dates = np.array(["2024-01-01", "2024-01-01", "2026-01-01", "2026-01-01"], dtype="datetime64[ns]")

    table = compare_by_upload_date(y, {"bueno": perfect, "falla_nuevas": wrong_after}, ids, dates, "2025-01-01")

    row = table.set_index(["modelo", "grupo"])
    assert row.loc[("bueno", "desde el corte"), "exactitud_grabacion"] == 1.0
    assert row.loc[("falla_nuevas", "antes del corte"), "exactitud_ventana"] == 1.0
    assert row.loc[("falla_nuevas", "desde el corte"), "exactitud_ventana"] == 0.0
    assert row.loc[("falla_nuevas", "desde el corte"), "grabaciones"] == 1
