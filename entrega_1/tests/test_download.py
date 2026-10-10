"""Pruebas de las utilidades de descarga (sin acceso a la red)."""

import pytest

from src.data.download import (
    ENV_KEY,
    audio_url,
    build_query,
    get_api_key,
    records_to_dataframe,
    safe_label,
)


def test_construccion_de_consulta():
    assert build_query("Turdus fuscater", quality="a") == (
        'sp:"Turdus fuscater" cnt:"Colombia" grp:"birds" q:A'
    )
    assert build_query(None, country=None, group="frogs") == 'grp:"frogs"'


def test_consulta_vacia():
    with pytest.raises(ValueError):
        build_query(None, country=None, group=None)


def test_clave_de_api_obligatoria(monkeypatch):
    monkeypatch.delenv(ENV_KEY, raising=False)
    with pytest.raises(RuntimeError):
        get_api_key()
    monkeypatch.setenv(ENV_KEY, "clave-de-prueba")
    assert get_api_key() == "clave-de-prueba"


def test_url_y_etiqueta():
    assert audio_url({"file": "//xeno-canto.org/123/download"}).startswith("https://")
    assert audio_url({}) is None
    assert safe_label("Turdus  fuscater") == "Turdus_fuscater"


def test_registros_a_tabla():
    frame = records_to_dataframe(
        [{"id": "1", "gen": "Turdus", "sp": "fuscater", "lic": "//creativecommons.org/x"}]
    )
    assert frame.loc[0, "species"] == "Turdus fuscater"
