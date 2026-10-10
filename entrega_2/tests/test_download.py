"""Pruebas de consultas, ranking de especies y descarga (sin acceso a la red)."""

import math

import pandas as pd
import pytest

from src.data import download as dl
from src.data.download import (
    ENV_KEY,
    audio_url,
    build_query,
    get_api_key,
    parse_length,
    quality_rank,
    rank_species,
    records_to_dataframe,
    safe_label,
    species_ranking,
)
from src.main import read_species_list


def fake_records():
    """Registros con la forma de la API: 3 buenas de especie a, 1 C de a, 1 A de b."""
    base = {"lic": "//creativecommons.org/licenses/by-nc-sa/4.0/", "loc": "Dapa, Yumbo, Valle del Cauca"}
    return [
        {**base, "id": "1", "gen": "Turdus", "sp": "ignobilis", "en": "Black-billed Thrush", "q": "A", "rec": "Ana", "length": "1:00", "also": ["Pitangus sulphuratus"], "file": "//xeno-canto.org/1/download", "file-name": "XC1-a.mp3"},
        {**base, "id": "2", "gen": "Turdus", "sp": "ignobilis", "en": "Black-billed Thrush", "q": "B", "rec": "Ana", "length": "0:30", "also": [], "file": "//xeno-canto.org/2/download", "file-name": "XC2-a.mp3"},
        {**base, "id": "3", "gen": "Turdus", "sp": "ignobilis", "en": "Black-billed Thrush", "q": "A", "rec": "Luis", "length": "0:30", "also": [], "file": "//xeno-canto.org/3/download", "file-name": "XC3-a.mp3"},
        {**base, "id": "4", "gen": "Turdus", "sp": "ignobilis", "en": "Black-billed Thrush", "q": "C", "rec": "Eva", "length": "2:00", "also": [], "file": "//xeno-canto.org/4/download", "file-name": "XC4-a.mp3"},
        {**base, "id": "5", "gen": "Pitangus", "sp": "sulphuratus", "en": "Great Kiskadee", "q": "A", "rec": "Eva", "length": "0:45", "also": [], "loc": "Popayán, Cauca", "file": "//xeno-canto.org/5/download", "file-name": "XC5-b.wav"},
    ]


def test_construccion_de_consulta():
    assert build_query("Turdus fuscater", quality="a") == 'sp:"Turdus fuscater" cnt:"Colombia" grp:"birds" q:A'
    assert build_query(None, country=None, group="frogs") == 'grp:"frogs"'
    assert build_query(box=(3.08, -77.55, 5.02, -75.7)) == 'cnt:"Colombia" grp:"birds" box:3.08,-77.55,5.02,-75.7'


def test_consulta_invalida():
    with pytest.raises(ValueError):
        build_query(None, country=None, group=None)
    with pytest.raises(ValueError):
        build_query(box=(5.0, -75.0, 3.0, -77.0))  # mínimos y máximos invertidos


def test_clave_desde_entorno_o_archivo_env(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_KEY, raising=False)
    with pytest.raises(RuntimeError):
        get_api_key(dotenv_path=tmp_path / ".env")
    (tmp_path / ".env").write_text("# clave personal\nXC_API_KEY = 'clave-de-archivo'\n", encoding="utf-8")
    assert get_api_key(dotenv_path=tmp_path / ".env") == "clave-de-archivo"
    monkeypatch.setenv(ENV_KEY, "clave-de-entorno")
    assert get_api_key(dotenv_path=tmp_path / ".env") == "clave-de-entorno"


def test_utilidades():
    assert audio_url({"file": "//xeno-canto.org/123/download"}).startswith("https://")
    assert audio_url({}) is None
    assert safe_label("Turdus  fuscater") == "Turdus_fuscater"
    assert parse_length("0:35") == 35
    assert parse_length("1:02:03") == 3723
    assert math.isnan(parse_length("sin dato"))
    assert quality_rank("a") == 0 and quality_rank("E") == 4 and quality_rank("no score") == 5


def test_registros_a_tabla():
    frame = records_to_dataframe(fake_records())
    assert frame.loc[0, "species"] == "Turdus ignobilis"


def test_ranking_de_especies():
    ranking = rank_species(records_to_dataframe(fake_records()), min_recordings=2)
    first = ranking.iloc[0]
    assert first["especie"] == "Turdus ignobilis"
    assert first["grabaciones"] == 3  # la grabación de calidad C no cuenta
    assert first["grabaciones_total"] == 4
    assert first["grabadores"] == 2
    assert first["minutos"] == 2.0
    assert first["con_otras_especies"] == 1
    assert bool(first["cumple"]) and not bool(ranking.iloc[1]["cumple"])


def test_ranking_filtrado_por_localidad():
    ranking = rank_species(records_to_dataframe(fake_records()), loc_contains="valle del cauca")
    assert list(ranking["especie"]) == ["Turdus ignobilis"]


def test_ranking_de_region_sin_red(monkeypatch, tmp_path):
    seen = {}

    def fake_search(query, key=None, max_pages=None):
        seen["query"] = query
        return fake_records()

    monkeypatch.setattr(dl, "search_recordings", fake_search)
    ranking = species_ranking(
        region="valle_del_cauca",
        min_recordings=2,
        metadata_csv=tmp_path / "meta.csv",
        ranking_csv=tmp_path / "ranking.csv",
    )
    assert "box:3.08,-77.55,5.02,-75.7" in seen["query"]
    assert (tmp_path / "meta.csv").exists() and (tmp_path / "ranking.csv").exists()
    assert pd.read_csv(tmp_path / "ranking.csv")["cumple"].sum() == 1
    with pytest.raises(ValueError):
        species_ranking(region="no_existe")


def test_descarga_filtra_y_prioriza_calidad(monkeypatch, tmp_path):
    requested = []

    class FakeResponse:
        content = b"audio"

        def raise_for_status(self):
            return None

    def fake_get(self, url, timeout=None):
        requested.append(url)
        return FakeResponse()

    monkeypatch.setattr(dl.requests.Session, "get", fake_get)
    saved = dl.download_recordings(fake_records(), tmp_path, max_per_species=2, pause_seconds=0)
    names = sorted(path.name for path in saved)
    assert names == ["XC1.mp3", "XC3.mp3", "XC5.wav"]  # 2 de calidad A para Turdus; nada de calidad C
    assert all("xeno-canto.org/4/" not in url for url in requested)


def test_lista_de_especies(tmp_path):
    path = tmp_path / "especies.txt"
    path.write_text("# especies objetivo\nTurdus_ignobilis\n\nPitangus sulphuratus  # común en Cali\n", encoding="utf-8")
    assert read_species_list(path) == ["Turdus ignobilis", "Pitangus sulphuratus"]
