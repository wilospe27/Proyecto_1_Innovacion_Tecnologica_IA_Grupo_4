"""Metadatos, ranking de especies y descarga de grabaciones desde xeno-canto (API v3).

La API v3 de xeno-canto exige una clave personal y gratuita, que se obtiene en
la página de la cuenta (https://xeno-canto.org/account). Se lee, en este orden,
de la variable de entorno ``XC_API_KEY`` o de un archivo ``.env`` en la carpeta
``entrega_2/`` con la línea ``XC_API_KEY=su_clave``. El archivo ``.env`` está
excluido de Git: nunca escriba la clave en el código ni en los notebooks.

Flujo de la Etapa 2 (desde la carpeta ``entrega_2/``)::

    # 1. Ranking de especies de la región (solo metadatos, sin descargar audio)
    python -m src.main especies --region valle_del_cauca

    # 2. Descarga de las especies elegidas por el equipo (grabaciones de toda Colombia)
    python -m src.main download --species "Turdus ignobilis" "Pitangus sulphuratus" --max-per-species 60

Las grabaciones de xeno-canto se publican con licencias Creative Commons: los
CSV de metadatos conservan la licencia (``lic``) y el autor (``rec``) de cada
archivo para poder atribuirlos correctamente.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import requests

from src.utils.helpers import PROJECT_ROOT, ensure_dir, get_logger

API_URL = "https://xeno-canto.org/api/3/recordings"
ENV_KEY = "XC_API_KEY"
TIMEOUT_SECONDS = 30
QUALITY_ORDER = ["A", "B", "C", "D", "E"]

# Rectángulos (lat_min, lon_min, lat_max, lon_max). El del Valle del Cauca usa sus
# coordenadas extremas (3°05'–5°01' N, 75°42'–77°33' O) e incluye franjas de los
# departamentos vecinos; para quedarse solo con el Valle use --filtrar-localidad.
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "valle_del_cauca": (3.08, -77.55, 5.02, -75.70),
}

logger = get_logger("download")


# ----------------------------------------------------------------------------- clave
def _read_dotenv(path: Path) -> dict[str, str]:
    """Lee pares ``CLAVE=valor`` de un archivo ``.env`` (ignora comentarios)."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def get_api_key(key: str | None = None, dotenv_path: str | Path | None = None) -> str:
    """Clave de la API: argumento, variable de entorno o archivo ``.env``."""
    key = key or os.environ.get(ENV_KEY)
    if not key:
        key = _read_dotenv(Path(dotenv_path or PROJECT_ROOT / ".env")).get(ENV_KEY)
    if not key:
        raise RuntimeError(
            f"No se encontró la clave de la API de xeno-canto. Defina la variable de entorno "
            f"{ENV_KEY} o cree el archivo .env con la línea {ENV_KEY}=su_clave "
            "(ver docs/instalacion.md)."
        )
    return key


# ----------------------------------------------------------------------------- consultas
def build_query(
    species: str | None = None,
    country: str | None = "Colombia",
    group: str | None = "birds",
    quality: str | None = None,
    box: tuple[float, float, float, float] | None = None,
    extra: Iterable[str] = (),
) -> str:
    """Construye una consulta con las etiquetas de búsqueda de xeno-canto.

    >>> build_query("Turdus fuscater", quality="A")
    'sp:"Turdus fuscater" cnt:"Colombia" grp:"birds" q:A'
    """
    parts = []
    if species:
        parts.append(f'sp:"{species.strip()}"')
    if country:
        parts.append(f'cnt:"{country}"')
    if group:
        parts.append(f'grp:"{group}"')
    if quality:
        parts.append(f"q:{quality.upper()}")
    if box:
        lat_min, lon_min, lat_max, lon_max = box
        if not (lat_min < lat_max and lon_min < lon_max):
            raise ValueError("El rectángulo debe ser (lat_min, lon_min, lat_max, lon_max)")
        parts.append(f"box:{lat_min:g},{lon_min:g},{lat_max:g},{lon_max:g}")
    parts.extend(extra)
    if not parts:
        raise ValueError("La consulta está vacía")
    return " ".join(parts)


def fetch_page(
    query: str,
    key: str,
    page: int = 1,
    session: requests.Session | None = None,
) -> dict:
    """Consulta una página de resultados de la API."""
    http = session or requests
    response = http.get(
        API_URL,
        params={"query": query, "key": key, "page": page},
        timeout=TIMEOUT_SECONDS,
    )
    if not response.ok:
        raise RuntimeError(
            f"La API de xeno-canto respondió {response.status_code} para la consulta "
            f"{query!r}: {response.text[:300]}"
        )
    return response.json()


def search_recordings(
    query: str,
    key: str | None = None,
    max_pages: int | None = None,
    pause_seconds: float = 1.0,
) -> list[dict]:
    """Retorna los registros (diccionarios) de todas las páginas de la consulta."""
    key = get_api_key(key)
    with requests.Session() as session:
        first = fetch_page(query, key, page=1, session=session)
        records = list(first.get("recordings", []))
        num_pages = int(first.get("numPages", 1) or 1)
        last_page = num_pages if max_pages is None else min(num_pages, max_pages)
        for page in range(2, last_page + 1):
            time.sleep(pause_seconds)  # respeta el servidor
            records.extend(fetch_page(query, key, page=page, session=session).get("recordings", []))
    logger.info("Consulta %s -> %d registros (%d páginas)", query, len(records), last_page)
    return records


# ----------------------------------------------------------------------------- utilidades
def species_name(record: dict) -> str:
    """Nombre científico ``Genero especie`` de un registro."""
    return f"{str(record.get('gen', '')).strip()} {str(record.get('sp', '')).strip()}".strip()


def safe_label(name: str) -> str:
    """Convierte un nombre científico en un nombre de carpeta: ``Turdus_fuscater``."""
    return "_".join(name.split())


def audio_url(record: dict) -> str | None:
    """URL absoluta del archivo de audio de un registro (si existe)."""
    url = record.get("file")
    if not url:
        return None
    return f"https:{url}" if url.startswith("//") else url


def parse_length(value) -> float:
    """Duración ``m:ss`` o ``h:mm:ss`` de xeno-canto en segundos (NaN si no se puede leer)."""
    try:
        parts = [float(p) for p in str(value).strip().split(":")]
    except ValueError:
        return float("nan")
    if not parts or len(parts) > 3:
        return float("nan")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def quality_rank(quality) -> int:
    """Posición de la calidad (A=0 ... E=4); las grabaciones sin calidad van al final."""
    quality = str(quality).strip().upper()
    return QUALITY_ORDER.index(quality) if quality in QUALITY_ORDER else len(QUALITY_ORDER)


def records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Convierte los registros en una tabla y agrega la columna ``species``."""
    frame = pd.json_normalize(records)
    if not frame.empty:
        frame["species"] = [species_name(record) for record in records]
    return frame


# ----------------------------------------------------------------------------- ranking
def rank_species(
    metadata: pd.DataFrame,
    qualities: Iterable[str] = ("A", "B"),
    min_recordings: int = 30,
    loc_contains: str | None = None,
) -> pd.DataFrame:
    """Resume los metadatos por especie para elegir las especies objetivo.

    Columnas: ``especie``, ``nombre_en``, ``grabaciones`` (con las calidades
    indicadas), ``grabaciones_total``, ``grabadores`` (autores distintos),
    ``minutos`` (de las grabaciones de buena calidad), ``con_otras_especies``
    (grabaciones que mencionan especies de fondo: etiquetas débiles) y
    ``cumple`` (``grabaciones >= min_recordings``). Ordenado de mayor a menor.
    """
    columns = [
        "especie", "nombre_en", "grabaciones", "grabaciones_total", "grabadores",
        "minutos", "con_otras_especies", "cumple",
    ]
    if metadata.empty:
        return pd.DataFrame(columns=columns)

    frame = metadata.copy()
    if "species" not in frame.columns:
        if not {"gen", "sp"}.issubset(frame.columns):
            raise ValueError("Los metadatos deben tener la columna 'species' o las columnas 'gen' y 'sp'")
        frame["species"] = (
            frame["gen"].astype(str).str.strip() + " " + frame["sp"].astype(str).str.strip()
        ).str.strip()
    if loc_contains:
        frame = frame[frame.get("loc", pd.Series("", index=frame.index)).astype(str).str.contains(loc_contains, case=False, na=False)]

    wanted = {q.upper() for q in qualities}
    quality = frame.get("q", pd.Series("", index=frame.index)).astype(str).str.strip().str.upper()
    frame["_buena"] = quality.isin(wanted)
    frame["_segundos"] = frame.get("length", pd.Series(None, index=frame.index)).map(parse_length)
    also = frame.get("also", pd.Series(None, index=frame.index))
    frame["_otras"] = also.map(lambda v: bool(v) and str(v) not in ("[]", "nan", "None", ""))
    frame["_rec"] = frame.get("rec", pd.Series("", index=frame.index)).astype(str)
    frame["_en"] = frame.get("en", pd.Series("", index=frame.index)).astype(str)

    good = frame[frame["_buena"]]
    summary = pd.DataFrame(
        {
            "nombre_en": frame.groupby("species")["_en"].first(),
            "grabaciones": good.groupby("species").size(),
            "grabaciones_total": frame.groupby("species").size(),
            "grabadores": good.groupby("species")["_rec"].nunique(),
            "minutos": good.groupby("species")["_segundos"].sum() / 60.0,
            "con_otras_especies": good.groupby("species")["_otras"].sum(),
        }
    ).fillna({"grabaciones": 0, "grabadores": 0, "minutos": 0.0, "con_otras_especies": 0})
    summary = summary[summary.index.astype(str).str.len() > 0]
    for column in ("grabaciones", "grabaciones_total", "grabadores", "con_otras_especies"):
        summary[column] = summary[column].astype(int)
    summary["minutos"] = summary["minutos"].round(1)
    summary["cumple"] = summary["grabaciones"] >= min_recordings
    summary = summary.rename_axis("especie").reset_index()
    return summary.sort_values(["grabaciones", "grabadores"], ascending=False, ignore_index=True)[columns]


def species_ranking(
    region: str | None = "valle_del_cauca",
    box: tuple[float, float, float, float] | None = None,
    country: str | None = "Colombia",
    group: str | None = "birds",
    qualities: Iterable[str] = ("A", "B"),
    min_recordings: int = 30,
    loc_contains: str | None = None,
    metadata_csv: str | Path | None = None,
    ranking_csv: str | Path | None = None,
    max_pages: int | None = None,
    key: str | None = None,
) -> pd.DataFrame:
    """Consulta los metadatos de una región (sin descargar audio) y rankea las especies."""
    qualities = tuple(qualities)
    if box is None and region is not None:
        try:
            box = REGIONS[region]
        except KeyError:
            raise ValueError(f"Región desconocida '{region}'. Opciones: {sorted(REGIONS)}") from None
    query = build_query(country=country, group=group, box=box)
    metadata = records_to_dataframe(search_recordings(query, key=key, max_pages=max_pages))
    if metadata_csv is not None:
        ensure_dir(Path(metadata_csv).parent)
        metadata.to_csv(metadata_csv, index=False)
    ranking = rank_species(metadata, qualities=qualities, min_recordings=min_recordings, loc_contains=loc_contains)
    if ranking_csv is not None:
        ensure_dir(Path(ranking_csv).parent)
        ranking.to_csv(ranking_csv, index=False)
    logger.info(
        "%d especies en la región; %d cumplen con al menos %d grabaciones de calidad %s",
        len(ranking),
        int(ranking["cumple"].sum()) if not ranking.empty else 0,
        min_recordings,
        "/".join(qualities),
    )
    return ranking


# ----------------------------------------------------------------------------- descarga
def download_recordings(
    records: list[dict],
    out_dir: str | Path,
    max_per_species: int | None = None,
    qualities: Iterable[str] | None = ("A", "B"),
    pause_seconds: float = 1.0,
    overwrite: bool = False,
) -> list[Path]:
    """Descarga los audios en ``out_dir/<Genero_especie>/XC<id>.<ext>``.

    Solo descarga las calidades indicadas (``None`` = todas) y prioriza las de
    mejor calidad cuando hay un máximo por especie.
    """
    out_dir = Path(out_dir)
    wanted = {q.upper() for q in qualities} if qualities else None
    candidates = [r for r in records if wanted is None or str(r.get("q", "")).strip().upper() in wanted]
    candidates.sort(key=lambda r: quality_rank(r.get("q")))

    saved: list[Path] = []
    per_species: dict[str, int] = {}
    with requests.Session() as session:
        for record in candidates:
            name, url = species_name(record), audio_url(record)
            if not name or not url:
                continue
            if max_per_species is not None and per_species.get(name, 0) >= max_per_species:
                continue
            extension = Path(str(record.get("file-name") or "audio.mp3")).suffix.lower() or ".mp3"
            target = out_dir / safe_label(name) / f"XC{record['id']}{extension}"
            per_species[name] = per_species.get(name, 0) + 1
            if target.exists() and not overwrite:
                saved.append(target)
                continue
            try:
                response = session.get(url, timeout=TIMEOUT_SECONDS)
                response.raise_for_status()
            except requests.RequestException as exc:
                logger.warning("Falló la descarga de XC%s: %s", record.get("id"), exc)
                per_species[name] -= 1
                continue
            ensure_dir(target.parent)
            target.write_bytes(response.content)
            saved.append(target)
            time.sleep(pause_seconds)
    logger.info("%d archivos disponibles en %s", len(saved), out_dir)
    return saved


def download_species(
    species_list: Iterable[str],
    raw_dir: str | Path,
    metadata_csv: str | Path,
    country: str | None = "Colombia",
    group: str | None = "birds",
    qualities: Iterable[str] | None = ("A", "B"),
    max_per_species: int | None = 60,
    key: str | None = None,
) -> pd.DataFrame:
    """Busca cada especie, guarda los metadatos en CSV y descarga los audios."""
    all_records: list[dict] = []
    for name in species_list:
        query = build_query(name, country=country, group=group)
        all_records.extend(search_recordings(query, key=key))

    metadata = records_to_dataframe(all_records)
    metadata_csv = Path(metadata_csv)
    ensure_dir(metadata_csv.parent)
    metadata.to_csv(metadata_csv, index=False)
    logger.info("Metadatos guardados en %s (%d filas)", metadata_csv, len(metadata))

    download_recordings(all_records, raw_dir, max_per_species=max_per_species, qualities=qualities)
    return metadata
