"""Descarga de metadatos y grabaciones desde xeno-canto (API v3).

La API v3 de xeno-canto exige una clave personal y gratuita, que se obtiene en
la página de la cuenta (https://xeno-canto.org/account). Defínala en la
variable de entorno ``XC_API_KEY``; nunca la escriba en el código ni la suba
al repositorio (ver ``docs/instalacion.md``).

Ejemplo desde la carpeta ``entrega_1/``::

    python -m src.main download --species "Turdus fuscater" "Pitangus sulphuratus" --max-per-species 30

Las grabaciones de xeno-canto se publican con licencias Creative Commons: el
CSV de metadatos conserva la licencia (``lic``) y el autor (``rec``) de cada
archivo para poder atribuirlos correctamente.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import requests

from src.utils.helpers import ensure_dir, get_logger

API_URL = "https://xeno-canto.org/api/3/recordings"
ENV_KEY = "XC_API_KEY"
TIMEOUT_SECONDS = 30

logger = get_logger("download")


def get_api_key(key: str | None = None) -> str:
    """Retorna la clave recibida o la de la variable de entorno ``XC_API_KEY``."""
    key = key or os.environ.get(ENV_KEY)
    if not key:
        raise RuntimeError(
            f"No se encontró la clave de la API de xeno-canto. Defina la variable de "
            f"entorno {ENV_KEY} (ver docs/instalacion.md)."
        )
    return key


def build_query(
    species: str | None = None,
    country: str | None = "Colombia",
    group: str | None = "birds",
    quality: str | None = None,
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
    response.raise_for_status()
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
    logger.info("Consulta %s -> %d registros", query, len(records))
    return records


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


def records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Convierte los registros en una tabla y agrega la columna ``species``."""
    frame = pd.json_normalize(records)
    if not frame.empty:
        frame["species"] = [species_name(record) for record in records]
    return frame


def download_recordings(
    records: list[dict],
    out_dir: str | Path,
    max_per_species: int | None = None,
    pause_seconds: float = 1.0,
    overwrite: bool = False,
) -> list[Path]:
    """Descarga los audios en ``out_dir/<Genero_especie>/XC<id>.<ext>``."""
    out_dir = Path(out_dir)
    saved: list[Path] = []
    per_species: dict[str, int] = {}
    with requests.Session() as session:
        for record in records:
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
    quality: str | None = None,
    max_per_species: int | None = 30,
    key: str | None = None,
) -> pd.DataFrame:
    """Busca cada especie, guarda los metadatos en CSV y descarga los audios."""
    all_records: list[dict] = []
    for name in species_list:
        query = build_query(name, country=country, group=group, quality=quality)
        all_records.extend(search_recordings(query, key=key))

    metadata = records_to_dataframe(all_records)
    metadata_csv = Path(metadata_csv)
    ensure_dir(metadata_csv.parent)
    metadata.to_csv(metadata_csv, index=False)
    logger.info("Metadatos guardados en %s (%d filas)", metadata_csv, len(metadata))

    download_recordings(all_records, raw_dir, max_per_species=max_per_species)
    return metadata
