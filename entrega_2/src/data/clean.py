"""Depuración de las grabaciones descargadas (tarea 2.3).

Revisa cada archivo de ``data/raw`` y decide si se conserva, **sin borrar
nada**: el resultado es un manifiesto (CSV) que el preprocesamiento respeta.

Reglas, en orden (la primera que se cumple define el motivo del descarte):

1. ``ilegible``: el archivo no se puede abrir como audio.
2. ``duplicado``: su contenido (MD5) es idéntico al de un archivo anterior.
3. ``muy_corto``: dura menos de ``min_seconds``.
4. ``fondo_objetivo``: los metadatos (campo ``also``) indican que al fondo canta
   otra de las especies objetivo, de modo que la etiqueta sería ambigua.

Además marca, sin descartar, las grabaciones cuyas observaciones mencionan
voces humanas (``posible_voz``) para revisarlas a oído.

Uso (desde ``entrega_2/``)::

    python -m src.main depurar
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from src.data.preprocess import list_audio_files
from src.utils.helpers import ensure_dir, get_logger

MANIFEST_COLUMNS = [
    "path", "label", "xc_id", "bytes", "duracion_s", "frecuencia_muestreo", "canales",
    "md5", "posible_voz", "conservar", "motivo",
]
REASONS = ["ilegible", "duplicado", "muy_corto", "fondo_objetivo"]
_VOICE_PATTERN = re.compile(
    r"\b(voice|voices|talking|speech|people|human|voz|voces|hablando|personas)\b", re.IGNORECASE
)

logger = get_logger("clean")


def file_md5(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """MD5 del contenido de un archivo (para detectar duplicados exactos)."""
    digest = hashlib.md5()  # noqa: S324 - huella de contenido, no uso criptográfico
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def xc_id_from_path(path: str | Path) -> str:
    """Identificador de xeno-canto a partir del nombre ``XC<id>.<ext>`` (o el nombre sin extensión)."""
    stem = Path(path).stem
    return stem[2:] if stem.upper().startswith("XC") else stem


def inspect_audio(path: str | Path) -> dict:
    """Lee la cabecera del audio: duración, frecuencia de muestreo y canales.

    Si el archivo no se puede abrir, retorna ``error`` con el mensaje.
    """
    import soundfile as sf

    info: dict = {"duracion_s": float("nan"), "frecuencia_muestreo": None, "canales": None, "error": None}
    try:
        header = sf.info(str(path))
        info.update(
            duracion_s=float(header.frames) / float(header.samplerate),
            frecuencia_muestreo=int(header.samplerate),
            canales=int(header.channels),
        )
    except Exception as exc:  # cabecera corrupta o formato no soportado
        info["error"] = str(exc)
    return info


def parse_also(value) -> list[str]:
    """Especies de fondo (campo ``also``) como lista de nombres científicos."""
    if isinstance(value, (list, tuple)):
        items = list(value)
    elif value is None or (isinstance(value, float) and pd.isna(value)):
        items = []
    else:
        text = str(value).strip()
        if text in ("", "[]", "nan", "None"):
            items = []
        else:
            try:
                parsed = ast.literal_eval(text)
                items = list(parsed) if isinstance(parsed, (list, tuple)) else [text]
            except (ValueError, SyntaxError):
                items = text.split(",")
    return [" ".join(str(item).split()) for item in items if str(item).strip()]


def mentions_voice(*texts) -> bool:
    """``True`` si alguna observación menciona voces humanas."""
    return any(
        bool(_VOICE_PATTERN.search(str(text)))
        for text in texts
        if text is not None and not (isinstance(text, float) and pd.isna(text))
    )


def build_manifest(
    raw_dir: str | Path,
    metadata: pd.DataFrame | None = None,
    target_species: Iterable[str] | None = None,
    min_seconds: float = 1.0,
) -> pd.DataFrame:
    """Inspecciona ``raw_dir`` y retorna el manifiesto con ``conservar`` y ``motivo``."""
    raw_dir = Path(raw_dir)
    files = list_audio_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"No se encontraron audios en {raw_dir}")

    by_id: dict[str, dict] = {}
    if metadata is not None and not metadata.empty and "id" in metadata.columns:
        by_id = {str(row["id"]): row for row in metadata.to_dict("records")}
    targets = {" ".join(name.replace("_", " ").split()).lower() for name in (target_species or [])}

    rows, seen_hashes = [], set()
    for path in files:
        label = path.parent.name
        xc_id = xc_id_from_path(path)
        record = by_id.get(xc_id, {})
        info = inspect_audio(path)
        md5 = file_md5(path)
        own_name = label.replace("_", " ").lower()
        background = {name.lower() for name in parse_also(record.get("also"))} - {own_name}

        if info["error"]:
            reason = "ilegible"
        elif md5 in seen_hashes:
            reason = "duplicado"
        elif info["duracion_s"] < min_seconds:
            reason = "muy_corto"
        elif background & targets:
            reason = "fondo_objetivo"
        else:
            reason = ""
        seen_hashes.add(md5)

        rows.append(
            {
                "path": path.relative_to(raw_dir).as_posix(),
                "label": label,
                "xc_id": xc_id,
                "bytes": path.stat().st_size,
                "duracion_s": info["duracion_s"],
                "frecuencia_muestreo": info["frecuencia_muestreo"],
                "canales": info["canales"],
                "md5": md5,
                "posible_voz": mentions_voice(record.get("rmk")),
                "conservar": reason == "",
                "motivo": reason,
            }
        )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def summarize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Resumen por especie: descargadas, descartadas por motivo, conservadas y minutos."""
    kept = manifest[manifest["conservar"]]
    summary = pd.DataFrame({"descargadas": manifest.groupby("label").size()})
    for reason in REASONS:
        summary[reason] = (
            manifest[manifest["motivo"] == reason].groupby("label").size().reindex(summary.index, fill_value=0)
        )
    summary["conservadas"] = kept.groupby("label").size().reindex(summary.index, fill_value=0)
    summary["minutos_conservados"] = (
        (kept.groupby("label")["duracion_s"].sum() / 60.0).reindex(summary.index, fill_value=0.0).round(1)
    )
    summary["posible_voz"] = (
        kept.groupby("label")["posible_voz"].sum().reindex(summary.index, fill_value=0).astype(int)
    )
    return summary.astype({c: int for c in ["descargadas", *REASONS, "conservadas"]}).rename_axis("especie").reset_index()


def clean_dataset(
    raw_dir: str | Path,
    metadata_csv: str | Path | None,
    manifest_csv: str | Path,
    summary_csv: str | Path | None = None,
    target_species: Iterable[str] | None = None,
    min_seconds: float = 1.0,
) -> pd.DataFrame:
    """Genera y guarda el manifiesto de depuración (y su resumen por especie)."""
    metadata = None
    if metadata_csv is not None and Path(metadata_csv).exists():
        metadata = pd.read_csv(metadata_csv, low_memory=False)
    manifest = build_manifest(raw_dir, metadata, target_species, min_seconds)

    manifest_csv = Path(manifest_csv)
    ensure_dir(manifest_csv.parent)
    manifest.to_csv(manifest_csv, index=False)
    if summary_csv is not None:
        ensure_dir(Path(summary_csv).parent)
        summarize_manifest(manifest).to_csv(summary_csv, index=False)

    discarded = manifest[~manifest["conservar"]]["motivo"].value_counts().to_dict()
    logger.info(
        "%d archivos revisados: %d conservados, %d descartados %s",
        len(manifest), int(manifest["conservar"].sum()), int((~manifest["conservar"]).sum()), discarded,
    )
    return manifest
