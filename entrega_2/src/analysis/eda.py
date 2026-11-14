"""Análisis exploratorio de datos (tarea 2.4): tablas y figuras del reporte 2.

Las funciones ``tabla_*`` y de cálculo retornan ``DataFrame``; las ``plot_*``
guardan una figura PNG y retornan su ruta. ``run_eda`` las ejecuta todas::

    python -m src.main eda

Entradas (las generan los comandos ``download``, ``depurar`` y ``preprocess``):

- ``data/metadata/xeno_canto.csv``: metadatos de todas las grabaciones de las especies objetivo.
- ``data/metadata/depuracion.csv``: manifiesto de los archivos descargados.
- ``data/processed/``: índice de ventanas, espectrogramas y estadísticas de MFCC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.clean import parse_also
from src.data.download import QUALITY_ORDER, REGIONS, parse_length
from src.data.preprocess import MFCC_FILE
from src.utils.helpers import (
    METADATA_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    ensure_dir,
    get_logger,
    load_json,
    save_json,
)

FIGURES_DIR = PROJECT_ROOT / "docs" / "figuras"

# Paleta: un solo tono para magnitudes, tonos categóricos en orden fijo, tinta neutra para textos.
BLUE = "#2a78d6"
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID, NEUTRAL = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"

METADATA_FIELDS = ["q", "length", "lat", "lon", "alt", "type", "sex", "stage", "date", "time", "rec", "lic", "smp", "rmk"]
VOCALIZATION_ORDER = ["canto", "llamado", "canto y llamado", "otro o sin dato"]

logger = get_logger("eda")


# ----------------------------------------------------------------------------- utilidades
def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
            "axes.edgecolor": NEUTRAL, "axes.labelcolor": INK, "text.color": INK,
            "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
            "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
            "axes.spines.top": False, "axes.spines.right": False,
            "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        }
    )
    return plt


def _save(fig, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("azul", [SURFACE, *BLUE_RAMP])


def short_name(label: str) -> str:
    """``Henicorhina_leucophrys`` -> ``H. leucophrys``."""
    parts = label.replace("_", " ").split()
    return f"{parts[0][0]}. {' '.join(parts[1:])}" if len(parts) > 1 else label


def vocalization_type(value) -> str:
    """Agrupa el campo libre ``type`` de xeno-canto en canto, llamado, ambos u otro."""
    text = str(value).lower()
    song, call = "song" in text, "call" in text
    if song and call:
        return "canto y llamado"
    if song:
        return "canto"
    if call:
        return "llamado"
    return "otro o sin dato"


def normalize_quality(value) -> str:
    """Calidad A–E; cualquier otro valor se agrupa en ``sin calificar``."""
    quality = str(value).strip().upper()
    return quality if quality in QUALITY_ORDER else "sin calificar"


def downloaded_metadata(metadata: pd.DataFrame, manifest: pd.DataFrame, only_kept: bool = True) -> pd.DataFrame:
    """Metadatos de los archivos descargados (por defecto, solo los conservados)."""
    files = manifest[manifest["conservar"].astype(bool)] if only_kept else manifest
    frame = metadata.copy()
    frame["xc_id"] = frame["id"].astype(str)
    files = files.assign(xc_id=files["xc_id"].astype(str))
    return files.merge(frame.drop_duplicates("xc_id"), on="xc_id", how="left")


# ----------------------------------------------------------------------------- tablas
def missing_values(metadata: pd.DataFrame, fields: list[str] | None = None) -> pd.DataFrame:
    """Porcentaje de valores faltantes por campo (vacíos, ``?`` o no numéricos donde aplica)."""
    rows = []
    for field in fields or METADATA_FIELDS:
        if field not in metadata.columns:
            continue
        column = metadata[field]
        text = column.astype(str).str.strip().str.lower()
        missing = column.isna() | text.isin(["", "nan", "none", "?", "??", "unknown", "uncertain"])
        if field in ("lat", "lon", "alt", "smp"):
            missing |= pd.to_numeric(column, errors="coerce").isna()
        elif field == "q":
            missing |= ~text.str.upper().isin(QUALITY_ORDER)
        elif field == "length":
            missing |= column.map(parse_length).isna()
        elif field == "time":
            missing |= ~text.str.match(r"^\d{1,2}:\d{2}")
        elif field == "date":
            missing |= text.str.contains("00-00|-00", regex=True)
        rows.append({"campo": field, "faltantes": int(missing.sum()), "porcentaje": round(100 * missing.mean(), 1)})
    return pd.DataFrame(rows).sort_values("porcentaje", ascending=False, ignore_index=True)


def quality_table(metadata: pd.DataFrame) -> pd.DataFrame:
    """Grabaciones disponibles por especie y calidad (todas las calidades)."""
    quality = metadata["q"].map(normalize_quality)
    table = pd.crosstab(metadata["species"], quality).reindex(columns=[*QUALITY_ORDER, "sin calificar"], fill_value=0)
    table["total"] = table.sum(axis=1)
    return table.sort_values("total", ascending=False).rename_axis("especie").reset_index()


def species_summary(files: pd.DataFrame, index: pd.DataFrame | None = None) -> pd.DataFrame:
    """Resumen por especie de las grabaciones conservadas (``downloaded_metadata``)."""
    frame = files.copy()
    frame["_otras"] = frame["also"].map(lambda value: len(parse_also(value)) > 0)
    frame["_tipo"] = frame["type"].map(vocalization_type)
    frame["_q"] = frame["q"].map(normalize_quality)
    group = frame.groupby("label")
    summary = pd.DataFrame(
        {
            "grabaciones": group.size(),
            "calidad_A": group["_q"].agg(lambda s: int((s == "A").sum())),
            "calidad_B": group["_q"].agg(lambda s: int((s == "B").sum())),
            "minutos": (group["duracion_s"].sum() / 60).round(1),
            "duracion_mediana_s": group["duracion_s"].median().round(1),
            "grabadores": group["rec"].nunique(),
            "max_por_grabador_pct": group["rec"].agg(lambda s: round(100 * s.value_counts(normalize=True).iloc[0], 1)),
            "con_otras_especies_pct": (100 * group["_otras"].mean()).round(1),
            "cantos_pct": group["_tipo"].agg(lambda s: round(100 * s.isin(["canto", "canto y llamado"]).mean(), 1)),
        }
    )
    if index is not None:
        summary["ventanas"] = index.groupby("label").size().reindex(summary.index, fill_value=0)
    return summary.sort_values("grabaciones", ascending=False).rename_axis("especie").reset_index()


def license_table(files: pd.DataFrame) -> pd.DataFrame:
    """Grabaciones conservadas por tipo de licencia Creative Commons."""
    code = files["lic"].astype(str).str.extract(r"licenses/([a-z\-]+)/([0-9.]+)", expand=True)
    name = ("CC " + code[0].str.upper() + " " + code[1]).fillna("otra o sin dato")
    table = name.value_counts().rename_axis("licencia").reset_index(name="grabaciones")
    table["porcentaje"] = (100 * table["grabaciones"] / table["grabaciones"].sum()).round(1)
    return table


# ----------------------------------------------------------------------------- figuras de metadatos
def plot_quality(metadata: pd.DataFrame, path: str | Path) -> Path:
    """Barras apiladas: grabaciones disponibles por especie y calidad."""
    plt = _style()
    table = quality_table(metadata).set_index("especie").drop(columns="total").iloc[::-1]
    colors = dict(zip([*QUALITY_ORDER, "sin calificar"], [*BLUE_RAMP[::-1], NEUTRAL]))
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(table) + 1.6))
    left = np.zeros(len(table))
    for quality in table.columns:
        ax.barh([short_name(s) for s in table.index], table[quality], left=left, color=colors[quality],
                edgecolor=SURFACE, linewidth=1.2, height=0.7, label=quality)
        left += table[quality].to_numpy()
    for y, total in enumerate(left):
        ax.text(total + left.max() * 0.01, y, f"{int(total)}", va="center", fontsize=8, color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Grabaciones en xeno-canto (Colombia)")
    ax.set_title("Calidad de las grabaciones disponibles por especie")
    ax.legend(title="Calidad", ncols=6, frameon=False, loc="lower right", fontsize=8)
    return _save(fig, path)


def plot_durations(files: pd.DataFrame, path: str | Path, max_seconds: float | None = None) -> Path:
    """Histograma de la duración de las grabaciones conservadas (eje logarítmico)."""
    plt = _style()
    duration = files["duracion_s"].dropna()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bins = np.logspace(np.log10(max(duration.min(), 0.5)), np.log10(duration.max()), 32)
    ax.hist(duration, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1.0)
    ax.set_xscale("log")
    ax.axvline(duration.median(), color=INK, linewidth=1.2, linestyle="--")
    ax.text(duration.median(), ax.get_ylim()[1] * 0.95, f"  mediana {duration.median():.0f} s", fontsize=8, va="top")
    if max_seconds:
        ax.axvline(max_seconds, color=MUTED, linewidth=1.2, linestyle=":")
        ax.text(max_seconds, ax.get_ylim()[1] * 0.80, f"  tope de {max_seconds:.0f} s", fontsize=8, va="top", color=INK)
    ax.set_xticks([1, 3, 10, 30, 60, 180, 600], ["1", "3", "10", "30", "60", "180", "600"])
    ax.set_xlabel("Duración de la grabación (s, escala logarítmica)")
    ax.set_ylabel("Grabaciones")
    ax.set_title("Duración de las grabaciones conservadas")
    return _save(fig, path)


def plot_balance(summary: pd.DataFrame, path: str | Path) -> Path:
    """Tres paneles con el mismo orden de especies: grabaciones, minutos y ventanas."""
    plt = _style()
    panels = [("grabaciones", "Grabaciones"), ("minutos", "Minutos de audio"), ("ventanas", "Ventanas de 5 s")]
    panels = [(column, title) for column, title in panels if column in summary.columns]
    data = summary.iloc[::-1]
    names = [short_name(s) for s in data["especie"]]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 0.32 * len(data) + 1.4), sharey=True)
    for ax, (column, title) in zip(np.atleast_1d(axes), panels):
        ax.barh(names, data[column], color=BLUE, height=0.7)
        for y, value in enumerate(data[column]):
            ax.text(value + data[column].max() * 0.015, y, f"{value:g}", va="center", fontsize=7.5)
        ax.set_xlim(0, data[column].max() * 1.16)
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
    fig.suptitle("Desbalance entre especies", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_recorders(summary: pd.DataFrame, path: str | Path) -> Path:
    """Grabadores distintos y concentración en el grabador principal, por especie."""
    plt = _style()
    data = summary.sort_values("grabadores").reset_index(drop=True)
    names = [short_name(s) for s in data["especie"]]
    fig, axes = plt.subplots(1, 2, figsize=(8, 0.32 * len(data) + 1.4), sharey=True)
    for ax, column, title, fmt in [
        (axes[0], "grabadores", "Grabadores distintos", "{:.0f}"),
        (axes[1], "max_por_grabador_pct", "Grabaciones del grabador principal (%)", "{:.0f} %"),
    ]:
        ax.barh(names, data[column], color=BLUE, height=0.7)
        for y, value in enumerate(data[column]):
            ax.text(value + data[column].max() * 0.015, y, fmt.format(value), va="center", fontsize=7.5)
        ax.set_xlim(0, data[column].max() * 1.18)
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
    fig.tight_layout()
    return _save(fig, path)


def plot_vocalization_types(files: pd.DataFrame, path: str | Path) -> Path:
    """Barras apiladas al 100 %: tipo de vocalización por especie."""
    plt = _style()
    table = pd.crosstab(files["label"], files["type"].map(vocalization_type), normalize="index") * 100
    table = table.reindex(columns=VOCALIZATION_ORDER, fill_value=0.0).sort_values("canto")
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(table) + 1.6))
    left = np.zeros(len(table))
    for kind, color in zip(VOCALIZATION_ORDER, CATEGORICAL):
        ax.barh([short_name(s) for s in table.index], table[kind], left=left, color=color,
                edgecolor=SURFACE, linewidth=1.2, height=0.7, label=kind)
        left += table[kind].to_numpy()
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Grabaciones conservadas (%)")
    ax.set_title("Tipo de vocalización por especie")
    ax.legend(ncols=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    return _save(fig, path)


def plot_map(files: pd.DataFrame, path: str | Path) -> Path:
    """Ubicación de las grabaciones conservadas y rectángulo del Valle del Cauca."""
    plt = _style()
    from matplotlib.patches import Rectangle

    lat = pd.to_numeric(files["lat"], errors="coerce")
    lon = pd.to_numeric(files["lon"], errors="coerce")
    valid = lat.notna() & lon.notna()
    lat_min, lon_min, lat_max, lon_max = REGIONS["valle_del_cauca"]
    inside = valid & lat.between(lat_min, lat_max) & lon.between(lon_min, lon_max)
    fig, ax = plt.subplots(figsize=(5.4, 6.4))
    ax.scatter(lon[valid & ~inside], lat[valid & ~inside], s=14, color=BLUE, alpha=0.55, linewidths=0,
               label=f"Resto de Colombia ({int((valid & ~inside).sum())})")
    ax.scatter(lon[inside], lat[inside], s=16, color=CATEGORICAL[1], alpha=0.8, linewidths=0,
               label=f"Dentro del rectángulo del Valle ({int(inside.sum())})")
    ax.add_patch(Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min, fill=False,
                           edgecolor=INK, linewidth=1.2))
    ax.set_aspect("equal")
    ax.set_xlabel("Longitud (°)")
    ax.set_ylabel("Latitud (°)")
    ax.set_title("Ubicación de las grabaciones conservadas")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8)
    return _save(fig, path)


# ----------------------------------------------------------------------------- exploración acústica
def band_profiles(index: pd.DataFrame, processed_dir: str | Path) -> np.ndarray:
    """Energía media en el tiempo de cada banda Mel, por ventana: ``(n_ventanas, n_mels)``.

    Los espectrogramas están estandarizados por ventana, de modo que el perfil
    describe la **forma** del espectro (qué bandas sobresalen), no el volumen.
    """
    processed_dir = Path(processed_dir)
    return np.stack([np.load(processed_dir / path).mean(axis=1) for path in index["path"]])


def mel_frequencies(audio_config: dict) -> np.ndarray:
    """Frecuencia central (Hz) de cada banda Mel de la configuración."""
    import librosa

    return librosa.mel_frequencies(
        n_mels=audio_config["n_mels"], fmin=audio_config["fmin"], fmax=audio_config["fmax"]
    )


def species_profiles(profiles: np.ndarray, index: pd.DataFrame) -> pd.DataFrame:
    """Perfil espectral medio por especie (filas) y banda Mel (columnas)."""
    return pd.DataFrame(profiles).groupby(index["label"].to_numpy()).mean()


def relative_profiles(profile_by_species: pd.DataFrame) -> pd.DataFrame:
    """Exceso de energía de cada especie frente al promedio de todas, por banda.

    El ruido de fondo (viento, tráfico, zumbido de baja frecuencia) es común a
    todas las especies y domina el perfil bruto; al restar el promedio queda lo
    que distingue a cada especie. Los valores negativos se recortan a cero.
    """
    return (profile_by_species - profile_by_species.mean(axis=0)).clip(lower=0.0)


def dominant_bands(profile_by_species: pd.DataFrame, frequencies: np.ndarray) -> pd.DataFrame:
    """Frecuencia del pico y banda dominante de cada especie.

    Se calcula sobre el perfil relativo (``relative_profiles``). La banda
    dominante es el intervalo contiguo alrededor del pico donde el perfil
    supera el punto medio entre el pico y la mediana del perfil.
    """
    rows = []
    for species, profile in relative_profiles(profile_by_species).iterrows():
        values = profile.to_numpy()
        peak = int(values.argmax())
        threshold = (values[peak] + np.median(values)) / 2
        low = high = peak
        while low > 0 and values[low - 1] >= threshold:
            low -= 1
        while high < len(values) - 1 and values[high + 1] >= threshold:
            high += 1
        rows.append(
            {
                "especie": species,
                "pico_khz": round(frequencies[peak] / 1000, 2),
                "banda_min_khz": round(frequencies[low] / 1000, 2),
                "banda_max_khz": round(frequencies[high] / 1000, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("pico_khz", ignore_index=True)


def plot_species_profiles(profile_by_species: pd.DataFrame, frequencies: np.ndarray, path: str | Path) -> Path:
    """Mapa de calor especie × frecuencia con el perfil espectral relativo."""
    plt = _style()
    order = dominant_bands(profile_by_species, frequencies)["especie"][::-1]
    data = relative_profiles(profile_by_species).loc[order]
    fig, ax = plt.subplots(figsize=(8.5, 0.36 * len(data) + 1.6))
    mesh = ax.pcolormesh(frequencies / 1000, np.arange(len(data)), data.to_numpy(), cmap=_sequential_cmap(), shading="nearest")
    ax.set_yticks(np.arange(len(data)), [short_name(s) for s in data.index])
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.3, 1, 2, 4, 8, 14], ["0,1", "0,3", "1", "2", "4", "8", "14"])
    ax.grid(visible=False)
    ax.set_xlabel("Frecuencia (kHz, escala logarítmica)")
    ax.set_title("Bandas de frecuencia que distinguen a cada especie (ordenadas por su pico)")
    fig.colorbar(mesh, ax=ax, pad=0.02, label="Exceso de energía frente al promedio de las 15 especies")
    return _save(fig, path)


def plot_example_spectrograms(
    index: pd.DataFrame, processed_dir: str | Path, audio_config: dict, path: str | Path, profiles: np.ndarray | None = None
) -> Path:
    """Un espectrograma de ejemplo por especie.

    Si se pasan los perfiles, elige la ventana más parecida al perfil medio de
    su especie (un ejemplo representativo); si no, la ventana central.
    """
    plt = _style()
    processed_dir = Path(processed_dir)
    labels = sorted(index["label"].unique())
    frequencies = mel_frequencies(audio_config) / 1000
    ncols = 5
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.9 * ncols, 2.3 * nrows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    reset = index.reset_index(drop=True)
    for ax, label in zip(axes.ravel(), labels):
        rows = reset.index[reset["label"] == label].to_numpy()
        if profiles is not None:
            distance = np.linalg.norm(profiles[rows] - profiles[rows].mean(axis=0), axis=1)
            chosen = rows[int(distance.argmin())]
        else:
            chosen = rows[len(rows) // 2]
        spectrogram = np.load(processed_dir / reset.loc[chosen, "path"])
        time = np.linspace(0, audio_config["segment_seconds"], spectrogram.shape[1])
        ax.pcolormesh(time, frequencies, spectrogram, cmap=_sequential_cmap(), shading="auto", rasterized=True)
        ax.set_title(short_name(label), fontsize=9, fontstyle="italic", fontweight="normal")
        ax.grid(visible=False)
    for ax in axes.ravel()[len(labels):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Tiempo (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("kHz")
    fig.suptitle("Espectrograma Mel representativo de cada especie (ventana de 5 s)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def species_distances(features: np.ndarray, index: pd.DataFrame) -> pd.DataFrame:
    """Distancia euclidiana entre los centroides de las especies en el espacio de MFCC estandarizado."""
    from sklearn.preprocessing import StandardScaler

    scaled = StandardScaler().fit_transform(features)
    centroids = pd.DataFrame(scaled).groupby(index["label"].to_numpy()).mean()
    difference = centroids.to_numpy()[:, None, :] - centroids.to_numpy()[None, :, :]
    return pd.DataFrame(np.linalg.norm(difference, axis=2), index=centroids.index, columns=centroids.index)


def closest_pairs(distances: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    """Pares de especies más cercanos (candidatos a confundirse)."""
    rows = [
        {"especie_1": a, "especie_2": b, "distancia": round(float(distances.loc[a, b]), 2)}
        for i, a in enumerate(distances.index)
        for b in distances.columns[i + 1:]
    ]
    return pd.DataFrame(rows).sort_values("distancia", ignore_index=True).head(top)


def plot_species_distances(distances: pd.DataFrame, path: str | Path) -> Path:
    """Mapa de calor de distancias entre especies (más oscuro = más parecidas)."""
    plt = _style()
    names = [short_name(s) for s in distances.index]
    size = 0.42 * len(names) + 2.4
    fig, ax = plt.subplots(figsize=(size + 0.8, size))
    values = distances.to_numpy().copy()
    np.fill_diagonal(values, np.nan)
    image = ax.imshow(values, cmap=_sequential_cmap().reversed())
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=8, fontstyle="italic")
    ax.set_yticks(range(len(names)), names, fontsize=8, fontstyle="italic")
    ax.grid(visible=False)
    ax.set_title("Distancia entre especies en el espacio de MFCC")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="Distancia entre centroides (menor = más parecidas)")
    return _save(fig, path)


def kmeans_exploration(features: np.ndarray, index: pd.DataFrame, seed: int = 42) -> tuple[dict, pd.DataFrame]:
    """K-means con k = número de especies sobre las estadísticas de MFCC.

    Retorna las medidas de acuerdo con las etiquetas reales (pureza, ARI, NMI)
    y la tabla de contingencia especie × grupo.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.preprocessing import StandardScaler

    labels = index["label"].to_numpy()
    k = len(np.unique(labels))
    clusters = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(StandardScaler().fit_transform(features))
    contingency = pd.crosstab(pd.Series(labels, name="especie"), pd.Series(clusters, name="grupo"))
    majority = pd.Series(labels).value_counts(normalize=True).iloc[0]
    measures = {
        "k": int(k),
        "n_ventanas": int(len(labels)),
        "pureza": float(contingency.max(axis=0).sum() / len(labels)),
        "pureza_azar_clase_mayoritaria": float(majority),
        "ari": float(adjusted_rand_score(labels, clusters)),
        "nmi": float(normalized_mutual_info_score(labels, clusters)),
    }
    return measures, contingency


def plot_pca_small_multiples(features: np.ndarray, index: pd.DataFrame, path: str | Path, seed: int = 42) -> Path:
    """Proyección PCA de las ventanas: un panel por especie, resaltada sobre el resto en gris."""
    plt = _style()
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    pca = PCA(n_components=2, random_state=seed)
    points = pca.fit_transform(StandardScaler().fit_transform(features))
    labels = index["label"].to_numpy()
    species = sorted(np.unique(labels))
    rng = np.random.default_rng(seed)
    background = rng.choice(len(points), size=min(len(points), 3000), replace=False)
    ncols = 5
    nrows = int(np.ceil(len(species) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.4 * nrows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    limits = np.percentile(points, [0.5, 99.5], axis=0)
    for ax, name in zip(axes.ravel(), species):
        ax.scatter(points[background, 0], points[background, 1], s=3, color=NEUTRAL, alpha=0.35, linewidths=0, rasterized=True)
        own = points[labels == name]
        ax.scatter(own[:, 0], own[:, 1], s=4, color=BLUE, alpha=0.6, linewidths=0, rasterized=True)
        ax.set_title(short_name(name), fontsize=9, fontstyle="italic", fontweight="normal")
        ax.set_xlim(limits[0, 0], limits[1, 0])
        ax.set_ylim(limits[0, 1], limits[1, 1])
        ax.grid(visible=False)
    for ax in axes.ravel()[len(species):]:
        ax.set_visible(False)
    explained = 100 * pca.explained_variance_ratio_
    fig.supxlabel(f"Componente 1 ({explained[0]:.0f} % de la varianza)", fontsize=9)
    fig.supylabel(f"Componente 2 ({explained[1]:.0f} %)", fontsize=9)
    fig.suptitle("Ventanas de cada especie (azul) frente a todas las demás (gris), PCA de MFCC", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


# ----------------------------------------------------------------------------- todo junto
def run_eda(
    metadata_csv: str | Path = METADATA_DIR / "xeno_canto.csv",
    manifest_csv: str | Path = METADATA_DIR / "depuracion.csv",
    processed_dir: str | Path = PROCESSED_DIR,
    figures_dir: str | Path = FIGURES_DIR,
    results_dir: str | Path = RESULTS_DIR,
    seed: int = 42,
) -> dict:
    """Genera todas las tablas (``results_dir/eda_*.csv``) y figuras (``figures_dir/eda_*.png``)."""
    figures_dir, results_dir, processed_dir = ensure_dir(figures_dir), ensure_dir(results_dir), Path(processed_dir)
    metadata = pd.read_csv(metadata_csv, low_memory=False)
    manifest = pd.read_csv(manifest_csv)
    index = pd.read_csv(processed_dir / "index.csv")
    features = np.load(processed_dir / MFCC_FILE)
    audio_config = load_json(processed_dir / "audio_config.json")
    files = downloaded_metadata(metadata, manifest)

    summary = species_summary(files, index)
    tables = {
        "eda_valores_faltantes": missing_values(metadata),
        "eda_calidad_por_especie": quality_table(metadata),
        "eda_resumen_por_especie": summary,
        "eda_licencias": license_table(files),
    }
    plot_quality(metadata, figures_dir / "eda_calidad.png")
    plot_durations(files, figures_dir / "eda_duraciones.png", audio_config.get("max_seconds"))
    plot_balance(summary, figures_dir / "eda_desbalance.png")
    plot_recorders(summary, figures_dir / "eda_grabadores.png")
    plot_vocalization_types(files, figures_dir / "eda_tipos_vocalizacion.png")
    plot_map(files, figures_dir / "eda_mapa.png")

    logger.info("Calculando perfiles espectrales de %d ventanas...", len(index))
    profiles = band_profiles(index, processed_dir)
    frequencies = mel_frequencies(audio_config)
    by_species = species_profiles(profiles, index)
    tables["eda_bandas_dominantes"] = dominant_bands(by_species, frequencies)
    plot_species_profiles(by_species, frequencies, figures_dir / "eda_perfil_espectral.png")
    plot_example_spectrograms(index, processed_dir, audio_config, figures_dir / "eda_espectrogramas.png", profiles)

    distances = species_distances(features, index)
    tables["eda_pares_parecidos"] = closest_pairs(distances)
    plot_species_distances(distances, figures_dir / "eda_distancias_especies.png")
    measures, contingency = kmeans_exploration(features, index, seed)
    tables["eda_kmeans_contingencia"] = contingency.reset_index()
    plot_pca_small_multiples(features, index, figures_dir / "eda_pca_especies.png", seed)

    for name, table in tables.items():
        table.to_csv(results_dir / f"{name}.csv", index=False)
    general = {
        "grabaciones_disponibles": int(len(metadata)),
        "grabaciones_descargadas": int(len(manifest)),
        "grabaciones_conservadas": int(manifest["conservar"].astype(bool).sum()),
        "minutos_conservados": round(float(files["duracion_s"].sum() / 60), 1),
        "ventanas": int(len(index)),
        "grabadores": int(files["rec"].nunique()),
        "duracion_mediana_s": round(float(files["duracion_s"].median()), 1),
        "con_otras_especies_pct": round(float(100 * files["also"].map(lambda v: len(parse_also(v)) > 0).mean()), 1),
        "frecuencias_de_muestreo": {str(k): int(v) for k, v in files["frecuencia_muestreo"].value_counts().items()},
        "kmeans": measures,
    }
    save_json(general, results_dir / "eda_resumen_general.json")
    logger.info("EDA completo: tablas en %s y figuras en %s", results_dir, figures_dir)
    return {"general": general, "tablas": tables}
