"""Genera y ejecuta los notebooks de la Entrega 2 a partir de celdas definidas aquí.

Los notebooks son delgados: la lógica vive en ``src/`` (con pruebas) y aquí solo
se llama y se comenta. Uso (desde ``entrega_2/``, con los datos ya procesados)::

    python notebooks/_construir_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent

SETUP = """\
import os, sys, json
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent   # carpeta entrega_2/
os.chdir(ROOT); sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from IPython.display import Image, display

pd.set_option("display.max_rows", 60, "display.width", 200, "display.float_format", "{:.3f}".format)
RESULTS, FIGURES, PROCESSED = ROOT / "experiments/results", ROOT / "docs/figuras", ROOT / "data/processed"
"""

EDA = [
    ("md", """# 02 · Análisis exploratorio de datos (tarea 2.4)

**Proyecto 1 · Innovación Tecnológica con IA · Grupo 4** — Identificación de especies de aves de Colombia por su canto.

En este notebook exploramos las grabaciones que descargamos de xeno-canto para las 15 especies objetivo: calidad de los datos, distribuciones y exploración acústica. La lógica está en `src/analysis/eda.py`; aquí la ejecutamos y comentamos. Las conclusiones las llevamos a las secciones II y III del [reporte 2](../docs/reporte_entrega_2.md).

> Requiere haber corrido `download`, `depurar` y `preprocess` (ver [`docs/guia_etapa_2.md`](../docs/guia_etapa_2.md))."""),
    ("code", SETUP),
    ("code", """\
from src.analysis import eda
from src.data.preprocess import MFCC_FILE
from src.utils.helpers import load_json

metadata = pd.read_csv("data/metadata/xeno_canto.csv", low_memory=False)
manifest = pd.read_csv("data/metadata/depuracion.csv")
index = pd.read_csv(PROCESSED / "index.csv")
features = np.load(PROCESSED / MFCC_FILE)
audio_config = load_json(PROCESSED / "audio_config.json")
files = eda.downloaded_metadata(metadata, manifest)

print(f"Grabaciones disponibles en xeno-canto (15 especies, Colombia): {len(metadata)}")
print(f"Descargadas (calidad A/B, máx. 60 por especie): {len(manifest)} | conservadas: {manifest['conservar'].sum()}")
print(f"Ventanas de 5 s: {len(index)} | características MFCC por ventana: {features.shape[1]}")"""),
    ("md", "## 1. Selección de especies\n\nLas 15 especies objetivo en el ranking nacional (criterio de 30 grabaciones A/B) y en el rectángulo del Valle del Cauca (presencia regional)."),
    ("code", """\
from src.main import read_species_list

targets = read_species_list("data/especies_objetivo.txt")
national = pd.read_csv(RESULTS / "ranking_especies.csv").set_index("especie")
valle = pd.read_csv(RESULTS / "ranking_especies_valle.csv").set_index("especie")
selection = national.loc[targets, ["nombre_en", "grabaciones", "grabadores", "minutos"]].join(
    valle["grabaciones"].rename("grabaciones_valle"))
print(f"Especies con ≥ 30 grabaciones A/B en Colombia: {int(national['cumple'].sum())} de {len(national)}")
selection.sort_values("grabaciones", ascending=False)"""),
    ("md", "## 2. Depuración (tarea 2.3)\n\nEl manifiesto marca qué archivos conservamos y por qué descartamos los demás; no borramos ningún archivo."),
    ("code", """\
from src.data.clean import summarize_manifest

summary_clean = summarize_manifest(manifest)
display(summary_clean)
print(summary_clean.drop(columns="especie").sum().to_string())"""),
    ("md", "## 3. Calidad de los datos\n\n### 3.1 Valores faltantes en los metadatos"),
    ("code", "eda.missing_values(metadata)"),
    ("md", "### 3.2 Calidad asignada por la comunidad (A–E)\n\nSolo descargamos A y B; la figura muestra todo lo disponible para dimensionar cuánto dejamos por fuera."),
    ("code", """\
display(Image(eda.plot_quality(metadata, FIGURES / "eda_calidad.png")))
quality = eda.quality_table(metadata)
print("Proporción por calidad (%):")
print((100 * quality.drop(columns=["especie", "total"]).sum() / quality["total"].sum()).round(1).to_string())"""),
    ("md", "### 3.3 Duración, frecuencia de muestreo y etiquetas débiles"),
    ("code", """\
display(Image(eda.plot_durations(files, FIGURES / "eda_duraciones.png", audio_config.get("max_seconds"))))
print(files["duracion_s"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(1).to_string())
print("\\nFrecuencia de muestreo de los archivos (Hz):")
print(files["frecuencia_muestreo"].value_counts().to_string())
also = files["also"].map(lambda v: len(eda.parse_also(v)) > 0)
print(f"\\nGrabaciones con otras especies de fondo (etiquetas débiles): {100 * also.mean():.1f} %")"""),
    ("md", "## 4. Distribuciones\n\n### 4.1 Resumen por especie y desbalance"),
    ("code", """\
summary = eda.species_summary(files, index)
display(summary)
display(Image(eda.plot_balance(summary, FIGURES / "eda_desbalance.png")))
for column in ("grabaciones", "minutos", "ventanas"):
    print(f"Razón máx/mín de {column}: {summary[column].max() / summary[column].min():.1f}")"""),
    ("md", "### 4.2 Grabadores por especie\n\nSi una sola persona aporta buena parte de las grabaciones de una especie, el modelo puede aprender su equipo o su sitio en lugar del canto."),
    ("code", 'display(Image(eda.plot_recorders(summary, FIGURES / "eda_grabadores.png")))'),
    ("md", "### 4.3 Tipo de vocalización"),
    ("code", 'display(Image(eda.plot_vocalization_types(files, FIGURES / "eda_tipos_vocalizacion.png")))'),
    ("md", "### 4.4 Distribución geográfica y licencias"),
    ("code", """\
display(Image(eda.plot_map(files, FIGURES / "eda_mapa.png")))
display(eda.license_table(files))"""),
    ("md", "## 5. Exploración acústica\n\n### 5.1 Espectrogramas representativos"),
    ("code", """\
profiles = eda.band_profiles(index, PROCESSED)
frequencies = eda.mel_frequencies(audio_config)
display(Image(eda.plot_example_spectrograms(index, PROCESSED, audio_config, FIGURES / "eda_espectrogramas.png", profiles)))"""),
    ("md", "### 5.2 Perfil espectral medio y bandas dominantes"),
    ("code", """\
by_species = eda.species_profiles(profiles, index)
display(Image(eda.plot_species_profiles(by_species, frequencies, FIGURES / "eda_perfil_espectral.png")))
eda.dominant_bands(by_species, frequencies)"""),
    ("md", "### 5.3 Especies parecidas\n\nDistancia entre los centroides de cada especie en el espacio de estadísticas de MFCC (estandarizado). Los pares más cercanos son los candidatos a confundirse."),
    ("code", """\
distances = eda.species_distances(features, index)
display(Image(eda.plot_species_distances(distances, FIGURES / "eda_distancias_especies.png")))
eda.closest_pairs(distances)"""),
    ("md", "### 5.4 Agrupamiento no supervisado (k-means)\n\n¿Las ventanas se separan por especie sin usar las etiquetas? Aplicamos k-means con k = 15 sobre las estadísticas de MFCC y comparamos los grupos con las especies."),
    ("code", """\
measures, contingency = eda.kmeans_exploration(features, index)
print(json.dumps(measures, indent=2))
display(Image(eda.plot_pca_small_multiples(features, index, FIGURES / "eda_pca_especies.png")))"""),
    ("md", "__CONCLUSIONES_EDA__"),
]

MODELOS = [
    ("md", """# 03 · Modelos de referencia y pruebas iniciales (tareas 2.6 y 2.7)

**Proyecto 1 · Innovación Tecnológica con IA · Grupo 4**

Comparamos, con la **misma partición por grabación** (70/15/15, semilla 42), la clase mayoritaria, cuatro clasificadores clásicos sobre estadísticas de MFCC y la CNN base sobre espectrogramas Mel, y ponemos a prueba las hipótesis H2, H3 y H4 del [reporte 2](../docs/reporte_entrega_2.md).

Los modelos clásicos se entrenan aquí mismo (tardan segundos). Las corridas de la CNN se entrenan con `python -m src.main train` (ver los comandos al final) y aquí leemos sus resultados."""),
    ("code", SETUP),
    ("md", "## 1. Partición\n\nTodas las ventanas de una grabación quedan en la misma partición."),
    ("code", """\
from src.data.dataset import WINDOW_SPLIT_COLUMN
from src.models.classical import MODEL_TITLES, evaluate_classical, load_mfcc_dataset

index, features = load_mfcc_dataset(PROCESSED)
assert {"split", WINDOW_SPLIT_COLUMN} <= set(index.columns), "Ejecute primero `python -m src.main baselines` con ambas particiones"
assert (index.groupby("recording_id")["split"].nunique() == 1).all()

sizes = pd.DataFrame({
    "grabaciones": index.groupby("split")["recording_id"].nunique(),
    "ventanas": index.groupby("split").size(),
}).loc[["train", "val", "test"]]
sizes["ventanas_pct"] = (100 * sizes["ventanas"] / sizes["ventanas"].sum()).round(1)
display(sizes)
pd.crosstab(index["label"], index["split"])[["train", "val", "test"]]"""),
    ("md", "## 2. Modelos clásicos sobre estadísticas de MFCC\n\nCada ventana se resume en 80 valores: media y desviación estándar en el tiempo de 20 MFCC y de sus deltas."),
    ("code", """\
METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "top3_accuracy"]
classical, predictions = evaluate_classical(features, index, "split")
classical_test = classical[classical["conjunto"] == "test"].set_index("modelo")[METRICS]
classical_test.rename(index=MODEL_TITLES)"""),
    ("md", "## 3. CNN base sobre espectrogramas Mel\n\nCuatro bloques convolucionales, ≈ 1,2 millones de parámetros, entrenada desde cero con pesos por clase y aumentación tipo SpecAugment."),
    ("code", """\
def cnn_metrics(run: str) -> dict:
    return json.loads((RESULTS / f"{run}_test_metricas.json").read_text(encoding="utf-8"))

def cnn_summary(run: str) -> dict:
    return json.loads((RESULTS / f"{run}_train_summary.json").read_text(encoding="utf-8"))

RUNS = {"cnn_base": "CNN base (con aumentación)", "cnn_sin_aug": "CNN sin aumentación", "cnn_split_ventana": "CNN, partición por ventana"}
runs = pd.DataFrame({title: {**{m: cnn_metrics(run)[m] for m in METRICS},
                             "mejor_epoca": cnn_summary(run)["best_epoch"], "epocas": cnn_summary(run)["epochs_run"]}
                     for run, title in RUNS.items()}).T
runs"""),
    ("code", """\
import matplotlib.pyplot as plt

history = pd.read_csv(RESULTS / "cnn_base_history.csv")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
axes[0].plot(history["epoch"], history["train_loss"], color="#2a78d6", linewidth=2, label="entrenamiento")
axes[0].plot(history["epoch"], history["val_loss"], color="#eb6834", linewidth=2, label="validación")
axes[0].set_title("Pérdida", loc="left"); axes[0].set_xlabel("Época"); axes[0].legend(frameon=False)
axes[1].plot(history["epoch"], history["f1_macro"], color="#2a78d6", linewidth=2)
axes[1].set_title("F1 macro en validación", loc="left"); axes[1].set_xlabel("Época"); axes[1].set_ylim(0, 1)
for ax in axes:
    ax.grid(color="#e1e0d9", linewidth=0.6); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIGURES / "cnn_base_curvas.png", dpi=130, bbox_inches="tight"); plt.show()"""),
    ("md", "## 4. Tabla comparativa (conjunto de prueba, partición por grabación)"),
    ("code", """\
table = pd.concat([classical_test.rename(index=MODEL_TITLES), runs.loc[["CNN base (con aumentación)"], METRICS].rename(index={"CNN base (con aumentación)": "CNN base (espectrogramas Mel)"})])
table.columns = ["Exactitud", "Exactitud balanceada", "F1 macro", "Top-3"]
table.to_csv(RESULTS / "comparacion_modelos_test.csv")
table.astype(float).round(3)"""),
    ("md", "### Matriz de confusión y F1 por especie de la CNN base"),
    ("code", """\
display(Image(RESULTS / "cnn_base_test_matriz_confusion.png", width=720))
per_species = pd.read_csv(RESULTS / "cnn_base_test_por_especie.csv", index_col=0)
per_species = per_species[per_species.index.isin(index["label"].unique())]
per_species["ventanas_train"] = index[index["split"] == "train"].groupby("label").size()
per_species.sort_values("f1-score")[["precision", "recall", "f1-score", "support", "ventanas_train"]]"""),
    ("md", "## 5. Hipótesis\n\n### H4 · ¿La CNN sobre espectrogramas supera a los clasificadores clásicos sobre MFCC?"),
    ("code", """\
best_classical = classical_test["f1_macro"].drop("mayoritaria").idxmax()
print(f"Mejor clásico: {MODEL_TITLES[best_classical]} -> F1 macro {classical_test.loc[best_classical, 'f1_macro']:.3f}")
print(f"CNN base -> F1 macro {runs.loc['CNN base (con aumentación)', 'f1_macro']:.3f}")"""),
    ("md", "### H2 · ¿La aumentación mejora las especies con menos datos?\n\nComparamos el F1 por especie de la CNN con y sin aumentación, separando el tercio de especies con menos ventanas de entrenamiento."),
    ("code", """\
def f1_by_species(run: str) -> pd.Series:
    report = pd.read_csv(RESULTS / f"{run}_test_por_especie.csv", index_col=0)
    return report.loc[report.index.isin(index["label"].unique()), "f1-score"]

h2 = pd.DataFrame({"ventanas_train": index[index["split"] == "train"].groupby("label").size(),
                   "f1_con_aug": f1_by_species("cnn_base"), "f1_sin_aug": f1_by_species("cnn_sin_aug")}).sort_values("ventanas_train")
h2["diferencia"] = h2["f1_con_aug"] - h2["f1_sin_aug"]
third = len(h2) // 3
h2["grupo"] = ["tercio con menos datos"] * third + ["resto"] * (len(h2) - third)
display(h2)
h2_summary = h2.groupby("grupo")[["f1_con_aug", "f1_sin_aug", "diferencia"]].mean()
h2_summary.loc["todas (F1 macro)"] = h2[["f1_con_aug", "f1_sin_aug", "diferencia"]].mean()
h2.to_csv(RESULTS / "h2_aumentacion_por_especie.csv")
h2_summary"""),
    ("md", "### H3 · ¿Particionar por ventana sobrestima el desempeño?\n\nRepetimos el entrenamiento con una partición al azar por ventana (`split_ventana`), en la que ventanas de una misma grabación caen en entrenamiento y en prueba, y comparamos con la partición por grabación."),
    ("code", """\
by_window, _ = evaluate_classical(features, index, WINDOW_SPLIT_COLUMN)
window_test = by_window[by_window["conjunto"] == "test"].set_index("modelo")["f1_macro"]
h3 = pd.DataFrame({"F1 por grabación": classical_test["f1_macro"], "F1 por ventana": window_test}).drop("mayoritaria").rename(index=MODEL_TITLES)
h3.loc["CNN base (espectrogramas Mel)"] = [cnn_metrics("cnn_base")["f1_macro"], cnn_metrics("cnn_split_ventana")["f1_macro"]]
h3["sobrestimación"] = h3["F1 por ventana"] - h3["F1 por grabación"]
h3.to_csv(RESULTS / "h3_particion.csv")
h3"""),
    ("md", "__CONCLUSIONES_MODELOS__"),
]


def build(cells: list[tuple[str, str]], replacements: dict[str, str]):
    notebook = new_notebook(metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    for kind, source in cells:
        source = replacements.get(source, source)
        notebook.cells.append(new_markdown_cell(source) if kind == "md" else new_code_cell(source))
    return notebook


def main() -> None:
    replacements = {
        "__CONCLUSIONES_EDA__": (HERE / "_conclusiones_eda.md").read_text(encoding="utf-8"),
        "__CONCLUSIONES_MODELOS__": (HERE / "_conclusiones_modelos.md").read_text(encoding="utf-8"),
    }
    for name, cells in [("02_eda.ipynb", EDA), ("03_modelos_referencia.ipynb", MODELOS)]:
        notebook = build(cells, replacements)
        ExecutePreprocessor(timeout=3600, kernel_name="python3").preprocess(notebook, {"metadata": {"path": str(HERE)}})
        nbformat.write(notebook, HERE / name)
        print(f"Generado {name}")


if __name__ == "__main__":
    main()
