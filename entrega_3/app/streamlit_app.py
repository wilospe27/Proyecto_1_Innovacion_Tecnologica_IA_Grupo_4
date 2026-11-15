"""Prototipo de demostración: ¿qué ave está cantando?

Ejecutar desde ``entrega_3/``::

    streamlit run app/streamlit_app.py

El usuario sube o graba un audio, o elige una grabación de prueba; la app lo
divide en ventanas de 5 s, las clasifica con un modelo entrenado y muestra las
especies más probables, qué escuchó el modelo en cada ventana y qué tan
confiable es con esa especie. Solo reconoce las 15 especies del proyecto.

Perch 2.0 aparece solo si TensorFlow y ``perch-hoplite`` están instalados en el
mismo entorno (ver ``docs/instalacion.md``); si no, la app usa EfficientNet-B0
y la CNN base.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # silencia los mensajes de TensorFlow en la consola

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import load_audio, mel_spectrogram  # noqa: E402
from src.inference.perch import PROBE_FILE, PerchPredictor, load_perch_predictor, predict_signal_perch  # noqa: E402
from src.inference.predict import Predictor, load_predictor, predict_signal, top_species  # noqa: E402
from src.models.embeddings import embeddings_file  # noqa: E402
from src.utils.helpers import CHECKPOINTS_DIR, PROCESSED_DIR, RAW_DIR, RESULTS_DIR  # noqa: E402

PERCH = "Perch 2.0"
# Modelo -> (checkpoint, prefijo de sus resultados en experiments/results), de mejor a peor F1 en prueba.
# El primero disponible es el predeterminado.
MODELS = {
    PERCH: (PROBE_FILE.name, "perch_v2_regresion_logistica"),
    "EfficientNet-B0": ("efficientnet_b0.pt", "efficientnet_b0"),
    "CNN base": ("cnn_base.pt", "cnn_base"),
}
# Nombre común (inglés, como en el reporte), grupo en español y endemismo en Colombia.
SPECIES = {
    "Cyphorhinus_thoracicus": ("Chestnut-breasted Wren", "Cucaracheros", None),
    "Henicorhina_leucophrys": ("Grey-breasted Wood Wren", "Cucaracheros", None),
    "Henicorhina_leucosticta": ("White-breasted Wood Wren", "Cucaracheros", None),
    "Myadestes_ralloides": ("Andean Solitaire", "Mirlos y solitarios", None),
    "Odontophorus_hyperythrus": ("Chestnut Wood Quail", "Perdices de monte", "Endémica"),
    "Ortalis_columbiana": ("Colombian Chachalaca", "Pavas y guacharacas", "Endémica"),
    "Pheugopedius_mystacalis": ("Whiskered Wren", "Cucaracheros", None),
    "Rupornis_magnirostris": ("Roadside Hawk", "Gavilanes y águilas", None),
    "Scytalopus_alvarezlopezi": ("Tatamá Tapaculo", "Tapaculos", "Endémica"),
    "Scytalopus_vicinior": ("Nariño Tapaculo", "Tapaculos", None),
    "Synallaxis_albescens": ("Pale-breasted Spinetail", "Horneros y rastrojeros", None),
    "Synallaxis_azarae": ("Azara's Spinetail", "Horneros y rastrojeros", None),
    "Thamnophilus_multistriatus": ("Bar-crested Antshrike", "Batarás", "Casi endémica"),
    "Troglodytes_aedon": ("House Wren", "Cucaracheros", None),
    "Zonotrichia_capensis": ("Rufous-collared Sparrow", "Gorriones y copetones", None),
}
# Comparación de los cinco enfoques (archivos de intervalos de confianza en experiments/results).
BENCHMARK = [
    ("Regresión logística · MFCC", "baseline_regresion_logistica", "Clásico sobre MFCC"),
    ("SVM · MFCC", "baseline_svm", "Clásico sobre MFCC"),
    ("Bosque aleatorio · MFCC", "baseline_bosque_aleatorio", "Clásico sobre MFCC"),
    ("Potenciación de gradiente · MFCC", "baseline_potenciacion_gradiente", "Clásico sobre MFCC"),
    ("CNN base", "cnn_base", "Red entrenada desde cero"),
    ("EfficientNet-B0", "efficientnet_b0", "Preentrenado"),
    ("Perch 2.0 + regresión logística", "perch_v2_regresion_logistica", "Preentrenado"),
    ("Perch 2.0 + SVM", "perch_v2_svm", "Preentrenado"),
]
MAX_SECONDS = 60  # igual que en el entrenamiento: solo el primer minuto
LOW_CONFIDENCE = 0.5
GOAL = 0.70


def species_name(label: str) -> str:
    return label.replace("_", " ")


def decimal(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


# --- Datos y modelos (en caché) -------------------------------------------------------------------


def model_available(name: str) -> bool:
    if name == PERCH:
        has_tf = all(importlib.util.find_spec(module) for module in ("tensorflow", "perch_hoplite"))
        return has_tf and (PROBE_FILE.exists() or (PROCESSED_DIR / embeddings_file("perch_v2")).exists())
    return (CHECKPOINTS_DIR / MODELS[name][0]).exists()


@st.cache_resource(max_entries=3, show_spinner=False)
def get_predictor(name: str) -> Predictor | PerchPredictor:
    if name == PERCH:
        return load_perch_predictor()
    return load_predictor(CHECKPOINTS_DIR / MODELS[name][0], device="cpu")


def run_model(predictor: Predictor | PerchPredictor, signal: np.ndarray) -> np.ndarray:
    if isinstance(predictor, PerchPredictor):
        return predict_signal_perch(predictor, signal)
    return predict_signal(predictor, signal)


@st.cache_data(max_entries=1)
def test_examples() -> pd.DataFrame:
    """Grabaciones del conjunto de prueba (no vistas en el entrenamiento) con audio disponible."""
    index = pd.read_csv(PROCESSED_DIR / "index.csv", usecols=["label", "recording_id", "split"])
    recordings = index[index["split"] == "test"].drop_duplicates("recording_id")
    files = {
        rec: next((p for p in (RAW_DIR / rec).parent.glob(Path(rec).name + ".*")), None)
        for rec in recordings["recording_id"]
    }
    recordings = recordings.assign(file=recordings["recording_id"].map(files)).dropna(subset=["file"])
    return recordings.sort_values(["label", "recording_id"]).reset_index(drop=True)


@st.cache_data(max_entries=8, show_spinner=False)
def load_example(path: str, sample_rate: int) -> np.ndarray:
    return load_audio(path, sample_rate)[: MAX_SECONDS * sample_rate]


@st.cache_data(max_entries=32, show_spinner=False)
def analyze(model_name: str, audio_key: str, _signal: np.ndarray) -> tuple[np.ndarray, float]:
    """Probabilidades por ventana y segundos de inferencia; ``audio_key`` identifica el audio en la caché."""
    predictor = get_predictor(model_name)
    start = time.perf_counter()
    probabilities = run_model(predictor, _signal)
    return probabilities, time.perf_counter() - start


@st.cache_data
def species_f1(prefix: str) -> dict[str, float]:
    """F1 por especie del modelo en el conjunto de prueba."""
    path = RESULTS_DIR / f"{prefix}_test_por_especie.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path, index_col=0)
    return table.loc[table.index.isin(SPECIES), "f1-score"].to_dict()


@st.cache_data
def benchmark() -> pd.DataFrame:
    rows = []
    for name, prefix, family in BENCHMARK:
        path = RESULTS_DIR / f"{prefix}_test_intervalos.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for level, key in (("Por grabación", "por_grabacion"), ("Por ventana", "por_ventana")):
            block = data[key]
            rows.append(
                {
                    "modelo": name,
                    "enfoque": family,
                    "nivel": level,
                    "f1": block["f1_macro"],
                    "inferior": block["ic_inferior"],
                    "superior": block["ic_superior"],
                }
            )
    return pd.DataFrame(rows)


def read_audio(data: bytes, suffix: str, sample_rate: int) -> np.ndarray:
    """Decodifica el audio subido (wav, mp3, flac u ogg) a mono y lo recorta al primer minuto."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(data)
        path = Path(handle.name)
    try:
        signal = load_audio(path, sample_rate)
    finally:
        path.unlink(missing_ok=True)
    return signal[: MAX_SECONDS * sample_rate]


def surprise_me() -> None:
    """Elige una grabación de prueba al azar (callback del botón «Sorpréndeme»)."""
    examples = test_examples()
    row = examples.iloc[random.randrange(len(examples))]
    st.session_state["fuente"] = "Ejemplo de prueba"
    st.session_state["especie"] = row["label"]
    st.session_state["ejemplo"] = row["recording_id"]


# --- Gráficos -------------------------------------------------------------------------------------


def activity_chart(probabilities: np.ndarray, labels: list[str], hop: float, segment: float) -> alt.Chart:
    """Mapa de calor: probabilidad de cada especie en cada ventana de 5 s.

    Las ventanas se solapan; cada celda va del inicio de su ventana al inicio de la siguiente
    (la última, hasta el final), para que las celdas no se tapen entre sí.
    """
    order = np.argsort(probabilities.mean(axis=0))[::-1]
    keep = [i for i in order if probabilities[:, i].max() >= 0.05][:8] or list(order[:3])
    last = len(probabilities) - 1
    rows = [
        {
            "inicio": w * hop,
            "fin": w * hop + (segment if w == last else hop),
            "ventana": f"{decimal(w * hop, 1)}–{decimal(w * hop + segment, 1)} s",
            "especie": species_name(labels[i]),
            "probabilidad": float(probabilities[w, i]),
        }
        for w in range(len(probabilities))
        for i in keep
    ]
    data = pd.DataFrame(rows)
    sort = [species_name(labels[i]) for i in keep]
    return (
        alt.Chart(data)
        .mark_rect(stroke="#0F1714", strokeWidth=1)
        .encode(
            x=alt.X("inicio:Q", title="Tiempo (s)", scale=alt.Scale(nice=False)),
            x2="fin:Q",
            y=alt.Y("especie:N", sort=sort, title=None, axis=alt.Axis(labelFontStyle="italic", labelLimit=220)),
            color=alt.Color("probabilidad:Q", title="Probabilidad", scale=alt.Scale(domain=[0, 1]), legend=alt.Legend(format="%")),
            tooltip=[
                alt.Tooltip("ventana:N", title="Ventana"),
                alt.Tooltip("especie:N", title="Especie"),
                alt.Tooltip("probabilidad:Q", title="Probabilidad", format=".0%"),
            ],
        )
        .properties(height=max(140, 34 * len(keep)))
    )


def spectrogram_figure(signal: np.ndarray, predictor: Predictor | PerchPredictor):
    import matplotlib.pyplot as plt

    cfg = predictor.audio_config
    mel_db = mel_spectrogram(signal, cfg)
    seconds = len(signal) / cfg.sample_rate
    fig, ax = plt.subplots(figsize=(10, 2.4), facecolor="#17231E")
    ax.imshow(mel_db, origin="lower", aspect="auto", cmap="magma", extent=(0, seconds, 0, cfg.n_mels))
    ax.set_facecolor("#17231E")
    ax.set_xlabel("Tiempo (s)", color="#9FB3A8")
    ax.set_ylabel("Banda Mel", color="#9FB3A8")
    ax.tick_params(colors="#9FB3A8")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def comma(field: str, fmt: str) -> str:
    """Expresión de Vega-Lite que formatea ``datum.<field>`` con coma decimal (``value`` en ejes y leyendas)."""
    return f"replace(format(datum.{field}, '{fmt}'), '.', ',')"


def benchmark_chart(data: pd.DataFrame) -> alt.Chart:
    order = [name for name, _, _ in BENCHMARK]
    families = ["Clásico sobre MFCC", "Red entrenada desde cero", "Preentrenado"]
    base = (
        alt.Chart(data)
        .transform_calculate(
            f1_txt=comma("f1", ".3f"),
            ic_txt=f"{comma('inferior', '.2f')} + ' – ' + {comma('superior', '.2f')}",
        )
        .encode(
            y=alt.Y("modelo:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
            color=alt.Color(
                "enfoque:N",
                title=None,
                scale=alt.Scale(domain=families, range=["#9AA5A0", "#F2B84B", "#3DDC97"]),
                legend=alt.Legend(orient="bottom", labelLimit=300),
            ),
            tooltip=[
                alt.Tooltip("modelo:N", title="Modelo"),
                alt.Tooltip("f1_txt:N", title="F1 macro"),
                alt.Tooltip("ic_txt:N", title="IC del 95 %"),
            ],
        )
    )
    interval = base.mark_rule(strokeWidth=3, opacity=0.55).encode(
        x=alt.X(
            "inferior:Q",
            title="F1 macro en prueba (intervalo de confianza del 95 %)",
            scale=alt.Scale(domain=[0.4, 1]),
            axis=alt.Axis(values=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], labelExpr=comma("value", ".1f")),
        ),
        x2="superior:Q",
    )
    point = base.mark_circle(size=140, opacity=1).encode(x="f1:Q")
    goal = alt.Chart(pd.DataFrame({"meta": [GOAL]})).mark_rule(strokeDash=[5, 4], color="#9FB3A8").encode(x=alt.X("meta:Q", scale=alt.Scale(domain=[0.4, 1])))
    label = (
        alt.Chart(pd.DataFrame({"meta": [GOAL], "texto": ["Meta ≈ 0,70"]}))
        .mark_text(align="left", dx=5, dy=-6, color="#9FB3A8", baseline="bottom")
        .encode(x="meta:Q", y=alt.value(0), text="texto:N")
    )
    return (interval + point + goal + label).properties(height=330)


def species_chart() -> alt.Chart:
    columns = {"CNN base": "cnn_base", "EfficientNet-B0": "efficientnet_b0", "Perch 2.0 + RL": "perch_v2_regresion_logistica"}
    rows = [
        {"modelo": model, "especie": species_name(label), "f1": f1}
        for model, prefix in columns.items()
        for label, f1 in species_f1(prefix).items()
    ]
    data = pd.DataFrame(rows)
    sort = data[data["modelo"] == "CNN base"].sort_values("f1")["especie"].tolist()
    base = (
        alt.Chart(data)
        .transform_calculate(f1_txt=comma("f1", ".2f"))
        .encode(
            x=alt.X("modelo:N", sort=list(columns), title=None, axis=alt.Axis(orient="top", labelAngle=0)),
            y=alt.Y("especie:N", sort=sort, title=None, axis=alt.Axis(labelFontStyle="italic", labelLimit=240)),
        )
    )
    cells = base.mark_rect(stroke="#0F1714", strokeWidth=2).encode(
        color=alt.Color("f1:Q", title="F1", scale=alt.Scale(domain=[0.5, 1]), legend=alt.Legend(labelExpr=comma("value", ".1f"))),
        tooltip=[alt.Tooltip("especie:N", title="Especie"), alt.Tooltip("modelo:N", title="Modelo"), alt.Tooltip("f1_txt:N", title="F1")],
    )
    text = base.mark_text(fontSize=12).encode(
        text=alt.Text("f1_txt:N"),
        color=alt.condition("datum.f1 > 0.8", alt.value("#0F1714"), alt.value("#E6EFE9")),
    )
    return (cells + text).properties(height=520)


# --- Página ---------------------------------------------------------------------------------------

st.set_page_config(page_title="¿Qué ave canta?", page_icon=":material/graphic_eq:", layout="wide")

st.title("¿Qué ave está cantando?", icon=":material/graphic_eq:")
st.caption(
    "Prototipo del Grupo 4 · Innovación Tecnológica con IA · Universidad Icesi. "
    "Escucha un canto y sugiere cuál de 15 aves del Valle del Cauca lo produjo."
)

available = {name: spec for name, spec in MODELS.items() if model_available(name)}
if not available:
    st.error(
        "Entrene un modelo antes de usar la app: `python -m src.main train --run-name cnn_base`.",
        title="No hay modelos entrenados",
    )
    st.stop()

scores = benchmark()
best_recording = scores[(scores["modelo"] == "Perch 2.0 + regresión logística") & (scores["nivel"] == "Por grabación")]
with st.container(horizontal=True):
    st.metric("Especies", "15", border=True, icon=":material/raven:")
    st.metric("Grabaciones de xeno-canto", "822", border=True, icon=":material/library_music:")
    st.metric("Minutos de audio", "665", border=True, icon=":material/graphic_eq:")
    st.metric(
        "Mejor F1 macro por grabación",
        decimal(best_recording["f1"].iloc[0]) if not best_recording.empty else "—",
        border=True,
        icon=":material/verified:",
        help="Perch 2.0 + regresión logística en 119 grabaciones que el modelo nunca vio.",
    )

tab_identify, tab_how, tab_quality = st.tabs(
    [":material/hearing: Identificar", ":material/schema: Cómo funciona", ":material/insights: Qué tan bueno es"]
)

# --- Identificar ----------------------------------------------------------------------------------
with tab_identify:
    left, right = st.columns([1, 1.6], gap="large")

    with left:
        with st.container(horizontal=True, vertical_alignment="bottom"):
            model_name = st.segmented_control("Modelo", list(available), default=next(iter(available)), key="modelo")
            st.button(
                "Sorpréndeme",
                icon=":material/shuffle:",
                on_click=surprise_me,
                help="Elige al azar una grabación que el modelo nunca vio.",
            )
        st.session_state.setdefault("fuente", "Ejemplo de prueba")  # «Sorpréndeme» también la cambia
        source = st.segmented_control("Audio", ["Subir archivo", "Grabar", "Ejemplo de prueba"], key="fuente")
        model_name = model_name or next(iter(available))
        results_prefix = available[model_name][1]
        loading = "Cargando Perch 2.0 (la primera vez tarda cerca de medio minuto)…" if model_name == PERCH else "Cargando el modelo…"
        with st.spinner(loading):
            predictor = get_predictor(model_name)
        sample_rate = predictor.audio_config.sample_rate

        signal: np.ndarray | None = None
        audio_key = ""
        true_label: str | None = None
        if source == "Subir archivo":
            uploaded = st.file_uploader("Grabación", type=["wav", "mp3", "flac", "ogg"], key="archivo")
            if uploaded is not None:
                data = uploaded.getvalue()
                signal = read_audio(data, Path(uploaded.name).suffix, sample_rate)
                audio_key = hashlib.md5(data).hexdigest()
        elif source == "Grabar":
            recorded = st.audio_input("Grabe el canto con el micrófono", key="grabacion_mic")
            if recorded is not None:
                data = recorded.getvalue()
                signal = read_audio(data, ".wav", sample_rate)
                audio_key = hashlib.md5(data).hexdigest()
        elif source == "Ejemplo de prueba":
            examples = test_examples()
            if examples.empty:
                st.info("Los audios de ejemplo están en `data/raw/`, que no se versiona en Git.", title="Sin ejemplos")
            else:
                true_label = st.selectbox(
                    "Especie real del ejemplo",
                    sorted(examples["label"].unique()),
                    format_func=lambda label: f"{species_name(label)} ({SPECIES[label][0]})",
                    key="especie",
                )
                options = examples[examples["label"] == true_label]
                if st.session_state.get("ejemplo") not in set(options["recording_id"]):
                    st.session_state["ejemplo"] = options["recording_id"].iloc[0]
                recording = st.selectbox(
                    "Grabación", options["recording_id"], format_func=lambda r: Path(r).name, key="ejemplo"
                )
                file = options.loc[options["recording_id"] == recording, "file"].iloc[0]
                signal = load_example(str(file), sample_rate)
                audio_key = str(recording)
                st.caption(
                    f"Grabación [{Path(recording).name}](https://xeno-canto.org/{Path(recording).name[2:]}) de xeno-canto, "
                    "del conjunto de prueba: el modelo no la escuchó durante el entrenamiento."
                )

        if signal is not None:
            st.audio(signal, sample_rate=sample_rate)
            compare = len(available) > 1 and st.toggle(
                "Comparar con los otros modelos", key="comparar", help="Analiza el mismo audio con todos los modelos disponibles."
            )

    with right:
        if signal is None:
            st.info(
                "Suba o grabe un audio de al menos 5 segundos, elija un ejemplo de prueba o pulse **Sorpréndeme**.",
                icon=":material/mic:",
            )
            st.stop()

        try:
            with st.spinner("Escuchando el audio…"):
                probabilities, seconds = analyze(model_name, audio_key, signal)
        except ValueError:
            st.warning("El audio dura menos de 5 segundos: no alcanza para una ventana de análisis.", title="Audio muy corto")
            st.stop()

        ranking = top_species(probabilities, predictor.labels, top_k=3)
        best_label = ranking["especie"].iloc[0].replace(" ", "_")
        best_p = float(ranking["probabilidad"].iloc[0])
        votes = int((probabilities.argmax(axis=1) == predictor.labels.index(best_label)).sum())
        common, group, endemic = SPECIES.get(best_label, ("", "", None))
        f1 = species_f1(results_prefix).get(best_label)

        with st.container(border=True):
            with st.container(horizontal=True, gap="small"):
                st.badge(group, icon=":material/forest:", color="green")
                if endemic:
                    st.badge(f"{endemic} de Colombia", icon=":material/star:", color="orange")
                if f1 is not None:
                    color = "green" if f1 >= 0.85 else "orange" if f1 >= 0.7 else "red"
                    st.badge(
                        f"F1 del modelo con esta especie: {decimal(f1)}",
                        icon=":material/target:",
                        color=color,
                        help="F1 por ventana en el conjunto de prueba. Indica qué tanto confiar en esta sugerencia.",
                    )
            st.header(f"*{species_name(best_label)}*")
            st.caption(common)
            with st.container(horizontal=True):
                st.metric("Probabilidad media", f"{best_p:.0%}", border=True)
                st.metric("Ventanas a su favor", f"{votes} de {len(probabilities)}", border=True)
                st.metric("Tiempo de análisis", f"{decimal(seconds, 1)} s", border=True)

            if true_label is not None:
                if best_label == true_label:
                    st.success("La especie sugerida coincide con la especie real de la grabación.", title="Acertó", icon=":material/check_circle:")
                else:
                    st.error(
                        f"La especie real es *{species_name(true_label)}*. "
                        + ("Aparece entre las tres sugerencias." if true_label in ranking["especie"].str.replace(" ", "_").tolist() else "No aparece entre las tres sugerencias."),
                        title="Se equivocó",
                        icon=":material/cancel:",
                    )
            if best_p < LOW_CONFIDENCE:
                st.warning(
                    "Puede ser una especie fuera de las 15 que conoce, un audio con mucho ruido o un canto poco común.",
                    title="Confianza baja",
                )

            st.dataframe(
                ranking.assign(**{"nombre común": ranking["especie"].str.replace(" ", "_").map(lambda s: SPECIES[s][0])})[
                    ["especie", "nombre común", "probabilidad"]
                ],
                hide_index=True,
                column_config={
                    "especie": st.column_config.TextColumn("Las tres más probables"),
                    "nombre común": st.column_config.TextColumn("Nombre común"),
                    "probabilidad": st.column_config.ProgressColumn("Probabilidad", min_value=0.0, max_value=1.0, format="percent"),
                },
                alt="Las tres especies más probables con su probabilidad",
            )

        if compare:
            verdicts = {model_name: (species_name(best_label), best_p)}
            with st.spinner("Analizando con los otros modelos…"):
                for other_name in available:
                    if other_name != model_name:
                        other_p, _ = analyze(other_name, audio_key, signal)
                        other_rank = top_species(other_p, get_predictor(other_name).labels, top_k=1)
                        verdicts[other_name] = (other_rank["especie"].iloc[0], float(other_rank["probabilidad"].iloc[0]))
            agreeing = sum(label == species_name(best_label) for label, _ in verdicts.values())
            with st.container(border=True):
                if agreeing == len(verdicts):
                    st.badge(f"Los {len(verdicts)} modelos coinciden", color="green", icon=":material/handshake:")
                else:
                    st.badge(f"{agreeing} de {len(verdicts)} modelos coinciden con {model_name}", color="orange", icon=":material/call_split:")
                for column, (name, (label, p)) in zip(st.columns(len(verdicts)), verdicts.items()):
                    with column:
                        st.caption(name)
                        st.markdown(f"**_{label}_**")
                        if true_label is None:
                            st.badge(f"{p:.0%}", color="gray")
                        elif label == species_name(true_label):
                            st.badge(f"{p:.0%} · acertó", icon=":material/check:", color="green")
                        else:
                            st.badge(f"{p:.0%} · se equivocó", icon=":material/close:", color="red")

    cfg = predictor.audio_config
    st.subheader("Qué escuchó el modelo, ventana por ventana", icon=":material/timeline:")
    st.caption(
        f"El audio se divide en ventanas de {decimal(cfg.segment_seconds, 0)} s que se solapan a la mitad. "
        "Cada fila es una especie; entre más clara la celda, más probable. La sugerencia final es el promedio de todas las ventanas."
    )
    st.altair_chart(
        activity_chart(probabilities, predictor.labels, cfg.hop_seconds, cfg.segment_seconds),
        alt="Mapa de calor con la probabilidad de cada especie en cada ventana de 5 segundos",
    )
    st.pyplot(spectrogram_figure(signal, predictor), alt="")
    st.caption(
        "Espectrograma Mel: así «ve» el modelo el sonido (tiempo en el eje horizontal, tono en el vertical). "
        "Las predicciones son sugerencias con una probabilidad asociada; no reemplazan la identificación de una persona experta."
    )

# --- Cómo funciona --------------------------------------------------------------------------------
with tab_how:
    st.subheader("Del canto a la sugerencia en cinco pasos")
    steps = [
        (":material/mic:", "1. Audio", "Se convierte a mono a 32 kHz y se usa el primer minuto."),
        (":material/view_week:", "2. Ventanas", "Se corta en ventanas de 5 s con 50 % de solapamiento."),
        (":material/gradient:", "3. Representación", "Perch 2.0, entrenado con millones de sonidos de animales, resume cada ventana en 1.536 números; EfficientNet-B0 y la CNN la ven como un espectrograma log-Mel."),
        (":material/neurology:", "4. Clasificación", "Una regresión logística (sobre Perch 2.0) o la propia red decide la especie de cada ventana."),
        (":material/functions:", "5. Promedio", "Se promedian las probabilidades de las ventanas y se muestran las tres especies más probables."),
    ]
    with st.container(horizontal=True):
        for icon, title, text in steps:
            with st.container(border=True, height="stretch"):
                st.markdown(f"#### {icon} {title}")
                st.caption(text)
    st.subheader("Cómo lo evaluamos")
    with st.container(horizontal=True):
        with st.container(border=True, height="stretch"):
            st.markdown("**:material/call_split: Partición por grabación**")
            st.write(
                "Ninguna grabación aparece a la vez en entrenamiento y prueba. Si se reparten ventanas sueltas, "
                "el modelo «reconoce» la grabación y el F1 sube hasta 0,34 sin que el modelo sea mejor."
            )
        with st.container(border=True, height="stretch"):
            st.markdown("**:material/database: Datos abiertos**")
            st.write(
                "822 grabaciones de xeno-canto (665 minutos de 178 grabadores), con su licencia Creative Commons. "
                "70 % para entrenar, 15 % para validar y 15 % para la prueba final."
            )
        with st.container(border=True, height="stretch"):
            st.markdown("**:material/shield: Sin fuga en Perch 2.0**")
            st.write(
                "Perch 2.0 se entrenó con xeno-canto. En las 12 grabaciones de prueba subidas después de su "
                "entrenamiento acierta las 12: su ventaja no viene de «recordar» los audios."
            )

# --- Qué tan bueno es -----------------------------------------------------------------------------
with tab_quality:
    st.subheader("Cinco enfoques, mismas grabaciones de prueba")
    level = st.segmented_control("Nivel", ["Por grabación", "Por ventana"], default="Por grabación", key="nivel")
    st.altair_chart(
        benchmark_chart(scores[scores["nivel"] == (level or "Por grabación")]),
        alt="F1 macro con intervalo de confianza de los ocho modelos comparados",
    )
    st.caption(
        "Por grabación se promedian las ventanas de cada grabación antes de decidir; por ventana se evalúa cada fragmento de 5 s. "
        "Los modelos preentrenados superan a la CNN entrenada desde cero, que a su vez supera a los clasificadores sobre MFCC. "
        "La app incluye Perch 2.0 + regresión logística, EfficientNet-B0 y la CNN base."
    )
    st.subheader("Qué especies se le dificultan")
    st.altair_chart(species_chart(), alt="F1 por especie de la CNN base, EfficientNet-B0 y Perch 2.0")
    st.caption(
        "F1 por ventana en prueba. Los cucaracheros (*Henicorhina*, *Pheugopedius*, *Troglodytes*) son los más difíciles "
        "para las redes propias; Perch 2.0 los reconoce mucho mejor."
    )
