# 🔌 Interfaces del proyecto (CLI y funciones de Python)

No exponemos una API web: la interfaz es la **línea de comandos** de `src/main.py`, las funciones de Python que esta usa y el prototipo en Streamlit (`app/streamlit_app.py`).

## Prototipo en Streamlit

```bash
streamlit run app/streamlit_app.py
```

| Pestaña | Qué muestra |
|---|---|
| Identificar | Elige el modelo (Perch 2.0, EfficientNet-B0 o CNN base) y el audio: subir un archivo, grabarlo o tomar un ejemplo de prueba. El botón «Sorpréndeme» elige una grabación de prueba al azar. Muestra la especie más probable con su grupo, su endemismo y el F1 del modelo con esa especie; si el audio es un ejemplo, dice si acertó. También muestra las tres especies más probables, un mapa de calor con la probabilidad de cada especie por ventana, el espectrograma y, si se activa, la comparación entre los modelos |
| Cómo funciona | Los cinco pasos del audio a la sugerencia y cómo evaluamos |
| Qué tan bueno es | F1 macro con intervalo de confianza de los ocho modelos comparados, por grabación o por ventana, y F1 por especie |

Perch 2.0 aparece solo si TensorFlow está instalado en el mismo entorno (ver [`instalacion.md`](instalacion.md)).

## Línea de comandos

Ejecutar desde `entrega_3/`:

```bash
python -m src.main <comando> [opciones]
```

| Comando | Qué hace | Opciones principales |
|---|---|---|
| `especies` | Consulta los metadatos de una región en xeno-canto (sin descargar audio) y rankea las especies por número de grabaciones de buena calidad | `--region` (`valle_del_cauca`), `--box LAT_MIN LON_MIN LAT_MAX LON_MAX`, `--filtrar-localidad`, `--grupo` (birds), `--calidades` (A B), `--min-grabaciones` (30), `--max-paginas`, `--salida`, `--top` |
| `download` | Busca grabaciones de las especies indicadas, guarda los metadatos y descarga los audios (prioriza la mejor calidad) | `--species` o `--lista` (una de las dos), `--country` (Colombia), `--group` (birds), `--calidades` (A B), `--max-per-species` (60), `--raw-dir`, `--metadata` |
| `depurar` | Revisa los audios descargados (ilegibles, duplicados, muy cortos, otra especie objetivo al fondo) y genera el manifiesto de depuración; no borra archivos | `--raw-dir`, `--metadata`, `--lista`, `--min-seconds` (1), `--manifest`, `--resumen` |
| `preprocess` | Divide los audios en ventanas y guarda espectrogramas Mel, estadísticas de MFCC (`mfcc_stats.npy`) e `index.csv` | `--raw-dir`, `--out-dir`, `--manifest`, `--max-seconds`, `--sample-rate` (32000), `--segment-seconds` (5), `--hop-seconds` (2.5), `--n-mels` (128), `--reduce-noise` |
| `eda` | Genera las tablas (`experiments/results/eda_*.csv`) y figuras (`docs/figuras/eda_*.png`) del análisis exploratorio | `--metadata`, `--manifest`, `--processed-dir` |
| `baselines` | Evalúa los modelos clásicos (clase mayoritaria, regresión logística, SVM, bosque aleatorio y potenciación de gradiente) sobre estadísticas de MFCC o sobre *embeddings*, con intervalos de confianza | `--processed-dir`, `--split-column` (`split` \| `split_ventana`), `--seed`, `--features` (`mfcc_stats.npy`), `--modelos`, `--prefijo` (`baseline`) |
| `embeddings` | Extrae un vector por ventana con un modelo bioacústico preentrenado (requiere el entorno con TensorFlow) | `--modelo` (`perch_v2`), `--processed-dir`, `--raw-dir`, `--batch-size` (32) |
| `train` | Entrena un modelo y guarda el mejor checkpoint | `--model` (`cnn_baseline` \| `efficientnet_b0`), `--epochs` (30), `--batch-size` (32), `--lr` (0.001), `--patience` (5), `--no-augment`, `--no-class-weights`, `--no-pretrained`, `--split-column`, `--threads`, `--seed`, `--run-name` |
| `evaluate` | Calcula métricas, reporte por especie y matriz de confusión | `--checkpoint` (obligatoria), `--split` (`train` \| `val` \| `test`), `--split-column`, `--processed-dir`, `--batch-size` |
| `intervalos` | F1 macro por ventana y por grabación, con intervalo de confianza por *bootstrap* | `--checkpoint` (obligatoria), `--split`, `--n-boot`, `--umbral`, `--seed` |
| `fuga` | Compara Perch 2.0 y la CNN base en grabaciones de prueba subidas antes y después de una fecha (verificación de fuga) | `--corte` (2025-08-01), `--checkpoint` |
| `predict` | Retorna las especies más probables para un audio | `--audio` y `--checkpoint` (obligatorias), `--top-k` (5) |

### Flujo de la Etapa 2

```bash
python -m src.main especies --region valle_del_cauca --filtrar-localidad "Valle del Cauca"
# elegimos las especies y las escribimos en data/especies_objetivo.txt
python -m src.main download --lista data/especies_objetivo.txt --max-per-species 60
python -m src.main depurar
python -m src.main preprocess --manifest data/metadata/depuracion.csv --max-seconds 60
python -m src.main eda
python -m src.main baselines
python -m src.main train --model cnn_baseline --epochs 30 --run-name cnn_base
python -m src.main evaluate --checkpoint experiments/checkpoints/cnn_base.pt --split test
```

La columna `split` del índice es la partición por grabación (la correcta). `split_ventana` reparte las ventanas al azar y solo la usamos para medir la fuga de información (hipótesis H3).

El paso a paso está en [`guia_etapa_2.md`](guia_etapa_2.md).

## Funciones de Python

| Módulo | Función / clase | Descripción |
|---|---|---|
| `src.data.download` | `build_query(species, country, group, quality, box)` | Construye la consulta con etiquetas de xeno-canto (`sp:`, `cnt:`, `grp:`, `q:`, `box:`) |
| | `search_recordings(query, key=None, max_pages=None)` | Retorna los registros de todas las páginas |
| | `rank_species(metadata, qualities, min_recordings, loc_contains)` | Resumen por especie: grabaciones, grabadores, minutos, etiquetas débiles |
| | `species_ranking(region, box, ...)` | Consulta una región y guarda metadatos y ranking |
| | `download_species(species_list, raw_dir, metadata_csv, ...)` | Busca, guarda metadatos y descarga audios |
| `src.data.clean` | `build_manifest`, `summarize_manifest`, `clean_dataset` | Depuración: manifiesto de archivos conservados y descartados, con el motivo |
| `src.data.preprocess` | `AudioConfig`, `load_audio`, `segment_signal`, `mel_spectrogram`, `mfcc_statistics`, `audio_to_features`, `preprocess_directory` | Preprocesamiento de audio |
| `src.data.dataset` | `SpectrogramDataset`, `assign_splits(index, val_size, test_size, by_recording)`, `ensure_splits` | Dataset de PyTorch y particiones (por grabación o al azar por ventana) |
| `src.analysis.eda` | `run_eda`, `species_summary`, `missing_values`, `dominant_bands`, `kmeans_exploration`, `plot_*` | Tablas y figuras del análisis exploratorio |
| `src.models.classical` | `build_classical_models`, `load_feature_dataset`, `evaluate_classical`, `run_baselines` | Clasificadores clásicos sobre MFCC o *embeddings* |
| `src.models.embeddings` | `PerchEmbedder`, `recording_windows`, `extract_embeddings` | *Embeddings* de Perch 2.0 alineados con el índice de ventanas |
| `src.models.cnn_baseline` | `build_model(name, n_classes, **kwargs)` | Fábrica de modelos (`cnn_baseline`, `efficientnet_b0`) |
| `src.training.train` | `TrainConfig`, `train(cfg)` | Configuración y bucle de entrenamiento |
| `src.evaluation.evaluate` | `compute_metrics`, `top_k_accuracy`, `predict_split`, `evaluate_checkpoint` | Métricas y evaluación de un checkpoint |
| `src.evaluation.uncertainty` | `aggregate_by_recording`, `bootstrap_f1_macro`, `intervals_from_scores`, `evaluate_with_intervals` | F1 por grabación e intervalos de confianza |
| `src.evaluation.leakage` | `upload_dates`, `compare_by_upload_date`, `run_leakage_check` | Verificación de fuga por fecha de subida |
| `src.inference.predict` | `load_predictor`, `predict_signal`, `top_species`, `predict_file(audio_path, checkpoint, top_k)` | Predicción para un audio (la usa también el prototipo `app/streamlit_app.py`) |
| `src.inference.perch` | `train_probe`, `load_perch_predictor`, `predict_signal_perch` | Predicción con Perch 2.0 + regresión logística; el clasificador se guarda en `experiments/checkpoints/perch_v2_regresion_logistica.joblib` |

## Clave de la API

| Fuente (en orden de prioridad) | Uso |
|---|---|
| Variable de entorno `XC_API_KEY` | Clave de la API v3 de xeno-canto |
| Archivo `entrega_3/.env` con `XC_API_KEY=...` | Alternativa recomendada en Windows; está excluido de Git |
