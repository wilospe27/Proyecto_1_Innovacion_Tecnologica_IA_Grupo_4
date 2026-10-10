# 🔌 Interfaces del proyecto (CLI y funciones de Python)

Todavía no exponemos una API web: la interfaz es la **línea de comandos** de `src/main.py` y las funciones de Python que esta usa. En la Entrega 3 documentaremos aquí la interfaz web de demostración (Streamlit o Gradio).

## Línea de comandos

Ejecutar desde `entrega_2/`:

```bash
python -m src.main <comando> [opciones]
```

| Comando | Qué hace | Opciones principales |
|---|---|---|
| `especies` | Consulta los metadatos de una región en xeno-canto (sin descargar audio) y rankea las especies por número de grabaciones de buena calidad | `--region` (`valle_del_cauca`), `--box LAT_MIN LON_MIN LAT_MAX LON_MAX`, `--filtrar-localidad`, `--grupo` (birds), `--calidades` (A B), `--min-grabaciones` (30), `--max-paginas`, `--salida`, `--top` |
| `download` | Busca grabaciones de las especies indicadas, guarda los metadatos y descarga los audios (prioriza la mejor calidad) | `--species` o `--lista` (una de las dos), `--country` (Colombia), `--group` (birds), `--calidades` (A B), `--max-per-species` (60), `--raw-dir`, `--metadata` |
| `preprocess` | Divide los audios en ventanas y guarda espectrogramas Mel + `index.csv` | `--raw-dir`, `--out-dir`, `--sample-rate` (32000), `--segment-seconds` (5), `--hop-seconds` (2.5), `--n-mels` (128), `--reduce-noise` |
| `train` | Entrena un modelo y guarda el mejor checkpoint | `--model` (`cnn_baseline` \| `efficientnet_b0`), `--epochs` (30), `--batch-size` (32), `--lr` (0.001), `--patience` (5), `--no-augment`, `--no-class-weights`, `--no-pretrained`, `--seed`, `--run-name` |
| `evaluate` | Calcula métricas, reporte por especie y matriz de confusión | `--checkpoint` (obligatoria), `--split` (`train` \| `val` \| `test`), `--processed-dir`, `--batch-size` |
| `predict` | Retorna las especies más probables para un audio | `--audio` y `--checkpoint` (obligatorias), `--top-k` (5) |

### Flujo de la Etapa 2

```bash
python -m src.main especies --region valle_del_cauca --filtrar-localidad "Valle del Cauca"
# elegimos las especies y las escribimos en data/especies_objetivo.txt
python -m src.main download --lista data/especies_objetivo.txt --max-per-species 60
python -m src.main preprocess
python -m src.main train --model cnn_baseline --epochs 30 --run-name cnn_v1
python -m src.main evaluate --checkpoint experiments/checkpoints/cnn_v1.pt --split test
```

El paso a paso está en [`guia_etapa_2.md`](guia_etapa_2.md).

## Funciones de Python

| Módulo | Función / clase | Descripción |
|---|---|---|
| `src.data.download` | `build_query(species, country, group, quality, box)` | Construye la consulta con etiquetas de xeno-canto (`sp:`, `cnt:`, `grp:`, `q:`, `box:`) |
| | `search_recordings(query, key=None, max_pages=None)` | Retorna los registros de todas las páginas |
| | `rank_species(metadata, qualities, min_recordings, loc_contains)` | Resumen por especie: grabaciones, grabadores, minutos, etiquetas débiles |
| | `species_ranking(region, box, ...)` | Consulta una región y guarda metadatos y ranking |
| | `download_species(species_list, raw_dir, metadata_csv, ...)` | Busca, guarda metadatos y descarga audios |
| `src.data.preprocess` | `AudioConfig`, `load_audio`, `segment_signal`, `mel_spectrogram`, `audio_to_features`, `preprocess_directory` | Preprocesamiento de audio |
| `src.data.dataset` | `SpectrogramDataset`, `assign_splits(index, val_size, test_size)` | Dataset de PyTorch y partición por grabación |
| `src.models.cnn_baseline` | `build_model(name, n_classes, **kwargs)` | Fábrica de modelos (`cnn_baseline`, `efficientnet_b0`) |
| `src.training.train` | `TrainConfig`, `train(cfg)` | Configuración y bucle de entrenamiento |
| `src.evaluation.evaluate` | `compute_metrics`, `top_k_accuracy`, `evaluate_checkpoint` | Métricas y evaluación de un checkpoint |
| `src.inference.predict` | `predict_file(audio_path, checkpoint, top_k)` | Predicción para un archivo de audio |

## Clave de la API

| Fuente (en orden de prioridad) | Uso |
|---|---|
| Variable de entorno `XC_API_KEY` | Clave de la API v3 de xeno-canto |
| Archivo `entrega_2/.env` con `XC_API_KEY=...` | Alternativa recomendada en Windows; está excluido de Git |
