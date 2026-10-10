# 🔌 Interfaces del proyecto (CLI y funciones de Python)

En la Entrega 1 todavía no exponemos una API web: la interfaz es la **línea de comandos** de `src/main.py` y las funciones de Python que esta usa. En la Entrega 3 documentaremos aquí la interfaz web de demostración (Streamlit o Gradio).

## Línea de comandos

Ejecutar desde `entrega_1/`:

```bash
python -m src.main <comando> [opciones]
```

| Comando | Qué hace | Opciones principales |
|---|---|---|
| `download` | Busca grabaciones en xeno-canto, guarda los metadatos y descarga los audios | `--species` (obligatoria, uno o más nombres científicos), `--country` (Colombia), `--group` (birds), `--quality` (A–E), `--max-per-species` (30), `--raw-dir`, `--metadata` |
| `preprocess` | Divide los audios en ventanas y guarda espectrogramas Mel + `index.csv` | `--raw-dir`, `--out-dir`, `--sample-rate` (32000), `--segment-seconds` (5), `--hop-seconds` (2.5), `--n-mels` (128), `--reduce-noise` |
| `train` | Entrena un modelo y guarda el mejor checkpoint | `--model` (`cnn_baseline` \| `efficientnet_b0`), `--epochs` (30), `--batch-size` (32), `--lr` (0.001), `--patience` (5), `--no-augment`, `--no-class-weights`, `--no-pretrained`, `--seed`, `--run-name` |
| `evaluate` | Calcula métricas, reporte por especie y matriz de confusión | `--checkpoint` (obligatoria), `--split` (`train` \| `val` \| `test`), `--processed-dir`, `--batch-size` |
| `predict` | Retorna las especies más probables para un audio | `--audio` y `--checkpoint` (obligatorias), `--top-k` (5) |

### Ejemplo de flujo completo

```bash
python -m src.main download --species "Turdus fuscater" "Pitangus sulphuratus" "Zonotrichia capensis" --max-per-species 30
python -m src.main preprocess
python -m src.main train --model cnn_baseline --epochs 30 --run-name cnn_v1
python -m src.main evaluate --checkpoint experiments/checkpoints/cnn_v1.pt --split test
python -m src.main predict --audio mi_grabacion.wav --checkpoint experiments/checkpoints/cnn_v1.pt
```

Salida de `predict` (formato; valores ilustrativos):

```
             especie  probabilidad
     Turdus fuscater        0.8123
 Zonotrichia capensis       0.1032
 ...
```

## Funciones de Python

| Módulo | Función / clase | Descripción |
|---|---|---|
| `src.data.download` | `build_query(species, country, group, quality)` | Construye la consulta con etiquetas de xeno-canto (`sp:`, `cnt:`, `grp:`, `q:`) |
| | `search_recordings(query, key=None, max_pages=None)` | Retorna los registros de todas las páginas |
| | `download_species(species_list, raw_dir, metadata_csv, ...)` | Busca, guarda metadatos y descarga audios |
| `src.data.preprocess` | `AudioConfig` | Parámetros del preprocesamiento |
| | `load_audio(path, sample_rate)` | Carga en mono y remuestrea |
| | `segment_signal(signal, segment_samples, hop_samples)` | Ventanas con solapamiento |
| | `mel_spectrogram(segment, cfg)` / `audio_to_features(signal, cfg)` | Espectrograma log-Mel de una ventana / de toda la grabación |
| | `preprocess_directory(raw_dir, output_dir, cfg)` | Procesa una carpeta completa y genera el índice |
| `src.data.dataset` | `SpectrogramDataset`, `assign_splits(index, val_size, test_size)` | Dataset de PyTorch y partición por grabación |
| `src.models.cnn_baseline` | `build_model(name, n_classes, **kwargs)` | Fábrica de modelos (`cnn_baseline`, `efficientnet_b0`) |
| `src.training.train` | `TrainConfig`, `train(cfg)` | Configuración y bucle de entrenamiento |
| `src.evaluation.evaluate` | `compute_metrics`, `top_k_accuracy`, `evaluate_checkpoint` | Métricas y evaluación de un checkpoint |
| `src.inference.predict` | `predict_file(audio_path, checkpoint, top_k)` | Predicción para un archivo de audio |

Ejemplo:

```python
from src.inference.predict import predict_file

tabla = predict_file("mi_grabacion.wav", "experiments/checkpoints/cnn_v1.pt", top_k=3)
print(tabla)
```

## Variables de entorno

| Variable | Uso |
|---|---|
| `XC_API_KEY` | Clave de la API v3 de xeno-canto (obligatoria para `download`) |
