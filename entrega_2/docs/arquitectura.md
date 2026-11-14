# 🏗️ Arquitectura y diseño del modelo

Este es el diseño técnico de nuestro sistema, versión de la Entrega 2. Partimos del diseño de la Entrega 1 y lo ajustamos con los resultados del análisis exploratorio y de los modelos de referencia (ver [`reporte_entrega_2.md`](reporte_entrega_2.md)).

## 1. Flujo de extremo a extremo

```mermaid
flowchart LR
    subgraph Datos["1 · Datos"]
        A["xeno-canto API v3<br/>+ otras fuentes"] --> B["data/raw/<br/>Genero_especie/XC123.mp3"]
        A --> M["data/metadata/<br/>licencia · autor · calidad"]
    end
    subgraph Prep["2 · Preprocesamiento"]
        B --> C["Mono · 32 kHz"] --> D["Ventanas de 5 s<br/>50 % de solapamiento"] --> E["log-Mel 128 × 501<br/>estandarizado"]
    end
    subgraph Modelo["3 · Modelado"]
        E --> F["Partición por grabación<br/>70 / 15 / 15"]
        F --> G1["SVM<br/>sobre MFCC"]
        F --> G2["Random Forest<br/>sobre MFCC"]
        F --> G3["CNN base<br/>sobre espectrograma"]
        F --> G4["EfficientNet-B0<br/>(transferencia)"]
        F --> G5["Embeddings bioacústicos<br/>BirdNET / Perch + clasificador lineal"]
    end
    subgraph Salida["4 · Evaluación e inferencia"]
        G1 & G2 & G3 & G4 & G5 --> H["F1 macro · top-3<br/>matriz de confusión"]
        G1 & G2 & G3 & G4 & G5 --> I["Audio nuevo → promedio de ventanas<br/>→ top-N especies"]
    end
```

| Paso | Módulo | Comando |
|---|---|---|
| Ranking de especies de la región | `src/data/download.py` | `python -m src.main especies --region valle_del_cauca` |
| Descarga | `src/data/download.py` | `python -m src.main download --lista data/especies_objetivo.txt` |
| Depuración | `src/data/clean.py` | `python -m src.main depurar` |
| Preprocesamiento | `src/data/preprocess.py` | `python -m src.main preprocess --manifest data/metadata/depuracion.csv --max-seconds 60` |
| Análisis exploratorio | `src/analysis/eda.py` | `python -m src.main eda` |
| Modelos clásicos | `src/models/classical.py` | `python -m src.main baselines` |
| Dataset y particiones | `src/data/dataset.py` | (lo usa el entrenamiento) |
| Modelos | `src/models/cnn_baseline.py` | — |
| Entrenamiento | `src/training/train.py` | `python -m src.main train` |
| Evaluación | `src/evaluation/evaluate.py` | `python -m src.main evaluate --checkpoint ...` |
| Inferencia | `src/inference/predict.py` | `python -m src.main predict --audio ... --checkpoint ...` |
| Utilidades | `src/utils/helpers.py` | — |

## 2. Datos

```
data/raw/<Genero_especie>/<id>.mp3          # la carpeta define la etiqueta
data/processed/<Genero_especie>/<id>_000.npy  # una ventana = un espectrograma
data/processed/index.csv                      # path, label, recording_id, segment, split, split_ventana
data/processed/mfcc_stats.npy                 # 80 estadísticas de MFCC por ventana (mismo orden del índice)
data/processed/audio_config.json              # parámetros usados (se reutilizan en la inferencia)
data/metadata/depuracion.csv                  # manifiesto: qué archivos conservamos y por qué descartamos el resto
```

- **Depuración:** descartamos archivos ilegibles, duplicados, menores de 1 s o con otra especie objetivo al fondo. No borramos archivos: el preprocesamiento lee el manifiesto.
- **Tope por grabación:** usamos los primeros 60 s de cada grabación para que las grabaciones largas no dominen su especie.

- **Etiqueta:** nombre de la carpeta de la especie (`Turdus_fuscater`).
- **`recording_id`:** identifica la grabación de origen de cada ventana; la usamos para agrupar en la partición.
- **`split`:** partición por grabación (`train` / `val` / `test`). La crea el primer comando que la necesita (`baselines` o `train`) y queda guardada para que todos los modelos comparen sobre la misma partición.
- **`split_ventana`:** partición al azar por ventana. Solo la usamos para medir la fuga de información (hipótesis H3); nunca para reportar desempeño.

## 3. Preprocesamiento

| Parámetro | Valor | Justificación |
|---|---|---|
| Frecuencia de muestreo | 32 kHz, mono | Cubre hasta 16 kHz, donde está la energía de la mayoría de cantos; valor usual en BirdCLEF |
| Duración de ventana | 5 s | Contiene frases completas de canto; es la unidad usada en BirdCLEF |
| Solapamiento | 50 % (salto de 2,5 s) | Evita cortar vocalizaciones en el borde de una ventana |
| FFT / salto | 1024 / 320 muestras (10 ms) | Buen compromiso entre resolución temporal y frecuencial |
| Bandas Mel | 128, de 50 Hz a 14 kHz | Reduce parte del ruido de muy baja frecuencia (viento, tráfico) |
| Escala | dB relativos al máximo, rango de 80 dB | Comprime el rango dinámico |
| Normalización | media 0, desviación 1 por ventana | Reduce diferencias de volumen entre grabadoras |
| Reducción de ruido | opcional (`--reduce-noise`) | Evaluaremos su efecto en la Entrega 2 |

Forma de cada ventana: **128 × 501** (bandas Mel × cuadros de tiempo).

## 4. Modelos

### 4.1 CNN base (`cnn_baseline`)

| Capa | Salida (para 128 × 501) |
|---|---|
| Entrada | 1 × 128 × 501 |
| Bloque 1: 2 × (Conv 3×3 → BatchNorm → ReLU), MaxPool 2×2, Dropout | 32 × 64 × 250 |
| Bloque 2 | 64 × 32 × 125 |
| Bloque 3 | 128 × 16 × 62 |
| Bloque 4 | 256 × 8 × 31 |
| *Pooling* promedio global | 256 |
| Dropout 0,3 + capa lineal | N especies |

≈ 1,2 millones de parámetros con 20 especies. El *pooling* global permite procesar ventanas de distinta duración.

**Versión liviana (`--base-channels 16`).** Con 16, 32, 64 y 128 canales tiene 295.647 parámetros (15 especies) y entrena unas 2,5 veces más rápido en CPU. Es la que usamos en las pruebas iniciales de la Entrega 2, porque entrenamos sin GPU.

### 4.0 Modelos clásicos de referencia (`src/models/classical.py`)

Clase mayoritaria, regresión logística, SVM (RBF), bosque aleatorio y potenciación de gradiente sobre 80 estadísticas de MFCC por ventana (media y desviación estándar en el tiempo de 20 MFCC y de sus deltas). Usan la misma partición y las mismas métricas que la CNN.

### 4.2 Transferencia de aprendizaje (`efficientnet_b0`)

Usamos EfficientNet-B0 de `torchvision` con pesos de ImageNet: replicamos el espectrograma en 3 canales y reemplazamos la capa final por una de N especies. Ajustamos todo el modelo con una tasa de aprendizaje menor.

### 4.3 *Embeddings* bioacústicos (Entrega 3)

Extraeremos vectores de características con un modelo bioacústico preentrenado (BirdNET o Perch 2.0) y entrenaremos un clasificador lineal (por ejemplo, regresión logística). Es la opción que recomendamos cuando hay pocas grabaciones por especie (ver estado del arte, trabajos 5 y 6).

## 5. Entrenamiento

| Elemento | Configuración inicial |
|---|---|
| Pérdida | Entropía cruzada con pesos por clase (inverso de la frecuencia) |
| Optimizador | AdamW, tasa de aprendizaje 1e-3, *weight decay* 1e-4 |
| Programación de la tasa | `ReduceLROnPlateau` sobre el F1 macro de validación (factor 0,5) |
| Lote / épocas | 32 / hasta 30, con parada temprana (paciencia 5) |
| Aumentación | Máscaras aleatorias de frecuencia y tiempo (tipo SpecAugment); *mixup* y mezcla de ruido en la Entrega 3 |
| Selección del modelo | Mejor F1 macro en validación |
| Reproducibilidad | Semilla fija (42) y partición guardada en `index.csv` |

Salidas de cada corrida (en `experiments/`): `checkpoints/<corrida>.pt`, `logs/<corrida>.log`, `logs/<corrida>_history.csv` y `results/<corrida>_train_summary.json`.

## 6. Evaluación

- **Partición por grabación** con `StratifiedGroupKFold`: ninguna grabación aparece en dos particiones, lo que evita la fuga de información entre ventanas de un mismo audio.
- **Métrica principal:** F1 macro. **Complementarias:** exactitud, exactitud balanceada, precisión y exhaustividad macro, exactitud top-3.
- **Artefactos:** `results/<corrida>_<split>_metricas.json`, `..._por_especie.csv` y `..._matriz_confusion.png`.

## 7. Inferencia

1. Cargamos el audio con la misma configuración del entrenamiento (guardada en el checkpoint).
2. Lo dividimos en ventanas de 5 s y calculamos la probabilidad de cada especie por ventana.
3. Promediamos las probabilidades y devolvemos las N especies más probables.

## 8. Decisiones tomadas en la Entrega 2

- **Especies:** 15 especies objetivo con datos de toda Colombia; el Valle del Cauca solo confirma la presencia regional.
- **Datos:** calidades A y B, máximo 60 grabaciones por especie, tope de 60 s por grabación y descarte de grabaciones con otra especie objetivo al fondo.
- **Evaluación:** partición por grabación como única forma válida de reportar desempeño.

## 9. Decisiones pendientes (Entrega 3)

- Filtro pasa-altos o frecuencia mínima mayor del espectrograma: el análisis exploratorio mostró ruido de baja frecuencia asociado a algunas especies.
- Efecto de la reducción de ruido y del descarte de ventanas silenciosas.
- Agregación por grabación: promedio frente a máximo de probabilidades.
- Modelo bioacústico para los *embeddings* (BirdNET o Perch 2.0) según licencia y facilidad de uso.
