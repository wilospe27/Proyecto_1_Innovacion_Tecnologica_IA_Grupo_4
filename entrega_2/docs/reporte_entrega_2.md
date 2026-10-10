# Identificación de especies de fauna colombiana por su canto mediante aprendizaje profundo

**Reporte de la segunda etapa · Entrega 2** — *plantilla en construcción*

Proyecto 1 · Innovación Tecnológica con IA · **Grupo 4**  
Universidad Icesi · Facultad Barberi de Ingeniería, Diseño y Ciencias Aplicadas  
Integrantes: Wilson Ospina Pérez · Santiago Osorio · Martín Pérez · Adriana Zapata  
Fecha de entrega: 19 de noviembre de 2026

> **Cómo usar esta plantilla:** cada sección indica qué debe contener (en cursiva) y las tareas del cronograma que la alimentan. Reemplacen las indicaciones por el contenido final y borren esta nota antes de entregar.

| Criterio de la rúbrica | Peso | Secciones |
|---|---|---|
| Análisis exploratorio: calidad de la exploración y justificación de la selección de datos y variables | 5 % | II y III |
| Modelos de referencia y experimentos: planteamiento de hipótesis y pruebas iniciales | 5 % | IV y V |

---

**Resumen.** *Un párrafo de 150–200 palabras: datos obtenidos, especies elegidas, hallazgos principales del EDA, hipótesis y resultados de los modelos de referencia.*

---

## I. Introducción

*Recordar en pocas líneas el problema, la pregunta de investigación y los objetivos de la Entrega 1 (enlazar `entrega_1/docs/reporte_entrega_1.md`) y explicar qué cubre esta etapa.*

## II. Datos

### A. Fuentes y consulta (tarea 2.1)

*Fuente (xeno-canto, API v3), fecha de la consulta, región (rectángulo del Valle del Cauca y filtro por localidad), grupo taxonómico y número de grabaciones y especies obtenidas.*

### B. Selección de especies objetivo (tarea 2.1)

*Criterios (≥ 30 grabaciones de calidad A/B, número de grabadores, 20–30 especies), tabla con el ranking resumido (`experiments/results/ranking_especies.csv`) y lista final con la razón de cada inclusión o exclusión relevante. Explicar por qué se descargan grabaciones de toda Colombia para las especies elegidas.*

### C. Descarga y depuración (tareas 2.2 y 2.3)

*Número de archivos descargados por especie, archivos descartados y por qué (corruptos, duplicados, etiquetas dudosas, voces humanas), y resumen de licencias y autores.*

## III. Análisis exploratorio de datos (tarea 2.4)

### A. Calidad de los datos

*Valores faltantes en los metadatos, distribución de calidades (A–E), duraciones, frecuencias de muestreo y proporción de grabaciones con otras especies de fondo (etiquetas débiles).*

### B. Distribuciones

*Grabaciones y minutos por especie (desbalance), grabadores por especie, tipo de vocalización (canto o llamado) y distribución geográfica (mapa).*

### C. Exploración acústica

*Espectrogramas de ejemplo y promedio por especie, rangos de frecuencia dominantes y especies con cantos parecidos que podrían confundirse.*

### D. Justificación de la selección de datos y variables

*Por qué las ventanas de 5 s, la frecuencia de muestreo de 32 kHz y los 128 filtros Mel entre 50 Hz y 14 kHz; filtros de calidad aplicados; variables usadas por los modelos de referencia (estadísticas de MFCC) y por la CNN (espectrogramas).*

## IV. Hipótesis y diseño experimental (tarea 2.6)

*Formalizar las hipótesis preliminares del reporte 1 con criterios medibles. Por ejemplo:*

| Hipótesis | Predicción medible | Cómo se pone a prueba | Entrega |
|---|---|---|---|
| H1. Las representaciones preentrenadas superan a una CNN entrenada desde cero con pocas grabaciones por especie | F1 macro mayor en prueba | CNN base (Entrega 2) frente a EfficientNet y *embeddings* (Entrega 3), misma partición | 2–3 |
| H2. La aumentación mejora las especies con menos grabaciones | Mayor F1 en el tercio de especies con menos datos | CNN con y sin aumentación | 2 |
| H3. Particionar por ventana sobrestima el desempeño | F1 por ventana aleatoria > F1 por grabación | Misma CNN con ambas particiones | 2 |

*Protocolo: partición estratificada por grabación 70/15/15, semilla fija, métrica principal F1 macro, métricas complementarias.*

## V. Modelos de referencia y pruebas iniciales (tarea 2.7)

### A. Modelos

*Clase mayoritaria; regresión logística sobre estadísticas de MFCC; CNN base (`src/models/cnn_baseline.py`). Hiperparámetros usados.*

### B. Resultados iniciales

| Modelo | Exactitud | Exactitud balanceada | F1 macro | Top-3 |
|---|---|---|---|---|
| Clase mayoritaria | | | | |
| Regresión logística (MFCC) | | | | |
| CNN base | | | | |

*Incluir la matriz de confusión de la CNN base (`experiments/results/`).*

### C. Discusión

*Qué dicen los resultados sobre H2 y H3, especies con más errores y por qué, y ajustes para la Etapa 3.*

## VI. Próximos pasos (Etapa 3)

*Transferencia de aprendizaje con EfficientNet-B0, *embeddings* de BirdNET o Perch 2.0, ajuste de hiperparámetros, prototipo y validación con expertos (tareas 3.1–3.6).*

## VII. Referencias

*Mantener el formato IEEE del reporte 1 y agregar las fuentes nuevas (por ejemplo, la documentación de la API de xeno-canto).*
