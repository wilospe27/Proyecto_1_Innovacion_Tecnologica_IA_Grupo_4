# 🐦 Identificación de especies por su canto
### Aves y fauna de Colombia mediante Inteligencia Artificial

**Proyecto 1 · Innovación Tecnológica con IA · Grupo 4**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow%20%2F%20PyTorch-DL-orange)
![Estado](https://img.shields.io/badge/estado-en%20desarrollo-yellow)
![Licencia](https://img.shields.io/badge/licencia-MIT-green)

---

## 📌 Descripción

Colombia es el país con mayor diversidad de aves del mundo (más de 1.900 especies registradas), además de una enorme riqueza de anfibios, insectos y mamíferos que se comunican mediante sonidos. Muchas de estas especies son difíciles de ver, pero fáciles de **escuchar**.

Este proyecto desarrolla un sistema basado en **aprendizaje profundo** que, a partir de una grabación de audio, **identifica la especie que está cantando o vocalizando**. Su objetivo es apoyar el monitoreo de biodiversidad, la ciencia ciudadana, la educación ambiental y el ecoturismo.

## 🎯 Objetivos

**General**

Construir un modelo de IA capaz de clasificar especies de fauna colombiana a partir de sus vocalizaciones.

**Específicos**

- Recopilar y depurar un conjunto de datos de audios etiquetados de especies colombianas.
- Transformar los audios en representaciones visuales (espectrogramas Mel) aptas para redes neuronales.
- Entrenar y comparar modelos de clasificación (CNN propia, transfer learning y modelos preentrenados como BirdNET).
- Evaluar el desempeño con métricas estándar (accuracy, precisión, recall, F1, matriz de confusión).
- Ofrecer una interfaz sencilla donde el usuario suba o grabe un audio y obtenga la especie probable.

## 🧠 ¿Cómo funciona?

```
 Audio (.wav/.mp3)
        │
        ▼
 1. Preprocesamiento ──► remuestreo, recorte en ventanas, reducción de ruido
        │
        ▼
 2. Extracción de características ──► espectrograma Mel / MFCC
        │
        ▼
 3. Modelo de IA (CNN / Transfer Learning)
        │
        ▼
 4. Predicción ──► Top-N especies + probabilidad
```

## 📂 Fuentes de datos

| Fuente | Descripción |
|---|---|
| [Xeno-canto](https://xeno-canto.org) | Base colaborativa de cantos de aves, con miles de registros de Colombia |
| [Macaulay Library – Cornell Lab](https://www.macaulaylibrary.org) | Archivo científico de sonidos de fauna |
| [Colección de Sonidos Ambientales – Instituto Humboldt](https://www.humboldt.org.co) | Grabaciones de fauna colombiana |
| [iNaturalist](https://www.inaturalist.org) | Observaciones ciudadanas con audio |

> ⚠️ Respetar la licencia de cada grabación (la mayoría Creative Commons) y citar a los autores.

## 🛠️ Tecnologías

- **Lenguaje:** Python 3.10+
- **Audio:** `librosa`, `torchaudio`, `pydub`, `noisereduce`
- **Modelado:** `TensorFlow/Keras` o `PyTorch`, `scikit-learn`
- **Modelos de referencia:** BirdNET, YAMNet, EfficientNet/ResNet (transfer learning)
- **Visualización:** `matplotlib`, `seaborn`
- **Interfaz:** `Streamlit` o `Gradio`
- **Entorno:** Jupyter Notebook / Google Colab

## 📦 Entregas

Cada entrega vive en su propia carpeta con la estructura estándar del curso (`docs/`, `src/`, `notebooks/`, `experiments/`, `tests/`, `requirements.txt`, `README.md`).

| Entrega | Fecha | Contenido | Peso | Carpeta | Estado |
|---|---|---|---|---|---|
| 1 | 19-oct-2026 | Análisis del problema, estado del arte y planeación | 20 % | [`entrega_1/`](entrega_1/) | ✅ Lista para revisión del equipo |
| 2 | 19-nov-2026 | Análisis exploratorio de datos, modelos de referencia e hipótesis | 10 % | `entrega_2/` (rama `entrega-2`) | 🚧 En progreso |
| 3 | 1-dic-2026 | Reporte final y video de 10 minutos | 70 % | `entrega_3/` | ⏳ Pendiente |

📄 **Reporte de la Entrega 1:** [`entrega_1/docs/reporte_entrega_1.md`](entrega_1/docs/reporte_entrega_1.md) · 📅 **Cronograma:** [`entrega_1/docs/cronograma.xlsx`](entrega_1/docs/cronograma.xlsx)

## 📁 Estructura del repositorio

```
Proyecto_1_innovaci-n_tecnol-gica_IA_Grupo_4/
├── README.md                 # Este archivo
├── LICENSE
├── .gitignore
└── entrega_1/
    ├── docs/                 # Reporte, cronograma (Excel), arquitectura, API e instalación
    ├── src/                  # data/ · models/ · training/ · evaluation/ · inference/ · utils/ · main.py
    ├── notebooks/            # 01_exploracion_inicial.ipynb
    ├── data/                 # raw/ · processed/ · metadata/ (no se versionan)
    ├── experiments/          # logs/ · checkpoints/ · results/
    ├── tests/                # Pruebas con pytest
    ├── requirements.txt
    ├── environment.yml
    ├── .gitignore
    ├── LICENSE
    └── README.md
```

## 🚀 Instalación y uso

```bash
# 1. Clonar el repositorio y entrar a la entrega
git clone https://github.com/wilospe27/Proyecto_1_innovaci-n_tecnol-gica_IA_Grupo_4.git
cd Proyecto_1_innovaci-n_tecnol-gica_IA_Grupo_4/entrega_1

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1

# 3. Instalar dependencias y verificar
pip install -r requirements.txt
python -m pytest
```

**Comandos del proyecto** (ver [`entrega_1/docs/api.md`](entrega_1/docs/api.md)):

```bash
python -m src.main download --species "Turdus fuscater" --max-per-species 30   # requiere XC_API_KEY
python -m src.main preprocess
python -m src.main train --model cnn_baseline --epochs 30 --run-name cnn_v1
python -m src.main predict --audio ruta/al/audio.wav --checkpoint experiments/checkpoints/cnn_v1.pt
```

La interfaz web (Streamlit/Gradio) se desarrollará en la Entrega 3.

## 📊 Resultados

| Modelo | Accuracy | F1-score | Observaciones |
|---|---|---|---|
| CNN base | — | — | Pendiente |
| Transfer learning (EfficientNet) | — | — | Pendiente |
| BirdNET (referencia) | — | — | Pendiente |

*Se actualizará a medida que avance el entrenamiento.*

## 🌱 Impacto y aplicaciones

- **Conservación:** monitoreo acústico pasivo de especies amenazadas o endémicas.
- **Ciencia ciudadana:** cualquier persona puede identificar lo que escucha con su celular.
- **Educación ambiental:** herramienta didáctica para colegios y universidades.
- **Ecoturismo:** apoyo a guías y observadores de aves.

## ⚠️ Retos y limitaciones

- Ruido de fondo (viento, lluvia, otras especies cantando a la vez).
- Desbalance de clases: especies comunes con muchos audios y especies raras con pocos.
- Variaciones regionales y dialectos en el canto de una misma especie.
- Necesidad de validar las predicciones con expertos.

## 🗺️ Hoja de ruta

- [ ] Definir la lista de especies objetivo
- [ ] Descargar y depurar el dataset
- [ ] Preprocesamiento y generación de espectrogramas
- [ ] Entrenamiento del modelo base
- [ ] Transfer learning y comparación de modelos
- [ ] Interfaz web de demostración
- [ ] Informe final y presentación

## 👥 Equipo – Grupo 4

| Integrante | Contacto |
|---|---|
| Wilson Ospina Pérez | wilospe27@gmail.com |
| Santiago Osorio | Santiago.osorio@hotmail.es |
| Martín Pérez | martinperez_lopez@hotmail.com |
| Adriana Zapata | adriana.zapata.gomez.2021@gmail.com |

## 📚 Referencias

- Kahl, S., Wood, C. M., Eibl, M., & Klinck, H. (2021). *BirdNET: A deep learning solution for avian diversity monitoring.* Ecological Informatics, 61.
- Xeno-canto Foundation. *Sharing wildlife sounds from around the world.* https://xeno-canto.org
- McFee, B. et al. *librosa: Audio and music signal analysis in Python.*

## 📄 Licencia

Este proyecto se distribuye bajo la licencia **MIT**. Los audios usados conservan la licencia original de sus autores.
