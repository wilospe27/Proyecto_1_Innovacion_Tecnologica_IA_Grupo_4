# ⚙️ Guía de instalación y dependencias

## Requisitos

| Herramienta | Versión | Notas |
|---|---|---|
| Python | 3.10 – 3.13 (recomendado 3.11) | Los resultados del reporte 2 los generamos con Python 3.13 |
| Git | cualquiera reciente | Para clonar y versionar |
| Conda (opcional) | Miniconda o Anaconda | Alternativa a `venv` |
| FFmpeg (opcional) | — | Solo si el sistema no puede leer archivos `.mp3` (ver *Problemas frecuentes*) |

## 1. Clonar el repositorio

```bash
git clone https://github.com/wilospe27/Proyecto_1_Innovacion_Tecnologica_IA_Grupo_4.git
cd Proyecto_1_Innovacion_Tecnologica_IA_Grupo_4
git checkout entrega-2      # rama de trabajo de la Entrega 2 hasta que la unamos a main
cd entrega_3
```

> Todos los comandos del proyecto los ejecutamos **desde la carpeta `entrega_3/`**.

## 2. Crear el entorno e instalar dependencias

### Opción A — `venv`

**Windows (PowerShell):**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Opción B — Conda

```bash
conda env create -f environment.yml
conda activate cantos-ia
```

### GPU (opcional)

`requirements.txt` instala PyTorch para CPU en Windows y macOS. Para entrenar con una GPU NVIDIA, primero instalamos la versión de PyTorch con CUDA que indica el selector oficial (https://pytorch.org/get-started/locally/) y **después** corremos `pip install -r requirements.txt`. En Google Colab ya viene instalada.

### Entorno para Perch 2.0 (opcional)

Perch 2.0 usa TensorFlow, que no conviene mezclar con PyTorch en el mismo entorno. Lo instalamos aparte, en una ruta corta:

```powershell
py -3.13 -m venv C:\venvs\perch
C:\venvs\perch\Scripts\python.exe -m pip install "perch-hoplite[tf]" librosa soundfile pandas scikit-learn tqdm
# desde entrega_3/: la primera vez descarga el modelo de Kaggle (unos 390 MB)
C:\venvs\perch\Scripts\python.exe -m src.main embeddings --modelo perch_v2
```

En CPU tarda alrededor de una hora para las 11.159 ventanas. Después, el clasificador sobre los *embeddings* se entrena con el entorno principal (`python -m src.main baselines --features embeddings_perch_v2.npy ...`).

### Perch 2.0 en el prototipo (opcional)

Para que la app de Streamlit ofrezca Perch 2.0, TensorFlow tiene que estar en el **mismo entorno** que PyTorch. Lo probamos con Python 3.13, `torch` 2.15 y `tensorflow` 2.21, y ambos funcionan en el mismo proceso. Instalamos `perch-hoplite` sin sus dependencias, porque las suyas bajarían de versión a `librosa` y `pandas`, y eso podría cambiar los espectrogramas con los que entrenamos la CNN y EfficientNet-B0:

```bash
pip install "tensorflow==2.21.0" "kagglehub==0.3.13" "ml_collections==0.1.1" "etils==1.15.0" "absl-py==1.4.0"
pip install --no-deps "perch-hoplite==1.0.2"
```

La primera vez que se elige Perch 2.0, la app entrena el clasificador lineal con `data/processed/embeddings_perch_v2.npy` y lo guarda en `experiments/checkpoints/perch_v2_regresion_logistica.joblib`. Las siguientes veces lo carga directamente. Cargar Perch tarda cerca de medio minuto. Si TensorFlow no está instalado, la app ofrece solo EfficientNet-B0 y la CNN base.

## 3. Verificar la instalación

```bash
python -m pytest            # deben pasar todas las pruebas (sin descargar datos)
python -m src.main --help   # lista los comandos disponibles
```

## 4. Clave de la API de xeno-canto

La API v3 de xeno-canto exige una clave personal y gratuita:

1. Creamos una cuenta en https://xeno-canto.org y copiamos la clave desde la página de la cuenta (https://xeno-canto.org/account).
2. **Opción recomendada:** creamos el archivo `entrega_3/.env` con la línea `XC_API_KEY=la_clave`. El código la lee automáticamente.
3. **Alternativa:** la definimos como variable de entorno `XC_API_KEY` (tiene prioridad sobre `.env`):

| Sistema | Solo para la terminal actual | Permanente |
|---|---|---|
| Windows (PowerShell) | `$env:XC_API_KEY = "la_clave"` | `setx XC_API_KEY "la_clave"` (y abrimos una terminal nueva) |
| Linux / macOS | `export XC_API_KEY="la_clave"` | Agregamos la línea anterior a `~/.bashrc` o `~/.zshrc` |

4. Probamos la conexión con una consulta corta de metadatos:

```bash
python -m src.main especies --max-paginas 1 --top 10
```

> ⚠️ **Nunca** escribimos la clave en el código, en notebooks ni en archivos versionados. El archivo `.env` está excluido en `.gitignore`.

## 5. Google Colab

```python
!git clone https://github.com/wilospe27/Proyecto_1_Innovacion_Tecnologica_IA_Grupo_4.git
%cd Proyecto_1_Innovacion_Tecnologica_IA_Grupo_4/entrega_2
!pip install -q librosa soundfile noisereduce

import os
from google.colab import userdata          # guardamos la clave en "Secrets" con el nombre XC_API_KEY
os.environ["XC_API_KEY"] = userdata.get("XC_API_KEY")

!python -m pytest -q
```

Para usar la GPU, la activamos en *Entorno de ejecución → Cambiar tipo de entorno de ejecución*.

## 6. Problemas frecuentes

| Síntoma | Causa probable | Solución |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | El comando se ejecutó fuera de `entrega_3/` o sin `-m` | Nos ubicamos en la carpeta (`cd entrega_3`) y usamos `python -m src.main ...` |
| Error al leer archivos `.mp3` | Falta un decodificador de audio | `conda install -c conda-forge ffmpeg` o, en Windows, `winget install Gyan.FFmpeg` |
| `pip` falla al instalar PyTorch con `OSError: [Errno 2] No such file or directory` (Windows) | La ruta del repositorio es muy larga y Windows limita las rutas a 260 caracteres | Creamos el entorno en una ruta corta (`py -3.11 -m venv C:\venvs\cantos` y `C:\venvs\cantos\Scripts\Activate.ps1`) o activamos las rutas largas de Windows |
| PowerShell no deja activar `.venv` | Política de ejecución de scripts | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `RuntimeError: No se encontró la clave de la API` | Falta `XC_API_KEY` | Ver la sección 4 |
| Memoria insuficiente al entrenar | Lote muy grande | `python -m src.main train --batch-size 16` |
