# ⚙️ Guía de instalación y dependencias

## Requisitos

| Herramienta | Versión | Notas |
|---|---|---|
| Python | 3.10 – 3.12 (recomendado 3.11) | |
| Git | cualquiera reciente | Para clonar y versionar |
| Conda (opcional) | Miniconda o Anaconda | Alternativa a `venv` |
| FFmpeg (opcional) | — | Solo si el sistema no puede leer archivos `.mp3` (ver *Problemas frecuentes*) |

## 1. Clonar el repositorio

```bash
git clone https://github.com/wilospe27/Proyecto_1_Innovacion_Tecnologica_IA_Grupo_4.git
cd Proyecto_1_Innovacion_Tecnologica_IA_Grupo_4/entrega_1
```

> Todos los comandos del proyecto los ejecutamos **desde la carpeta `entrega_1/`**.

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

## 3. Verificar la instalación

```bash
python -m pytest            # deben pasar todas las pruebas (sin descargar datos)
python -m src.main --help   # lista los comandos disponibles
```

## 4. Clave de la API de xeno-canto

La API v3 de xeno-canto exige una clave personal y gratuita:

1. Creamos una cuenta en https://xeno-canto.org y copiamos la clave desde la página de la cuenta (https://xeno-canto.org/account).
2. La definimos como variable de entorno `XC_API_KEY`:

| Sistema | Solo para la terminal actual | Permanente |
|---|---|---|
| Windows (PowerShell) | `$env:XC_API_KEY = "la_clave"` | `setx XC_API_KEY "la_clave"` (y abrimos una terminal nueva) |
| Linux / macOS | `export XC_API_KEY="la_clave"` | Agregamos la línea anterior a `~/.bashrc` o `~/.zshrc` |

3. Probamos la descarga con una especie:

```bash
python -m src.main download --species "Turdus fuscater" --max-per-species 5
```

> ⚠️ **Nunca** escribimos la clave en el código, en notebooks ni en archivos versionados. El archivo `.env` está excluido en `.gitignore`.

## 5. Google Colab

```python
!git clone https://github.com/wilospe27/Proyecto_1_Innovacion_Tecnologica_IA_Grupo_4.git
%cd Proyecto_1_Innovacion_Tecnologica_IA_Grupo_4/entrega_1
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
| `ModuleNotFoundError: No module named 'src'` | El comando se ejecutó fuera de `entrega_1/` o sin `-m` | Nos ubicamos en la carpeta (`cd entrega_1`) y usamos `python -m src.main ...` |
| Error al leer archivos `.mp3` | Falta un decodificador de audio | `conda install -c conda-forge ffmpeg` o, en Windows, `winget install Gyan.FFmpeg` |
| PowerShell no deja activar `.venv` | Política de ejecución de scripts | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `RuntimeError: No se encontró la clave de la API` | Falta `XC_API_KEY` | Ver la sección 4 |
| Memoria insuficiente al entrenar | Lote muy grande | `python -m src.main train --batch-size 16` |
