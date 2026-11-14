# 🧭 Guía de la Etapa 2 · De los datos a los modelos de referencia

Este es nuestro paso a paso para reproducir la Entrega 2: selección de especies (tarea 2.1), descarga (2.2), depuración (2.3), análisis exploratorio (2.4), preprocesamiento (2.5) y modelos de referencia (2.7).

> Estos comandos los ejecutamos en nuestros computadores o en Google Colab: la descarga necesita la clave personal de xeno-canto y conexión directa a xeno-canto.org.

## Paso 1 · Clave de la API (5 minutos)

1. Creamos una cuenta en https://xeno-canto.org (basta una para todo el grupo).
2. Copiamos la clave desde la página de la cuenta: https://xeno-canto.org/account.
3. La guardamos en un archivo llamado `.env` dentro de `entrega_2/`, con una sola línea:

   ```
   XC_API_KEY=la_clave
   ```

   `.env` está en `.gitignore`, así que no se sube a GitHub. No compartimos la clave por chat ni la escribimos en notebooks.

## Paso 2 · Entorno

Desde la raíz del repositorio, en la rama `entrega-2` (ver [`instalacion.md`](instalacion.md)):

```powershell
git checkout entrega-2
cd entrega_2
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest
```

## Paso 3 · Ranking de especies de la región (tarea 2.1)

```bash
python -m src.main especies --region valle_del_cauca --filtrar-localidad "Valle del Cauca"
```

- Solo consulta **metadatos** (no descarga audio). Para una prueba rápida agregamos `--max-paginas 2`.
- El rectángulo `valle_del_cauca` usa las coordenadas extremas del departamento e incluye franjas de los departamentos vecinos; `--filtrar-localidad` deja solo las grabaciones cuya localidad menciona el Valle del Cauca.
- Salidas:
  - `experiments/results/ranking_especies.csv` → **lo versionamos**, porque sustenta la selección en el reporte 2.
  - `data/metadata/xeno_canto_region.csv` → metadatos completos (no los versionamos).

| Columna del ranking | Significado |
|---|---|
| `especie`, `nombre_en` | Nombre científico y nombre en inglés |
| `grabaciones` | Grabaciones de calidad A o B en la región |
| `grabaciones_total` | Grabaciones de cualquier calidad |
| `grabadores` | Autores distintos de las grabaciones A/B |
| `minutos` | Minutos de audio de calidad A/B |
| `con_otras_especies` | Grabaciones A/B que mencionan especies de fondo (etiquetas débiles) |
| `cumple` | `grabaciones` ≥ 30 |

> **Lo que encontramos al correrlo (18-oct-2026):** en el rectángulo del Valle ninguna especie identificada alcanza las 30 grabaciones A/B. Por eso corrimos también un ranking **nacional** (`python -m src.main especies --box -4.3 -79.1 13.5 -66.8`), guardamos el regional como `ranking_especies_valle.csv`, y aplicamos el mínimo de 30 sobre el conteo nacional, usando el ranking regional solo para confirmar la presencia de la especie en el Valle.

## Paso 4 · Elegir las especies objetivo (reunión del equipo)

Usamos los criterios que acordamos en el reporte 1 (sección VI.B) y dos más que sugerimos:

1. Al menos **30 grabaciones de calidad A o B en el ranking nacional** y presencia en el ranking regional (≥ 2 grabaciones A/B en el rectángulo del Valle).
2. *Sugerido:* varios grabadores por especie (por ejemplo, 5 o más), para que el modelo no aprenda el equipo de una sola persona.
3. *Sugerido:* incluir algunos pares de especies con cantos parecidos (por ejemplo, del mismo género), para que la evaluación sea exigente.
4. Total de **10 a 20 especies**, el subconjunto que define el planteamiento del curso.

Escribimos la lista en [`../data/especies_objetivo.txt`](../data/especies_objetivo.txt), una especie por línea, y registramos la decisión y sus razones en el reporte 2 (sección II.B).

## Paso 5 · Descargar las grabaciones (tarea 2.2)

```bash
python -m src.main download --lista data/especies_objetivo.txt --max-per-species 60
```

- Descarga grabaciones de **toda Colombia** para las especies elegidas: más datos y más variación regional. Prioriza calidad A, luego B.
- Hace una pausa de un segundo entre descargas para respetar el servidor: nuestras 870 grabaciones tardaron 27 minutos y ocupan 2,1 GB.
- Resultado: `data/raw/<Genero_especie>/XC<id>.mp3` y `data/metadata/xeno_canto.csv` (con licencia y autor de cada archivo).

## Paso 6 · Compartir los datos en el equipo

Los audios no van a GitHub. Tenemos dos opciones:

- **Carpeta compartida** (Google Drive u OneDrive) con `data/raw/` comprimido.
- **Repetir la descarga** con la misma lista: cada integrante obtiene los mismos archivos, salvo grabaciones publicadas después.

## Paso 7 · Depurar (tarea 2.3)

```bash
python -m src.main depurar
```

Revisa cada archivo (ilegible, duplicado, menor de 1 s u otra especie objetivo al fondo) y **no borra nada**: escribe el manifiesto `data/metadata/depuracion.csv` con la decisión y el motivo, y el resumen por especie `experiments/results/depuracion_resumen.csv`, que sí versionamos.

## Paso 8 · Preprocesar (tarea 2.5)

```bash
python -m src.main preprocess --manifest data/metadata/depuracion.csv --max-seconds 60
```

Procesa solo los archivos conservados y usa los primeros 60 s de cada grabación. Genera en `data/processed/` un espectrograma `.npy` por ventana, `index.csv`, `mfcc_stats.npy` (características de los modelos clásicos) y `audio_config.json`. Con 822 grabaciones tarda unos 7 minutos y ocupa cerca de 3 GB.

## Paso 9 · Análisis exploratorio (tarea 2.4)

```bash
python -m src.main eda
```

Escribe las tablas `experiments/results/eda_*.csv` y las figuras `docs/figuras/eda_*.png` que usa el reporte.

## Paso 10 · Modelos de referencia (tarea 2.7)

```bash
# Clásicos sobre MFCC: crean y guardan la partición por grabación (y la de ventana, para H3)
python -m src.main baselines
python -m src.main baselines --split-column split_ventana

# CNN base, versión liviana para CPU (unas 4 horas las tres corridas en paralelo, sin GPU)
python -m src.main train --base-channels 16 --epochs 12 --patience 4 --run-name cnn_base
python -m src.main train --base-channels 16 --epochs 12 --patience 4 --run-name cnn_sin_aug --no-augment
python -m src.main train --base-channels 16 --epochs 12 --patience 4 --run-name cnn_split_ventana --split-column split_ventana

python -m src.main evaluate --checkpoint experiments/checkpoints/cnn_base.pt
python -m src.main evaluate --checkpoint experiments/checkpoints/cnn_sin_aug.pt
python -m src.main evaluate --checkpoint experiments/checkpoints/cnn_split_ventana.pt

# F1 macro por ventana y por grabación, con intervalo de confianza (bootstrap)
python -m src.main intervalos --checkpoint experiments/checkpoints/cnn_base.pt
```

## Paso 11 · Notebooks

```bash
python notebooks/_construir_notebooks.py
```

Genera y ejecuta `02_eda.ipynb` y `03_modelos_referencia.ipynb` con los resultados anteriores.

## Alternativa: Google Colab

```python
from google.colab import drive, userdata
drive.mount("/content/drive")

!git clone -b entrega-2 https://github.com/wilospe27/Proyecto_1_Innovacion_Tecnologica_IA_Grupo_4.git
%cd Proyecto_1_Innovacion_Tecnologica_IA_Grupo_4/entrega_2
!pip install -q -r requirements.txt

import os
os.environ["XC_API_KEY"] = userdata.get("XC_API_KEY")   # la guardamos en "Secrets" de Colab

!python -m src.main especies --region valle_del_cauca --filtrar-localidad "Valle del Cauca"
!python -m src.main download --lista data/especies_objetivo.txt --raw-dir /content/drive/MyDrive/cantos/raw
```

## Si algo falla

| Síntoma | Qué hacer |
|---|---|
| `No se encontró la clave de la API` | Revisamos que `.env` esté dentro de `entrega_2/` y tenga `XC_API_KEY=...` |
| `La API de xeno-canto respondió 401` o `403` | Clave inválida o vencida: la copiamos de nuevo desde la página de la cuenta |
| La API rechaza la consulta (error 400) | Revisamos la sintaxis vigente en https://xeno-canto.org/help/search y ajustamos `build_query` en `src/data/download.py` |
| El ranking sale vacío | Probamos sin `--filtrar-localidad` o ampliamos el rectángulo con `--box` |
