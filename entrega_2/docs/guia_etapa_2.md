# 🧭 Guía de la Etapa 2 · Datos y selección de especies

Este es nuestro paso a paso para las tareas **2.1** (clave de API y selección de especies, 12–17 oct) y **2.2** (descarga, 16–20 oct) del cronograma.

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
- Hace una solicitud por segundo para respetar el servidor: unas 1.500 grabaciones pueden tardar más de media hora y ocupar varios GB.
- Resultado: `data/raw/<Genero_especie>/XC<id>.mp3` y `data/metadata/xeno_canto.csv` (con licencia y autor de cada archivo).

## Paso 6 · Compartir los datos en el equipo

Los audios no van a GitHub. Tenemos dos opciones:

- **Carpeta compartida** (Google Drive u OneDrive) con `data/raw/` comprimido.
- **Repetir la descarga** con la misma lista: cada integrante obtiene los mismos archivos, salvo grabaciones publicadas después.

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
