# 🧭 Guía de la Etapa 2 · Datos y selección de especies

Paso a paso de las tareas **2.1** (clave de API y selección de especies, 12–17 oct) y **2.2** (descarga, 16–20 oct) del cronograma.

> Estos comandos los ejecuta el equipo en su computador o en Google Colab: la descarga necesita la clave personal de xeno-canto y conexión directa a xeno-canto.org.

## Paso 1 · Clave de la API (5 minutos)

1. Cree una cuenta en https://xeno-canto.org (basta una cuenta para el equipo).
2. Copie la clave en la página de la cuenta: https://xeno-canto.org/account.
3. Guárdela en un archivo llamado `.env` dentro de `entrega_2/`, con una sola línea:

   ```
   XC_API_KEY=su_clave
   ```

   `.env` está en `.gitignore`, así que no se sube a GitHub. No compartan la clave por chat ni la escriban en notebooks.

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

- Solo consulta **metadatos** (no descarga audio). Para una prueba rápida agregue `--max-paginas 2`.
- El rectángulo `valle_del_cauca` usa las coordenadas extremas del departamento e incluye franjas de los departamentos vecinos; `--filtrar-localidad` deja solo las grabaciones cuya localidad menciona el Valle del Cauca.
- Salidas:
  - `experiments/results/ranking_especies.csv` → **se versiona**: sustenta la selección en el reporte 2.
  - `data/metadata/xeno_canto_region.csv` → metadatos completos (no se versionan).

| Columna del ranking | Significado |
|---|---|
| `especie`, `nombre_en` | Nombre científico y nombre en inglés |
| `grabaciones` | Grabaciones de calidad A o B en la región |
| `grabaciones_total` | Grabaciones de cualquier calidad |
| `grabadores` | Autores distintos de las grabaciones A/B |
| `minutos` | Minutos de audio de calidad A/B |
| `con_otras_especies` | Grabaciones A/B que mencionan especies de fondo (etiquetas débiles) |
| `cumple` | `grabaciones` ≥ 30 |

## Paso 4 · Elegir las especies objetivo (reunión del equipo)

Criterios acordados en el reporte 1 (sección VI.B), con dos sugerencias adicionales:

1. Al menos **30 grabaciones de calidad A o B** (columna `cumple`).
2. *Sugerido:* varios grabadores por especie (por ejemplo, 5 o más), para que el modelo no aprenda el equipo de una sola persona.
3. *Sugerido:* incluir algunos pares de especies con cantos parecidos (por ejemplo, del mismo género), para que la evaluación sea exigente.
4. Total de **20 a 30 especies**.

Escriban la lista en [`../data/especies_objetivo.txt`](../data/especies_objetivo.txt), una especie por línea, y registren la decisión y sus razones en el reporte 2 (sección II.B).

## Paso 5 · Descargar las grabaciones (tarea 2.2)

```bash
python -m src.main download --lista data/especies_objetivo.txt --max-per-species 60
```

- Descarga grabaciones de **toda Colombia** para las especies elegidas: más datos y más variación regional. Prioriza calidad A, luego B.
- Hace una solicitud por segundo para respetar el servidor: unas 1.500 grabaciones pueden tardar más de media hora y ocupar varios GB.
- Resultado: `data/raw/<Genero_especie>/XC<id>.mp3` y `data/metadata/xeno_canto.csv` (con licencia y autor de cada archivo).

## Paso 6 · Compartir los datos en el equipo

Los audios no van a GitHub. Dos opciones:

- **Carpeta compartida** (Google Drive u OneDrive) con `data/raw/` comprimido.
- **Repetir la descarga** con la misma lista: cada integrante obtiene los mismos archivos, salvo grabaciones publicadas después.

## Alternativa: Google Colab

```python
from google.colab import drive, userdata
drive.mount("/content/drive")

!git clone -b entrega-2 https://github.com/wilospe27/Proyecto_1_innovaci-n_tecnol-gica_IA_Grupo_4.git
%cd Proyecto_1_innovaci-n_tecnol-gica_IA_Grupo_4/entrega_2
!pip install -q -r requirements.txt

import os
os.environ["XC_API_KEY"] = userdata.get("XC_API_KEY")   # guárdela en "Secrets" de Colab

!python -m src.main especies --region valle_del_cauca --filtrar-localidad "Valle del Cauca"
!python -m src.main download --lista data/especies_objetivo.txt --raw-dir /content/drive/MyDrive/cantos/raw
```

## Si algo falla

| Síntoma | Qué hacer |
|---|---|
| `No se encontró la clave de la API` | Revise que `.env` esté dentro de `entrega_2/` y tenga `XC_API_KEY=...` |
| `La API de xeno-canto respondió 401` o `403` | Clave inválida o vencida: cópiela de nuevo desde la página de la cuenta |
| La API rechaza la consulta (error 400) | Revise la sintaxis vigente en https://xeno-canto.org/help/search y ajuste `build_query` en `src/data/download.py` |
| El ranking sale vacío | Pruebe sin `--filtrar-localidad` o amplíe el rectángulo con `--box` |
