# 🐦 Entrega 2 · Datos y modelos de referencia

**Proyecto 1 · Innovación Tecnológica con IA · Grupo 4** — Reporte de la segunda etapa (10 % de la nota) · Fecha de entrega: **19 de noviembre de 2026**

Partimos del código de la Entrega 1 y le agregamos la selección de especies, la depuración, el análisis exploratorio y los modelos de referencia.

## 📄 Qué contiene esta entrega

| Criterio de la rúbrica | Peso | Archivo |
|---|---|---|
| Análisis exploratorio: calidad de la exploración y justificación de la selección de datos y variables | 5 % | [`docs/reporte_entrega_2.md`](docs/reporte_entrega_2.md) — secciones II y III · [`notebooks/02_eda.ipynb`](notebooks/02_eda.ipynb) |
| Modelos de referencia y experimentos: hipótesis y pruebas iniciales | 5 % | [`docs/reporte_entrega_2.md`](docs/reporte_entrega_2.md) — secciones IV y V · [`notebooks/03_modelos_referencia.ipynb`](notebooks/03_modelos_referencia.ipynb) |

## 📊 Resultados principales

- **Datos:** 15 especies de aves, 822 grabaciones de xeno-canto tras la depuración (665 minutos, 178 grabadores) y 11.159 ventanas de 5 s.
- **Desempeño en prueba, con partición por grabación:**

| Modelo | Exactitud | F1 macro | Top-3 |
|---|---|---|---|
| Clase mayoritaria | 0,083 | 0,010 | 0,205 |
| Regresión logística (MFCC) | 0,578 | 0,567 | 0,769 |
| SVM RBF (MFCC) | 0,620 | 0,623 | 0,802 |
| Bosque aleatorio (MFCC) | 0,610 | 0,608 | 0,794 |
| Potenciación de gradiente (MFCC) | 0,593 | 0,590 | 0,803 |
| **CNN base (espectrogramas Mel)** | **0,734** | **0,735** | **0,871** |

- **Hipótesis:** la CNN supera a los modelos clásicos (H4, +0,11 de F1 macro); particionar por ventana sobrestima el F1 macro entre 0,12 y 0,34 (H3); la aumentación no mostró ventaja en las especies con menos datos (H2). H1 queda para la Entrega 3.

## 🚀 Cómo reproducirla

```bash
cd entrega_2
# guardamos la clave de xeno-canto en entrega_2/.env  →  XC_API_KEY=la_clave
python -m src.main download --lista data/especies_objetivo.txt --max-per-species 60
python -m src.main depurar
python -m src.main preprocess --manifest data/metadata/depuracion.csv --max-seconds 60
python -m src.main eda
python -m src.main baselines
python -m src.main train --base-channels 16 --epochs 12 --patience 4 --run-name cnn_base
python -m src.main evaluate --checkpoint experiments/checkpoints/cnn_base.pt
```

El paso a paso completo, con tiempos y las corridas de las hipótesis, está en [`docs/guia_etapa_2.md`](docs/guia_etapa_2.md); la instalación, en [`docs/instalacion.md`](docs/instalacion.md).

## ✅ Estado

- [x] 2.1 Selección de las 15 especies objetivo (`data/especies_objetivo.txt`)
- [x] 2.2 Descarga de grabaciones · 2.3 Depuración · 2.4 Análisis exploratorio · 2.5 Preprocesamiento
- [x] 2.6 Hipótesis · 2.7 Modelos de referencia y CNN base · 2.8 Reporte
- [ ] 2.9 Publicación: unir la rama `entrega-2` a `main`

## 📁 Estructura

```
entrega_2/
├── docs/          # reporte_entrega_2.md · guia_etapa_2.md · arquitectura.md · api.md · instalacion.md · figuras/
├── src/           # data/ (download, clean, preprocess, dataset) · analysis/ (eda) · models/ (classical, cnn_baseline)
│                  # training/ · evaluation/ · inference/ · utils/ · main.py
├── notebooks/     # 02_eda.ipynb · 03_modelos_referencia.ipynb · _construir_notebooks.py
├── data/          # especies_objetivo.txt · raw/ · processed/ · metadata/ (los audios no van a Git)
├── experiments/   # logs/ · checkpoints/ · results/ (tablas y métricas que cita el reporte)
├── tests/         # 40 pruebas con pytest (sin acceso a la red)
├── requirements.txt · environment.yml · pytest.ini · .gitignore · LICENSE
└── README.md
```

## 📄 Licencia

Publicamos el código y la documentación bajo licencia [MIT](LICENSE). Las grabaciones conservan la licencia de sus autores y no las redistribuimos.
