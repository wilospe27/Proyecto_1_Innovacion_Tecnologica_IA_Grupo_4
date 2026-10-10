# 🐦 Entrega 2 · Datos y modelos de referencia (en construcción)

**Proyecto 1 · Innovación Tecnológica con IA · Grupo 4** — Reporte de la segunda etapa (10 % de la nota) · Fecha de entrega: **19 de noviembre de 2026**

Trabajamos esta carpeta en la rama `entrega-2` y la uniremos a `main` al publicar la entrega (tarea 2.9). Partimos del código de la Entrega 1 y le agregamos la consulta de metadatos de xeno-canto para seleccionar las especies objetivo.

## 📄 Qué contendrá esta entrega

| Criterio de la rúbrica | Peso | Archivo |
|---|---|---|
| Análisis exploratorio: calidad de la exploración y justificación de la selección de datos y variables | 5 % | [`docs/reporte_entrega_2.md`](docs/reporte_entrega_2.md) — secciones II y III |
| Modelos de referencia y experimentos: hipótesis y pruebas iniciales | 5 % | [`docs/reporte_entrega_2.md`](docs/reporte_entrega_2.md) — secciones IV y V |

## 🚀 Primer paso: elegir las especies (tarea 2.1, 12–17 oct)

```bash
cd entrega_2
# guardamos la clave de xeno-canto en entrega_2/.env  →  XC_API_KEY=la_clave
python -m src.main especies --region valle_del_cauca --filtrar-localidad "Valle del Cauca"
```

Instrucciones completas en [`docs/guia_etapa_2.md`](docs/guia_etapa_2.md).

## ✅ Estado

- [x] Estructura de la carpeta y código heredado de la Entrega 1
- [x] Comando `especies`: ranking de especies de la región con metadatos de xeno-canto
- [x] Descarga por lista de especies (`--lista`), priorizando calidad A y B
- [x] Plantilla del reporte 2 según la rúbrica
- [ ] 2.1 Clave de API y selección de especies (`data/especies_objetivo.txt`)
- [ ] 2.2 Descarga de grabaciones · 2.3 Depuración · 2.4 EDA · 2.5 Preprocesamiento
- [ ] 2.6 Hipótesis · 2.7 Modelos de referencia y CNN base · 2.8 Reporte · 2.9 Publicación

## 📁 Estructura

```
entrega_2/
├── docs/          # guia_etapa_2.md · reporte_entrega_2.md · arquitectura.md · api.md · instalacion.md
├── src/           # data/ (download con ranking de especies) · models/ · training/ · evaluation/ · inference/ · utils/ · main.py
├── notebooks/     # 02_eda.ipynb y 03_modelos_referencia.ipynb (pendientes)
├── data/          # especies_objetivo.txt · raw/ · processed/ · metadata/
├── experiments/   # logs/ · checkpoints/ · results/ (aquí versionamos el ranking de especies)
├── tests/         # Pruebas con pytest (sin acceso a la red)
├── requirements.txt · environment.yml · pytest.ini · .gitignore · LICENSE
└── README.md
```

## 📄 Licencia

Publicamos el código y la documentación bajo licencia [MIT](LICENSE). Las grabaciones conservan la licencia de sus autores y no las redistribuimos.
