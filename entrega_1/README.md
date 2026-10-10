# 🐦 Entrega 1 · Identificación de especies por su canto

**Proyecto 1 · Innovación Tecnológica con IA · Grupo 4** — Reporte de la primera etapa (20 % de la nota) · Fecha de entrega: **19 de octubre de 2026**

Estamos desarrollando un sistema basado en aprendizaje profundo que, a partir de una grabación de audio, estima qué especie de fauna colombiana está vocalizando. En esta entrega presentamos el **análisis del problema, el estado del arte y la planeación**, junto con la estructura del repositorio y un *pipeline* base que ya probamos.

## 📄 Qué contiene esta entrega

| Criterio de la rúbrica | Peso | Archivo |
|---|---|---|
| Análisis del problema (problema, objetivos y árbol de problemas) | 10 % | [`docs/reporte_entrega_1.md`](docs/reporte_entrega_1.md) — secciones II a IV |
| Estado del arte (revisión bibliográfica y soluciones existentes) | 5 % | [`docs/reporte_entrega_1.md`](docs/reporte_entrega_1.md) — sección V |
| Planeación (tareas y cronograma) | 5 % | [`docs/reporte_entrega_1.md`](docs/reporte_entrega_1.md) — sección VII y [`docs/cronograma.xlsx`](docs/cronograma.xlsx) |

## 📁 Estructura

```
entrega_1/
├── docs/                         # Documentación
│   ├── README.md                 # Índice de la documentación
│   ├── reporte_entrega_1.md      # Reporte de la primera etapa (documento evaluado)
│   ├── cronograma.xlsx           # Plan de trabajo con fórmulas y Gantt
│   ├── arquitectura.md           # Diseño del pipeline y de los modelos
│   ├── api.md                    # Comandos y funciones del proyecto
│   ├── instalacion.md            # Instalación y clave de xeno-canto
│   └── figuras/                  # Árbol de problemas y Gantt (PNG)
├── src/                          # Código fuente
│   ├── data/                     # download.py · preprocess.py · dataset.py
│   ├── models/                   # cnn_baseline.py (CNN base y EfficientNet-B0)
│   ├── training/                 # train.py
│   ├── evaluation/               # evaluate.py
│   ├── inference/                # predict.py
│   ├── utils/                    # helpers.py
│   └── main.py                   # Punto de entrada (CLI)
├── notebooks/
│   └── 01_exploracion_inicial.ipynb
├── data/                         # raw/ · processed/ · metadata/ (no los subimos a Git)
├── experiments/                  # logs/ · checkpoints/ · results/
├── tests/                        # Pruebas con pytest
├── requirements.txt
├── environment.yml
├── pytest.ini
├── .gitignore
└── LICENSE
```

## 🚀 Inicio rápido

```bash
cd entrega_1
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   ·   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python -m pytest              # 25 pruebas, sin descargar datos
python -m src.main --help     # comandos: download, preprocess, train, evaluate, predict
```

Flujo completo (requiere la clave `XC_API_KEY` de xeno-canto; ver [`docs/instalacion.md`](docs/instalacion.md)):

```bash
python -m src.main download --species "Turdus fuscater" "Pitangus sulphuratus" --max-per-species 30
python -m src.main preprocess
python -m src.main train --model cnn_baseline --epochs 30 --run-name cnn_v1
python -m src.main evaluate --checkpoint experiments/checkpoints/cnn_v1.pt --split test
python -m src.main predict --audio mi_grabacion.wav --checkpoint experiments/checkpoints/cnn_v1.pt
```

## ✅ Estado

- [x] Análisis del problema, objetivos y árbol de problemas
- [x] Estado del arte y análisis de soluciones existentes
- [x] Cronograma y gestión de riesgos
- [x] Estructura del repositorio, *pipeline* base y pruebas automáticas
- [x] Cronograma ajustado a las fechas oficiales (19-oct, 19-nov y 1-dic-2026)
- [ ] Revisión del reporte entre todos los integrantes (tarea 1.9, hasta el 15-oct)
- [ ] Asignar responsables a las tareas «Por asignar» en `docs/cronograma.xlsx`
- [ ] Versión final y publicación (tarea 1.10, hasta el 18-oct)

## 👥 Equipo — Grupo 4

Wilson Ospina Pérez · Santiago Osorio · Martín Pérez · Adriana Zapata

## 📄 Licencia

Publicamos el código y la documentación bajo licencia [MIT](LICENSE). Las grabaciones conservan la licencia de sus autores y no las redistribuimos.
