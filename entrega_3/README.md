# 🐦 Entrega 3 · Reporte final, video y prototipo (avanzada)

**Proyecto 1 · Innovación Tecnológica con IA · Grupo 4** — Entrega final (70 % de la nota) · Fecha de entrega: **1 de diciembre de 2026** · Presentación oral en la última sesión

Partimos del código de la Entrega 2 y agregamos la transferencia de aprendizaje con EfficientNet-B0, los *embeddings* de Perch 2.0, los intervalos de confianza para todos los modelos y un prototipo en Streamlit.

## 📄 Qué contiene

| Entregable | Archivo |
|---|---|
| Reporte final (introducción, metodología, resultados, discusión, conclusiones y referencias) | [`docs/reporte_final.md`](docs/reporte_final.md) |
| Guion del video de 10 minutos y esquema de la presentación oral | [`docs/guion_video.md`](docs/guion_video.md) |
| Presentación para la clase (29 diapositivas con notas del orador) | [`docs/presentacion_proyecto.pptx`](docs/presentacion_proyecto.pptx) |
| Prototipo de demostración | [`app/streamlit_app.py`](app/streamlit_app.py) |

## 🚀 Cómo ejecutarla

Los datos son los de la Entrega 2: si ya los tenemos, basta con copiar o enlazar `entrega_2/data/raw`, `processed` y `metadata` dentro de `entrega_3/data/`.

```bash
cd entrega_3
python -m src.main baselines                       # clásicos sobre MFCC, con intervalos de confianza
python -m src.main train --model efficientnet_b0 --lr 0.0003 --epochs 10 --patience 3 --run-name efficientnet_b0
python -m src.main intervalos --checkpoint experiments/checkpoints/efficientnet_b0.pt
streamlit run app/streamlit_app.py                 # prototipo
```

Perch 2.0 necesita TensorFlow. Para extraer los *embeddings* usamos un entorno aparte; para usarlo en la app, se instala en el mismo entorno de PyTorch. Las dos opciones están en [`docs/instalacion.md`](docs/instalacion.md).

## ✅ Estado

- [x] Carpeta de la Entrega 3 con el código de la Entrega 2
- [x] Intervalos de confianza para los modelos clásicos y la CNN base
- [x] Prototipo en Streamlit con Perch 2.0, EfficientNet-B0 y la CNN base: subir, grabar o elegir un ejemplo de prueba («Sorpréndeme»), mapa de calor por ventana, comparación entre modelos y desempeño por especie
- [x] Borrador del reporte final y del guion del video
- [x] 3.1 EfficientNet-B0: F1 macro 0,80 por ventana y 0,90 por grabación
- [x] 3.2 *Embeddings* de Perch 2.0 + clasificador lineal: F1 macro 0,88 por ventana y 0,97 por grabación, con verificación de fuga
- [x] 3.4 Evaluación comparativa y análisis de errores (tabla con intervalos, F1 por especie, verificación de fuga)
- [ ] 3.3 Ajuste de hiperparámetros (opcional)
- [ ] 3.6 Reporte final · 3.7 Video · 3.8 Presentación oral · 3.9 Publicación y subida a la plataforma

## 📁 Estructura

```
entrega_3/
├── app/           # streamlit_app.py: prototipo de demostración
├── docs/          # reporte_final.md · guion_video.md · arquitectura.md · api.md · instalacion.md
├── src/           # data/ · analysis/ · models/ (classical, cnn_baseline, embeddings) · training/ · evaluation/ · inference/
├── data/          # especies_objetivo.txt · raw/ · processed/ · metadata/ (los audios no van a Git)
├── experiments/   # results/ (métricas e intervalos que cita el reporte) · checkpoints/ · logs/
├── tests/         # pruebas con pytest
└── requirements.txt · environment.yml · pytest.ini · .gitignore · LICENSE
```

## 📄 Licencia

Publicamos el código y la documentación bajo licencia [MIT](LICENSE). Las grabaciones conservan la licencia de sus autores y no las redistribuimos.
