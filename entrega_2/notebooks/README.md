# 📓 Notebooks de la Entrega 2

| Notebook | Tarea | Contenido |
|---|---|---|
| [`02_eda.ipynb`](02_eda.ipynb) | 2.3 y 2.4 | Selección de especies, depuración, calidad de los datos, distribuciones (desbalance, grabadores, tipos de vocalización, mapa, licencias) y exploración acústica (espectrogramas, bandas dominantes, especies parecidas y k-means) |
| [`03_modelos_referencia.ipynb`](03_modelos_referencia.ipynb) | 2.6 y 2.7 | Partición por grabación, modelos clásicos sobre MFCC, CNN base y pruebas de las hipótesis H2, H3 y H4 |

Los notebooks son delgados: la lógica está en `src/` (con pruebas en `tests/`) y aquí la ejecutamos y comentamos. Los generamos y ejecutamos con:

```bash
python notebooks/_construir_notebooks.py     # desde entrega_2/, con los datos ya procesados
```

`_construir_notebooks.py` define las celdas, y `_conclusiones_eda.md` y `_conclusiones_modelos.md` contienen el texto de las conclusiones. Para reproducirlos hay que seguir antes los pasos de [`docs/guia_etapa_2.md`](../docs/guia_etapa_2.md) (descarga, depuración, preprocesamiento y entrenamiento).

El notebook de exploración inicial del pipeline está en `entrega_1/notebooks/`.
