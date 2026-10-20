# 📚 Documentación de la Entrega 1

| Documento | Contenido |
|---|---|
| [`reporte_entrega_1.md`](reporte_entrega_1.md) | **Reporte de la primera etapa** (documento evaluado): análisis del problema, objetivos, árbol de problemas, estado del arte, metodología propuesta, plan de actividades y referencias |
| [`cronograma.xlsx`](cronograma.xlsx) | Plan de trabajo con fechas, estado, avance y diagrama de Gantt calculados con fórmulas (se recalcula al cambiar la fecha de inicio o las duraciones) |
| [`arquitectura.md`](arquitectura.md) | Diseño técnico: flujo de datos, preprocesamiento, modelos, entrenamiento y evaluación |
| [`instalacion.md`](instalacion.md) | Instalación del entorno (Windows, Linux/macOS, Conda, Colab) y clave de la API de xeno-canto |
| [`api.md`](api.md) | Comandos de la línea de comandos y funciones de Python del proyecto |
| [`figuras/`](figuras/) | Árbol de problemas y Gantt en PNG (y la fuente `.dot` del árbol) |

## Correspondencia con la rúbrica de la Entrega 1 (20 %)

| Criterio | Peso | Dónde está |
|---|---|---|
| Análisis del problema: identificación del problema, objetivos y árbol de problemas | 10 % | Reporte, secciones [II](reporte_entrega_1.md#ii-análisis-del-problema), [III](reporte_entrega_1.md#iii-objetivos) y [IV](reporte_entrega_1.md#iv-árbol-de-problemas) |
| Estado del arte: revisión bibliográfica y análisis de soluciones existentes | 5 % | Reporte, sección [V](reporte_entrega_1.md#v-estado-del-arte) |
| Planeación: tareas y cronograma | 5 % | Reporte, sección [VII](reporte_entrega_1.md#vii-plan-de-actividades-y-cronograma), y [`cronograma.xlsx`](cronograma.xlsx) |

## Cómo trabajamos en equipo

- **Ramas:** trabajamos con una rama por tarea (`feature/2.2-descarga`, `docs/reporte-2`) y abrimos un *pull request* hacia `main` que revisa otro integrante.
- **Mensajes de commit:** los escribimos en español y en modo imperativo (`Agrega EDA de metadatos`, `Corrige partición por grabación`).
- **Datos y modelos:** no subimos audios, espectrogramas ni checkpoints (ver `.gitignore`); sí versionamos métricas, figuras y notebooks con sus salidas.
- **Pruebas:** antes de cada *pull request* corremos `python -m pytest` desde `entrega_1/`.
- **Cronograma:** cada semana actualizamos *% avance* y *Responsable* en `cronograma.xlsx`.

Documentación del proyecto

Organizamos la documentación técnica del proyecto para facilitar la instalación, comprender la arquitectura, consultar el cronograma y revisar los resultados de la Entrega 1.
