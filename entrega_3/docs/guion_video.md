# 🎬 Guion del video (10 minutos) y de la presentación oral

Borrador para que el equipo lo ajuste. Repartimos el tiempo entre los cuatro integrantes; los nombres de cada bloque son una propuesta. Los resultados son los del reporte final.

## Video · 10 minutos

| Tiempo | Bloque | Quién | Qué se ve en pantalla | Qué decimos |
|---|---|---|---|---|
| 0:00–0:45 | Gancho | Wilson | Audio de *Cyphorhinus thoracicus* con su espectrograma | Colombia tiene 1.966 especies de aves, más que cualquier otro país; muchas se oyen antes de verse. ¿Puede una IA reconocerlas por su canto? |
| 0:45–2:00 | Problema | Santiago | Árbol de problemas (reporte 1) | Depende de expertos escasos; las grabadoras producen más audio del que se puede escuchar; las herramientas globales fallan más en el Neotrópico |
| 2:00–2:45 | Objetivo y alcance | Santiago | Objetivo general y mapa de las 15 especies | 15 especies del Valle del Cauca; meta aproximada de F1 macro de 0,70 en grabaciones no vistas, con intervalo de confianza |
| 2:45–4:15 | Datos | Adriana | Figuras de calidad, desbalance y espectrogramas | 822 grabaciones de xeno-canto, 665 minutos, 178 grabadores; depuración; ventanas de 5 s; etiquetas débiles y ruido de fondo |
| 4:15–5:15 | Cómo evaluamos | Adriana | Esquema de la partición por grabación frente a la de ventana | Ninguna grabación aparece en entrenamiento y prueba; con partición por ventana el F1 sube hasta 0,34 sin que el modelo sea mejor (H3) |
| 5:15–7:00 | Modelos y resultados | Martín | Tabla comparativa con intervalos; matriz de confusión | Cinco enfoques; la CNN supera a los clásicos (H4); los modelos preentrenados superan a la CNN (H1): por grabación, Perch 2.0 llega a 0,97 y EfficientNet-B0 a 0,90, frente a 0,82; Perch acierta 12 de 12 grabaciones subidas después de su entrenamiento, así que no es fuga; Perch corrige a los cucaracheros, el punto débil de la CNN |
| 7:00–8:45 | Demostración | Wilson | La app en Streamlit: un ejemplo fácil, uno difícil comparando los tres modelos y uno con «Sorpréndeme» | La app sugiere las tres especies con su probabilidad, dice si acertó, muestra qué escuchó en cada ventana y avisa cuando no está segura; en el ejemplo difícil solo Perch 2.0 acierta |
| 8:45–9:40 | Conclusiones y limitaciones | Martín | Lista de conclusiones | Qué logramos, qué no (una semilla, solo xeno-canto, posible fuga en modelos preentrenados) y qué sigue |
| 9:40–10:00 | Cierre | Todos | Créditos: integrantes, fuentes y licencias | Agradecimiento a los grabadores de xeno-canto |

**Recomendaciones para grabar**

- Grabar la demostración con la app ya abierta y los modelos cargados (la primera predicción tarda más).
- Usar ejemplos del conjunto de prueba: muestran grabaciones que el modelo no vio. Sugerencias: *C. thoracicus* XC1083725 (los tres modelos aciertan con probabilidad cercana al 100 %) y *T. aedon* XC414148 con «Comparar con los otros modelos» activado. Perch 2.0 acierta con 100 %, mientras que EfficientNet-B0 (68 %) y la CNN base (39 %, con aviso de confianza baja) lo confunden con *Z. capensis*. Ese ejemplo resume H1 en diez segundos, y muestra que una probabilidad alta no garantiza un acierto. El botón «Sorpréndeme» sirve para mostrar que no escogimos los ejemplos a dedo. Mostrar un error es más honesto y más interesante que mostrar solo aciertos.
- Citar a los autores de los audios que suenen en el video (xeno-canto exige atribución en las licencias Creative Commons).
- Exportar en 1080p y verificar que el audio de los cantos se escuche claro.

## Presentación oral · última sesión

Esquema de 8 a 10 diapositivas, reutilizando las figuras del video:

1. **Título** y equipo.
2. **El problema** en una frase y por qué importa en Colombia.
3. **Pregunta y objetivo** con la meta aproximada (F1 macro ≈ 0,70 con intervalo de confianza).
4. **Datos:** 15 especies, 822 grabaciones; una figura de espectrogramas.
5. **Cómo evaluamos:** partición por grabación y la lección de H3.
6. **Resultados:** tabla de los cinco enfoques con intervalos.
7. **Hipótesis:** H1 a H4 con su decisión.
8. **Demostración en vivo** (con un video de respaldo por si falla la conexión o el equipo).
9. **Conclusiones, limitaciones y trabajo futuro.**
10. **Preguntas.**

**Preguntas que conviene preparar**

- ¿Por qué no usaron BirdNET directamente? (Comparamos con Perch 2.0, un modelo del mismo tipo; además, su desempeño baja en el Neotrópico.)
- ¿Perch no había visto ya sus grabaciones de prueba? (Lo verificamos: en las 12 grabaciones subidas después de su entrenamiento acierta todas.)
- ¿Por qué dejaron EfficientNet y la CNN en la app si Perch es mejor? (Son 4 a 5 veces más rápidos y no necesitan TensorFlow; la comparación muestra cuándo coinciden los modelos.)
- ¿Cómo saben que el modelo no aprendió el ruido del sitio? (No lo descartamos: está entre las limitaciones; seis especies tienen su banda más distintiva por debajo de 1 kHz.)
- ¿Por qué bajaron la meta de 0,80 a 0,70? (La recalibramos con el análisis exploratorio, como habíamos previsto; la reportamos con intervalo de confianza.)
- ¿Funciona con grabaciones de celular? (No lo evaluamos de forma sistemática; es trabajo futuro.)
