# 📂 data/

Los audios y sus derivados **no se versionan** en Git (pesan mucho y cada grabación tiene su propia licencia). Esta carpeta solo guarda la estructura; cada integrante la llena localmente con los comandos del proyecto.

```
data/
├── raw/         # Audios originales: raw/<Genero_especie>/XC<id>.mp3
├── processed/   # Ventanas de 5 s como espectrogramas Mel (.npy) + index.csv + audio_config.json
└── metadata/    # Metadatos de las fuentes (p. ej. xeno_canto.csv con licencia y autor)
```

## Cómo poblarla

```bash
# 1. Descargar grabaciones de Colombia desde xeno-canto (requiere XC_API_KEY, ver docs/instalacion.md)
python -m src.main download --species "Turdus fuscater" "Pitangus sulphuratus" --max-per-species 30

# 2. Generar espectrogramas e índice
python -m src.main preprocess
```

## Fuentes previstas

| Fuente | Uso en el proyecto | Licencia |
|---|---|---|
| [xeno-canto](https://xeno-canto.org) | Fuente principal de cantos de aves (y anfibios) de Colombia | Creative Commons, según cada grabación |
| [Macaulay Library](https://www.macaulaylibrary.org) | Complemento para especies con pocas grabaciones | No es Creative Commons: acceso mediante solicitud y uso según sus términos |
| [Colección de Sonidos Ambientales – Instituto Humboldt](https://www.humboldt.org.co) | Grabaciones de fauna colombiana | Acceso mediante solicitud a la colección |
| [iNaturalist](https://www.inaturalist.org) | Observaciones ciudadanas con audio | Creative Commons, según cada observación |

> Conserve siempre la licencia (`lic`) y el autor (`rec`) de cada grabación: se necesitan para citar las fuentes en los reportes.
