# APP_DERMICA

Tesis: app Android nativa (Kotlin) que detecta Carcinoma Basocelular (BCC) en imágenes dermatoscópicas mediante un modelo de deep learning embebido. Herramienta de tamizaje, no de diagnóstico.

## Estructura

- `BACKEND/`
  - `data_pipeline/` — pipeline de datos (formación de subconjuntos TESIS, splits agrupados por `lesion_id` sobre ISIC 2019)
  - `model/` — fine-tuning de MobileNetV3-Small, cuantización int8, exportación a `.tflite`
  - `server/` — servidor FastAPI (actualizaciones habilitables del modelo)
- `FRONTEND/` — app Android nativa en Kotlin (LiteRT)

## Convenciones

- Ramas de trabajo: `feat/<unidad>`; el dueño hace los merges a `main` manualmente.
- Commits atómicos y pequeños; Conventional Commits con prefijo en inglés y descripción en español (ej. `feat(pipeline): add TESIS subset formation`).
- La rama `main` solo recibe cambios vía merge del dueño o el commit inicial de estructura.

## Atribución y licencia

- Dataset: ISIC 2019, licenciado bajo CC BY-NC 4.0. Uso no comercial con citación de ISIC/HAM10000/BCN20000/MSK.
- Repositorio privado; ningún dato de imágenes se redistribuye en este repo (solo manifiestos/seeds cuando aplique).
- Las imágenes ISIC pertenecen a sus fuentes originales; este proyecto es de investigación académica.
