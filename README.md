# APP_DERMICA

Aplicación Android nativa para **apoyo al tamizaje del carcinoma basocelular (BCC)** en imágenes dermatoscópicas, con un modelo de deep learning embebido en el dispositivo. Herramienta de tamizaje, no de diagnóstico: la app nunca sustituye el criterio médico profesional.

Estado: **release 1.0.0 publicada** (modelo float16 + APK v1.0.0).

## Qué productos contiene este repositorio

Monorepo con 4 productos. Hoy viven bajo `BACKEND/` y `FRONTEND/`; la Fase 1 los reorganizará en carpetas de producto (`pipeline/`, `model/`, `server/`, `android/`) sin cambiar su lógica ni sus contratos (ver `ARCHITECTURE.md`).

| Producto | Ubicación actual | Ubicación objetivo (Fase 1) | Qué hace |
|----------|------------------|-----------------------------|----------|
| **pipeline** | `BACKEND/data_pipeline/` | `pipeline/` | Auditoría de ISIC 2019, formación de subconjuntos TESIS, splits 70/20/10 agrupados por `lesion_id`, manifiestos para el dataset remoto |
| **model** | `BACKEND/model/` | `model/` | Fine-tuning de MobileNetV3-Small, evaluación, exportación a TFLite (float16/int8), publicación de métricas |
| **server** | `BACKEND/server/` | `server/` | FastAPI pública: distribución del modelo (`/healthz`, `/model/version`, `/model/latest`, `/model/download`) |
| **android** | `FRONTEND/` | `android/` | App Android nativa (Kotlin + Jetpack Compose Material 3, LiteRT) |

## Release 1.0.0 (alcance)

- **Modelo**: MobileNetV3-Small fine-tuned, cuantización **float16**, empaquetado en la app y actualizable por backend. AUPRC: **0.7788 (val) / 0.7709 (test)**. Clasificación binaria BCC vs no-BCC.
- **Android**: APK release v1.0.0 (`APP_DERMICA-v1.apk`, 22.26 MB, R8, firma v2, tag `v1.0.0`, sha256 `e5eb7077…`). Inferencia 100 % on-device; historial local (Room); sin telemetría.
- **Server**: FastAPI desplegada en Render (free). `/healthz` y `/model/version` públicos; `/model/latest` y `/model/download` requieren `Authorization: Bearer`.
- **Pipeline**: división 70/20/10 agrupada por `lesion_id` (sin fuga entre particiones); imágenes TESIS excluidas por contrato (`tesis_exclude.csv`).

## Cómo compilar y probar

**Interfaz objetivo (Fase 1, aún no disponible en `main`)**: `make apk` (build + tests de la app) y `make serve` (servidor local). No existen todavía; se introducirán con la reorganización de carpetas.

**Comandos que funcionan hoy (`main`)**:

```bash
# App Android (requiere JDK 21 + Android SDK con android-35 y build-tools 35.0.0)
cd FRONTEND
./gradlew :app:assembleDebug          # APK debug
./gradlew :app:testDebugUnitTest      # tests JVM (verdes, obligatorio antes de push)
./gradlew :app:assembleRelease        # APK release firmado (minify R8)

# Servidor FastAPI (requiere Python; secretos en .env: APP_API_KEY, HF_TOKEN, HF_OWNER)
cd BACKEND/server
pip install -r requirements.txt
uvicorn main:app --reload             # escucha en :8000; healthz en /healthz

# Pipeline y modelo (scripts directos; consultá cada script para su CLI)
uv run python BACKEND/data_pipeline/splits/make_splits.py --help
uv run python BACKEND/model/evaluate.py --help
```

Notas:
- Sin `APP_API_KEY` en `local.properties` la app compila y predice con normalidad; solo los botones de descarga/actualización del modelo quedan deshabilitados (ver `SECURITY.md`).
- La lógica del pipeline y del modelo se ejecuta desde fuentes Python con `uv`; sus dependencias viven en los `requirements.txt` de cada producto.
- Nada en este repo redistribuye imágenes de ISIC 2019 (ver `LICENSE`).

## Mapa del repositorio

```
dermica-bcc/
├── BACKEND/                      # hoy: pipeline + model + server
│   ├── data_pipeline/            # audit/ · prep/ · splits/ · TESIS · manifests/ · config.json
│   ├── model/                    # train_bcn.py · evaluate.py · metrics.py · push_hf.py · …
│   └── server/                   # main.py · render.yaml · requirements.txt
├── FRONTEND/                     # hoy: app Android (app/ · gradle wrapper · local.properties.example)
├── README.md                     # este archivo
├── ARCHITECTURE.md               # flujo, contratos de artefactos y reglas de dependencia
├── CHANGELOG.md                  # cronología S0 → S5 y release 1.0.0
├── CONTRIBUTING.md               # convenciones de desarrollo
├── SECURITY.md                   # manejo de secretos y reporte de vulnerabilidades
├── CODE_OF_CONDUCT.md
├── LICENSE
├── doc-adr/                      # decisiones de arquitectura (ADR 0001–0005)
└── .github/                      # plantillas de PR e issues
```

Layout objetivo (Fase 1): `pipeline/`, `model/`, `server/`, `android/` en la raíz, con los mismos contenidos movidos y sin cambios de lógica.

## Convenciones rápidas

- Ramas `feat/<unidad>`, `docs/<unidad>`, `refactor/<fase>`; el dueño del repo mergea los PRs.
- Commits atómicos, Conventional Commits (tipo en inglés, descripción en español); sin atribución de IA.
- Comentarios y docstrings internos en español; identificadores en inglés.
- Detalle completo: `CONTRIBUTING.md`.

## Documentos

`ARCHITECTURE.md` · `CHANGELOG.md` · `CONTRIBUTING.md` · `SECURITY.md` · `CODE_OF_CONDUCT.md` · `LICENSE` · `doc-adr/`

## Atribución y licencia

- Código bajo **MIT** (ver `LICENSE`).
- Dataset ISIC 2019 bajo **CC BY-NC 4.0** (uso no comercial con citación de ISIC/HAM10000/BCN20000/MSK). Las imágenes no se redistribuyen en este repo; los manifiestos derivados llevan atribución en `BACKEND/data_pipeline/manifests/DATASET_CARD.md`.
