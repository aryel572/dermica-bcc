# APP_DERMICA

Aplicación Android nativa para **apoyo al tamizaje del carcinoma basocelular (BCC)** en imágenes dermatoscópicas, con un modelo de deep learning embebido en el dispositivo. Herramienta de tamizaje, no de diagnóstico: la app nunca sustituye el criterio médico profesional.

Estado: **release 1.0.0 publicada** (modelo float16 + APK v1.0.0).

CI: **GitHub Actions** (`.github/workflows/ci.yml`) — compileall Python, smoke del servidor y build/tests de Android en push a `main` y en cada PR.

## Qué productos contiene este repositorio

Monorepo con 4 productos, cada uno en su carpeta de raíz (`pipeline/`, `model/`, `server/`, `android/`), sin cambios de lógica ni de contratos entre sí (ver `ARCHITECTURE.md`).

| Producto | Ubicación | Qué hace |
|----------|-----------|----------|
| **pipeline** | `pipeline/` | Auditoría de ISIC 2019, formación de subconjuntos TESIS, splits 70/20/10 agrupados por `lesion_id`, manifiestos para el dataset remoto |
| **model** | `model/` | Fine-tuning de MobileNetV3-Small, evaluación, exportación a TFLite (float16/int8), publicación de métricas |
| **server** | `server/` | FastAPI pública: distribución del modelo (`/healthz`, `/model/version`, `/model/latest`, `/model/download`) |
| **android** | `android/` | App Android nativa (Kotlin + Jetpack Compose Material 3, LiteRT) |

## Release 1.0.0 (alcance)

- **Modelo**: MobileNetV3-Small fine-tuned, cuantización **float16**, empaquetado en la app y actualizable por backend. AUPRC: **0.7788 (val) / 0.7709 (test)**. Clasificación binaria BCC vs no-BCC.
- **Android**: APK release v1.0.0 (`APP_DERMICA-v1.apk`, 22.26 MB, R8, firma v2, tag `v1.0.0`, sha256 `e5eb7077…`). Inferencia 100 % on-device; historial local (Room); sin telemetría.
- **Server**: FastAPI desplegada en Render (free). `/healthz` y `/model/version` públicos; `/model/latest` y `/model/download` requieren `Authorization: Bearer`.
- **Pipeline**: división 70/20/10 agrupada por `lesion_id` (sin fuga entre particiones); imágenes TESIS excluidas por contrato (`tesis_exclude.csv`).

## Cómo compilar y probar

```bash
# App Android (requiere JDK 21 + Android SDK con android-35 y build-tools 35.0.0)
cd android
./gradlew :app:assembleDebug          # APK debug
./gradlew :app:testDebugUnitTest      # tests JVM (verdes, obligatorio antes de push)
./gradlew :app:assembleRelease        # APK release firmado (minify R8)

# Servidor FastAPI (desde la raíz; secretos en .env: APP_API_KEY, HF_TOKEN, HF_OWNER)
uv run uvicorn --app-dir server main:app --reload   # escucha en :8000; healthz en /healthz

# Pipeline y modelo (scripts directos; consultá cada script para su CLI)
uv run python pipeline/splits/make_splits.py --help
uv run python model/push_hf.py --help
```

## Entorno Python

Un único entorno para los cuatro productos Python (`pipeline/`, `model/`, `server/`, `shared/`), gestionado con **uv** (probado con uv 0.13.x; `requires-python >= 3.12`, `shared/dermica-core` admite >= 3.10):

```bash
uv sync --all-packages   # crea/actualiza .venv en la raíz e instala los 4 miembros
```

- `shared/dermica-core` es el único paquete real (name `dermica-core`, 1.0.0, cero dependencias); los demás miembros son virtuales: scripts con deps pineadas, sin `[build-system]`.
- Las versiones están fijadas exactas en los `pyproject.toml` de cada miembro y las transitivas en `uv.lock` (commiteado). Nada de rangos abiertos.
- **TensorFlow no entra por defecto** (el entrenamiento y `model/evaluate.py` corren en Colab). Para un entorno con TF: `uv sync --extra colab` o `pip install tensorflow==2.21.0` en Colab.
- Verificación de la instalación: `uv run python -c "import dermica_core; print(dermica_core.SHORT_CIRCUIT_OK)"` → `True`.
- `server/requirements.txt` se conserva **solo** para Render (`render.yaml` lo instala); contiene las mismas versiones que `server/pyproject.toml`.
- Desde un clon limpio: `git clone … && cd dermica-bcc && uv sync --all-packages` — no hace falta crear ningún `.venv` a mano ni `pip install` nada.

Notas:
- Sin `APP_API_KEY` en `local.properties` la app compila y predice con normalidad; solo los botones de descarga/actualización del modelo quedan deshabilitados (ver `SECURITY.md`).
- Las dependencias del pipeline y del modelo viven en el workspace uv raíz (`pyproject.toml` + `uv.lock`), no en `requirements.txt` sueltos.
- Nada en este repo redistribuye imágenes de ISIC 2019 (ver `LICENSE`).

## Mapa del repositorio

```
dermica-bcc/
├── pipeline/                     # audit/ · prep/ · splits/ · tesis/ · manifests/ · ground-truth/ · config.json
├── model/                        # train_bcn.py · evaluate.py · metrics.py · push_hf.py · …
├── server/                       # main.py · render.yaml · requirements.txt
├── android/                      # app Android (app/ · gradle wrapper · local.properties.example)
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

## Convenciones rápidas

- Ramas `feat/<unidad>`, `docs/<unidad>`, `refactor/<fase>`; el dueño del repo mergea los PRs.
- Commits atómicos, Conventional Commits (tipo en inglés, descripción en español); sin atribución de IA.
- Comentarios y docstrings internos en español; identificadores en inglés.
- Detalle completo: `CONTRIBUTING.md`.

## Documentos

`ARCHITECTURE.md` · `CHANGELOG.md` · `CONTRIBUTING.md` · `SECURITY.md` · `CODE_OF_CONDUCT.md` · `LICENSE` · `doc-adr/`

## Atribución y licencia

- Código bajo **MIT** (ver `LICENSE`).
- Dataset ISIC 2019 bajo **CC BY-NC 4.0** (uso no comercial con citación de ISIC/HAM10000/BCN20000/MSK). Las imágenes no se redistribuyen en este repo; los manifiestos derivados llevan atribución en `pipeline/manifests/DATASET_CARD.md`.
