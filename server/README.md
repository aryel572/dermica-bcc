# server — APP_DERMICA API

FastAPI de distribución del modelo de tamizaje de **carcinoma basocelular (BCC)** de APP_DERMICA.
Sirve el release vigente (`1.0.0`, cuantización `float16`) desde Hugging Face privado.

- Documentación pública: **`/docs`** (Swagger UI), **`/redoc`** y el esquema **`/openapi.json`** (importable en Postman: Import → Link / File).
- El esquema y los ejemplos no contienen secretos: el acceso usa el placeholder `Bearer <APP_API_KEY>`.
- Aviso: herramienta de **apoyo al tamizaje**, no de diagnóstico; ningún resultado sustituye el criterio médico profesional.

## Correr local

Requiere Python ≥ 3.12 y [uv](https://docs.astral.sh/uv/). Desde la **raíz del monorepo**:

```bash
uv sync --all-packages
# Variables de entorno (o un .env en la raíz, nunca commiteado):
export APP_API_KEY=... HF_TOKEN=... HF_OWNER=...
uv run uvicorn --app-dir server main:app --reload
```

Sin `APP_API_KEY` arranca igual: `/healthz` y `/model/version` responden; las rutas protegidas devuelven `401`. Sin `HF_TOKEN`/`HF_OWNER` devuelven `500`.

## Endpoints

| Método | Ruta | Auth | Qué hace |
|--------|------|------|----------|
| `GET` | `/healthz` | público | Liveness (`healthCheckPath` de Render) |
| `GET` | `/model/version` | público | Versión y cuantización del release (`1.0.0` / `float16`) |
| `GET` | `/model/latest` | `Bearer <APP_API_KEY>` | Métricas del release (`dermica-metrics/1.0.0.json`, servido tal cual) |
| `GET` | `/model/download` | `Bearer <APP_API_KEY>` | Stream de `1.0.0/model-float16.tflite` (1.945.356 bytes ≈ 1.95 MB) |

Autenticación: `Authorization: Bearer <APP_API_KEY>`. Sin token o con token inválido: `401 {"detail": "unauthorized"}`.

## Variables de entorno

| Variable | Uso |
|----------|-----|
| `APP_API_KEY` | Bearer de las rutas protegidas (`/model/latest`, `/model/download`) |
| `HF_TOKEN` | Token de lectura (read-only) de los repos privados de Hugging Face |
| `HF_OWNER` | Cuenta/organización dueña de los repos HF |

Opcionales (defaults entre paréntesis): `HF_REPO_MODELS` (`dermica-models`), `HF_REPO_METRICS` (`dermica-metrics`).

## Notas operativas

- **Cold start (Render free)**: una instancia inactiva se apaga; la primera request puede tardar hasta **~60 s** y el cliente puede recibir `503` mientras despierta.
- El SHA-256 del artefacto (`d62c401c5f66a7eccd5f752b2d7aa73964fe6f75a92acde3cae951b0d911cc0a`) coincide con `float16_sha256` de `/model/latest`; la app lo verifica tras descargar.
- Despliegue: `render.yaml` (rootDir `server/`, health check `/healthz`).
