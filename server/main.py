#!/usr/bin/env python3
"""FastAPI — distribución de la ayuda de tamizaje (APP_DERMICA).

Sirve el artefacto entrenado de BCC (1.0.0, float16) desde Hugging Face:
- /healthz         → liveness de Render (sin auth)
- /model/version   → versión y cuantización del release (sin auth)
- /model/latest    → metadata del release (metrics JSON), requiere auth
- /model/download  → stream del .tflite float16, requiere auth

Documentación pública por decisión del proyecto: /docs, /redoc y
/openapi.json no exigen token y el esquema no contiene secretos (los
ejemplos usan placeholders tipo `Bearer <APP_API_KEY>`).

Secretos: APP_API_KEY (Bearer de las rutas autenticadas) y HF_TOKEN (lectura
read-only de HF) vienen del entorno; nunca del código ni del repo.
"""

import os
import secrets
from typing import Optional

import huggingface_hub
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field

VERSION = "1.0.0"
QUANTIZATION = "float16"
REPO_MODELS = os.environ.get("HF_REPO_MODELS", "dermica-models")
REPO_METRICS = os.environ.get("HF_REPO_METRICS", "dermica-metrics")
TFLITE_PATH_IN_REPO = f"{VERSION}/model-{QUANTIZATION}.tflite"
METRIC_FILE = f"{VERSION}.json"
# SHA-256 del artefacto float16 del release (verificado contra HF en la
# publicación y replicado en la app Android como ModelManager.EXPECTED_SHA256).
TFLITE_SHA256 = "d62c401c5f66a7eccd5f752b2d7aa73964fe6f75a92acde3cae951b0d911cc0a"
TFLITE_SIZE_BYTES = 1945356  # ~1.95 MB; la app compara tamaño antes de verificar el SHA

# Etiquetas de la documentación (Swagger/Redoc): agrupan los endpoints por
# propósito y declaran el nivel de acceso de cada grupo.
TAGS_METADATA = [
    {
        "name": "🏷️ Estado",
        "description": "Liveness y versión del release vigente; públicos, sin autenticación.",
    },
    {
        "name": "🗂️ Modelo",
        "description": "Métricas y descarga del artefacto del release; requieren `Authorization: Bearer <APP_API_KEY>`.",
    },
]

APP_DESCRIPTION = """\
API pública de distribución del modelo de tamizaje de **carcinoma basocelular (BCC)** de **APP_DERMICA**. Expone el liveness, la versión del release vigente y el artefacto empaquetado en la app Android (`1.0.0`, cuantización `float16`), alojados en Hugging Face privado.

- `GET /healthz` y `GET /model/version` son **públicos**.
- `GET /model/latest` y `GET /model/download` requieren `Authorization: Bearer <APP_API_KEY>`.

**Distribución**: el modelo solo se distribuye a través de esta API a clientes autorizados (la app oficial de APP_DERMICA); no es un artefacto de acceso abierto y el token no debe compartirse ni publicarse.

**Aviso**: APP_DERMICA es una herramienta de **apoyo al tamizaje**, no de diagnóstico; ningún resultado sustituye el criterio médico profesional.

El esquema `/openapi.json` puede importarse directamente en Postman (Import → Link / File).
"""

app = FastAPI(
    title="APP_DERMICA API",
    description=APP_DESCRIPTION,
    version=VERSION,
    openapi_tags=TAGS_METADATA,
    # Documentación pública por decisión del usuario: sin auth en /docs, /redoc y
    # /openapi.json. El esquema no contiene secretos (placeholders <APP_API_KEY>).
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    servers=[
        {"url": "https://dermica-api.onrender.com", "description": "Producción (Render free, Oregon)"},
        {"url": "http://localhost:8000", "description": "Desarrollo local"},
    ],
)

# Candado solo documental: HTTPBearer con auto_error=False nunca lanza. La
# verificación real la hace `auth_guard` (Header crudo, lógica intacta); así
# /docs y el esquema muestran el candado sin alterar códigos ni orden de
# validación (401 {"detail":"unauthorized"} en todos los fallos).
bearer_scheme = HTTPBearer(
    scheme_name="bearerAuth",
    description="Token de la app oficial (`APP_API_KEY`). Enviar `Authorization: Bearer <APP_API_KEY>`.",
    auto_error=False,
)

# Respuesta 401 común: mismo cuerpo que lanza `auth_guard` hoy.
UNAUTHORIZED_RESPONSE = {
    "description": "`Authorization` ausente o inválida: falta el header, no es `Bearer`, o la clave no coincide con `APP_API_KEY`.",
    "content": {"application/json": {"example": {"detail": "unauthorized"}}},
}
# Respuesta 503 común: comportamiento del tier free de Render mientras la instancia despierta.
COLD_START_RESPONSE = {
    "description": "Instancia aún despertando en Render free: la primera request puede tardar hasta ~60 s; reintentar unos segundos después.",
}
# Respuesta 500 común: el servidor no tiene la configuración de Hugging Face.
MISCONFIGURED_RESPONSE = {
    "description": "Configuración incompleta del servidor (faltan `HF_TOKEN`/`HF_OWNER` en el entorno).",
}


class HealthResponse(BaseModel):
    """Liveness del proceso (/healthz)."""

    status: str = Field(description="Estado del proceso", examples=["ok"])


class VersionResponse(BaseModel):
    """Versión y cuantización del release vigente (/model/version)."""

    version: str = Field(description="Tag del release vigente", examples=["1.0.0"])
    quantization: str = Field(description="Cuantización del artefacto del release", examples=["float16"])


def auth_guard(authorization: Optional[str] = Header(default=None)) -> bool:
    """Valida el header Authorization: Bearer <APP_API_KEY>.

    Parámetros: authorization — header Authorization completo (p.ej.
    'Bearer abc'); None si no fue enviado.
    Retorna: True si coincide.
    Lanza: 401 con detail 'unauthorized' en caso contrario.
    Por qué compare_digest: comparación en tiempo constante para secretos.
    """
    if authorization is None:
        raise HTTPException(status_code=401, detail="unauthorized")
    kind, _, key = authorization.partition(" ")
    if kind != "Bearer" or not key:
        raise HTTPException(status_code=401, detail="unauthorized")
    expected = os.environ.get("APP_API_KEY")
    if not expected or not secrets.compare_digest(key, expected):
        raise HTTPException(status_code=401, detail="unauthorized")
    return True


@app.get(
    "/healthz",
    tags=["🏷️ Estado"],
    summary="Liveness del servicio",
    description=(
        "Comprueba que el proceso está arriba. No consulta Hugging Face ni el modelo: responde "
        "`200` mientras el proceso viva. Render lo usa como `healthCheckPath` para su probe.\n\n"
        "En Render free, una instancia inactiva se apaga y la primera request (o el probe) puede tardar hasta ~60 s "
        "en despertarla; mientras tanto el cliente puede recibir `503`."
    ),
    response_model=HealthResponse,
    responses={
        200: {
            "description": "El proceso está arriba y atiende requests.",
            "content": {"application/json": {"example": {"status": "ok"}}},
        },
        503: COLD_START_RESPONSE,
    },
)
def healthz():
    """Liveness simple (200 siempre que el proceso esté arriba)."""
    return {"status": "ok"}


@app.get(
    "/model/version",
    tags=["🏷️ Estado"],
    summary="Versión y cuantización del release",
    description=(
        "Ping ligero y público que usa la app para conocer el release vigente: tag semver y cuantización del artefacto "
        "empaquetado/distribuido (`1.0.0` / `float16`).\n\n"
        "No requiere token ni consulta Hugging Face. En Render free, la primera request tras inactividad puede tardar "
        "hasta ~60 s (instancia despertando); mientras tanto el cliente puede recibir `503`."
    ),
    response_model=VersionResponse,
    responses={
        200: {
            "description": "Release vigente y su cuantización.",
            "content": {"application/json": {"example": {"version": "1.0.0", "quantization": "float16"}}},
        },
        503: COLD_START_RESPONSE,
    },
)
def model_version():
    """Versión y cuantización del release (público; ping ligero para la app)."""
    return {"version": VERSION, "quantization": QUANTIZATION}


def _download(repo: str, filename: str, repo_type: str) -> str:
    """Descarga un archivo desde HF privado con el token de entorno.

    Parámetros: repo — nombre del repo HF (sin owner); filename — ruta en el repo;
                repo_type — 'model'|'dataset'.
    Retorna: ruta local del archivo descargado (cache HF).
    Por qué hf_hub_download: maneja cache propio y autentica con el token; no
    re-descargar si ya está en cache.
    """
    token = os.environ.get("HF_TOKEN")
    owner = os.environ.get("HF_OWNER")
    if not token or not owner:
        raise HTTPException(status_code=500, detail="server misconfigured (HF_TOKEN/HF_OWNER)")
    return huggingface_hub.hf_hub_download(f"{owner}/{repo}", repo_type=repo_type,
                                           filename=filename, token=token)


# Payload real del release 1.0.0 (dermica-metrics/1.0.0.json, servido tal cual).
# El ejemplo redondea las métricas a 4 decimales para legibilidad; la API
# conserva la precisión completa guardada en HF. Conteos TP/FP/TN/FN enteros.
LATEST_EXAMPLE = {
    "tag": "1.0.0",
    "seed": 42,
    "auprc": 0.7788,
    "recall": 0.777,
    "sensibilidad": 0.777,
    "especificidad": 0.9493,
    "balanced_acc": 0.863,
    "f2": 0.7581,
    "precision": 0.69,
    "tp": 494,
    "fp": 220,
    "tn": 4121,
    "fn": 142,
    "quantization": "float16",
    "float16_artifact": "1.0.0/model-float16.tflite",
    "float16_auprc_val": 0.7786,
    "float16_delta_auprc_val": -0.0002,
    "float16_size_bytes": TFLITE_SIZE_BYTES,
    "float16_sha256": TFLITE_SHA256,
    "int8_disposition": "rejected_delta_-0.331",
}


@app.get(
    "/model/latest",
    tags=["🗂️ Modelo"],
    summary="Métricas del release vigente",
    # Sin response_model a propósito: la respuesta es el JSON de HF servido tal
    # cual (passthrough); fijar un esquema descartaría campos futuros del release.
    dependencies=[Depends(bearer_scheme)],
    description=(
        "Devuelve el JSON de métricas del release vigente (`dermica-metrics/1.0.0.json`) tal cual está guardado en "
        "Hugging Face privado: AUPRC (métrica primaria), recall/sensibilidad, especificidad, balanced_acc, F2, precision "
        "y los conteos TP/FP/TN/FN, más los campos del artefacto float16 (`quantization`, `float16_artifact`, "
        "`float16_auprc_val`, `float16_delta_auprc_val`, `float16_size_bytes`, `float16_sha256`) y `int8_disposition` "
        "(variante int8 rechazada por caída de AUPRC).\n\n"
        "La app usa `float16_sha256` y `float16_size_bytes` para verificar el archivo descargado en `/model/download`.\n\n"
        "La primera llamada tras inactividad puede tardar hasta ~60 s: Render free despierta la instancia y el servidor "
        "lee el JSON desde Hugging Face (la cache HF evita re-descargas sucesivas) mientras tanto el cliente puede "
        "recibir `503`.\n\n"
        "El ejemplo redondea las métricas a 4 decimales; la API sirve la precisión completa guardada en HF."
    ),
    responses={
        200: {
            "description": "JSON de métricas del release vigente (precisión completa; ejemplo redondeado a 4 decimales).",
            "content": {"application/json": {"example": LATEST_EXAMPLE}},
        },
        401: UNAUTHORIZED_RESPONSE,
        500: MISCONFIGURED_RESPONSE,
        503: COLD_START_RESPONSE,
    },
)
def model_latest(_: bool = Depends(auth_guard)):
    """Metadata del release: lee 1.0.0.json desde dermica-metrics vía HF.

    Requiere Bearer APP_API_KEY. Retorna el dict completo de métricas públicas
    (AUPRC, recall, especificidad, BA, F2, SHA-256 del artefacto, etc.).
    """
    import json
    path = _download(REPO_METRICS, METRIC_FILE, "dataset")
    with open(path) as f:
        meta = json.load(f)
    return meta


@app.get(
    "/model/download",
    tags=["🗂️ Modelo"],
    summary="Descarga del artefacto del release (.tflite)",
    dependencies=[Depends(bearer_scheme)],
    description=(
        "Stream del artefacto del release (`1.0.0/model-float16.tflite`, MobileNetV3-Small fine-tuned, cuantización "
        "float16) desde el repo privado `dermica-models` en Hugging Face.\n\n"
        "- **Tamaño**: 1.945.356 bytes (≈ 1.95 MB). La app compara el tamaño y luego verifica el SHA-256.\n"
        "- **SHA-256 esperado**: `d62c401c5f66a7eccd5f752b2d7aa73964fe6f75a92acde3cae951b0d911cc0a` "
        "(coincide con `float16_sha256` en `/model/latest`; si no coincide, la app descarta el archivo).\n"
        "- **Respuesta**: `application/octet-stream` con `Content-Disposition: attachment; filename=model-float16.tflite`.\n\n"
        "La primera descarga tras inactividad puede tardar hasta ~60 s: Render free despierta la instancia y el servidor "
        "obtiene el archivo desde Hugging Face; mientras tanto el cliente puede recibir `503`."
    ),
    responses={
        200: {
            "description": "Stream binario del `.tflite` (1.945.356 bytes ≈ 1.95 MB).",
            "content": {"application/octet-stream": {}},
        },
        401: UNAUTHORIZED_RESPONSE,
        500: MISCONFIGURED_RESPONSE,
        503: COLD_START_RESPONSE,
    },
)
def model_download(_: bool = Depends(auth_guard)):
    """Stream del .tflite float16 desde dermica-models (1.0.0/model-float16.tflite).

    Requiere Bearer APP_API_KEY. Retorna bytes crudo con Content-Disposition
    attachment (la app los guarda y verifica SHA-256 contra /model/latest).
    """
    path = _download(REPO_MODELS, TFLITE_PATH_IN_REPO, "model")
    def _stream():
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                yield chunk
    return StreamingResponse(_stream(), media_type="application/octet-stream",
                             headers={"Content-Disposition": "attachment; filename=model-float16.tflite"})
