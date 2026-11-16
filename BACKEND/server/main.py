#!/usr/bin/env python3
"""FastAPI mínimo — distribución de la ayuda de tamizaje (APP_DERMICA).

Sirve el artefacto entrenado de BCC (1.0.0, float16) desde Hugging Face:
- /healthz         → liveness de Render (sin auth)
- /model/version   → versión y cuantización del release (sin auth)
- /model/latest    → metadata del release (metrics JSON), requiere auth
- /model/download  → stream del .tflite float16, requiere auth

Secretos: APP_API_KEY (Bearer de las rutas autenticadas) y HF_TOKEN (lectura
read-only de HF) vienen del entorno; nunca del código ni del repo.
"""

import os
import secrets
from typing import Optional

import huggingface_hub
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

VERSION = "1.0.0"
QUANTIZATION = "float16"
REPO_MODELS = os.environ.get("HF_REPO_MODELS", "dermica-models")
REPO_METRICS = os.environ.get("HF_REPO_METRICS", "dermica-metrics")
TFLITE_PATH_IN_REPO = f"{VERSION}/model-{QUANTIZATION}.tflite"
METRIC_FILE = f"{VERSION}.json"

app = FastAPI(title="APP_DERMICA — Distribución modelo BCC", version=VERSION)


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


@app.get("/healthz")
def healthz():
    """Liveness simple (200 siempre que el proceso esté arriba)."""
    return {"status": "ok"}


@app.get("/model/version")
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


@app.get("/model/latest")
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


@app.get("/model/download")
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
