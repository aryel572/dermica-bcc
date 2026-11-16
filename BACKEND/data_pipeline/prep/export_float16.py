#!/usr/bin/env python3
"""S4 paso 2b: exporta MobileNetV3-Small 1.0.0 a .tflite float16 para el release.

Ante el fracaso de la cuantización int8 completa (ΔAUPRC −0.331 vs float en
val; se abortó y no se publicó nada), la decisión de release fue float16:
AUPRC 0.7786 vs 0.7788 (float), 1.95 MB, ideal para LiteRT en Android sin
pérdida medible sobre el checkpoint 1.0.0.

Flujo:
1. Descarga `1.0.0.keras` desde dermica-models (HF_TOKEN/HF_OWNER del .env se
   leen inline; nunca se imprimen ni se escriben al disco compartido).
2. Carga el modelo y descarta la capa `augment` (identidad en inferencia):
   el .tflite solo lleva graph de inferencia (Base → Dropout → Dense).
3. Convierte con TFLiteConverter SOLO con target_spec.supported_types=[tf.float16]
   (sin optimizations int8; es el checkpoint float almacenado en fp16).
4. Guarda `model-float16.tflite` en resultados/s4-int8/ e imprime tamaño y SHA-256.

Por qué float16: MobileNetV3-Small fine-tuneado sobre 24k imágenes es
extremadamente sensible a int8 weights (val AUPRC 0.78 → 0.45, recall 0.24);
fp16 conserva la precisión float al 100% y reduce el archivo a la mitad del
float32.

Uso:
    uv run python export_float16.py --keras /content/1.0.0.keras \
        --out ../resultados/s4-int8/model-float16.tflite
    uv run python export_float16.py --verificar  # comprueba que el .tflite es cargável
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

APP_ROOT = None
for _p in Path(__file__).resolve().parents:
    if (_p / "dermica-bcc").is_dir():
        APP_ROOT = _p
        break
if APP_ROOT is None:
    APP_ROOT = Path("/content")
ENV_PATH = APP_ROOT / "dermica-bcc" / ".env"
if not ENV_PATH.exists():
    ENV_PATH = Path("/content/dermica-bcc/.env")
REPO_MODELS = "dermica-models"
MODEL_FILE = "1.0.0.keras"
DEFAULT_OUT = APP_ROOT / "resultados" / "s4-int8" / "model-float16.tflite"


def _read_env() -> dict:
    """Lee .env KEY=VALUE (inline, sin dotenv). Nunca imprime valores."""
    env = {}
    if not ENV_PATH.exists():
        raise SystemExit(f"ERROR: falta .env en dermica-bcc ({ENV_PATH.name})")
    for raw in ENV_PATH.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def download_keras(dest: Path) -> Path:
    """Descarga 1.0.0.keras desde dermica-models con HF_TOKEN inline.

    Parámetros: dest — ruta local destino. Retorna Path del .keras descargado.
    Por qué token explícito: hf_hub_download sin token falla en repos privados.
    """
    env = _read_env()
    token, owner = env.get("HF_TOKEN"), env.get("HF_OWNER")
    if not token or not owner:
        raise SystemExit("ERROR: HF_TOKEN/HF_OWNER sin definir en .env")
    from huggingface_hub import hf_hub_download
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(f"{owner}/{REPO_MODELS}", repo_type="model",
                           filename=MODEL_FILE, token=token, local_dir=str(dest.parent))
    p = Path(path)
    print(f"descargado: {p} ({p.stat().st_size} bytes)")
    return p


def strip_augment(model):
    """Devuelve el modelo de inferencia SIN la capa augment.

    Parámetros: model — tf.keras.Model completo (con augment).
    Retorna: tf.keras.Model con Input → MobileNetV3Small → Dropout → Dense.
    Por qué: las capas Random* son ruido en inferencia y complican la
    conversión; dropout es identidad en eval.
    """
    base = next((l for l in model.layers if l.name == "MobileNetV3Small"), None)
    if base is None:
        raise SystemExit("ERROR: no se encontró la capa MobileNetV3Small en el modelo")
    new_input = tf.keras.Input(shape=model.input_shape[1:])
    x = base(new_input)
    idx = model.layers.index(base)
    for layer in model.layers[idx + 1:]:
        x = layer(x)
    return tf.keras.Model(new_input, x, name="inference_only")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keras", type=Path, default=Path(os.environ.get("KERAS_PATH", "/content/1.0.0.keras")),
                    help=".keras local (default: descarga desde dermica-models)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--verificar", action="store_true", help="solo comprueba que el .tflite es cargável")
    args = ap.parse_args()

    if args.verificar:
        if not args.out.exists():
            print(f"FAIL: no existe {args.out}")
            return 1
        interpreter = tf.lite.Interpreter(model_path=str(args.out))
        interpreter.allocate_tensors()
        print(f"VERIFICACIÓN: OK — {args.out} cargável ({args.out.stat().st_size} bytes)")
        return 0

    keras_path = args.keras
    if not keras_path.exists():
        keras_path = download_keras(Path("/content/1.0.0.keras"))
    print(f"cargando {keras_path} ...")
    model = strip_augment(tf.keras.models.load_model(str(keras_path), compile=False))

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # float16: Optimize.DEFAULT + float16 = la misma receta medida sobre VAL
    # (AUPRC 0.7786, 1.95 MB, verificada con val-metrics); el modelo fp16 exacto
    # del release es este
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(tflite_model)
    size_bytes = out.stat().st_size
    sha = hashlib.sha256()
    with open(out, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    meta = {
        "_generado_utc": datetime.now(timezone.utc).isoformat(),
        "tflite": str(out),
        "float16_size_bytes": size_bytes,
        "float16_size_mb": round(size_bytes / 1024 / 1024, 2),
        "float16_sha256": sha.hexdigest(),
        "quantization": "float16",
    }
    (out.parent / "float16-meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
