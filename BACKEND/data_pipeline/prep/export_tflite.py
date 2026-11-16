#!/usr/bin/env python3
"""S4 paso 2a: exporta MobileNetV3-Small 1.0.0 a TFLite int8 full-integer.

Flujo:
1. Descarga `1.0.0.keras` desde el repo privado dermica-models (HF_TOKEN/HF_OWNER
   del .env se leen inline; nunca se imprimen).
2. Carga el .keras (float) y convierte vía TFLiteConverter con:
   - optimizations DEFAULT (int8 full-integer);
   - RepresentativeDataGen sobre ~150 imágenes del split TRAIN (sin val ni test),
     [0,255] float32 — MISMO preprocess del entrenamiento S3 (no normalización);
   - supported_ops = TFLITE_BUILTINS_INT8, supported_types = [int8];
   - input/output en tf.int8 (quantización completa in-out).
3. Guarda `model-int8.tflite` en resultados/s4-int8/ y reporta tamaño (bytes)
   y latencia media (ms) sobre ~64 imágenes de val (proxy de eval).

Por qué RepresentativeDataGen de TRAIN: la cuantización necesita calibración con
detos reales; S3 usó [0,255] sin normalizar — desviación invalidaría las muestras.

Uso:
    uv run python export_tflite.py --data prepared/224 --out resultados/s4-int8/model-int8.tflite
    uv run python export_tflite.py --verificar  # comprueba que el .tflite es cargável
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

APP_ROOT = Path(__file__).resolve().parents[4]  # .../APP_DERMICA: monorepo
ENV_PATH = APP_ROOT / "dermica-bcc" / ".env"
TRAIN_PREPARED = APP_ROOT / "resultados" / "s3-training" / "prepared" / "224" / "train"
REPO_MODELS = "dermica-models"
MODEL_FILE = "1.0.0.keras"    # float release publicado en S3
IMG_SIZE = 224                # input del modelo
REP_SAMPLES = 150             # imágenes para calibrar la cuantización
SEED = 42                     # reproducibilidad del muestreo de calibración
LATENCY_SAMPLES = 64          # imágenes val para medir latencia media


def _read_env() -> dict:
    """Lee .env KEY=VALUE (inline, sin dotenv). El dict queda en memoria, nunca se imprime."""
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
    """Descarga 1.0.0.keras desde dermica-models (token inline).

    Parámetros: dest — ruta local destino. Retorna el Path del .keras descargado.
    Por qué token explícito: hf_hub_download sin token pediría anónimamente y fallaría.
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


def representative_dataset(train_dir: Path):
    """Generador de batches de calibración desde TRAIN.

    Params: train_dir — carpeta raíz con BCC/NO_BCC.
    Yields: listas [tensor] float32 [1,224,224,3] con valores [0,255].
    Por qué [0,255]: el modelo se entrenó sin normalizar; calibrar con otra escala
    produciría int8 insignificantes y degradación de AUPRC evitable.
    """
    files = sorted(Path(train_dir).rglob("*.jpg"))
    rng = random.Random(SEED)
    chosen = rng.sample(files, min(REP_SAMPLES, len(files)))
    print(f"calibration: {len(chosen)} imágenes TRAIN (subset seed {SEED})")
    for p in chosen:
        img = tf.keras.preprocessing.image.load_img(str(p), target_size=(IMG_SIZE, IMG_SIZE))
        arr = np.expand_dims(np.asarray(img, dtype=np.float32), axis=0)  # [1,224,224,3] valores [0,255]
        yield [arr]


def convert_to_tflite(model, train_dir: Path, out_path: Path) -> Path:
    """Convierte .keras → .tflite int8 full-integer.

    Params: model — tf.keras.Model cargado; train_dir — carpeta con TRAIN;
            out_path — ruta destino del .tflite.
    Retorna: Path del .tflite generado.
    Por qué full int8: input/output int8 elimina la necesidad de capas float
    extra en LiteRT y es el formato soportado por LiteRT/Android en gama media.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(train_dir)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)
    return out_path


def report_latency(tflite_path: Path, val_dir: Path) -> dict:
    """Mide latencia media sobre ~64 imágenes de val (proxy).

    Params: tflite_path — ruta al .tflite; val_dir — carpeta raíz con val (BCC/NO_BCC).
    Retorna: dict con latencia media/p50 en ms, int8-tensor sizes.
    Por qué val como proxy (no TRAIN ni TEST): prueba realista sobre datos que
    el modelo no vio durante calibración; NO es la medición final de evaluación.
    """
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    files = sorted(Path(val_dir).rglob("*.jpg"))
    rng = random.Random(SEED)
    chosen = rng.sample(files, min(LATENCY_SAMPLES, len(files)))
    in_scale, in_zp = in_det.get("quantization", (0.0, 0))
    latencies = []
    for p in chosen:
        img = tf.keras.preprocessing.image.load_img(str(p), target_size=(IMG_SIZE, IMG_SIZE))
        arr = np.asarray(img, dtype=np.float32)[np.newaxis, ...]
        if in_det["dtype"] == np.int8 and in_scale not in (None, 0.0, 1.0):
            arr = np.round(arr / in_scale + in_zp).astype(np.int8)
        t0 = time.perf_counter()
        interpreter.set_tensor(in_det["index"], arr)
        interpreter.invoke()
        _ = interpreter.get_tensor(out_det["index"])
        latencies.append((time.perf_counter() - t0) * 1000.0)
    lat = sorted(latencies)
    return {"latencia_media_ms": round(float(np.mean(lat)), 2),
            "latencia_p50_ms": round(float(lat[len(lat) // 2]), 2),
            "latencia_min_ms": round(float(lat[0]), 2),
            "latencia_max_ms": round(float(lat[-1]), 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=TRAIN_PREPARED,
                    help="carpeta raíz BCC/NO_BCC del split TRAIN preparado")
    ap.add_argument("--val-dir", type=Path, default=None,
                    help="carpeta raíz del split VAL preparado (para latencia)")
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("OUT_DIR", "resultados/s4-int8/model-int8.tflite")))
    ap.add_argument("--keras", type=Path, default=Path(os.environ.get("KERAS_PATH", "/content/1.0.0.keras")),
                    help=".keras local (default: descarga desde dermica-models)")
    ap.add_argument("--verificar", action="store_true", help="solo comprueba que el .tflite es cargável")
    args = ap.parse_args()

    if args.verificar:
        p = Path(os.environ.get("OUT_TFLITE", str(args.out)))
        if not p.exists():
            print(f"FAIL: no existe {p}")
            return 1
        interpreter = tf.lite.Interpreter(model_path=str(p))
        interpreter.allocate_tensors()
        print(f"VERIFICACIÓN: OK — {p} cargável ({p.stat().st_size} bytes)")
        return 0

    keras_path = args.keras
    if not keras_path.exists():
        keras_path = download_keras(Path("/content/1.0.0.keras"))
    print(f"cargando {keras_path} ...")
    model = tf.keras.models.load_model(str(keras_path), compile=False)

    out = convert_to_tflite(model, args.data, args.out)
    size = out.stat().st_size
    print(f"tflite: {out} ({size} bytes, {size / 1024 / 1024:.2f} MB)")

    lat = {}
    if args.val_dir and args.val_dir.exists():
        lat = report_latency(out, args.val_dir)
        print("latencia int8 (val):", json.dumps(lat))

    meta = {"_generado_utc": datetime.now(timezone.utc).isoformat(),
            "tflite": str(out), "bytes": size, "bytes_mb": round(size / 1024 / 1024, 2),
            **lat}
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
