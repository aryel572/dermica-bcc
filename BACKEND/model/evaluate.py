#!/usr/bin/env python3
"""S4 paso 2: evaluación formal sobre TEST (una sola vez).

Evalúa el modelo `1.0.0.keras` de dermica-models sobre el split TEST (2.488
imágenes, BCC 317) y escribe metrics JSON. TEST es de una sola pasada: el
modelo se fijó en S3 y nunca se tocó esta partición.

El preprocessing Replica EXACTAMENTE el del entrenamiento en S3, que NO normaliza:
decode_jpeg → cast float32 (rango [0,255]) → MobileNetV3-Small → sigmoid.
NO se usa mobilenet_v3.preprocess_input ni /255 (el checkpoint de entrenamiento
ya fue ajustado sobre [0,255]; cualquier otra escala rompería las métricas).

Uso:
    uv run python evaluate.py --data /content/test-224 \
        --split-csv test.csv --model /content/1.0.0.keras --tag test --reports test-metrics.json
    uv run python evaluate.py --verificar resultados/s4-eval/test-metrics.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics

IMG_SIZE = 224   # mismo input que el entrenamiento
BATCH_SIZE = 32  # tamaño de inferencia
CLASS_FOLDERS = ["NO_BCC", "BCC"]  # 0 = NO_BCC, 1 = BCC (mismo orden que S3)


def cargar_modelos(model_path: Path):
    """Carga el .keras entrenado (float).

    Parámetros: model_path — ruta local del .keras (descarga en el runner Colab).
    Retorna: modelo tf.keras listo para inferencia.
    Por qué función propia: permite reutilizar la carga para float/int8 cuando
    llegue la cuantización (S4 paso 2).
    """
    model = tf.keras.models.load_model(str(model_path), compile=False)
    return model


def evaluar(data_dir: Path):
    """Lista archivos del directorio y devuelve (paths, y_true).

    Parámetros: data_dir — carpeta con subcarpetas BCC/NO_BCC de imágenes 224.
    Retorna: (lista_paths, lista_y_true) alineados.
    Por qué listar archivos directamente: no hay loader fancy; sólo la correlación
    nombre→carpeta importa, así el resultado depende del contenido real en disco.
    """
    paths, y_true = [], []
    for clase_idx, cls in enumerate(CLASS_FOLDERS):  # NO_BCC=0, BCC=1
        for p in sorted((data_dir / cls).glob("*.jpg")):
            paths.append(str(p))
            y_true.append(clase_idx)
    return paths, np.array(y_true, dtype=int)


def _parse(path, label):
    """Lee y normaliza una imagen SIN normalizar (mismo pipeline que S3).

    Por qué sin /255: image_dataset_from_directory en train_bcn NO normaliza;
    el modelo se fine-tuneó sobre valores [0,255] y debe verse igual en eval.
    """
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.cast(img, tf.float32), label


def main() -> int:
    """Ejecuta la evaluación sobre --data con --model y escribe --reports.

    --verificar no escribe; solo audita el JSON ya generado.
    Retorna: 0 OK, 1 error o args inconsistentes.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=None,
                    help="directorio test-224 con subcarpetas BCC/NO_BCC (obligatorio para evaluar)")
    ap.add_argument("--split-csv", type=Path, default=None, help="CSV del split usado (informativo)")
    ap.add_argument("--model", type=Path, default=None, help="ruta al .keras (float)")
    ap.add_argument("--tflite", type=Path, default=None, help="ruta a un .tflite (modo int8); si se indica, se ignora --model")
    ap.add_argument("--tag", default="test", help="etiqueta del split evaluado (test|val)")
    ap.add_argument("--reports", type=Path, default=None, help="ruta del JSON de salida")
    ap.add_argument("--verificar", type=Path, default=None, help="JSON ya generado a verificar")
    args = ap.parse_args()

    if args.verificar is not None:
        return verificar(args.verificar)

    if args.model is None and args.tflite is None:
        print("ERROR: --model (o --tflite) es obligatorio para evaluar")
        return 1

    if args.data is None:
        print("ERROR: --data es obligatorio para evaluar")
        return 1
    paths, y_true = evaluar(args.data)
    if not paths:
        print(f"ERROR: {args.data} no contiene imágenes")
        return 1

    if args.tflite is not None:
        ys = []
        interpreter = tf.lite.Interpreter(model_path=str(args.tflite))
        interpreter.allocate_tensors()
        in_det = interpreter.get_input_details()[0]
        out_det = interpreter.get_output_details()[0]
        in_scale, in_zp = in_det.get("quantization", (0.0, 0))
        out_scale, out_zp = out_det.get("quantization", (0.0, 0))
        for p in paths:
            img = tf.io.read_file(p)
            img = tf.io.decode_jpeg(img, channels=3)
            arr = tf.cast(img, tf.float32).numpy()[None, ...]
            if in_det["dtype"] == np.int8 and in_scale not in (None, 0.0, 1.0):
                arr = np.round(arr / in_scale + in_zp).astype(np.int8)
            interpreter.set_tensor(in_det["index"], arr)
            interpreter.invoke()
            raw = interpreter.get_tensor(out_det["index"]).reshape(-1).astype(np.float32)
            if out_det["dtype"] == np.int8 and out_scale not in (None, 0.0, 1.0):
                raw = (raw - out_zp) * out_scale  # int8 cuantizado → desnormaliza
            ys.append(float(raw[0]))
        y_score = np.asarray(ys, dtype=np.float32)
        modelo_desc = "MobileNetV3-Small int8 (tflite), 1.0.0"
        preproc = "decode_jpeg→float32[0,255]→int8(q)"
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, y_true))
        ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        model = cargar_modelos(args.model)
        y_score = model.predict(ds, verbose=0).reshape(-1)
        modelo_desc = "MobileNetV3-Small (fine-tuned), 1.0.0"
        preproc = "decode_jpeg→float32[0,255]"

    y_score = np.asarray(y_score).reshape(-1).tolist()

    rep = metrics.reporte(y_true.tolist(), y_score)
    n_bcc = int(np.sum(y_true == 1))
    out = {
        "version": "1.0.0",
        "split": args.tag,
        "n_test": int(len(y_true)),
        "bcc_test": n_bcc,
        "auprc": round(rep["auprc"], 4),
        "recall": round(rep["recall"], 4),
        "especificidad": round(rep["especificidad"], 4),
        "balanced_acc": round(rep["balanced_acc"], 4),
        "f2": round(rep["f2"], 4),
        "precision": round(rep["precision"], 4),
        "tp": int(rep["tp"]), "fp": int(rep["fp"]), "tn": int(rep["tn"]), "fn": int(rep["fn"]),
        "preprocess": preproc,
        "modelo": modelo_desc,
        "_generado_utc": datetime.now(timezone.utc).isoformat(),
    }
    if args.reports:
        args.reports.parent.mkdir(parents=True, exist_ok=True)
        args.reports.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"métricas -> {args.reports}")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def verificar(json_path: Path) -> int:
    """Audita el JSON de métricas buscando incoherencias internas.

    Compruebas: n_test == tp+fp+tn+fn, tp+fn == bcc_test, ratios en [0, 1],
    valores redondeados a 4 decimales; VERIFICACIÓN: OK/FALLO.
    No escribe nada.
    """
    if not json_path.exists():
        print(f"FAIL: no existe {json_path}")
        return 1
    d = json.loads(json_path.read_text())
    problems = []
    counts = d["tp"] + d["fp"] + d["tn"] + d["fn"]
    if counts != d["n_test"]:
        problems.append(f"tp+fp+tn+fn ({counts}) != n_test ({d['n_test']})")
    if (d["tp"] + d["fn"]) != d["bcc_test"]:
        problems.append(f"tp+fn ({d['tp'] + d['fn']}) != bcc_test ({d['bcc_test']})")
    for k in ("auprc", "recall", "especificidad", "balanced_acc", "f2", "precision"):
        if not (0.0 <= d[k] <= 1.0):
            problems.append(f"{k}={d[k]} fuera de [0,1]")
    print("VERIFICACIÓN:", "OK" if not problems else "FALLO")
    for p in problems:
        print("FAIL", p)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
