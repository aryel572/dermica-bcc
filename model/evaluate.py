#!/usr/bin/env python3
"""S4 paso 2: evaluación formal sobre TEST (una sola vez).

Propósito
---------
Evalúa el modelo released del proyecto sobre el split TEST (una sola pasada,
sin retocar nada) y escribe el JSON de métricas finales: la evidencia numérica
del release 1.0.0.

Qué recibe
----------
Flags de CLI (origen: línea de comandos):
    --data ~DIR      Carpeta TEST con subcarpetas BCC/NO_BCC de imágenes 224 (obligatoria
                     salvo en modo --verificar).
    --model ~KERAS   Ruta local del .keras released (float). Sin ella ni --tflite, error.
    --tflite ~PATH   Alternativa int8: evalúa un .tflite y esta opción manda sobre --model.
    --split-csv      CSV del split evaluado (informativo; aceptado pero hoy no se usa en el código).
    --tag            Etiqueta del split en el JSON (default "test"; p.ej. "val").
    --reports ~JSON  Ruta del JSON de salida (el padre se crea si falta); además se imprime a stdout.
    --verificar ~JSON  Modo auditoría: no evalúa nada, solo chequea coherencia del JSON dado.
El modelo no lo descarga este script: la ruta local la resuelve el operador/entorno.

Qué produce
-----------
JSON de métricas (stdout y, opcionalmente, --reports) con la AUPRC primaria,
recall, especificidad, balanced_acc, f2 y precision del split, conteos
TP/FP/TN/FN, el preprocess usado y la descripción del modelo (float o int8).

Quién consume
-------------
Nadie importa este módulo. Lo ejecuta el operador; su salida la audita
`verificar()` (mismo módulo, modo --verificar) y queda como evidencia del
release en la documentación de S4.

Evalúa el modelo `1.0.0.keras` de dermica-models sobre el split TEST (2.488
imágenes, BCC 317) y escribe metrics JSON. TEST es de una sola pasada: el
modelo se fijó en S3 y nunca se tocó esta partición.

El preprocessing Replica EXACTAMENTE el del entrenamiento en S3, que NO normaliza:
decode_jpeg → cast float32 (rango [0,255]) → MobileNetV3-Small → sigmoid.
NO se usa mobilenet_v3.preprocess_input ni /255 (el checkpoint de entrenamiento
ya fue ajustado sobre [0,255]; cualquier otra escala rompería las métricas).

Uso:
    uv run python evaluate.py --data ~DIR_TEST --split-csv ~CSV \
        --model ~KERAS --tag test --reports ~JSON
    uv run python evaluate.py --verificar ~JSON

Fase: S4 (CRISP-DM Evaluation).
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

# Origen: mismo input que el entrenamiento en S3 (contrato 224 px). Nota: hoy no se
# referencia en este módulo (las imágenes ya llegan a 224); se mantiene por paridad
# con el pipeline de entrenamiento, no por un consumidor.
IMG_SIZE = 224
BATCH_SIZE = 32  # tamaño de inferencia; consumo: ds.batch() en la rama float (el modo int8 procesa imagen a imagen)
CLASS_FOLDERS = ["NO_BCC", "BCC"]  # 0 = NO_BCC, 1 = BCC (mismo orden que S3 y que class_weight); consumidor: evaluar()


def cargar_modelos(model_path: Path):
    """Carga el .keras entrenado (float).

    Parámetros: model_path — ruta local del .keras (lo descarga el entorno/operador;
                este módulo nunca va al remoto por su cuenta).
    Retorna: modelo tf.keras listo para inferencia (compile=False: no requiere loss).
    Lanza: error de tf.keras si el archivo no es un modelo Keras válido o no existe.
    Globales: ninguna.
    Por qué función propia: permite reutilizar la carga para float/int8 cuando
    llegue la cuantización (S4 paso 2).
    """
    model = tf.keras.models.load_model(str(model_path), compile=False)
    return model


def evaluar(data_dir: Path):
    """Lista archivos del directorio y devuelve (paths, y_true).

    Parámetros: data_dir — Path con subcarpetas BCC/NO_BCC de imágenes 224.
    Retorna: (lista_paths, lista_y_true) alineados; y_true es np.ndarray int
             (0 = NO_BCC, 1 = BCC) en el orden NO_BCC → BCC, sorted por carpeta.
    Lanza: nada propio; una subcarpeta BCC/NO_BCC ausente da lista vacía
           (el llamador detecta `if not paths` y falla con mensaje claro).
    Globales: CLASS_FOLDERS (orden fijo 0/1).
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
    """Lee y decodifica una imagen SIN normalizar (mismo pipeline que S3).

    Parámetros: path — tensor/string con la ruta de la imagen; label — etiqueta 0/1.
    Retorna: tuple (img, label) con img float32 en [0,255] shape (H, W, 3).
    Lanza: error de tf.io si el archivo no existe o no es un JPEG decodificable.
    Globales: ninguna.
    Por qué sin /255: image_dataset_from_directory en train_bcn NO normaliza;
    el modelo se fine-tuneó sobre valores [0,255] y debe verse igual en eval.
    """
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.cast(img, tf.float32), label


def main() -> int:
    """Ejecuta la evaluación sobre --data con --model/--tflite y escribe --reports.

    Parámetros: ninguno (los flags llegan por línea de comandos vía argparse);
    `description=__doc__` publica el docstring del módulo como ayuda -h.
    --verificar no escribe; solo audita el JSON ya generado.
    Retorna: 0 OK, 1 error o args inconsistentes (modelo/data ausentes o JSON inexistente).
    Lanza: propaga errores de inferencia (modelo corrupto, TFLite inválido, OOM);
           los flags incompletos SÍ devuelven 1 con mensaje claro en stdout.
    Globales: BATCH_SIZE (rama float), CLASS_FOLDERS vía evaluar().
    Notas de la rama int8: `in_det`/`out_det` son los descriptores del tensor;
    la quantization (scale, zero_point) del int8 se aplica a mano porque el
    modelo float16/int8 del release no lleva preprocesamiento dentro.
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
        # LiteRT sin delegados por defecto: el runtime puede fallar con XNNPACK
        # sobre grafos int8; el fallback tf.lite usa kernels de referencia.
        try:
            from ai_edge_litert.interpreter import Interpreter as LiteInterpreter, OpResolverType
            interpreter = LiteInterpreter(
                model_path=str(args.tflite),
                experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
                num_threads=4,
            )
        except ImportError:
            interpreter = tf.lite.Interpreter(model_path=str(args.tflite))
        interpreter.allocate_tensors()
        in_det = interpreter.get_input_details()[0]
        out_det = interpreter.get_output_details()[0]
        # scale/zp del int8 (vienen del descriptor del tensor; si no hay cuantización, (0.0, 0))
        in_scale, in_zp = in_det.get("quantization", (0.0, 0))
        out_scale, out_zp = out_det.get("quantization", (0.0, 0))
        ys = []
        for p in paths:
            img = tf.io.read_file(p)
            img = tf.io.decode_jpeg(img, channels=3)
            arr = tf.cast(img, tf.float32).numpy()[None, ...]
            if in_det["dtype"] == np.int8:
                arr = np.round(arr / in_scale + in_zp).astype(np.int8)
            interpreter.set_tensor(in_det["index"], arr)
            interpreter.invoke()
            raw = interpreter.get_tensor(out_det["index"]).reshape(-1).astype(np.float32)
            if out_det["dtype"] == np.int8 and out_scale not in (None, 0.0, 1.0):
                raw = (raw - out_zp) * out_scale  # int8 cuantizado → desnormaliza; si no, las métricas saldrían fuera de [0,1]
            ys.append(float(raw[0]))
        y_score = np.asarray(ys, dtype=np.float32)
        # descripción del artefacto y preprocess usados; quedan grabados en el JSON (auditoría del release)
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
    n_bcc = int(np.sum(y_true == 1))  # positivos reales del TEST (denominador clínico: 317 en 1.0.0)
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

    Parámetros: json_path — Path del JSON generado por la rama de evaluación.
    Retorna: 0 si el JSON es internamente coherente; 1 en caso contrario (o si no existe).
    Lanza: nada propio; un JSON malformado hace fallar json.loads (excepción propaga).
    Globales: ninguna.
    Comprueba: n_test == tp+fp+tn+fn, tp+fn == bcc_test, y
    auprc/recall/especificidad/balanced_acc/f2/precision dentro de [0, 1];
    VERIFICACIÓN: OK/FALLO. No escribe nada: es una auditoría de solo lectura.
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
