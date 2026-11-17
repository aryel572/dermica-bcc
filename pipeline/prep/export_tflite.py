#!/usr/bin/env python3
"""S4 paso 2a: exporta MobileNetV3-Small 1.0.0 a TFLite int8 full-integer.

Propósito
---------
Convierte el checkpoint float entrenado en la fase de entrenamiento
(`1.0.0.keras`) a un `.tflite` int8 full-integer y mide su latencia media
sobre el split val preparado (proxy de evaluación, nunca el test).

Flujo:
1. Localiza `1.0.0.keras`: usa `--keras` si existe; si no, lo descarga del
   repo remoto privado `dermica-models` (git-remote en Hugging Face; las
   credenciales se leen de las variables de entorno del proceso o del
   archivo `--env-file`; nunca se imprimen ni se escriben fuera de memoria).
2. Carga el .keras (float) y convierte vía TFLiteConverter con:
   - optimizations DEFAULT (int8 full-integer);
   - RepresentativeDataGen sobre ~150 imágenes del split TRAIN preparado
     (sin val ni test), [0,255] float32 — MISMO preprocess del entrenamiento
     (no normalización);
   - supported_ops = TFLITE_BUILTINS_INT8, supported_types = [int8];
   - input/output en tf.int8 (cuantización completa in-out).
3. Guarda el .tflite en `--out` y reporta tamaño (bytes) y latencia media
   (ms) sobre ~64 imágenes de val (proxy de eval) si se pasa `--val-dir`.

Por qué RepresentativeDataGen de TRAIN: la cuantización necesita calibración con
datos reales; el entrenamiento usó [0,255] sin normalizar — desviarse invalidaría
las muestras.

Entradas
--------
* `--data ~DIR` (obligatorio): carpeta raíz BCC/NO_BCC del TRAIN preparado de
  224 px (origen externo al repo; SOLO LECTURA).
* `--env-file ~ENV` (opcional): archivo KEY=VALUE (p. ej. el .env del repo,
  solo lectura) con HF_TOKEN/HF_OWNER; si falta un valor se toma de las
  variables de entorno del proceso. Si no existe, se ignoran.
* `--keras ~KERAS` (opcional): .keras local; si se omite o no existe, se
  descarga del git-remote `dermica-models` a la carpeta de `--out`.
* `--val-dir ~DIR` (opcional): carpeta raíz BCC/NO_BCC del VAL preparado; si
  se indica, se mide latencia media sobre una muestra fija.
* `--out ~TFLITE` (obligatorio): ruta de destino del .tflite.

Salidas
-------
El archivo .tflite indicado en `--out`, un dict de metadatos impreso por
stdout (tamaño, latencia) y — si se indica --val-dir — la latencia int8 sobre val.
El checkpoint y el .tflite se publican en el repo git-remote de modelos por
otro paso del pipeline; este script no publica nada por sí solo.

Uso:
    uv run python export_tflite.py --data ~DIR_TRAIN --out ~TFLITE
    uv run python export_tflite.py --data ~DIR_TRAIN --val-dir ~DIR_VAL --out ~TFLITE
    uv run python export_tflite.py --out ~TFLITE --verificar

Cubre el trabajo de la fase S4, paso 2a — exportación del modelo a TFLite int8
(CRISP-DM / MOBILE-D).
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

# TensorFlow solo lo necesitan las rutas de conversión/verificación, no el CLI
# ni --help: se importa de forma perezosa para que `--help` funcione en
# entornos sin TF; cada ruta que lo usa falla con un mensaje claro en español.
try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None

# Raíz del repo (este archivo vive en pipeline/prep/, dos
# niveles por debajo de la raíz). Solo se usa para el default repo-relativo de
# --env-file (el .env del repo, gitignored); nada del repo se modifica.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_POR_DEFECTO = REPO_ROOT / ".env"
# Nombre del repo git-remote (Hugging Face) que aloja el checkpoint float
# publicado en la fase de entrenamiento; NO es una ruta local. Lo consumen
# download_keras() y el error de credenciales.
REPO_MODELS = "dermica-models"
MODEL_FILE = "1.0.0.keras"    # float release publicado en la fase de entrenamiento (lo consume download_keras())
IMG_SIZE = 224                # input del modelo (lo consumen representative_dataset, report_latency y main())
REP_SAMPLES = 150             # imágenes para calibrar la cuantización (lo consume representative_dataset())
SEED = 42                     # reproducibilidad del muestreo de calibración (lo consumen representative_dataset() y report_latency())
LATENCY_SAMPLES = 64          # imágenes val para medir latencia media (lo consume report_latency())


class _ParserEspanol(argparse.ArgumentParser):
    """ArgumentParser con los mensajes de error en español neutro.

    argparse redacta por defecto en inglés; el pipeline de la tesis es en
    español, así que los fallos de invocación se traducen aquí (solo el
    texto mostrado, nunca el comportamiento ni los códigos de salida:
    sigue siendo exit 2 ante argumentos inválidos o faltantes).
    """

    def error(self, message: str) -> None:
        """Imprime el uso y un error en español; sale con código 2.

        Parámetros
        ----------
        message : str
            Detalle redactado por argparse (se conserva el dato útil, p. ej.
            los nombres de los flags faltantes, pero se enmarca en español).

        Retorno
        -------
        No retorna: termina el proceso con exit 2.
        """
        self.print_usage(sys.stderr)
        if "the following arguments are required" in message:
            faltan = message.split(":", 1)[1].strip()
            self.exit(2, f"ERROR: faltan los flags obligatorios: {faltan}\n"
                         f"Consulta --help para la ayuda completa en español.\n")
        self.exit(2, f"ERROR: {message}\nConsulta --help para la ayuda completa en español.\n")


def _read_env(env_file: Path) -> dict:
    """Lee un archivo KEY=VALUE estilo dotenv (inline, sin dotenv).

    Parámetros
    ----------
    env_file : Path
        Archivo KEY=VALUE de origen; si no existe, se devuelve {} (la
        resolución de credenciales cae entonces al entorno del proceso).

    Retorno
    -------
    dict
        Pares clave→valor con comillas simples/dobles desprendidas; claves
        ausentes o vacías no aparecen. El dict queda en memoria, nunca se imprime.

    Efectos colaterales
    -------------------
    Ninguno (solo lectura; jamás imprime valores).
    """
    env = {}
    if not env_file.exists():
        return env
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _hf_token_owner(env_file: Path) -> tuple[str, str]:
    """Resuelve HF_TOKEN y HF_OWNER desde el entorno del proceso o el archivo --env-file.

    Prioridad: las variables del proceso ganan sobre el archivo (lo explícito
    domina al implícito). El archivo nunca se imprime ni se copia.

    Parámetros
    ----------
    env_file : Path
        Archivo KEY=VALUE consultado como segunda fuente de credenciales.

    Retorno
    -------
    tuple[str, str]
        (token, owner) listos para autenticar contra Hugging Face.

    Errores
    -------
    SystemExit
        Código 1 con mensaje en español si falta cualquiera de las dos
        variables en ambas fuentes.
    """
    file_env = _read_env(env_file)
    token = os.environ.get("HF_TOKEN") or file_env.get("HF_TOKEN")
    owner = os.environ.get("HF_OWNER") or file_env.get("HF_OWNER")
    if not token or not owner:
        raise SystemExit(
            "ERROR: faltan HF_TOKEN/HF_OWNER. Defínelas como variables de entorno del proceso "
            "o en el archivo indicado por --env-file (p. ej. el .env del repo; solo lectura)."
        )
    return token, owner


def download_keras(dest_dir: Path, env_file: Path) -> Path:
    """Descarga 1.0.0.keras del repo git-remote privado dermica-models.

    Parámetros
    ----------
    dest_dir : Path
        Directorio destino local; se crea si no existe.
    env_file : Path
        Archivo KEY=VALUE con las credenciales (ver _hf_token_owner).

    Retorno
    -------
    Path
        Ruta local del .keras descargado.

    Errores
    -------
    SystemExit
        Si faltan las credenciales (HF_TOKEN/HF_OWNER) en proceso o archivo.

    Efectos colaterales
    -------------------
    Descarga desde Hugging Face (red) y escribe el archivo dentro de dest_dir.

    Por qué token explícito: hf_hub_download sin token pediría anónimamente y fallaría
    en el repo privado.
    """
    token, owner = _hf_token_owner(env_file)
    from huggingface_hub import hf_hub_download
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(f"{owner}/{REPO_MODELS}", repo_type="model",
                           filename=MODEL_FILE, token=token, local_dir=str(dest_dir))
    p = Path(path)
    print(f"descargado: {p} ({p.stat().st_size} bytes)")
    return p


def representative_dataset(train_dir: Path):
    """Generador de batches de calibración desde TRAIN.

    Parámetros
    ----------
    train_dir : Path
        Carpeta raíz con BCC/NO_BCC del TRAIN preparado.

    Yields
    ------
    list[np.ndarray]
        [tensor] float32 [1,224,224,3] con valores [0,255].

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

    Parámetros
    ----------
    model : tf.keras.Model
        Modelo ya cargado en memoria (float).
    train_dir : Path
        Carpeta raíz con el TRAIN preparado (origen de la calibración).
    out_path : Path
        Ruta destino del .tflite; su padre se crea si no existe.

    Retorno
    -------
    Path
        Ruta del .tflite generado.

    Errores
    -------
    ValueError / RuntimeError
        Propagados desde TFLiteConverter si el grafo no es convertible.

    Efectos colaterales
    -------------------
    Escribe el archivo .tflite en out_path.

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

    Parámetros
    ----------
    tflite_path : Path
        Ruta al .tflite ya generado.
    val_dir : Path
        Carpeta raíz con el VAL preparado (BCC/NO_BCC).

    Retorno
    -------
    dict
        Latencia media/p50/min/max en ms sobre la muestra fija (semilla SEED).

    Errores
    -------
    FileNotFoundError
        Si tflite_path no existe o val_dir no contiene imágenes .jpg.

    Efectos colaterales
    -------------------
    Ninguno (solo lectura; carga el .tflite en un intérprete local).

    Por qué val como proxy (no TRAIN ni TEST): prueba realista sobre datos que
    el modelo no vio durante calibración; NO es la medición final de evaluación.
    """
    # El delegate XNNPACK falla en este runtime (nodo 117 del grafo int8).
    # Se usa LiteRT (paquete oficial, sin delegados por defecto); si no está
    # instalado se cae a tf.lite con sus delegados por defecto.
    try:
        from ai_edge_litert.interpreter import Interpreter as LiteInterpreter, OpResolverType
        interpreter = LiteInterpreter(
            model_path=str(tflite_path),
            experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
            num_threads=4,
        )
    except ImportError:
        if tf is None:
            raise SystemExit("ERROR: no está ai_edge_litert ni TensorFlow en este entorno; falta una de las dos para medir latencia.")
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
        if in_det["dtype"] == np.int8:
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
    """CLI: convierte y mide; con --verificar solo audita que el .tflite es cargável.

    Retorno
    -------
    int
        0 en caso de éxito (o verificación OK); 1 si el .tflite no existe o no carga.

    Efectos colaterales
    -------------------
    Lee el .keras (o lo descarga), escribe el .tflite indicado en --out e
    imprime un dict de metadatos por stdout. Con --verificar no escribe nada.
    """
    ap = _ParserEspanol(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, metavar="~DIR_TRAIN",
                    help="carpeta raíz BCC/NO_BCC del TRAIN preparado de 224 px (origen de la calibración)")
    ap.add_argument("--val-dir", type=Path, default=None, metavar="~DIR_VAL",
                    help="carpeta raíz BCC/NO_BCC del VAL preparado; si se indica, mide latencia media")
    ap.add_argument("--out", type=Path, required=True, metavar="~TFLITE",
                    help="ruta de destino del .tflite int8 (p. ej. .../model-int8.tflite)")
    ap.add_argument("--keras", type=Path, default=None, metavar="~KERAS",
                    help=".keras local; si se omite o no existe, se descarga del git-remote dermica-models")
    ap.add_argument("--env-file", type=Path, default=ENV_POR_DEFECTO, metavar="~ENV",
                    help="archivo KEY=VALUE con HF_TOKEN/HF_OWNER (p. ej. el .env del repo; default: .env de la raíz)")
    ap.add_argument("--verificar", action="store_true",
                    help="solo comprueba que el .tflite indicado en --out es cargável")
    args = ap.parse_args()

    if args.verificar:
        p = args.out
        if not p.exists():
            print(f"FAIL: no existe {p}")
            return 1
        try:
            from ai_edge_litert.interpreter import Interpreter as LiteInterpreter, OpResolverType
            interpreter = LiteInterpreter(
                model_path=str(p),
                experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
                num_threads=4,
            )
        except ImportError:
            if tf is None:
                raise SystemExit("ERROR: no está ai_edge_litert ni TensorFlow en este entorno; falta una de las dos para cargar el .tflite.")
            interpreter = tf.lite.Interpreter(model_path=str(p))
        interpreter.allocate_tensors()
        print(f"VERIFICACIÓN: OK — {p} cargável ({p.stat().st_size} bytes)")
        return 0

    if tf is None:
        raise SystemExit("ERROR: falta TensorFlow en este entorno; la conversión int8 solo puede correr donde TF está instalado.")
    keras_path = args.keras
    if keras_path is None or not keras_path.exists():
        keras_path = download_keras(Path(args.out).parent, args.env_file)
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
