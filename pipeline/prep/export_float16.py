#!/usr/bin/env python3
"""S4 paso 2b: exporta MobileNetV3-Small 1.0.0 a .tflite float16 para el release.

Propósito
---------
Convierte el checkpoint float entrenado en la fase de entrenamiento
(`1.0.0.keras`) a un `.tflite` float16 y escribe su SHA-256 para el release.

Ante el fracaso de la cuantización int8 completa (ΔAUPRC −0.331 vs float en
val; se abortó y no se publicó nada), la decisión de release fue float16:
AUPRC 0.7786 vs 0.7788 (float), 1.95 MB, ideal para LiteRT en Android sin
pérdida medible sobre el checkpoint 1.0.0.

Flujo:
1. Localiza `1.0.0.keras`: usa `--keras` si existe; si no, lo descarga del
   repo remoto privado `dermica-models` (git-remote en Hugging Face; las
   credenciales se leen de las variables de entorno del proceso o del
   archivo `--env-file`; nunca se imprimen ni se escriben fuera de memoria).
2. Carga el modelo y descarta la capa `augment` (identidad en inferencia):
   el .tflite solo lleva graph de inferencia (Base → Dropout → Dense).
3. Convierte con TFLiteConverter SOLO con target_spec.supported_types=[tf.float16]
   (sin optimizations int8; es el checkpoint float almacenado en fp16).
4. Guarda el .tflite en `--out` e imprime tamaño y SHA-256.

Por qué float16: MobileNetV3-Small fine-tuneado sobre 24k imágenes es
extremadamente sensible a int8 weights (val AUPRC 0.78 → 0.45, recall 0.24);
fp16 conserva la precisión float al 100% y reduce el archivo a la mitad del
float32.

Entradas
--------
* `--env-file ~ENV` (opcional): archivo KEY=VALUE (p. ej. el .env del repo,
  solo lectura) con HF_TOKEN/HF_OWNER; si falta un valor se toma de las
  variables de entorno del proceso. Si no existe, se ignoran.
* `--keras ~KERAS` (opcional): .keras local; si se omite o no existe, se
  descarga del git-remote `dermica-models` a la carpeta de `--out`.
* `--out ~TFLITE` (obligatorio): ruta de destino del .tflite.

Salidas
-------
El archivo .tflite indicado en `--out`, un `float16-meta.json` en su misma
carpeta (tamaño, SHA-256, timestamp) y el dict de metadatos por stdout.
El checkpoint y el .tflite se publican en el repo git-remote de modelos por
otro paso del pipeline; este script no publica nada por sí solo.

Uso:
    uv run python export_float16.py --out ~TFLITE
    uv run python export_float16.py --keras ~KERAS --out ~TFLITE
    uv run python export_float16.py --out ~TFLITE --verificar

Cubre el trabajo de la fase S4, paso 2b — exportación del modelo a TFLite
float16 para el release (CRISP-DM / MOBILE-D).
"""

import argparse
from dermica_core.hashing import sha256_file  # import único desde dermica_core
from dermica_core.cli import ParserEspanol  # import único desde dermica_core
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

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
MODEL_FILE = "1.0.0.keras"   # float release publicado en la fase de entrenamiento (lo consume download_keras())




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

    Por qué token explícito: hf_hub_download sin token falla en repos privados.
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


def strip_augment(model):
    """Devuelve el modelo de inferencia SIN la capa augment.

    Parámetros
    ----------
    model : tf.keras.Model
        Modelo completo cargado (con la capa augment de entrenamiento).

    Retorno
    -------
    tf.keras.Model
        Modelo con Input → MobileNetV3Small → Dropout → Dense.

    Errores
    -------
    SystemExit
        Código 1 si no se encuentra la capa MobileNetV3Small en el modelo.

    Efectos colaterales
    -------------------
    Ninguno (construye un modelo nuevo en memoria; el original no se toca).

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
    """CLI: convierte a float16; con --verificar solo audita que el .tflite es cargável.

    Retorno
    -------
    int
        0 en caso de éxito (o verificación OK); 1 si el .tflite no existe o no carga.

    Efectos colaterales
    -------------------
    Lee el .keras (o lo descarga), escribe el .tflite indicado en --out y
    float16-meta.json en su misma carpeta, e imprime los metadatos por
    stdout. Con --verificar no escribe nada.
    """
    ap = ParserEspanol(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keras", type=Path, default=None, metavar="~KERAS",
                    help=".keras local; si se omite o no existe, se descarga del git-remote dermica-models")
    ap.add_argument("--out", type=Path, required=True, metavar="~TFLITE",
                    help="ruta de destino del .tflite float16 (p. ej. .../model-float16.tflite)")
    ap.add_argument("--env-file", type=Path, default=ENV_POR_DEFECTO, metavar="~ENV",
                    help="archivo KEY=VALUE con HF_TOKEN/HF_OWNER (p. ej. el .env del repo; default: .env de la raíz)")
    ap.add_argument("--verificar", action="store_true",
                    help="solo comprueba que el .tflite indicado en --out es cargável")
    args = ap.parse_args()

    if args.verificar:
        if not args.out.exists():
            print(f"FAIL: no existe {args.out}")
            return 1
        if tf is None:
            raise SystemExit("ERROR: falta TensorFlow en este entorno; sin él no se puede cargar el .tflite para la verificación.")
        interpreter = tf.lite.Interpreter(model_path=str(args.out))
        interpreter.allocate_tensors()
        print(f"VERIFICACIÓN: OK — {args.out} cargável ({args.out.stat().st_size} bytes)")
        return 0

    if tf is None:
        raise SystemExit("ERROR: falta TensorFlow en este entorno; la conversión float16 solo puede correr donde TF está instalado.")
    keras_path = args.keras
    if keras_path is None or not keras_path.exists():
        keras_path = download_keras(Path(args.out).parent, args.env_file)
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
    # sha256 del .tflite vía dermica_core (mismo patrón de chunks de 1 MiB)
    sha256_hex = sha256_file(out)
    meta = {
        "_generado_utc": datetime.now(timezone.utc).isoformat(),
        "tflite": str(out),
        "float16_size_bytes": size_bytes,
        "float16_size_mb": round(size_bytes / 1024 / 1024, 2),
        "float16_sha256": sha256_hex,
        "quantization": "float16",
    }
    (out.parent / "float16-meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
