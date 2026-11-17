#!/usr/bin/env python3
"""S3 paso 1: re-escalado del train+val a 224 px (preparación para entrenamiento).

Propósito
---------
Lee las imágenes crudas de ISIC 2019 (solo lectura) y los CSV particionados
70/20/10 de la fase de splits, y genera la copia de entrenamiento re-escalada
a 224 px organizada por partición y clase binaria:
`<out-dir>/{train,val}/<BCC|NO_BCC>/<image>.jpg`.

El TEST oficial de ISIC 2019 nunca se toca y TESIS tampoco aparece (sus ids
ya fueron excluidos de train/val en la fase de splits).

Entradas
--------
* `--data-dir ~DIR` (obligatorio): directorio con los archivos crudos de
  ISIC 2019 (el zip de training). Origen EXTERNO al repo; SOLO LECTURA.
* `--splits-dir ~DIR_CSV` (obligatorio): directorio con train.csv y val.csv
  (columnas image, split, label_full, label_bcc, lesion_id).
* `--out-dir ~DIR_OUT` (obligatorio): destino de la copia preparada.

Salidas
-------
`<out-dir>/{train,val}/<BCC|NO_BCC>/<image>.jpg` (fuera del repo),
`<out-dir>/../resize-summary.json` con el recibo de la corrida y —
con `--verificar` — un chequeo de conteos sin escritura.

Quién consume sus salidas: el entrenamiento de la fase S3 (la copia
preparada es la entrada de la DataPipeline de Keras/ImageNet) y, al cierre,
la tesis registra el recibo en `resize-summary.json`; nadie más escribe ni
lee el árbol 224 px.

Uso:
    uv run python resize_224.py --data-dir ~DIR --splits-dir ~DIR_CSV --out-dir ~DIR_OUT
    uv run python resize_224.py --data-dir ~DIR --splits-dir ~DIR_CSV --out-dir ~DIR_OUT --verificar

Cubre el trabajo de la fase S3, paso 1 — preparación de datos para el
entrenamiento (CRISP-DM / MOBILE-D).
"""

import argparse
from dermica_core.io import leer_csv_csv_splits  # import único desde dermica_core
from dermica_core.cli import ParserEspanol  # import único desde dermica_core
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image

# Nombres canónicos de ISIC 2019 dentro del directorio de entrada. TRAIN_ZIP
# es el archivo de origen; ZIP_PREFIX es la carpeta de primer nivel DENTRO
# del zip (las entradas se direccionan como f"{ZIP_PREFIX}{image_id}.jpg").
# Los consumen main() y verificar().
TRAIN_ZIP = "ISIC_2019_Training_Input.zip"
ZIP_PREFIX = "ISIC_2019_Training_Input/"
IMG_SIZE = 224  # input del modelo MobileNetV3-Small (checkpoint imagenet); lo consumen main() y verificar()
JPEG_Q = 85     # q85: suficiente para 224 px, ~35-50 KB por imagen; lo consume main()




def load_splits(splits_dir: Path) -> pd.DataFrame:
    """Concatena train y val (el test de ISIC y la lista de exclusión no entran aquí).

    Parámetros
    ----------
    splits_dir : Path
        Directorio con los CSV particionados de la fase de splits.

    Retorno
    -------
    pd.DataFrame
        DataFrame con image, split, label_full, label_bcc, lesion_id.

    Errores
    -------
    FileNotFoundError / pd.errors.ParserError / ValueError
        Si falta train.csv o val.csv o no son parseables.

    Efectos colaterales
    -------------------
    Ninguno (solo lectura).

    Globales
    --------
    Ninguna (usa `leer_csv_csv_splits` importada de dermica_core.io).
    """
    dfs = []
    for name in ("train", "val"):
        dfs.append(leer_csv_csv_splits(splits_dir / f"{name}.csv"))
    return pd.concat(dfs, ignore_index=True)


def main() -> int:
    """Ejecuta el re-escalado binario de train+val (usada como paso previo del entrenamiento).

    Retorno
    -------
    int
        0 si no hubo imágenes saltadas, 1 si alguna falta en el zip.

    Errores
    -------
    zipfile.BadZipFile
        Si el zip de entrada no es válido.

    Efectos colaterales
    -------------------
    Escribe <out-dir>/{train,val}/<BCC|NO_BCC>/<image>.jpg y
    resize-summary.json en el directorio padre de <out-dir>; con --verificar
    audita conteos y tamaño sin escribir.

    Globales
    --------
    `TRAIN_ZIP`, `ZIP_PREFIX`, `IMG_SIZE`, `JPEG_Q` (constantes del módulo).
    """
    ap = ParserEspanol(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True, metavar="~DIR",
                    help="directorio con los archivos crudos de ISIC 2019 (zip de training; solo lectura)")
    ap.add_argument("--splits-dir", type=Path, required=True, metavar="~DIR_CSV",
                    help="directorio con train.csv y val.csv de la fase de splits")
    ap.add_argument("--out-dir", type=Path, required=True, metavar="~DIR_OUT",
                    help="directorio donde se escribe la copia preparada (<DIR_OUT>/{train,val}/<CLASE>/<image>.jpg)")
    ap.add_argument("--verificar", action="store_true",
                    help="solo audita la salida ya generada (sin escribir)")
    args = ap.parse_args()

    if args.verificar:
        return verificar(args.out_dir, args.splits_dir)

    t0 = time.time()
    df = load_splits(args.splits_dir)
    print(f"imágenes a procesar: {len(df)} (train+val)")

    # abrir el zip UNA sola vez y buscar en entries; más rápido que reabrir por imagen
    zip_path = args.data_dir / TRAIN_ZIP
    written, skipped = 0, []
    with zipfile.ZipFile(zip_path) as zf:
        entries = set(zf.namelist())
        for _, row in df.iterrows():
            image_id = row["image"]
            entry = f"{ZIP_PREFIX}{image_id}.jpg"
            if entry not in entries:
                skipped.append(image_id)
                continue
            label = "BCC" if row["label_bcc"] == 1 else "NO_BCC"
            split = row["split"]
            dest = args.out_dir / split / label / f"{image_id}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(entry) as src:
                img = Image.open(BytesIO(src.read())).convert("RGB")
            # BILINEAR para preservar bordes de lesiones; nearest produciría aliasing visible
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            img.save(dest, "JPEG", quality=JPEG_Q)
            written += 1
            if written % 2500 == 0:
                print(f"  {written}/{len(df)} procesadas ({time.time() - t0:.0f}s)")

    print(f"escritas: {written} · saltadas: {len(skipped)}")
    if skipped:
        print(f"SIN COINCIDENCIA EN ZIP ({len(skipped)}): {skipped[:10]}")

    resumen = {
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "imagenes": int(len(df)),
        "escritas": int(written),
        "saltadas": [str(s) for s in skipped],
        "img_size": IMG_SIZE,
        "jpeg_quality": JPEG_Q,
        "segundos": round(time.time() - t0, 2),
    }
    (args.out_dir.parent / "resize-summary.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    return 0 if not skipped else 1


def verificar(out_dir: Path, splits_dir: Path) -> int:
    """Verificación de conformidad: conteos + legalidad de archivos generados.

    Parámetros
    ----------
    out_dir : Path
        Directorio ya generado con la copia preparada.
    splits_dir : Path
        Directorio con los CSV particionados de la fase de splits.

    Retorno
    -------
    int
        0 si los conteos coinciden con los CSV y cada muestra es legible y de
        224×224; 1 si falla alguna compuerta. No escribe nada.

    Efectos colaterales
    -------------------
    Ninguno (solo lectura).

    Por qué comparar contra los CSV particionados: garantiza 1:1 zip↔GT tras el
    filtrado (la auditoría previa lo hizo; se re-chequea aquí).

    Globales
    --------
    `IMG_SIZE` (constante del módulo; tamaño esperado 224×224).
    """
    df = load_splits(splits_dir)
    problems = []
    for split in ("train", "val"):
        g = df[df["split"] == split]
        for label in ("BCC", "NO_BCC"):
            expected = int((g["label_bcc"] == (1 if label == "BCC" else 0)).sum())
            found = len(list((out_dir / split / label).glob("*.jpg")))
            estado = "OK " if found == expected else "FAIL"
            print(f"{estado} {split}/{label}: {found}/{expected}")
            if found != expected:
                problems.append(f"{split}/{label}: {found} != {expected}")
        # legibilidad: abre una imagen de cada split/clase
        for label in ("BCC", "NO_BCC"):
            sample = next((out_dir / split / label).glob("*.jpg"), None)
            try:
                img = Image.open(sample)
                if img.size != (IMG_SIZE, IMG_SIZE):
                    problems.append(f"{split}/{label}: tamaño {img.size}")
            except Exception as e:
                problems.append(f"{split}/{label}: ilegible: {e}")
    print("VERIFICACIÓN:", "OK" if not problems else "FALLO")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
