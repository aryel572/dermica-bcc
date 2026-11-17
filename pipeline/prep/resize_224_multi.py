#!/usr/bin/env python3
"""Prepara la variante multi-clase (8 clases ISIC) para el entrenamiento de comparación.

Propósito
---------
Lee las imágenes crudas de ISIC 2019 (solo lectura) y los CSV particionados
70/20/10 de la fase de splits, y genera la copia de entrenamiento re-escalada
a 224 px organizada por partición y clase completa:
`<out-dir>/{train,val}/<CLASE>/<image>.jpg` con las 8 clases de ISIC 2019 que
tienen presencia en training (MEL, NV, BCC, AK, BKL, DF, VASC, SCC — sin UNK,
ausente en train).

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
`<out-dir>/{train,val}/<CLASE>/<image>.jpg` (fuera del repo).

Uso:
    uv run python resize_224_multi.py --data-dir ~DIR --splits-dir ~DIR_CSV --out-dir ~DIR_OUT

Cubre el trabajo de la fase S3, variante de entrenamiento de comparación
multi-clase (MOBILE-D / CRISP-DM).
"""

import argparse
from dermica_core.io import leer_csv_csv_splits  # import único desde dermica_core
from dermica_core.cli import ParserEspanol  # import único desde dermica_core
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import pandas as pd
from PIL import Image

# Nombres canónicos de ISIC 2019 dentro del directorio de entrada. TRAIN_ZIP
# es el archivo de origen; ZIP_PREFIX es la carpeta de primer nivel DENTRO
# del zip. Los consume main().
TRAIN_ZIP = "ISIC_2019_Training_Input.zip"
ZIP_PREFIX = "ISIC_2019_Training_Input/"
IMG_SIZE = 224   # input MobileNetV3-Small (checkpoint imagenet); lo consume main()
JPEG_Q = 85      # q85: suficiente para 224 px, ~35-50 KB por imagen; lo consume main()




def main() -> int:
    """Ejecuta el re-escalado multi-clase de train+val.

    Retorno
    -------
    int
        0 si cada imagen tuvo entry en el zip, 1 si alguna faltó (saltadas).

    Errores
    -------
    FileNotFoundError / zipfile.BadZipFile
        Si falta el zip, no es válido o faltan train.csv/val.csv.

    Efectos colaterales
    -------------------
    Escribe <out-dir>/{train,val}/<CLASE>/<image>.jpg (fuera del repo).

    Por qué marcar falla 1: un archivo faltante es señal de que el zip/versión cambió;
    no debe pasar desapercibido en la preparación del entrenamiento.
    """
    ap = ParserEspanol(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True, metavar="~DIR",
                    help="directorio con los archivos crudos de ISIC 2019 (zip de training; solo lectura)")
    ap.add_argument("--splits-dir", type=Path, required=True, metavar="~DIR_CSV",
                    help="directorio con train.csv y val.csv de la fase de splits")
    ap.add_argument("--out-dir", type=Path, required=True, metavar="~DIR_OUT",
                    help="directorio donde se escribe la copia multi-clase (<DIR_OUT>/{train,val}/<CLASE>/<image>.jpg)")
    args = ap.parse_args()

    t0 = time.time()
    dfs = []
    for name in ("train", "val"):
        dfs.append(leer_csv_csv_splits(args.splits_dir / f"{name}.csv"))
    df = pd.concat(dfs, ignore_index=True)
    print(f"imágenes: {len(df)} (train+val)")

    # set para lookup O(1) de nombres del zip; abrir el zip una sola vez es más barato que por imagen
    entries = set(zipfile.ZipFile(args.data_dir / TRAIN_ZIP).namelist())
    written, skipped = 0, []
    with zipfile.ZipFile(args.data_dir / TRAIN_ZIP) as zf:
        for _, row in df.iterrows():
            image_id = row["image"]
            entry = f"{ZIP_PREFIX}{image_id}.jpg"
            if entry not in entries:
                skipped.append(image_id)
                continue
            label = row["label_full"]
            dest = args.out_dir / row["split"] / label / f"{image_id}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(entry) as src:
                img = Image.open(BytesIO(src.read())).convert("RGB")
            # BILINEAR: suficiente para 224 px; mejor que nearest en piel con riesgo de aliasing
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            img.save(dest, "JPEG", quality=JPEG_Q)
            written += 1
            if written % 2500 == 0:
                print(f"  {written}/{len(df)} ({time.time() - t0:.0f}s)")

    print(f"escritas: {written} · saltadas: {len(skipped)} · {time.time() - t0:.0f}s")
    return 0 if not skipped else 1


if __name__ == "__main__":
    sys.exit(main())
