#!/usr/bin/env python3
"""Prepara variante multi-clase (8 clases ISIC) para el entrenamiento de comparación.

Copia train+val (70/20/10 splits) re-escalados a 224 px con carpetas por clase
(MEL, NV, BCC, AK, BKL, DF, VASC, SCC — sin UNK, ausente en train). TESIS y
TEST ISIC fuera. Salida fuera del repo: resultados/s3-training/prepared-multi/.

Uso: uv run python resize_224_multi.py   (ejecutar desde este directorio)
"""

import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image

APP_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = APP_ROOT / "DATA_CRUDA"
SPLITS_DIR = APP_ROOT / "resultados" / "s2-splits"
OUT_DIR = APP_ROOT / "resultados" / "s3-training" / "prepared-multi" / "224"
TRAIN_ZIP = "ISIC_2019_Training_Input.zip"
ZIP_PREFIX = "ISIC_2019_Training_Input/"
IMG_SIZE = 224
JPEG_Q = 85


def main() -> int:
    t0 = time.time()
    dfs = []
    for name in ("train", "val"):
        dfs.append(pd.read_csv(SPLITS_DIR / f"{name}.csv"))
    df = pd.concat(dfs, ignore_index=True)
    print(f"imágenes: {len(df)} (train+val)")

    entries = set(zipfile.ZipFile(DATA_DIR / TRAIN_ZIP).namelist())
    written, skipped = 0, []
    with zipfile.ZipFile(DATA_DIR / TRAIN_ZIP) as zf:
        for _, row in df.iterrows():
            image_id = row["image"]
            entry = f"{ZIP_PREFIX}{image_id}.jpg"
            if entry not in entries:
                skipped.append(image_id)
                continue
            label = row["label_full"]
            dest = OUT_DIR / row["split"] / label / f"{image_id}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(entry) as src:
                img = Image.open(BytesIO(src.read())).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            img.save(dest, "JPEG", quality=JPEG_Q)
            written += 1
            if written % 2500 == 0:
                print(f"  {written}/{len(df)} ({time.time() - t0:.0f}s)")

    print(f"escritas: {written} · saltadas: {len(skipped)} · {time.time() - t0:.0f}s")
    return 0 if not skipped else 1


if __name__ == "__main__":
    sys.exit(main())
