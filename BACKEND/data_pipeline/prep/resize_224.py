#!/usr/bin/env python3
"""S3 paso 1: re-escalado del train+val a 224 px (preparación para entrenamiento).

Lee DATA_CRUDA (solo lectura), los splits 70/20/10 de resultados/s2-splits/ y
genera prepared/224/{train,val}/<BCC|NO_BCC>/<image>.jpg en resultados/s3-training/.
El test de ISIC nunca se toca y TESIS no aparece (ya están excluidos de train/val).

Salida fuera del repo: resultados/s3-training/prepared/224/... (no se commitea).

Uso:
    uv run python resize_224.py            # escribe prepared/224/
    uv run python resize_224.py --verificar  # comprueba conteos y legibilidad
"""

import argparse
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image

APP_ROOT = Path(__file__).resolve().parents[4]  # .../APP_DERMICA
DATA_DIR = APP_ROOT / "DATA_CRUDA"
SPLITS_DIR = APP_ROOT / "resultados" / "s2-splits"
DEFAULT_OUT = APP_ROOT / "resultados" / "s3-training" / "prepared" / "224"
TRAIN_ZIP = "ISIC_2019_Training_Input.zip"
ZIP_PREFIX = "ISIC_2019_Training_Input/"
IMG_SIZE = 224  # input del modelo MobileNetV3-Small (checkpoint imagenet)
JPEG_Q = 85     # q85: suficiente para 224 px, ~35-50 KB por imagen


def load_splits(splits_dir: Path) -> pd.DataFrame:
    """Concatena train y val (el test de ISIC y TESIS no entran en S3).

    Params: splits_dir — directorio con train.csv/val.csv del step S2.
    Returns: DataFrame con image, split, label_full, label_bcc, lesion_id.
    """
    dfs = []
    for name in ("train", "val"):
        dfs.append(pd.read_csv(splits_dir / f"{name}.csv"))
    return pd.concat(dfs, ignore_index=True)


def main() -> int:
    """Ejecuta el re-escalado binario de train+val (usada como paso previo de S3).

    Returns: 0 si no hubo imágenes saltadas, 1 si alguna falta en el zip.
    Side effects: escribe prepared/224/{train,val}/<BCC|NO_BCC>/<image>.jpg;
    con --verificar audita conteos y tamaño sin escribir.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--verificar", action="store_true", help="solo audita la salida ya generada")
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

    Params: out_dir — dir ya generado (prepared/224); splits_dir — CSV 70/20/10.
    Returns: 0 si conteos coinciden con splits, 1 si falla alguna; no escribe nada.
    Por qué comparar contra splits: garantiza 1:1 zip↔GT tras el filtrado (S1 lo hizo; se re-chequea).
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
        # legibilidad: abre una imagen de cada split/claso
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
