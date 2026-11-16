#!/usr/bin/env python3
"""S4 paso 1: re-escalado del TEST a 224 px (preparación para evaluación final).

Lee DATA_CRUDA (solo lectura), el test 70/20/10 de S2 en resultados/s2-splits/
y genera test-224/{BCC,NO_BCC}/<image>.jpg en resultados/s4-eval/. Este paso es
la ÚNICA vez que se toca el test: el modelo se evalúa sobre él una sola vez.
El split de los datos de prueba lo hizo S2; aquí solo se preparan imágenes 224.

Salida fuera del repo: resultados/s4-eval/test-224/... (no se commitea).

Uso:
    uv run python resize_224_test.py            # escribe test-224/
    uv run python resize_224_test.py --verificar  # comprueba conteos y legibilidad
"""

import argparse
import sys
import time
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import json

import pandas as pd
from PIL import Image

APP_ROOT = Path(__file__).resolve().parents[4]  # .../APP_DERMICA
DATA_DIR = APP_ROOT / "DATA_CRUDA"
SPLIT_CSV = APP_ROOT / "resultados" / "s2-splits" / "test.csv"
DEFAULT_OUT = APP_ROOT / "resultados" / "s4-eval" / "test-224"
# los ids del split 70/20/10 son del GT de TRAINING de ISIC 2019; TEST-oficial nunca entra
TRAIN_ZIP = "ISIC_2019_Training_Input.zip"
ZIP_PREFIX = "ISIC_2019_Training_Input/"
IMG_SIZE = 224  # mismo input que el entrenamiento (MobileNetV3-Small)
JPEG_Q = 85     # q85, mismo JPEG que el train/val preparado en S3


def load_test(split_csv: Path) -> pd.DataFrame:
    """Carga test.csv con image/split/label_* y devuelve el DataFrame.

    Parámetros: split_csv — ruta a resultados/s2-splits/test.csv.
    Retorna: DataFrame con el test (2.488 imágenes, BCC 317 según S2).
    Por qué sólo test.csv: la composición del test la definió S2 y es fija;
    leer el GT original de ISIC arriesgaría re-barajar; no tocamos DATA_CRUDA.
    """
    df = pd.read_csv(split_csv)
    assert set(df["split"].unique()) == {"test"}, "el CSV debe ser únicamente del split test"
    return df


def main() -> int:
    """Ejecuta el re-escalado del test y escribe test-224/ + resize-summary.

    Retorna: 0 si todas las imágenes se escribieron (test completo), 1 si faltó
    alguna (señal de mismatch zip/GT). Efectos colaterales: escribe
    resultados/s4-eval/test-224/<label>/<image>.jpg y resize-summary.json.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--split-csv", type=Path, default=SPLIT_CSV)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--verificar", action="store_true", help="solo audita la salida ya generada")
    args = ap.parse_args()

    if args.verificar:
        return verificar(args.out_dir, args.split_csv)

    t0 = time.time()
    df = load_test(args.split_csv)
    print(f"TEST: {len(df)} imágenes (BCC {int(df['label_bcc'].sum())})")

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
            dest = args.out_dir / label / f"{image_id}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(entry) as src:
                img = Image.open(BytesIO(src.read())).convert("RGB")
            # BILINEAR como train/val preparados en S3
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            img.save(dest, "JPEG", quality=JPEG_Q)
            written += 1
            if written % 500 == 0:
                print(f"  {written}/{len(df)} ({time.time() - t0:.0f}s)")

    print(f"escritas: {written} · saltadas: {len(skipped)} · {time.time() - t0:.0f}s")
    resumen = {
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "imagenes": int(len(df)),
        "escritas": int(written),
        "saltadas": [str(s) for s in skipped],
        "img_size": IMG_SIZE,
        "jpeg_quality": JPEG_Q,
        "segundos": round(time.time() - t0, 2),
    }
    (args.out_dir.parent).mkdir(parents=True, exist_ok=True)
    (args.out_dir.parent / "resize-summary.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    return 0 if not skipped else 1


def verificar(out_dir: Path, split_csv: Path) -> int:
    """Verificación de conformidad: conteos + legibilidad de la salida.

    Parámetros: out_dir — dir ya generado; split_csv — test.csv del split.
    Retorna: 0 si los conteos coinciden con splits, 1 si no. No escribe nada.
    Por qué conteos vs split: asegura 1:1 test↔zip tras el filtrado (barato).
    """
    df = load_test(split_csv)
    problems = []
    for label in ("BCC", "NO_BCC"):
        expected = int((df["label_bcc"] == (1 if label == "BCC" else 0)).sum())
        found = len(list((out_dir / label).glob("*.jpg"))) if (out_dir / label).exists() else 0
        estado = "OK " if found == expected else "FAIL"
        print(f"{estado} {label}: {found}/{expected}")
        if found != expected:
            problems.append(f"{label}: {found} != {expected}")
    for label in ("BCC", "NO_BCC"):
        sample_dir = out_dir / label
        sample = next(sample_dir.glob("*.jpg"), None)
        if sample is None:
            problems.append(f"{label}: sin archivos para probar legibilidad")
            continue
        try:
            img = Image.open(sample)
            if img.size != (IMG_SIZE, IMG_SIZE):
                problems.append(f"{label}: tamaño {img.size}")
        except Exception as e:
            problems.append(f"{label}: ilegible: {e}")
    print("VERIFICACIÓN:", "OK" if not problems else "FALLO")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
