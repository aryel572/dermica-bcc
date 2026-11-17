#!/usr/bin/env python3
"""S4 paso 1: re-escalado del test 70/20/10 a 224 px (preparación para evaluación final).

Propósito
---------
Lee las imágenes crudas de ISIC 2019 (solo lectura) y el test particionado
70/20/10 de la fase de splits, y genera la copia de evaluación re-escalada a
224 px: `<out-dir>/{BCC,NO_BCC}/<image>.jpg`. Este paso es la ÚNICA vez que
se toca el test: el modelo se evalúa sobre él una sola vez. La composición de
la partición la definió la fase de splits; aquí solo se preparan imágenes 224.

El TEST oficial de ISIC 2019 nunca entra en este script: los ids del test
70/20/10 provienen del GT de TRAINING de ISIC 2019.

Entradas
--------
* `--data-dir ~DIR` (obligatorio): directorio con los archivos crudos de
  ISIC 2019 (el zip de training, que es de donde provienen los ids del test
  70/20/10). Origen EXTERNO al repo; SOLO LECTURA.
* `--split-csv ~CSV` (obligatorio): CSV del test particionado 70/20/10
  (columnas image, split, label_full, label_bcc, lesion_id; solo filas test).
* `--out-dir ~DIR_OUT` (obligatorio): destino de la copia preparada.

Salidas
-------
`<out-dir>/{BCC,NO_BCC}/<image>.jpg` (fuera del repo) y
`<out-dir>/../resize-summary.json` con el recibo de la corrida.

Uso:
    uv run python resize_224_test.py --data-dir ~DIR --split-csv ~CSV --out-dir ~DIR_OUT
    uv run python resize_224_test.py --data-dir ~DIR --split-csv ~CSV --out-dir ~DIR_OUT --verificar

Cubre el trabajo de la fase S4, paso 1 — preparación de datos para la
evaluación final (CRISP-DM).
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

# Nombres canónicos de ISIC 2019 dentro del directorio de entrada. Los ids
# del test 70/20/10 provienen del GT de TRAINING de ISIC 2019 (el test oficial
# nunca entra en este script). TRAIN_ZIP es el archivo de origen; ZIP_PREFIX
# es la carpeta de primer nivel DENTRO del zip. Los consumen load_test(),
# main() y verificar().
TRAIN_ZIP = "ISIC_2019_Training_Input.zip"
ZIP_PREFIX = "ISIC_2019_Training_Input/"
IMG_SIZE = 224  # mismo input que el entrenamiento (MobileNetV3-Small); lo consumen main() y verificar()
JPEG_Q = 85     # q85, mismo JPEG que el train/val preparado en la fase de entrenamiento; lo consume main()


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


def load_test(split_csv: Path) -> pd.DataFrame:
    """Carga el CSV del test particionado y devuelve el DataFrame.

    Parámetros
    ----------
    split_csv : Path
        Ruta al test.csv de la fase de splits (2.488 imágenes, BCC 317 según
        la partición producida).

    Retorno
    -------
    pd.DataFrame
        DataFrame con el test (columnas image, split, label_full, label_bcc,
        lesion_id); se exige que todas las filas sean del split test.

    Errores
    -------
    FileNotFoundError / pd.errors.ParserError
        Si el CSV no existe o no es parseable.
    AssertionError
        Si alguna fila no pertenece al split test (CSV equivocado o editado).

    Efectos colaterales
    -------------------
    Ninguno (solo lectura).

    Por qué solo test.csv: la composición del test la definió la fase de
    splits y es fija; leer el GT original de ISIC arriesgaría re-barajar;
    nunca se lee el dataset crudo más allá del zip fuente.
    """
    df = pd.read_csv(split_csv)
    assert set(df["split"].unique()) == {"test"}, "el CSV debe ser únicamente del split test"
    return df


def main() -> int:
    """Ejecuta el re-escalado del test y escribe test-224/ + resize-summary.

    Retorno
    -------
    int
        0 si todas las imágenes se escribieron (test completo), 1 si faltó
        alguna (señal de mismatch zip/GT).

    Errores
    -------
    zipfile.BadZipFile / AssertionError
        Si el zip no es válido o el CSV contiene filas ajenas al test.

    Efectos colaterales
    -------------------
    Escribe <out-dir>/<label>/<image>.jpg y resize-summary.json en el
    directorio padre de <out-dir>; con --verificar audita conteos y tamaño
    sin escribir.
    """
    ap = _ParserEspanol(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True, metavar="~DIR",
                    help="directorio con los archivos crudos de ISIC 2019 (zip de training; solo lectura)")
    ap.add_argument("--split-csv", type=Path, required=True, metavar="~CSV",
                    help="test.csv de la fase de splits (partición 70/20/10, solo split test)")
    ap.add_argument("--out-dir", type=Path, required=True, metavar="~DIR_OUT",
                    help="directorio donde se escribe la copia preparada (<DIR_OUT>/<CLASE>/<image>.jpg)")
    ap.add_argument("--verificar", action="store_true",
                    help="solo audita la salida ya generada (sin escribir)")
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
            # BILINEAR como train/val preparados en la fase de entrenamiento;
            # nearest produciría aliasing visible en piel
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

    Parámetros
    ----------
    out_dir : Path
        Directorio ya generado con la copia preparada del test.
    split_csv : Path
        CSV del test particionado (test.csv de la fase de splits).

    Retorno
    -------
    int
        0 si los conteos coinciden con el CSV y cada muestra es legible y de
        224×224; 1 si falla alguna compuerta. No escribe nada.

    Efectos colaterales
    -------------------
    Ninguno (solo lectura).

    Por qué conteos vs CSV: asegura 1:1 test↔zip tras el filtrado (barato).
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
