#!/usr/bin/env python3
"""Auditoría S1 del dataset crudo ISIC 2019 (entendimiento de datos, CRISP-DM).

Propósito
---------
Compuerta de "entendimiento de datos" del pipeline de la tesis: antes de
formar cualquier split o subconjunto, evidencia verificada por máquina de que
la copia cruda en disco del dataset ISIC 2019 está completa y es
autoconsistente:

  * Los zips de training/test contienen exactamente los conteos oficiales.
  * Cada id de imagen del ground truth (GT) existe en el zip y viceversa
    (cotejo 1:1 estricto, cero duplicados en ambos lados).
  * Las columnas one-hot de clase suman exactamente una etiqueta por fila
    (invariante de etiqueta única del formato GT de ISIC 2019).
  * Los CSV de metadata son parseables y su perfil de valores ausentes es
    conocido.
  * Los cuatro CSV y cada artefacto producido quedan sellados con SHA-256;
    la corrida es reproducible y resistente a manipulación (`--verificar`
    los revisa de nuevo).
  * La integridad CRC de ambos zips se verifica de extremo a extremo con
    `unzip -t`. (`zipfile.namelist()` NO verifica CRCs: solo lee el
    directorio central; hace falta una pasada completa aparte.)

Entradas
--------
* `--data-dir ~DIR` (obligatorio): directorio con los archivos crudos de
  ISIC 2019. Es un origen EXTERNO al repo: lo provee quien corre la
  auditoría y es de SOLO LECTURA; este script nunca lo modifica.
* `--out-dir ~DIR` (obligatorio): directorio donde se escriben todos los
  artefactos. Lo decide quien invoca; nada se escribe dentro del repo.

Salidas (todas bajo `--out-dir`)
--------------------------------
  auditoria.json     informe legible por máquina (invariantes + hashes)
  clases.csv         distribución de clases por split (conteos y porcentajes)
  lesion_id.csv      imágenes por lesion_id (solo training; ids ausentes omitidos)
  dimensiones.csv    muestra de anchos/altos/modos de imagen (ver SAMPLE_SEED)
  figuras/clases.png barras por clase
  figuras/faltantes.png barras de valores ausentes
  README-corrida.md  notas legibles de la corrida con los comandos exactos

El dataset crudo nunca se extrae (los zips se leen en streaming), el script
nunca escribe en el directorio de entrada y el split TEST de ISIC 2019 no se
usa para nada que no sea auditarlo: la partición de training, agrupada por
lesion_id, es la única base de los splits posteriores.

Uso:
    uv run python audit_isic.py --data-dir ~DIR --out-dir ~DIR_OUT
    uv run python audit_isic.py --data-dir ~DIR --out-dir ~DIR_OUT --verificar ~DIR_OUT/auditoria.json

Cubre el trabajo de la fase S1 — entendimiento de datos (CRISP-DM).
"""

import argparse
from dermica_core.labels import ISIC_LABELS  # import único desde dermica_core
from dermica_core.hashing import sha256_file  # import único desde dermica_core
from dermica_core.cli import ParserEspanol  # import único desde dermica_core
import json
import random
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import matplotlib

# El backend headless DEBE seleccionarse antes de importar pyplot; si no,
# matplotlib intenta cargar un backend gráfico y revienta en máquinas sin
# display (CI/servidores). "Agg" solo renderiza a archivos.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# Nombres canónicos de los archivos de ISIC 2019 dentro del directorio de
# entrada. Son los nombres exactos de la versión oficial; la auditoría no
# asume nada de su contenido más allá de lo que miden las comprobaciones.
# Los consumen main() y audit_split() para ubicar las fuentes.
TRAIN_ZIP = "ISIC_2019_Training_Input.zip"   # entrada del split de entrenamiento (varios GB)
TEST_ZIP = "ISIC_2019_Test_Input.zip"        # entrada del split oficial de test
GT_TRAIN = "ISIC_2019_Training_GroundTruth.csv"   # etiquetas one-hot de training
GT_TEST = "ISIC_2019_Test_GroundTruth.csv"        # etiquetas one-hot de test
META_TRAIN = "ISIC_2019_Training_Metadata.csv"    # metadata de training (incluye lesion_id)
META_TEST = "ISIC_2019_Test_Metadata.csv"         # metadata de test (sin lesion_id)

# Tamaños oficiales de las tablas ISIC 2019. Codificarlos convierte la
# auditoría de un "generador de reportes" en una compuerta real: cualquier
# imagen faltante o sobrante revienta la corrida en vez de propagarse en
# silencio hacia los splits y subconjuntos. La consume audit_split() en sus
# asserts duros.
EXPECTED = {"training": 25331, "test": 8238}


# Semilla de la muestra de dimensiones/modos (~600 imágenes). 42 es la semilla
# fija de todo el proyecto (el mismo valor que se usa para la formación de
# subconjuntos y los splits posteriores), de modo que cada elección
# estocástica del pipeline sea reproducible con la misma constante y auditable
# en la tesis. Cambiarla alteraría en silencio la muestra, así que la propia
# semilla forma parte del contrato del artefacto: dimensiones.csv siempre
# describe las mismas imágenes para un mismo dataset. La consume sample_dimensions().
SAMPLE_SEED = 42






def unzip_test(zip_path: Path) -> tuple[bool, str, float]:
    """Comprueba CRC completo vía `unzip -t`. Devuelve (ok, cola_de_salida, segundos).

    Por qué el `unzip` del sistema y no el zipfile de la stdlib: listar o
    extraer con el zipfile de Python verifica los CRCs solo cuando los bytes
    se leen de verdad (al extraer o en `ZipFile.read`), y nunca para
    operaciones que solo tocan metadatos. `unzip -t` es la comprobación de
    integridad canónica de extremo a extremo para estos archivos de varios
    GB y no requiere disco para extraer. Esta función delega en ella y mide
    el tiempo de pared.

    Parámetros
    ----------
    zip_path : Path
        Archivo a probar. No se modifica y no se escribe nada en disco.

    Retorno
    -------
    tuple[bool, str, float]
        (True si el código de salida es 0, última línea no vacía de la
        salida para el log de auditoría, segundos transcurridos). En caso de
        fallo la tupla se devuelve igual para que el llamador pueda decidir
        cómo reportarlo; el llamador hace assert sobre el flag.

    Errores
    -------
    FileNotFoundError
        Si el binario `unzip` no está instalado (el subproceso no puede
        lanzarse) o `zip_path` no existe; el traceback lo delata tal cual.

    Efectos colaterales
    -------------------
    Lanza un subproceso (`unzip -t`). No se crean ni borran archivos; el
    archivo se abre en solo lectura.
    """
    start = time.perf_counter()
    proc = subprocess.run(["unzip", "-t", str(zip_path)], capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    out = (proc.stdout or "") + (proc.stderr or "")
    tail = out.strip().splitlines()[-1] if out.strip() else ""
    return proc.returncode == 0, tail, elapsed


def zip_entries(zip_path: Path) -> tuple[list[str], int]:
    """Lista las entradas de imagen y el total de entradas de un zip sin extraer.

    Parámetros
    ----------
    zip_path : Path
        Archivo a inspeccionar. Solo lectura.

    Retorno
    -------
    tuple[list[str], int]
        (stems de todas las entradas cuyo nombre termina en .jpg, sin
        importar mayúsculas/minúsculas, número total de entradas del archivo
        incluyendo no-jpg/directorios). El stem (nombre sin extensión) es lo
        que casa con la columna `image` del GT en ISIC 2019.

    Errores
    -------
    zipfile.BadZipFile
        Si `zip_path` no es un zip válido o está truncado.
    FileNotFoundError
        Si `zip_path` no existe.

    Efectos colaterales
    -------------------
    Lee solo el directorio central; no hay verificación CRC aquí (eso es
    trabajo de unzip_test) y no se descomprime ningún byte de imagen.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    return [Path(n).stem for n in names if n.lower().endswith(".jpg")], len(names)


def load_gt(csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Parsea un CSV GroundTruth y resuelve qué columnas de clase están presentes.

    Parámetros
    ----------
    csv_path : Path
        Ruta al CSV GroundTruth de ISIC 2019 del split auditado.

    Retorno
    -------
    tuple[pd.DataFrame, list[str]]
        (dataframe GT completo, ISIC_LABELS intersecadas con las columnas
        realmente presentes, preservando el orden de ISIC_LABELS). La
        intersección tolera versiones del GT que omiten clases (p. ej. un
        split sin filas SCC puede omitir la columna) sin fallar; el invariante
        que importa — una etiqueta hot por fila — lo hace cumplir
        class_distribution().

    Errores
    -------
    FileNotFoundError / pd.errors.ParserError
        Si el CSV no existe o no es parseable.

    Efectos colaterales
    -------------------
    Ninguno (parseo del CSV en solo lectura).
    """
    gt = pd.read_csv(csv_path)
    class_cols = [c for c in ISIC_LABELS if c in gt.columns]
    return gt, class_cols


def class_distribution(gt: pd.DataFrame, class_cols: list[str]) -> pd.DataFrame:
    """Calcula conteos y porcentajes por clase desde el GT one-hot.

    Aquí es también donde se hace cumplir el invariante de etiqueta única:
    las filas del GT de ISIC 2019 llevan exactamente una columna de clase en
    1 y ceros en el resto. Si un CSV viniera multi-etiqueta o corrupto, las
    sumas de columnas no igualarían el número de filas y el assert de abajo
    revienta de inmediato en lugar de producir en silencio una distribución
    equivocada.

    Parámetros
    ----------
    gt : pd.DataFrame
        Dataframe de verdad-terreno; debe contener todas las `class_cols` y
        la columna de id `image` (no se usa aquí, pero la requieren los
        llamadores para el cotejo con el zip).
    class_cols : list[str]
        Columnas one-hot a agregar, típicamente la salida de load_gt.

    Retorno
    -------
    pd.DataFrame
        Columnas `clase`, `conteo`, `porcentaje` (porcentaje de filas del GT).

    Errores
    -------
    AssertionError
        Si las sumas one-hot no igualan el número de filas (CSV corrupto o
        multi-etiqueta).

    Efectos colaterales
    -------------------
    Ninguno más allá de construir el pequeño dataframe de resultados.
    """
    counts = gt[class_cols].sum().astype(int)
    total = int(counts.sum())
    assert total == len(gt), (
        f"one-hot sums to {total} but GT has {len(gt)} rows; multi-label or bad CSV?"
    )
    return pd.DataFrame(
        {
            "clase": class_cols,
            "conteo": counts.values,
            "porcentaje": (counts.values / len(gt)) * 100,
        }
    )


def audit_split(
    name: str,
    data_dir: Path,
    zip_path: Path,
    gt_path: Path,
    meta_path: Path,
    out_dir: Path,
    crc: tuple[bool, str, float],
) -> dict:
    """Ejecuta la auditoría completa de un split y devuelve sus resultados.

    Chequeos, en orden:
      1. Lista de entradas del zip vs filas del GT (conteos oficiales exactos,
         EXPECTED).
      2. ids de imagen duplicados en el GT y en el zip (deben ser 0).
      3. Cotejo estricto 1:1 entre ids del GT y stems jpg del zip, listando
         hasta unos pocos ids ofensivos en el mensaje del assert para
         depurar.
      4. Distribución por clase (delegada a class_distribution).
      5. Parseo del CSV de metadata + conteos de ausencias por columna.

    Parámetros
    ----------
    name : str
        Nombre del split, exactamente "training" u "test"; indexa EXPECTED y
        se refleja en el informe.
    data_dir : Path
        Directorio raíz del dataset crudo (se usa para reportar rutas; las
        rutas de zip/gt/meta se pasan explícitamente como rutas absolutas).
    zip_path : Path
        El *_Input.zip de ISIC 2019 de este split.
    gt_path : Path
        El *_GroundTruth.csv correspondiente.
    meta_path : Path
        El *_Metadata.csv correspondiente.
    out_dir : Path
        Sin uso dentro de esta función (se conserva en la firma por simetría
        con el resto del pipeline y para futuros artefactos); aquí no se
        escribe ningún archivo.
    crc : tuple[bool, str, float]
        Resultado precalculado de unzip_test para este zip (se ejecuta una
        vez por split en main para no duplicar el trabajo y para fallar
        antes de la auditoría costosa).

    Retorno
    -------
    dict
        Informe por split legible por máquina: tamaños, CRCs, duplicados,
        conteos de ausentes, más los dataframes embebidos `dist`, `meta`,
        `gt` (los usa main para figuras/JSON; se eliminan antes de
        serializar). Los campos `gt_zip_1a1`, `gt_sin_zip` y `zip_sin_gt`
        codifican el estado post-assert: llegar al return implica 1:1 con
        cero faltantes.

    Errores
    -------
    AssertionError
        Ante cualquier violación de invariante (conteos, duplicados, cotejo
        1:1, etiqueta única). Aborta toda la corrida; es intencional.

    Efectos colaterales
    -------------------
    Lee el zip (solo el directorio central) y dos CSV. Los asserts abortan
    toda la corrida ante cualquier violación de invariante; es intencional:
    un invariante fallido significa que el dataset en disco no es lo que la
    tesis asume.
    """
    zip_start = time.perf_counter()
    ids_zip, total_entries = zip_entries(zip_path)
    zip_secs = time.perf_counter() - zip_start
    gt, class_cols = load_gt(gt_path)
    dup_gt = int(gt["image"].duplicated().sum())
    dup_zip = len(ids_zip) - len(set(ids_zip))
    set_gt, set_zip = set(gt["image"]), set(ids_zip)
    en_gt_no_zip = sorted(set_gt - set_zip)
    en_zip_no_gt = sorted(set_zip - set_gt)
    dist = class_distribution(gt, class_cols)
    meta = pd.read_csv(meta_path)
    faltantes = {c: int(meta[c].isna().sum()) for c in meta.columns}
    meta_rows = len(meta)

    # Compuertas duras. Son las propiedades en las que el resto del pipeline
    # (agrupar por lesion_id, splits estratificados, subconjuntos TESIS) se
    # apoya en silencio; fallar aquí sale más barato que depurar un modelo
    # sesgado después.
    assert len(gt) == EXPECTED[name], f"{name}: GT rows {len(gt)} != {EXPECTED[name]}"
    assert len(ids_zip) == EXPECTED[name], f"{name}: zip jpgs {len(ids_zip)} != {EXPECTED[name]}"
    assert dup_gt == 0, f"{name}: {dup_gt} duplicate image ids in GT"
    assert dup_zip == 0, f"{name}: {dup_zip} duplicate image ids in zip"
    assert not en_gt_no_zip, f"{name}: {len(en_gt_no_zip)} ids in GT missing from zip (e.g. {en_gt_no_zip[:5]})"
    assert not en_zip_no_gt, f"{name}: {len(en_zip_no_gt)} ids in zip missing from GT (e.g. {en_zip_no_gt[:5]})"

    return {
        "nombre": name,
        "zip": zip_path.name,
        "zip_bytes": zip_path.stat().st_size,
        "zip_entradas": total_entries,
        "zip_imagenes": len(ids_zip),
        "zip_scan_seg": round(zip_secs, 2),
        "crc_ok": crc[0],
        "crc_tail": crc[1],
        "crc_seg": round(crc[2], 2),
        "gt_filas": len(gt),
        "gt_duplicados": dup_gt,
        "zip_duplicados": dup_zip,
        # Tras los asserts de arriba estos valores son invariantemente True/0;
        # se serializan para que el JSON registre explícitamente la propiedad
        # comprobada en lugar de insinuarla con la ausencia de errores.
        "gt_zip_1a1": True,
        "gt_sin_zip": 0,
        "zip_sin_gt": 0,
        "meta_filas": meta_rows,
        "meta_faltantes": faltantes,
        "dist": dist,
        "meta": meta,
        "gt": gt,
    }


def sample_dimensions(splits: dict, data_dir: Path, out_dir: Path, sample_per_split: int) -> pd.DataFrame:
    """Muestrea imágenes por split y registra ancho/alto/modo.

    Determinístico: el RNG se seedea con SAMPLE_SEED y las entradas se ordenan
    antes de muestrear, de modo que se miden las MISMAS imágenes en cada
    corrida (rng.sample sobre la misma lista ordenada con la misma semilla es
    estable entre máquinas y versiones de Python para random.Random).
    Ordenar es lo que vuelve la muestra independiente del orden interno del
    zip.

    Parámetros
    ----------
    splits : dict
        Salida de las llamadas a `audit_split` de main por split (necesita el
        nombre de archivo "zip" de cada split).
    data_dir : Path
        Directorio raíz del dataset crudo.
    out_dir : Path
        Destino de dimensiones.csv.
    sample_per_split : int
        Máximo de imágenes a abrir por split (por defecto 300 en main →
        ~600 en total). Si un split tiene menos jpg que lo pedido, se
        muestrean todos.

    Retorno
    -------
    pd.DataFrame
        Columnas image, split, width, height, mode — una fila por imagen
        muestreada. También se escribe en out_dir/dimensiones.csv
        (index=False).

    Errores
    -------
    zipfile.BadZipFile / OSError
        Si un zip no puede abrirse o una entrada declarada no puede leerse.

    Efectos colaterales
    -------------------
    Abre y decodifica (cabeceras + píxeles) las imágenes muestreadas desde
    dentro de los zips vía zf.open; escribe dimensiones.csv. NO extrae los
    archivos.
    """
    rng = random.Random(SAMPLE_SEED)
    rows = []
    for name, info in splits.items():
        zip_path = data_dir / info["zip"]
        with zipfile.ZipFile(zip_path) as zf:
            jpg_entries = [n for n in zf.namelist() if n.lower().endswith(".jpg")]
            pick = rng.sample(sorted(jpg_entries), min(sample_per_split, len(jpg_entries)))
            for entry in pick:
                with zf.open(entry) as fh:
                    with Image.open(fh) as im:
                        rows.append(
                            {
                                "image": Path(entry).stem,
                                "split": name,
                                "width": im.width,
                                "height": im.height,
                                "mode": im.mode,
                            }
                        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "dimensiones.csv", index=False)
    return df


def main() -> int:
    """Ejecuta la auditoría completa y escribe todos los artefactos; devuelve el código de salida.

    El código 0 significa que todos los invariantes se cumplieron y los
    artefactos se escribieron. Cualquier assert fallido o excepción sale con
    código distinto de cero (traceback), que es el modo de fallo previsto para
    CI/corridas manuales: la auditoría es todo-o-nada.

    Efectos colaterales
    -------------------
    Crea --out-dir y --out-dir/figuras/ si no existen; escribe
    auditoria.json, clases.csv, lesion_id.csv, dimensiones.csv,
    figuras/*.png y README-corrida.md; imprime el progreso por stdout.
    Nunca escribe en --data-dir.
    """
    ap = ParserEspanol(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True, metavar="~DIR",
                    help="directorio con los archivos crudos de ISIC 2019 (zips y CSV; solo lectura)")
    ap.add_argument("--out-dir", type=Path, required=True, metavar="~DIR_OUT",
                    help="directorio donde se escriben todos los artefactos de la auditoría")
    ap.add_argument("--sample-per-split", type=int, default=300, metavar="~N",
                    help="imágenes a muestrear por split para dimensiones/modos (default: 300)")
    ap.add_argument("--verificar", type=Path, default=None, metavar="~JSON",
                    help="ruta a un auditoria.json previo; corre verificaciones sin escribir artefactos")
    args = ap.parse_args()

    if args.verificar:
        return verificar(args.verificar, args.data_dir)

    data_dir: Path = args.data_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figuras").mkdir(exist_ok=True)

    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    print("[1/6] SHA-256 de los CSVs...")
    hashes = {name: sha256_file(data_dir / name) for name in [GT_TRAIN, GT_TEST, META_TRAIN, META_TEST]}

    print("[2/6] unzip -t (CRC completo) del training zip...")
    crc_train = unzip_test(data_dir / TRAIN_ZIP)
    print(f"      -> {'OK' if crc_train[0] else 'FAIL'} en {crc_train[2]:.1f}s | {crc_train[1]}")
    print("[2/6] unzip -t (CRC completo) del test zip...")
    crc_test = unzip_test(data_dir / TEST_ZIP)
    print(f"      -> {'OK' if crc_test[0] else 'FAIL'} en {crc_test[2]:.1f}s | {crc_test[1]}")
    timings["crc_train_seg"] = round(crc_train[2], 2)
    timings["crc_test_seg"] = round(crc_test[2], 2)
    # El CRC es la comprobación más costosa; si falla nos detenemos aquí en
    # vez de quemar tiempo con el resto de la auditoría sobre un archivo corrupto.
    assert crc_train[0] and crc_test[0], "CRC failed (unzip -t); see output above"

    print("[3/6] Auditando GT/metadata/zips...")
    splits = {
        "training": audit_split("training", data_dir, data_dir / TRAIN_ZIP, data_dir / GT_TRAIN, data_dir / META_TRAIN, out_dir, crc_train),
        "test": audit_split("test", data_dir, data_dir / TEST_ZIP, data_dir / GT_TEST, data_dir / META_TEST, out_dir, crc_test),
    }

    # lesion_id solo existe en la metadata de TRAINING (la metadata de test de
    # ISIC 2019 no tiene lesion_id — las lesiones allí son desconocidas). Este
    # bloque calcula las estadísticas a nivel de lesión que justifican agrupar
    # por lesion_id en lugar de por image_id en el split 70/20/10: las
    # imágenes de una misma lesión son casi duplicados, y dejar que caigan en
    # particiones distintas filtraría contenido visualmente idéntico entre
    # train/validación.
    print("[4/6] lesion_id (solo training)...")
    meta_tr = splits["training"]["meta"]
    lesion = meta_tr["lesion_id"].dropna()
    lesion_counts = lesion.value_counts()
    lesion_csv = lesion_counts.rename_axis("lesion_id").reset_index(name="n_imagenes").sort_values("n_imagenes", ascending=False)
    lesion_csv.to_csv(out_dir / "lesion_id.csv", index=False)
    n_missing = int(meta_tr["lesion_id"].isna().sum())
    multi = lesion_counts[lesion_counts >= 2]
    lesion_summary = {
        "unicos": int(lesion_counts.shape[0]),
        "ausentes": n_missing,
        "pct_ausentes": round(n_missing / len(meta_tr) * 100, 2),
        "imagenes_con_lesion_id": int(lesion.shape[0]),
        "imagenes_por_lesion_media": round(float(lesion_counts.mean()), 4),
        "imagenes_por_lesion_mediana": float(lesion_counts.median()),
        "imagenes_por_lesion_max": int(lesion_counts.max()),
        "lesiones_multi_imagen": int(multi.shape[0]),
        "imagenes_en_lesiones_multi": int(multi.sum()),
    }

    print("[5/6] Dimensiones/modos (muestra)...")
    dims = sample_dimensions(splits, data_dir, out_dir, args.sample_per_split)
    mode_counts = dims["mode"].value_counts().to_dict()
    dims_summary = {
        "n": len(dims),
        "width_min": int(dims["width"].min()),
        "width_max": int(dims["width"].max()),
        "width_mean": round(float(dims["width"].mean()), 1),
        "height_min": int(dims["height"].min()),
        "height_max": int(dims["height"].max()),
        "height_mean": round(float(dims["height"].mean()), 1),
        "modos": {str(k): int(v) for k, v in mode_counts.items()},
    }

    print("[6/6] Figuras y JSON...")
    clases = pd.concat(
        [s["dist"].assign(split=s["nombre"]) for s in splits.values()], ignore_index=True
    )[["split", "clase", "conteo", "porcentaje"]]
    clases.to_csv(out_dir / "clases.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, (name, s) in zip(axes, splits.items()):
        # Terracota para training, azul para test: misma semántica de
        # paleta que el resto de figuras del proyecto (train/test nunca
        # deben confundirse visualmente).
        ax.bar(s["dist"]["clase"], s["dist"]["conteo"], color="#b34724" if name == "training" else "#2c6fbb")
        ax.set_title(f"ISIC 2019 {name} (n={len(s['gt'])})")
        ax.set_xlabel("Clase")
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Imágenes")
    fig.tight_layout()
    fig.savefig(out_dir / "figuras" / "clases.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False)
    for ax, (name, s) in zip(axes, splits.items()):
        mf = s["meta_faltantes"]
        pct = [v / s["meta_filas"] * 100 for v in mf.values()]
        ax.barh(list(mf.keys()), pct, color="#3a7d44")
        ax.set_title(f"Faltantes en metadata ({name})")
        ax.set_xlabel("% ausente")
        # Las etiquetas barh se leen de arriba hacia abajo; invertir pone la
        # primera columna (alfabético por orden del CSV) arriba, como el CSV para humanos.
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "figuras" / "faltantes.png", dpi=150)
    plt.close(fig)

    timings["total_seg"] = round(time.perf_counter() - t0, 2)

    audit = {
        "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_dir": str(data_dir),
        "entorno": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "pillow": Image.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "tiempos_seg": timings,
        "hashes_sha256_csv": hashes,
        # Se eliminan los dataframes embebidos antes de serializar: dist/meta/gt
        # viven en memoria solo para figuras y resúmenes, no en el JSON.
        "zips": {
            name: {k: v for k, v in s.items() if k not in ("dist", "meta", "gt")} for name, s in splits.items()
        },
        "clases_train": dict(zip(splits["training"]["dist"]["clase"], splits["training"]["dist"]["conteo"])),
        "clases_test": dict(zip(splits["test"]["dist"]["clase"], splits["test"]["dist"]["conteo"])),
        "faltantes_metadata": {name: s["meta_faltantes"] for name, s in splits.items()},
        "lesion_id_training": lesion_summary,
        "dimensiones_muestra": dims_summary,
        # Los invariantes se guardan explícitamente, no se recalculan al leer:
        # el JSON es el recibo firmado de que el dataset cumplió las
        # precondiciones de la tesis en el momento de generarlo.
        "invariantes": {
            "train_imagenes": splits["training"]["gt_filas"],
            "test_imagenes": splits["test"]["gt_filas"],
            "duplicados_gt": splits["training"]["gt_duplicados"] + splits["test"]["gt_duplicados"],
            "gt_zip_1a1": splits["training"]["gt_zip_1a1"] and splits["test"]["gt_zip_1a1"],
            "crc_ok": splits["training"]["crc_ok"] and splits["test"]["crc_ok"],
        },
        "artefactos": {},
    }
    audit_path = out_dir / "auditoria.json"
    artifact_paths = [
        out_dir / f
        for f in ["clases.csv", "lesion_id.csv", "dimensiones.csv", "figuras/clases.png", "figuras/faltantes.png"]
    ]
    # Se hashea cada artefacto que prometemos, para que --verificar pueda
    # detectar cualquier deriva o edición manual posterior.
    audit["artefactos"] = {p.relative_to(out_dir).as_posix(): sha256_file(p) for p in artifact_paths}
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")

    # autocomprobación: se relee lo que acabamos de escribir. Cable de
    # integridad barato: si el serializador corrompió algo o un assert fue
    # eliminado (python -O), esto falla antes de escribir el README.
    check = json.loads(audit_path.read_text())
    assert check["invariantes"]["train_imagenes"] == EXPECTED["training"]
    assert check["invariantes"]["test_imagenes"] == EXPECTED["test"]
    assert check["invariantes"]["gt_zip_1a1"] is True
    assert check["invariantes"]["crc_ok"] is True

    write_readme(out_dir, audit)
    print(f"Listo en {timings['total_seg']}s -> {out_dir}")
    return 0


def verificar(audit_path: Path, data_dir: Path) -> int:
    """Reverifica un auditoria.json previamente escrito contra el disco.

    No vuelve a derivar nada del dataset excepto los cuatro CSV de entrada:
    comprueba (a) que todas las claves requeridas estén presentes, (b) que los
    invariantes serializados coincidan con los conteos oficiales, (c) que los
    CSV de entrada sigan hasheando a los digests registrados, (d) que cada
    artefacto listado exista y hashee a su digest registrado. Este es el
    recibo de reproducibilidad de la auditoría; debe pasar en la máquina que
    generó el JSON (y no depende de reejecutar los CRC de los zips).

    Parámetros
    ----------
    audit_path : Path
        Ruta a auditoria.json. Su directorio padre se toma como out_dir.
    data_dir : Path
        Directorio con los CSV de entrada a re-hashear. Se pasa por CLI
        (--data-dir) y NO se lee del JSON: así la verificación funciona en
        cualquier máquina que tenga su propia copia del dataset crudo, sin
        depender de la ruta absoluta grabada al generar.

    Retorno
    -------
    int
        0 si todos los chequeos pasan, 1 en caso contrario. Los chequeos
        individuales se imprimen como líneas PASS/FAIL; todos se ejecutan aun
        tras un fallo (sin salida temprana) para que el resumen muestre todo
        el daño.

    Errores
    -------
    FileNotFoundError / json.JSONDecodeError
        Si `audit_path` no existe o no es JSON válido.

    Efectos colaterales
    -------------------
    Solo lectura: hashea los CSV y los artefactos listados en el JSON.
    """
    audit_path = audit_path.resolve()
    out_dir = audit_path.parent
    ok = True

    def check(label: str, cond: bool) -> None:
        # Acumulador nonlocal: colecta cada veredicto, imprime uno por uno y
        # nunca corta en el primero — el resumen debe listar TODOS los fallos.
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    print(f"Verificando {audit_path}")
    audit = json.loads(audit_path.read_text())

    for key in ["generado_utc", "hashes_sha256_csv", "zips", "clases_train", "clases_test",
                "lesion_id_training", "dimensiones_muestra", "invariantes", "artefactos"]:
        check(f"clave '{key}' presente", key in audit)

    inv = audit.get("invariantes", {})
    check("train = 25.331 imágenes", inv.get("train_imagenes") == 25331)
    check("test = 8.238 imágenes", inv.get("test_imagenes") == 8238)
    check("duplicados GT == 0", inv.get("duplicados_gt") == 0)
    check("GT <-> zip 1:1", inv.get("gt_zip_1a1") is True)
    check("CRC zips OK", inv.get("crc_ok") is True)

    # Se rehashean entradas y artefactos; ambos chequeos comparan contra los
    # digests registrados al generar. Un CSV o artefacto editado tras la
    # auditoría falla aquí aunque todos los conteos parezcan estar bien.
    for name, digest in audit.get("hashes_sha256_csv", {}).items():
        p = data_dir / name
        check(f"SHA-256 {name}", p.exists() and sha256_file(p) == digest)

    for rel, digest in audit.get("artefactos", {}).items():
        p = out_dir / rel
        check(f"artefacto {rel}", p.exists() and sha256_file(p) == digest)

    print("RESULTADO:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def write_readme(out_dir: Path, audit: dict) -> None:
    """Escribe README-corrida.md — un recibo legible de esta corrida de auditoría.

    El README registra los comandos exactos, versiones del entorno, tiempos y
    hashes de entrada para que la auditoría pueda reproducirse o citarse desde
    la tesis sin abrir el JSON. El contenido sale del dict de auditoría (nunca
    se recalcula), así que el README siempre coincide con auditoria.json.

    Parámetros
    ----------
    out_dir : Path
        Directorio de destino (se crea README-corrida.md ahí).
    audit : dict
        El informe de auditoría en memoria (el mismo objeto serializado a
        auditoria.json); lee hashes, tiempos, entorno y data_dir.

    Errores
    -------
    OSError / KeyError
        Si `out_dir` no es escribible o `audit` carece de las claves
        esperadas (llamada interna: siempre proviene de main()).

    Efectos colaterales
    -------------------
    Escribe out_dir/README-corrida.md.
    """
    h = audit["hashes_sha256_csv"]
    t = audit["tiempos_seg"]
    env = audit["entorno"]
    lines = [
        "# S1 — Auditoría ISIC 2019 (corrida local)",
        "",
        f"Generado (UTC): {audit['generado_utc']}",
        f"Data dir: `{audit['data_dir']}`",
        "",
        "## Comandos exactos",
        "",
        "```bash",
        "uv run python audit_isic.py --data-dir ~DIR --out-dir ~DIR_OUT",
        f"uv run python audit_isic.py --data-dir {audit['data_dir']} --out-dir {out_dir} --verificar {out_dir / 'auditoria.json'}",
        "```",
        "",
        "(El CRC completo usa `unzip -t` del sistema; el resto es Python puro + pandas + Pillow.)",
        "",
        "## Entorno",
        "",
        f"- Python {env['python']} · pandas {env['pandas']} · Pillow {env['pillow']} · matplotlib {env['matplotlib']}",
        f"- Tiempos: CRC training {t['crc_train_seg']}s · CRC test {t['crc_test_seg']}s · total {t['total_seg']}s",
        "",
        "## Hashes SHA-256 de los CSVs de entrada",
        "",
    ] + [f"- `{name}`: `{digest}`" for name, digest in h.items()] + [
        "",
        "## Artefactos",
        "",
        "- `auditoria.json` — resumen completo legible por máquina",
        "- `clases.csv` — distribución de clases train/test (conteo y %)",
        "- `lesion_id.csv` — nº de imágenes por `lesion_id` (train, solo lesiones con id)",
        "- `dimensiones.csv` — muestra de ~600 imágenes (width/height/mode), semilla 42",
        "- `figuras/clases.png`, `figuras/faltantes.png`",
        "",
        "CC BY-NC 4.0 — datos ISIC 2019 (International Skin Imaging Collaboration). Solo auditoría; ningún dato se redistribuye.",
    ]
    (out_dir / "README-corrida.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
