#!/usr/bin/env python3
"""Paso 1 de S2: formación de TESIS desde el ISIC 2019 crudo (preparación, CRISP-DM).

Propósito
---------
Forma TESIS: 30 subconjuntos × 15 imágenes muestreadas del set de training de
ISIC 2019 con una semilla fija desde `config.json` (semilla 42). Cada
subconjunto contiene al menos `min_bcc_per_subset` imágenes BCC (muestreo
guiado, no random crudo). Las imágenes pueden repetirse entre subconjuntos.
El split TEST de ISIC 2019 nunca se toca.

QUÉ ES TESIS Y POR QUÉ EXISTE
-----------------------------
TESIS es el paquete de demo/evidencia de la tesis: 30 subconjuntos pequeños y
autocontenidos de 15 imágenes dermatoscópicas cada uno (450 imágenes elegidas,
pueden repetirse entre subconjuntos). Existe para que la app final pueda
demostrarse y medirse sobre imágenes selladas, hasheadas y excluidas del
entrenamiento — de lo contrario cualquier número de "exactitud de demo" carecería
de sentido, porque el modelo podría haber memorizado esas imágenes durante el
fine-tuning.

Tres consecuencias de ese propósito guían cada decisión de diseño aquí:

  1. Reproducibilidad: la selección de subconjuntos queda totalmente
     determinada por la semilla fija del proyecto (config.json -> "seed": 42).
     Reejecutar este script sobre el mismo dataset produce el manifiesto
     idéntico. 42 es la semilla de todo el proyecto (auditorías, splits,
     entrenamiento), de modo que una sola constante explica todo el pipeline en
     la tesis. Cualquier cambio en el orden de muestreo, en el contenido de
     las listas o en las llamadas al RNG altera los subconjuntos — por eso el
     código es estable y nunca debe "limpiarse" cosméticamente.
  2. Exclusión del entrenamiento: cada id elegido queda listado en
     `tesis_exclude.csv`. La formación posterior de splits (70/20/10, agrupada
     por lesion_id, estratificada) DEBE excluir estos ids para que TESIS siga
     siendo un conjunto sonda limpio, nunca entrenado. La misma semilla fija es
     lo que permite a cualquier tercero regenerar exactamente esta lista de
     exclusión y verificarla (semilla -> subconjuntos -> lista de exclusión es
     una función pura del dataset crudo).
  3. Muestreo guiado, no random crudo: BCC es ~13,1 % del set de training
     (3.323 / 25.331). Un subconjunto uniforme de 15 imágenes llevaría un
     conteo de BCC distribuido alrededor de ~2 por subconjunto, con frecuencia
     0-1 — inútil para una demo cuyo propósito es el tamizaje de BCC. En su
     lugar cada subconjunto se compone de exactamente `min_bcc_per_subset`
     imágenes BCC más el remate extraído del pool no-BCC, es decir, muestreamos
     la composición, no la equidad. Es un sobremuestreo deliberado de la clase
     positiva SOLO en el paquete de demo; no afecta a los datos de
     entrenamiento ni a las métricas.

Este script lee ÚNICAMENTE el zip de training y el CSV GT de training. El split
TEST de ISIC 2019 nunca se abre, lista, hashea ni copia — no
tiene lesion_id, ~24,9 % de etiquetas UNK y se reserva exclusivamente para la
evaluación final. No agregues aquí ningún camino de código que referencie los
archivos de test.

Entradas
--------
* `--data-dir ~DIR` (obligatorio): directorio con los archivos crudos de
  ISIC 2019 de training (zip + GT). Origen EXTERNO al repo;
  SOLO LECTURA.
* `--out-dir ~DIR_OUT` (obligatorio): destino de los artefactos
  manifiesto/exclusión/resumen (texto, fuera del repo).
* `--tesis-dir ~DIR_TESIS` (obligatorio): destino de las imágenes copiadas
  (`<tesis-dir>/<subset_id>/<image_id>.jpg`, fuera del repo).

Salidas (bajo `--out-dir`)
--------------------------
  tesis_manifest.csv  una fila por pick (subset_id, image_id) con etiqueta,
                      sha256 del archivo copiado y la ruta de entrada original
                      dentro del zip
  tesis_exclude.csv   lista deduplicada de image_id a excluir de los splits de
                      entrenamiento (columnas image_id, label_full)
  summary.json        recibo de la corrida: semilla, config, conteos, mínimo de
                      BCC por subconjunto, distribución de clases de los picks,
                      procedencia de los artefactos
  README-corrida.md   (lo escribe un paso posterior, no este script)

Quién consume sus salidas
-------------------------
* `tesis_manifest.csv` → export_ground_truth.py (JSON de etiquetas por
  subconjunto, para la demo de la app), verificar() de este mismo script y
  hf_upload.py (subida como manifiesto al git-remote del dataset).
* `tesis_exclude.csv` → make_splits.py, como lista de exclusión del split
  70/20/10; es el contrato que mantiene TESIS fuera del entrenamiento.
* `summary.json` → la evidencia de S2 paso 1 (recibo de la corrida: semilla,
  conteos, distribución de clases de los picks).

Uso:
    uv run python form_tesis.py --data-dir ~DIR --out-dir ~DIR_OUT --tesis-dir ~DIR_TESIS
    uv run python form_tesis.py --data-dir ~DIR --out-dir ~DIR_OUT --tesis-dir ~DIR_TESIS --force
    uv run python form_tesis.py --data-dir ~DIR --out-dir ~DIR_OUT --tesis-dir ~DIR_TESIS --verificar ~DIR_OUT/tesis_manifest.csv

Cubre el trabajo de la fase S2, paso 1 — preparación de datos (CRISP-DM).
"""

import argparse
from dermica_core.labels import ISIC_LABELS  # import único desde dermica_core
from dermica_core.hashing import sha256_file  # import único desde dermica_core
from dermica_core.cli import ParserEspanol  # import único desde dermica_core
import csv
import json
import random
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# config.json vive junto al código de este pipeline (pipeline/) y forma
# parte del repo: es la única fuente de verdad para la semilla y la forma de
# TESIS; no incrustes estos valores — la tesis cita config.json como contrato
# congelado. La resuelven form() y verificar().
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
TRAIN_ZIP = "ISIC_2019_Training_Input.zip"   # zip de training de ISIC 2019; nunca se extrae masivamente; lo abre form() en streaming
GT_TRAIN = "ISIC_2019_Training_GroundTruth.csv"   # etiquetas one-hot de training; las carga load_train_labels()


# Carpeta de primer nivel DENTRO del zip de training; las entradas se direccionan
# como f"{ZIP_PREFIX}{image_id}.jpg". El zip nunca se extrae masivamente: cada
# entrada elegida se transmite individualmente (ver el bucle de copia en form()).
# La consume form().
ZIP_PREFIX = "ISIC_2019_Training_Input/"






def load_train_labels(data_dir: Path) -> pd.Series:
    """Carga el GT de training y resuelve la etiqueta única por imagen.

    Parámetros
    ----------
    data_dir : Path
        Directorio con los CSV crudos de ISIC 2019 que contiene GT_TRAIN.

    Retorno
    -------
    pd.Series
        Serie indexada por id de imagen (gt["image"]) cuyos valores son el
        nombre de la clase cuya columna one-hot está en hot. idxmax funciona
        porque el GT es mono-etiqueta (una columna = 1 por fila); los empates
        son imposibles por ese invariante, y la auditoría S1 lo garantiza antes
        de que esto se ejecute.

    Errores
    -------
    FileNotFoundError / pd.errors.ParserError
        Si el GT no existe o no es parseable.

    Efectos colaterales
    -------------------
    Parseo del CSV en solo lectura. NUNCA abre el GT de test de ISIC.

    Globales
    --------
    `GT_TRAIN` (nombre del CSV de training, de este módulo) e `ISIC_LABELS`
    (orden de columnas one-hot, de dermica_core.labels).
    """
    gt = pd.read_csv(data_dir / GT_TRAIN)
    labels = gt[ISIC_LABELS].idxmax(axis=1)
    labels.index = gt["image"]
    return labels


def pick_subset(rng: random.Random, bcc_ids: list, rest_ids: list, n_bcc: int, n_total: int) -> list:
    """Extrae un subconjunto: n_bcc ids BCC + (n_total - n_bcc) ids no-BCC, barajados.

    Parámetros
    ----------
    rng : random.Random
        Generador seedeado (semilla 42 desde config.json). El RNG es propiedad
        del LLAMADOR, de modo que los 30 subconjuntos salen de una única
        secuencia determinista; cada llamada a pick_subset consume estado del
        rng en un orden fijo.
    bcc_ids : list
        Todos los ids de imagen de training cuya etiqueta es BCC (~3.323). El
        muestreo es con reemplazo ENTRE subconjuntos (los subconjuntos se
        sortean de forma independiente), así que la misma imagen BCC puede
        aparecer en varios subconjuntos — previsto por
        "repeat_across_subsets": true en config.json. Dentro de un mismo
        subconjunto, rng.sample garantiza cero duplicados.
    rest_ids : list
        Todos los ids de imagen de training cuya etiqueta NO es BCC.
    n_bcc : int
        Imágenes BCC a extraer (config "min_bcc_per_subset", default 5). Esta
        es la parte guiada: composición fija de la clase positiva por
        subconjunto.
    n_total : int
        Tamaño del subconjunto (config "images_per_subset", default 15).

    Retorno
    -------
    list
        n_total ids de imagen, barajados para que el orden de filas del
        manifiesto no codifique "primero los picks BCC, luego el resto" (el
        orden barajado es el que conserva el bucle de copia y el manifiesto).

    Errores
    -------
    ValueError
        Si bcc_ids o rest_ids tienen menos elementos que n_bcc / n_total - n_bcc
        (rng.sample falla con un mensaje claro; ver nota en form()).

    Efectos colaterales
    -------------------
    Avanza el estado de rng. No toca archivos.

    Globales
    --------
    Ninguna.
    """
    picks = rng.sample(bcc_ids, n_bcc) + rng.sample(rest_ids, n_total - n_bcc)
    rng.shuffle(picks)
    return picks


def form(data_dir: Path, out_dir: Path, tesis_dir: Path, force: bool) -> int:
    """Forma los subconjuntos TESIS, copia las imágenes y escribe todos los artefactos.

    Pasos:
      1. Lee config.json (semilla + forma de TESIS). Nota: los valores de
         config se confían tal cual, sin validarse — un config con
         subsets*images_per_subset desproporcionado respecto a los ids BCC
         disponibles fallaría en rng.sample con un error claro, aceptable para
         un paso de pipeline de una sola sentada.
      2. Carga las etiquetas de training; separa los ids en pools BCC / resto.
      3. Se niega a sobrescribir un tesis_dir no vacío salvo que force=True — la
         guarda existe para que una reejecución no mezcle en silencio copias
         viejas y nuevas con semillas distintas.
      4. Sortea los 30 subconjuntos desde UN único RNG seedeado
         (orden: subset_01..30).
      5. Copia por streaming cada entrada elegida desde el zip de training a
         tesis_dir/<subset_id>/<image_id>.jpg (chunks de 1 MiB, nunca extractall) y
         hashea el archivo copiado en disco.
      6. Escribe tesis_manifest.csv, tesis_exclude.csv (ids deduplicados) y
         summary.json; imprime un recibo de una línea.

    Parámetros
    ----------
    data_dir : Path
        Directorio con los archivos crudos de ISIC 2019 de training (zip + GT).
    out_dir : Path
        Destino de los artefactos manifiesto/exclusión/resumen.
    tesis_dir : Path
        Destino de las imágenes copiadas (tesis_dir/<subset_id>/...).
    force : bool
        Si True, permite escribir en un tesis_dir existente no vacío.

    Retorno
    -------
    int
        0 en caso de éxito; 1 si tesis_dir existe y no está vacío sin force.

    Errores
    -------
    zipfile.BadZipFile / KeyError
        Si el zip no es válido o una entrada esperada no existe dentro de él
        (dataset corrupto o zip equivocado; el traceback lo delata).

    Efectos colaterales
    -------------------
    Crea tesis_dir y out_dir; escribe ~450+ copias jpg y tres artefactos.
    Nunca escribe en data_dir. Nunca abre el zip de test de ISIC ni los CSV de
    test.

    Globales
    --------
    `CONFIG_PATH` (semilla y forma de TESIS desde config.json), `TRAIN_ZIP`
    y `ZIP_PREFIX` (zip de training y su prefijo de primer nivel). El GT llega
    indirectamente vía load_train_labels() (`GT_TRAIN`, `ISIC_LABELS`).
    """
    t0 = time.time()
    cfg = json.loads(CONFIG_PATH.read_text())
    seed = int(cfg["seed"])
    n_subsets = int(cfg["tesis"]["subsets"])
    per_subset = int(cfg["tesis"]["images_per_subset"])
    min_bcc = int(cfg["tesis"]["min_bcc_per_subset"])
    print(f"config: seed={seed} subsets={n_subsets} x {per_subset} "
          f"(min BCC={min_bcc})")

    labels = load_train_labels(data_dir)
    bcc_ids = labels[labels == "BCC"].index.tolist()
    rest_ids = labels[labels != "BCC"].index.tolist()
    print(f"train: {len(labels)} imgs | BCC {len(bcc_ids)} | rest {len(rest_ids)}")

    if tesis_dir.exists() and any(tesis_dir.iterdir()) and not force:
        print(f"ERROR: {tesis_dir} ya existe y no está vacío. Usa --force para sobrescribir.")
        return 1

    rng = random.Random(seed)
    subset_picks = {}   # subset_id -> lista de picks de ese subconjunto (orden subset_01..30)
    for i in range(1, n_subsets + 1):
        subset_picks[f"subset_{i:02d}"] = pick_subset(rng, bcc_ids, rest_ids, min_bcc, per_subset)

    # Copia las imágenes del zip de training una entrada a la vez. Sin
    # extracción masiva: solo se escriben los picks (~450, menos únicos).
    zip_path = data_dir / TRAIN_ZIP
    copied = 0          # contador de copias jpg escritas (≈450; duplicados entre subconjuntos cuentan aparte)
    rows = []           # una fila por pick: subset_id, image_id, label_full, sha256 del archivo copiado y entrada original dentro del zip
    with zipfile.ZipFile(zip_path) as zf:
        for subset_id in sorted(subset_picks):
            dest_dir = tesis_dir / subset_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            for image_id in subset_picks[subset_id]:
                entry = f"{ZIP_PREFIX}{image_id}.jpg"
                dest = dest_dir / f"{image_id}.jpg"
                with zf.open(entry) as src, open(dest, "wb") as dst:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        dst.write(chunk)
                copied += 1
                rows.append({
                    "subset_id": subset_id,
                    "image_id": image_id,
                    "label_full": labels[image_id],
                    "sha256": sha256_file(dest),
                    "original_path": entry,
                })

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "tesis_manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subset_id", "image_id", "label_full", "sha256", "original_path"])
        w.writeheader()
        w.writerows(rows)

    # Deduplicar: el manifiesto lleva 450 filas (30 × 15), pero una imagen
    # repetida entre subconjuntos aparece bajo varias filas. La lista de
    # exclusión es sobre ids, así que colapsa al conjunto único; las etiquetas
    # se buscan en la serie del GT (misma fuente que el manifiesto — consistentes
    # por construcción).
    unique_ids = sorted({r["image_id"] for r in rows})   # ids únicos para la lista de exclusión (colapsa repeticiones entre subconjuntos)
    exclude = out_dir / "tesis_exclude.csv"
    ex_labels = labels[unique_ids]
    ex_unique = pd.DataFrame({"image_id": unique_ids,
                              "label_full": [ex_labels[i] for i in unique_ids]})
    ex_unique.to_csv(exclude, index=False)

    per_subset_bcc = {sid: sum(1 for r in rows if r["subset_id"] == sid and r["label_full"] == "BCC")
                      for sid in sorted(subset_picks)}   # BCC por subconjunto; alimenta el recibo y la compuerta min_bcc
    class_counts = pd.Series([r["label_full"] for r in rows]).value_counts().to_dict()
    summary = {
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "config": cfg["tesis"],
        "picks": len(rows),
        "unique_images": len(unique_ids),
        "subsets": n_subsets,
        "images_per_subset": per_subset,
        "bcc_per_subset_min": min(per_subset_bcc.values()),
        "bcc_per_subset": per_subset_bcc,
        "class_distribution_picks": {k: int(v) for k, v in class_counts.items()},
        "bcc_picks": int(sum(1 for r in rows if r["label_full"] == "BCC")),
        # Postcondiciones registradas explícitamente para que el recibo muestre
        # que se comprobaron al generar, no que se dieron por sentadas: cada
        # subconjunto cumple min_bcc y cada id elegido existe en el GT de training.
        "min_bcc_ok": all(v >= min_bcc for v in per_subset_bcc.values()),
        "all_ids_in_train": all(i in labels.index for i in unique_ids),
        "zip_source": TRAIN_ZIP,
        "segundos": round(time.time() - t0, 2),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"picks={summary['picks']} unique={summary['unique_images']} "
          f"BCC/picks={summary['bcc_picks']} min_bcc_subset={summary['bcc_per_subset_min']}")
    print(f"copied {copied} files -> {tesis_dir}")
    print(f"artifacts -> {out_dir}")
    return 0


def verificar(manifest_path: Path, data_dir: Path, tesis_dir: Path) -> int:
    """Reverifica un tesis_manifest.csv contra el GT, la config y los archivos copiados.

    Chequeos (todos se ejecutan; sin salida temprana, así se imprimen todos los
    fallos):
      1. el manifiesto tiene exactamente subsets*images_per_subset filas (30*15=450);
      2. el conjunto de subset_id es exactamente subset_01..subset_<n_subsets>;
      3. cada subconjunto tiene exactamente images_per_subset filas;
      4. cada image_id existe en el GT de training (sin contaminación del split
         test ni ids ajenos);
      5. el label_full del manifiesto coincide con la etiqueta del GT para cada id;
      6. cada subconjunto contiene >= min_bcc imágenes BCC (el contrato de
         muestreo guiado);
      7. cada archivo copiado existe en <tesis-dir>/<subset_id>/<image_id>.jpg y su
         SHA-256 coincide con el del manifiesto — es decir, las copias en disco
         son exactamente lo que se eligió, no sobreescrituras viejas de una
         corrida anterior.

    Por qué StratifiedGroupKFold (y no un shuffle random plano) en el split
    posterior — contexto de lo que este manifiesto protege: tras excluir los
    ids del manifiesto, la partición de training sigue conteniendo varias imágenes
    por lesion_id (la misma lesión, tomas distintas). Un shuffle aleatorio a
    nivel de imagen dispersaría recortes casi idénticos de la misma lesión
    entre train/val/test, filtrando señal específica de la lesión e inflando
    todas las métricas. El split debe agrupar por lesion_id (ninguna lesión en
    dos particiones) y mantener balanceada la prevalencia de BCC (~13,1 %)
    entre particiones — de ahí un fold ESTRATIFICADO Y AGRUPADO
    (StratifiedGroupKFold), nunca RandomShuffleSplit. La exclusión de los ids
    del manifiesto es la primera mitad de esa higiene; este verificador es el
    recibo de que TESIS es un conjunto sonda limpio y retenido.

    Parámetros
    ----------
    manifest_path : Path
        Ruta a tesis_manifest.csv producido por form().
    data_dir : Path
        Directorio con los CSV crudos de ISIC 2019 de training (GT); se
        resuelve por CLI (--data-dir), no se asume ninguna ubicación fija.
    tesis_dir : Path
        Directorio con las imágenes copiadas (<tesis-dir>/<subset_id>/...); se
        resuelve por CLI (--tesis-dir), no se asume ninguna ubicación fija.

    Retorno
    -------
    int
        0 si todos los chequeos pasan, 1 en caso contrario.

    Errores
    -------
    FileNotFoundError / pd.errors.ParserError
        Si el manifiesto o el GT no existen o no son parseables.

    Efectos colaterales
    -------------------
    Solo lectura: hashea los archivos copiados bajo tesis_dir y
    parsea el CSV GT. No escribe nada.

    Globales
    --------
    `CONFIG_PATH` (aspectos verificados contra config.json: semilla, forma de
    TESIS, mínimo de BCC). El GT llega indirectamente vía load_train_labels()
    (`GT_TRAIN`, `ISIC_LABELS`).
    """
    t0 = time.time()
    cfg = json.loads(CONFIG_PATH.read_text())
    min_bcc = int(cfg["tesis"]["min_bcc_per_subset"])
    per_subset = int(cfg["tesis"]["images_per_subset"])
    n_subsets = int(cfg["tesis"]["subsets"])

    if not manifest_path.exists():
        print(f"FAIL: no existe {manifest_path}")
        return 1
    df = pd.read_csv(manifest_path)
    print(f"filas: {len(df)}")
    ok = True

    def check(cond, msg):
        # Acumulador en lugar de retorno temprano: reporta TODOS los chequeos
        # fallidos, para que un manifiesto roto muestre todo el daño en una pasada.
        nonlocal ok
        print(("OK  " if cond else "FAIL") + "  " + msg)
        ok = ok and cond

    check(len(df) == n_subsets * per_subset, f"{n_subsets * per_subset} filas (450)")
    check(set(df["subset_id"]) == {f"subset_{i:02d}" for i in range(1, n_subsets + 1)},
          f"{n_subsets} subconjuntos válidos")
    check(df.groupby("subset_id").size().eq(per_subset).all(), f"{per_subset} imágenes por subconjunto")

    labels = load_train_labels(data_dir)
    check(df["image_id"].isin(labels.index).all(), "todas las image_id ⊆ ground truth train")
    check(labels[df["image_id"]].eq(df["label_full"].values).all(), "label_full coincide con el GT")

    bcc = df[df["label_full"] == "BCC"].groupby("subset_id").size()
    bcc = bcc.reindex([f"subset_{i:02d}" for i in range(1, n_subsets + 1)], fill_value=0)
    check(bcc.ge(min_bcc).all(), f"≥{min_bcc} BCC por subconjunto (min={int(bcc.min())})")

    missing, bad_hash = [], []
    for r in df.itertuples():
        p = tesis_dir / r.subset_id / f"{r.image_id}.jpg"
        if not p.exists():
            missing.append(str(p))
            continue
        if sha256_file(p) != r.sha256:
            bad_hash.append(str(p))
    check(not missing, f"archivos presentes ({len(missing)} faltantes)")
    check(not bad_hash, f"hashes SHA-256 coinciden ({len(bad_hash)} distintos)")

    print(f"tiempo: {time.time() - t0:.1f}s")
    print("VERIFICACIÓN:", "OK" if ok else "FALLO")
    return 0 if ok else 1


def main() -> int:
    """Punto de entrada CLI: parsea args y despacha a form() o verificar().

    Efectos colaterales
    -------------------
    Ver form()/verificar(); aquí no ocurre nada más allá de parsear argumentos
    y delegar.

    Globales
    --------
    Ninguna (las rutas se resuelven en form()/verificar()).
    """
    ap = ParserEspanol(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True, metavar="~DIR",
                    help="directorio con los archivos crudos de ISIC 2019 de training (zip + GT)")
    ap.add_argument("--out-dir", type=Path, required=True, metavar="~DIR_OUT",
                    help="directorio donde se escriben tesis_manifest.csv, tesis_exclude.csv y summary.json")
    ap.add_argument("--tesis-dir", type=Path, required=True, metavar="~DIR_TESIS",
                    help="directorio donde se copian las imágenes (<DIR_TESIS>/<subset_id>/<image_id>.jpg)")
    ap.add_argument("--force", action="store_true",
                    help="permite escribir en un --tesis-dir existente no vacío (sobrescribe copias)")
    ap.add_argument("--verificar", type=Path, default=None, metavar="~CSV",
                    help="ruta a tesis_manifest.csv; corre verificaciones sin escribir")
    args = ap.parse_args()
    if args.verificar is not None:
        return verificar(args.verificar, args.data_dir, args.tesis_dir)
    return form(args.data_dir, args.out_dir, args.tesis_dir, args.force)


if __name__ == "__main__":
    sys.exit(main())
