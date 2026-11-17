#!/usr/bin/env python3
"""S2 paso 2: split 70/20/10 estratificado por BCC/no-BCC y agrupado por lesion_id.

Propósito
---------
Construye la partición de entrenamiento del proyecto: 70 % train / 20 % val /
10 % test, estratificada por la etiqueta binaria BCC/no-BCC y agrupada por
`lesion_id` (ninguna lesión cae en dos particiones). Las imágenes sin
`lesion_id` se tratan como singletons (cada imagen es su propio grupo). El
muestreo usa la semilla fija de `config.json` (42),
`StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)` y una
reunión de folds 7/2/1 para obtener train/val/test = 70/20/10.

Entradas
--------
* `--data-dir ~DIR` (obligatorio): directorio con los CSV crudos de
  ISIC 2019 (GT y metadata de training). Origen EXTERNO al repo; solo lectura.
* `--exclude-csv ~CSV` (obligatorio): lista de exclusión con columna
  `image_id` producida por la fase de formación de subconjuntos; los ids
  listados nunca entran al split.
* `--out-dir ~DIR_OUT` (obligatorio): destino de los artefactos.

Salidas (bajo `--out-dir`)
--------------------------
  train.csv, val.csv, test.csv   (columnas: image, split, label_full,
                                  label_bcc, lesion_id)
  summary.json                   recibo de la corrida (semilla, conteos,
                                 ratios por partición, % BCC)
  README-corrida.md              (lo escribe un paso posterior, no este script)

Nunca toca el split TEST oficial de ISIC 2019 ni escribe en las fuentes de
entrada: solo lee el GT/metadata de training y la lista de exclusión.

Uso:
    uv run python make_splits.py --data-dir ~DIR --exclude-csv ~CSV --out-dir ~DIR_OUT
    uv run python make_splits.py --data-dir ~DIR --exclude-csv ~CSV --out-dir ~DIR_OUT --verificar ~DIR_OUT/summary.json

Cubre el trabajo de la fase S2, paso 2 — preparación de datos (CRISP-DM).
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# config.json vive junto al código de este pipeline (pipeline/) y forma
# parte del repo: es la única fuente de verdad para la semilla y los ratios
# del split. Se resuelve desde __file__ para que el CWD no importe. La
# consumen make(), verify_outputs() y las compuertas de ratios.
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
GT_TRAIN = "ISIC_2019_Training_GroundTruth.csv"   # etiquetas one-hot de training (única fuente de etiquetas)
META_TRAIN = "ISIC_2019_Training_Metadata.csv"    # metadata de training; aporta image↔lesion_id
# Columnas one-hot de clases ISIC 2019 (un 1.0 por fila). El orden es fijo para
# que idxmax(...) de más abajo sea determinista aunque haya empates. La
# consumen load_frames().
CLASS_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]


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


def load_frames(data_dir: Path) -> pd.DataFrame:
    """Carga y une GT + metadata de TRAIN de ISIC 2019 (solo lectura).

    Parámetros
    ----------
    data_dir : Path
        Directorio con los CSV crudos de ISIC 2019 (GT y metadata de training).

    Retorno
    -------
    pd.DataFrame
        DataFrame con columnas image, lesion_id, label_full, label_bcc, group.

    Errores
    -------
    FileNotFoundError / pd.errors.MergeError
        Si falta un CSV o el merge GT↔metadata deja de ser 1:1.

    Efectos colaterales
    -------------------
    Ninguno (solo lectura).

    El merge fuerza relación 1:1 GT↔metadata (validate="1:1"): si el dataset
    cambia y un image aparece duplicado o sin pareja, el fallo ocurre aquí y no
    silenciosamente más abajo. label_full se deriva del one-hot con idxmax;
    label_bcc es el binario BCC vs resto (problema del proyecto).
    """
    gt = pd.read_csv(data_dir / GT_TRAIN)
    meta = pd.read_csv(data_dir / META_TRAIN)[["image", "lesion_id"]]
    df = pd.merge(gt, meta, on="image", validate="1:1")
    df["label_full"] = df[CLASS_COLS].idxmax(axis=1)
    df["label_bcc"] = (df["label_full"] == "BCC").astype(int)
    # Los sin lesion_id son singletons: cada imagen es su propio grupo.
    # Truco: fillna("") normaliza NaN a "" y mask sustituye las vacías por
    # image (único). Así ninguna fila queda con group = "" (inválido como
    # grupo verdadero: colapsaría todas las imágenes sin lesion_id en un
    # único súper-grupo y las partiría junto).
    df["group"] = df["lesion_id"].fillna("").mask(lambda s: s == "", df["image"])
    return df


def build_splits(df: pd.DataFrame, seed: int) -> pd.Series:
    """Asigna cada imagen a train/val/test y devuelve la serie resultante.

    Parámetros
    ----------
    df : pd.DataFrame
        Salida de load_frames() tras excluir los ids de la lista de exclusión
        (debe contener label_bcc y group).
    seed : int
        Semilla del proyecto (42, de config.json). Determina el shuffle del
        StratifiedGroupKFold.

    Retorno
    -------
    pd.Series
        Serie indexada como df con valores en {"train", "val", "test"}.

    Errores
    -------
    KeyError
        Si df carece de las columnas label_bcc o group.

    Efectos colaterales
    -------------------
    Ninguno (cálculo en memoria).

    Por qué StratifiedGroupKFold y no train_test_split×3: mantiene (a) balance
    de clases en cada fold y (b) grupos lesion_id cerrados (ninguna lesión cae
    en dos folds), dos garantías que splits independientes y aleatorios no dan.

    shuffle=True con random_state=seed (42, de config.json) hace la
    asignación estratificada-agrupada reproducible y no dependiente del orden
    del CSV. Los 10 folds se reúnen 7 (train) / 2 (val) / 1 (test) tras el
    shuffle, lo que da 70/20/10 exacto y determinista.
    """
    skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    assign = pd.Series("", index=df.index, dtype=object)
    for fold, (_, test_idx) in enumerate(skf.split(df, df["label_bcc"], df["group"])):
        assign.iloc[test_idx] = f"fold_{fold:02d}"
    # 7 folds -> train, 2 -> val, 1 -> test (determinista tras shuffle)
    assign = assign.map(lambda f: "train" if int(f[5:]) < 7 else ("val" if int(f[5:]) < 9 else "test"))
    return assign


def ratios_report(df: pd.DataFrame) -> dict:
    """Reporte por partición: cardinalidad, BCC y % del total.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas split y label_bcc (salida de make()).

    Retorno
    -------
    dict
        {split: {"imagenes": int, "bcc": int, "bcc_pct": float, "pct_total": float}}

    Efectos colaterales
    -------------------
    Ninguno (cálculo en memoria).

    label_bcc es 0/1, por eso label_bcc.mean() es exactamente la tasa BCC de la
    partición (suma de BCC / total). Se reporta para auditar el balance tras el
    split: no falla si varía ±1%, pero deja evidencia numérica en summary.json.
    """
    rep = {}
    for split, g in df.groupby("split"):
        rep[split] = {
            "imagenes": int(len(g)),
            "bcc": int(g["label_bcc"].sum()),
            # media de 0/1 == fracción de BCC; x100 = tasa porcentual
            "bcc_pct": round(100.0 * g["label_bcc"].mean(), 2),
            "pct_total": round(100.0 * len(g) / len(df), 2),
        }
    return rep


def verify(df: pd.DataFrame, cfg: dict, source_exclude: set) -> list[str]:
    """Compuertas inline del split; devuelve la lista de problemas (vacía si OK).

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame particionado (debe contener image, split, group).
    cfg : dict
        Contenido de config.json (usa cfg["splits"] para los objetivos de ratio).
    source_exclude : set
        Conjunto de image_id de la lista de exclusión; se usa para detectar
        contaminación de subconjuntos excluidos en las particiones.

    Retorno
    -------
    list[str]
        Descripciones de los problemas detectados; [] si todo cuadra.

    Efectos colaterales
    -------------------
    Imprime avisos (no fallos) si una partición se desvía >1 p.p. del objetivo.

    Cada compuerta cubre un modo de fuga/defecto distinto:
    - intersección de imágenes entre particiones -> misma foto en train y val/test
      (fuga directa de datos): debe ser ∅.
    - lesion_id compartido entre particiones -> imágenes de la misma lesión en
      distintas particiones, el clásico fallo de "multiple image lesions" de
      ISIC; garantiza que val/test miden generalización real.
    - subconjuntos excluidos presentes -> imágenes de la lista de exclusión
      contaminando train/val/test.
    - image_id duplicados -> cardinalidad inflada; rompe ratios y métricas.
    - ratios 70/20/10: se avisa (no se falla) si alguna partición se desvía
      >1 p.p.; el tamaño exacto depende del dataset y no debe ser una compuerta dura.
    """
    problems = []
    splits = {s: set(df[df["split"] == s]["image"]) for s in ("train", "val", "test")}
    groups = {s: set(df[df["split"] == s]["group"]) for s in ("train", "val", "test")}

    if set.intersection(splits["train"], splits["val"]):
        problems.append("hay imágenes de train también en val")
    if set.intersection(splits["train"], splits["test"]):
        problems.append("hay imágenes de train también en test")
    if set.intersection(splits["val"], splits["test"]):
        problems.append("hay imágenes de val también en test")

    overlap_tv = set.intersection(groups["train"], groups["val"])
    overlap_tt = set.intersection(groups["train"], groups["test"])
    overlap_vt = set.intersection(groups["val"], groups["test"])
    n_ov = len(overlap_tv) + len(overlap_tt) + len(overlap_vt)
    if n_ov:
        problems.append(f"{n_ov} grupos lesion_id compartidos entre particiones")

    tesis_leak = source_exclude & set(df["image"])
    if tesis_leak:
        problems.append(f"{len(tesis_leak)} imágenes de la lista de exclusión presentes en una partición")

    if len(df) != len(set(df["image"])):
        problems.append("hay image_id duplicados")

    ratios = {s: len(g) / len(df) for s, g in df.groupby("split")}
    for name, target in (("train", cfg["splits"]["train"]), ("val", cfg["splits"]["val"]), ("test", cfg["splits"]["test"])):
        dev = abs(ratios.get(name, 0) - target)
        if dev > 0.01:
            print(f"AVISO: {name} desviado {(dev * 100):.2f} p.p. del objetivo {target:.2f}")

    return problems


def write_outputs(df: pd.DataFrame, out_dir: Path) -> dict:
    """Escribe train/val/test.csv con el contrato de columnas del pipeline.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame particionado con image, split, label_full, label_bcc, lesion_id.
    out_dir : Path
        Directorio de destino; se crea si no existe.

    Retorno
    -------
    dict
        {split: n_imágenes_escritas}.

    Efectos colaterales
    -------------------
    Crea out_dir y escribe train.csv, val.csv y test.csv.

    lesion_id ausente se escribe como cadena vacía (no "NaN" ni null), de modo
    que el consumidor (entrenamiento) siempre vea strings. El orden de columnas
    es el contrato: image, split, label_full, label_bcc, lesion_id.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "val", "test"):
        g = df[df["split"] == split][["image", "split", "label_full", "label_bcc", "lesion_id"]].copy()
        g["lesion_id"] = g["lesion_id"].fillna("")
        g.to_csv(out_dir / f"{split}.csv", index=False)
        counts[split] = len(g)
    return counts


def make(data_dir: Path, out_dir: Path, exclude_csv: Path) -> int:
    """Construye el split y escribe artefactos en out_dir.

    Parámetros
    ----------
    data_dir : Path
        Directorio con los CSV crudos de ISIC 2019 (GT y metadata de training).
    out_dir : Path
        Destino de los artefactos (train/val/test.csv, summary.json).
    exclude_csv : Path
        CSV con columna `image_id` cuyos ids se excluyen del split.

    Retorno
    -------
    int
        0 si las compuertas inline pasaron y los artefactos se escribieron;
        1 si falta la lista de exclusión o cualquier compuerta falló.

    Efectos colaterales
    -------------------
    Crea out_dir y escribe los CSV particionados y summary.json (solo si todas
    las compuertas pasan). Nunca escribe en data_dir.

    Contrato: si cualquier compuerta inline falla, NO se escribe ningún
    artefacto (evita summary/CSVs engañosos a medio construir); mensajes de
    error listan los problemas uno a uno. Los datos crudos son solo lectura.
    """
    t0 = time.time()
    cfg = json.loads(CONFIG_PATH.read_text())
    seed = int(cfg["seed"])

    if not exclude_csv.exists():
        print(f"ERROR: no existe la lista de exclusión: {exclude_csv}")
        return 1
    exclude = set(pd.read_csv(exclude_csv)["image_id"])
    print(f"exclusión: {len(exclude)} imágenes")

    df = load_frames(data_dir)
    before = len(df)
    df = df[~df["image"].isin(exclude)].reset_index(drop=True)
    print(f"train GT {before} -> sin exclusión {len(df)}")

    df["split"] = build_splits(df, seed)

    problems = verify(df, cfg, exclude)
    if problems:
        print("FALLO de verificación inline:")
        for p in problems:
            print(" -", p)
        print("No se escriben artefactos.")
        return 1

    counts = write_outputs(df, out_dir)
    rep = ratios_report(df)
    summary = {
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "fuente": {"gt": GT_TRAIN, "metadata": META_TRAIN, "exclude": str(exclude_csv.name)},
        "candidatos": int(len(df)),
        "excluidas_tesis": before - len(df),
        "particiones": counts,
        "bcc_pct_global": round(100.0 * df["label_bcc"].mean(), 2),
        "bcc_pct_por_particion": {s: rep[s]["bcc_pct"] for s in rep},
        "pct_por_particion": {s: rep[s]["pct_total"] for s in rep},
        "bcc_por_particion": {s: rep[s]["bcc"] for s in rep},
        "grupo_overlap_imagen": 0,
        "grupo_overlap_lesion": 0,
        "segundos": round(time.time() - t0, 2),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"artefactos -> {out_dir}")
    return 0


def verify_outputs(summary_path: Path, exclude_csv: Path) -> int:
    """Lee artefactos ya escritos y verifica conformidad (no escribe nada).

    Parámetros
    ----------
    summary_path : Path
        Ruta al summary.json producido por make(); su directorio padre se
        toma como out_dir para localizar train/val/test.csv.
    exclude_csv : Path
        CSV con columna `image_id` que ninguna partición puede contener;
        si no existe, la compuerta de exclusión se omite con aviso.

    Retorno
    -------
    int
        0 si todos los chequeos pasan, 1 en caso contrario.

    Errores
    -------
    json.JSONDecodeError / pd.errors.ParserError / KeyError
        Si summary.json o los CSV particionados no son válidos o les faltan
        claves (el mensaje lo delata tal cual).

    Efectos colaterales
    -------------------
    Ninguno (solo lectura). Imprime PASS/FAIL por compuerta.

    Reconstruye el concat train+val+test y comprueba: cardinalidad y
    cardinalidades por partición contra summary.json, valores de split válidos,
    columnas esperadas, exclusiones ausentes, image únicos y ninguna lesión en
    más de un split; ratios con aviso ±1%.
    """
    cfg = json.loads(CONFIG_PATH.read_text())
    if not summary_path.exists():
        print(f"FAIL: no existe {summary_path}")
        return 1
    s = json.loads(summary_path.read_text())
    out_dir = summary_path.parent
    df = pd.concat([pd.read_csv(out_dir / f"{p}.csv") for p in ("train", "val", "test")],
                   ignore_index=True)
    exclude = set(pd.read_csv(exclude_csv)["image_id"]) if exclude_csv.exists() else set()

    problems = []
    if len(df) != s["candidatos"]:
        problems.append(f"cardinalidad {len(df)} != {s['candidatos']} en summary.json")
    split_vals = set(df["split"])
    if split_vals - {"train", "val", "test"}:
        problems.append(f"valores de split desconocidos: {split_vals - {'train', 'val', 'test'}}")
    for p in ("train", "val", "test"):
        g = df[df["split"] == p]
        if len(g) == 0:
            problems.append(f"partición {p} vacía")
        if len(g) != s["particiones"][p]:
            problems.append(f"{p}: {len(g)} != {s['particiones'][p]} en summary.json")
        if not g.columns.isin(["image", "split", "label_full", "label_bcc", "lesion_id"]).all():
            problems.append(f"{p}: columnas inesperadas")
    if exclude & set(df["image"]):
        problems.append("lista de exclusión presente en una partición")
    images = df["image"]
    if len(images) != images.nunique():
        problems.append("image duplicados")
    # sensibilidad ±: si alguna lesión aparece en >1 split, los CVs entre
    # train/val/test dejarían de ser independientes (fuga de múltiples).
    lesiones = df["lesion_id"].fillna("")
    lesiones = lesiones[lesiones != ""]
    ok_lesiones = all(
        len(set(df[df["lesion_id"] == l]["split"])) <= 1
        for l in lesiones.unique()
    )
    if not ok_lesiones:
        problems.append("una lesión aparece en más de un split")
    ratios = df.groupby("split").size() / len(df)
    for name, target in (("train", cfg["splits"]["train"]), ("val", cfg["splits"]["val"]), ("test", cfg["splits"]["test"])):
        if abs(ratios[name] - target) > 0.01:
            print(f"AVISO: {name} desviado {(abs(ratios[name] - target) * 100):.2f} p.p. del objetivo")

    for p in problems:
        print("FAIL", p)
    print("VERIFICACIÓN:", "OK" if not problems else "FALLO")
    return 0 if not problems else 1


def main() -> int:
    """CLI: sin --verificar construye y escribe; --verificar solo audita.

    Contrato: make() es la única ruta que escribe artefactos; verificar no
    escribe. Código de retorno 0 = OK, 1 = fallo (errores en pantalla).
    """
    ap = _ParserEspanol(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True, metavar="~DIR",
                    help="directorio con los CSV crudos de ISIC 2019 de training (GT y metadata)")
    ap.add_argument("--out-dir", type=Path, required=True, metavar="~DIR_OUT",
                    help="directorio donde se escriben train/val/test.csv y summary.json")
    ap.add_argument("--exclude-csv", type=Path, required=True, metavar="~CSV",
                    help="CSV con columna image_id a excluir del split (lista de subconjuntos excluidos)")
    ap.add_argument("--verificar", type=Path, default=None, metavar="~JSON",
                    help="ruta a summary.json; corre verificaciones sin escribir")
    args = ap.parse_args()
    if args.verificar is not None:
        return verify_outputs(args.verificar, args.exclude_csv)
    return make(args.data_dir, args.out_dir, args.exclude_csv)


if __name__ == "__main__":
    sys.exit(main())
