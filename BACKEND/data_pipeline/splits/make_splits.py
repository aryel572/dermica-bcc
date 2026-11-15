#!/usr/bin/env python3
"""S2 paso 2: split 70/20/10 estratificado por BCC/no-BCC y agrupado por lesion_id.

Lee DATA_CRUDA (solo lectura) y resultados/s2-tesis/tesis_exclude.csv; nunca
toca el TEST de ISIC 2019. Las imágenes sin lesion_id se tratan como singletons
(cada imagen es su propio grupo). El muestreo usa la semilla 42 de
BACKEND/data_pipeline/config.json, StratifiedGroupKFold(n_splits=10,
shuffle=True, random_state=42) y una reunión de folds 7/2/1 para obtener
train/val/test = 70/20/10.

Artefactos (fuera del repo): train.csv, val.csv, test.csv, summary.json,
README-corrida.md — columnas: image, split, label_full, label_bcc, lesion_id.
Nunca escribe en DATA_CRUDA ni en TESIS/; solo lectura.

Uso:
    uv run python make_splits.py            # construye el split y verifica inline
    uv run python make_splits.py --verificar resultados/s2-splits/summary.json
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Raíz del monorepo APP_DERMICA: este archivo vive en BACKEND/data_pipeline/splits/,
# por lo que parents[4] resuelve a .../APP_DERMICA (dos repos DATA_CRUDA y resultados/
# cuelgan de ella; si el árbol del repo cambia, ajustar aquí).
APP_ROOT = Path(__file__).resolve().parents[4]  # .../APP_DERMICA
DEFAULT_DATA_DIR = APP_ROOT / "DATA_CRUDA"
DEFAULT_OUT_DIR = APP_ROOT / "resultados" / "s2-splits"
EXCLUDE_CSV = APP_ROOT / "resultados" / "s2-tesis" / "tesis_exclude.csv"
# parents[1] = BACKEND/data_pipeline: config.json vive junto a audit/ y tesis/.
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"
GT_TRAIN = "ISIC_2019_Training_GroundTruth.csv"
META_TRAIN = "ISIC_2019_Training_Metadata.csv"
# Columnas one-hot de clases ISIC 2019 (un 1.0 por fila). El orden es fijo para
# que idxmax(...) de más abajo sea determinista aunque haya empates.
CLASS_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
# Tasa BCC auditada del train de ISIC 2019 (13.12%, ver S1 en CRISP-DM):
# referencia de sanity check, no compuerta de fallo (el split puede desviarse ±1%).
BCC_TARGET_PCT = 13.12


def load_frames(data_dir: Path) -> pd.DataFrame:
    """Carga y une GT + metadata de TRAIN de ISIC 2019 (solo lectura).

    Params: data_dir — directorio con DATA_CRUDA (GT y metadata train).
    Returns: DataFrame con columnas image, lesion_id, label_full, label_bcc, group.

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

    Cada compuerta cubre un modo de fuga/defecto distinto:
    - intersección de imágenes entre particiones -> misma foto en train y val/test
      (fuga directa de datos): debe ser ∅.
    - lesion_id compartido entre particiones -> imágenes de la misma lesión en
      distintas particiones, el clásico fallo de "multiple image lesions" de
      ISIC; garantiza que val/test miden generalización real.
    - TESIS presente -> imágenes del conjunto excluido contaminando train/val/test.
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
        problems.append(f"{len(tesis_leak)} imágenes de TESIS presentes en una partición")

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

    lesion_id ausente se escribe como cadena vacía (no "NaN" ni null), de modo
    que el consumidor (S2-paso 3 / entrenamiento) siempre vea strings. El orden
    de columnas es el contrato: image, split, label_full, label_bcc, lesion_id.
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

    Contrato: si cualquier compuerta inline falla, NO se escribe ningún
    artefacto (evita summary/CSVs engañosos a medio construir); mensajes de
    error listan los problemas uno a uno. Datos crudos = solo lectura.
    """
    t0 = time.time()
    cfg = json.loads(CONFIG_PATH.read_text())
    seed = int(cfg["seed"])

    if not exclude_csv.exists():
        print(f"ERROR: no existe la lista de exclusión TESIS: {exclude_csv}")
        return 1
    exclude = set(pd.read_csv(exclude_csv)["image_id"])
    print(f"exclusión TESIS: {len(exclude)} imágenes")

    df = load_frames(data_dir)
    before = len(df)
    df = df[~df["image"].isin(exclude)].reset_index(drop=True)
    print(f"train GT {before} -> sin TESIS {len(df)}")

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


def verify_outputs(summary_path: Path) -> int:
    """Lee artefactos ya escritos y verifica conformidad (no escribe nada).

    Reconstruye el concat train+val+test y comprueba: cardinalidad y
    cardinalidades por partición contra summary.json, valores de split válidos,
    columnas esperadas, TESIS ausente, image únicos y ninguna lesión en más de
    un split; ratios con aviso ±1%. Devuelve 0/1 según OK/FALLO.
    """
    cfg = json.loads(CONFIG_PATH.read_text())
    if not summary_path.exists():
        print(f"FAIL: no existe {summary_path}")
        return 1
    s = json.loads(summary_path.read_text())
    out_dir = summary_path.parent
    df = pd.concat([pd.read_csv(out_dir / f"{p}.csv") for p in ("train", "val", "test")],
                   ignore_index=True)
    exclude_path = EXCLUDE_CSV
    exclude = set(pd.read_csv(exclude_path)["image_id"]) if exclude_path.exists() else set()

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
        problems.append("TESIS presente en una partición")
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--exclude-csv", type=Path, default=EXCLUDE_CSV)
    ap.add_argument("--verificar", type=Path, default=None,
                    help="ruta a summary.json; corre verificaciones sin escribir")
    args = ap.parse_args()
    if args.verificar is not None:
        return verify_outputs(args.verificar)
    return make(args.data_dir, args.out_dir, args.exclude_csv)


if __name__ == "__main__":
    sys.exit(main())
