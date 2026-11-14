#!/usr/bin/env python3
"""S2 step 1: TESIS selection from raw ISIC 2019 (preparation phase, CRISP-DM).

Forms TESIS: 30 subsets x 15 images sampled from the ISIC 2019 training set
with a fixed seed from BACKEND/data_pipeline/config.json. Each subset holds
>= min_bcc_per_subset BCC images (guided sampling, not raw random). Images may
repeat across subsets. The TEST split of ISIC 2019 is never touched.

Chosen images are copied out of the training zip with zipfile (no mass
extraction) into TESIS/<subset_id>/<isic_id>.jpg and SHA-256 hashed.
Artifacts go to resultados/s2-tesis/ (outside the repo):
tesis_manifest.csv, tesis_exclude.csv, summary.json, README-corrida.md.

Usage:
    uv run python form_tesis.py                 # form TESIS and write artifacts
    uv run python form_tesis.py --force         # overwrite existing TESIS/ copies
    uv run python form_tesis.py --verificar resultados/s2-tesis/tesis_manifest.csv
"""

import argparse
import csv
import hashlib
import json
import random
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[4]  # .../APP_DERMICA
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"
DEFAULT_DATA_DIR = APP_ROOT / "DATA_CRUDA"
DEFAULT_OUT_DIR = APP_ROOT / "resultados" / "s2-tesis"
DEFAULT_TESIS_DIR = APP_ROOT / "TESIS"
TRAIN_ZIP = "ISIC_2019_Training_Input.zip"
GT_TRAIN = "ISIC_2019_Training_GroundTruth.csv"
META_TRAIN = "ISIC_2019_Training_Metadata.csv"
CLASS_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
ZIP_PREFIX = "ISIC_2019_Training_Input/"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_train_labels(data_dir: Path) -> pd.Series:
    gt = pd.read_csv(data_dir / GT_TRAIN)
    labels = gt[CLASS_COLS].idxmax(axis=1)
    labels.index = gt["image"]
    return labels


def pick_subset(rng: random.Random, bcc_ids: list, rest_ids: list, n_bcc: int, n_total: int) -> list:
    picks = rng.sample(bcc_ids, n_bcc) + rng.sample(rest_ids, n_total - n_bcc)
    rng.shuffle(picks)
    return picks


def form(data_dir: Path, out_dir: Path, tesis_dir: Path, force: bool) -> int:
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
    subset_picks = {}
    for i in range(1, n_subsets + 1):
        subset_picks[f"subset_{i:02d}"] = pick_subset(rng, bcc_ids, rest_ids, min_bcc, per_subset)

    # copy images out of the training zip
    zip_path = data_dir / TRAIN_ZIP
    copied = 0
    rows = []
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

    unique_ids = sorted({r["image_id"] for r in rows})
    exclude = out_dir / "tesis_exclude.csv"
    ex_labels = labels[unique_ids]
    ex_unique = pd.DataFrame({"image_id": unique_ids,
                              "label_full": [ex_labels[i] for i in unique_ids]})
    ex_unique.to_csv(exclude, index=False)

    per_subset_bcc = {sid: sum(1 for r in rows if r["subset_id"] == sid and r["label_full"] == "BCC")
                      for sid in sorted(subset_picks)}
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


def verificar(manifest_path: Path) -> int:
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
        nonlocal ok
        print(("OK  " if cond else "FAIL") + "  " + msg)
        ok = ok and cond

    check(len(df) == n_subsets * per_subset, f"{n_subsets * per_subset} filas (450)")
    check(set(df["subset_id"]) == {f"subset_{i:02d}" for i in range(1, n_subsets + 1)},
          f"{n_subsets} subconjuntos válidos")
    check(df.groupby("subset_id").size().eq(per_subset).all(), f"{per_subset} imágenes por subconjunto")

    labels = load_train_labels(DEFAULT_DATA_DIR)
    check(df["image_id"].isin(labels.index).all(), "todas las image_id ⊆ ground truth train")
    check(labels[df["image_id"]].eq(df["label_full"].values).all(), "label_full coincide con el GT")

    bcc = df[df["label_full"] == "BCC"].groupby("subset_id").size()
    bcc = bcc.reindex([f"subset_{i:02d}" for i in range(1, n_subsets + 1)], fill_value=0)
    check(bcc.ge(min_bcc).all(), f"≥{min_bcc} BCC por subconjunto (min={int(bcc.min())})")

    tesis_dir = DEFAULT_TESIS_DIR
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--tesis-dir", type=Path, default=DEFAULT_TESIS_DIR)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verificar", type=Path, default=None,
                    help="ruta a tesis_manifest.csv; corre verificaciones sin escribir")
    args = ap.parse_args()
    if args.verificar is not None:
        return verificar(args.verificar)
    return form(args.data_dir, args.out_dir, args.tesis_dir, args.force)


if __name__ == "__main__":
    sys.exit(main())
