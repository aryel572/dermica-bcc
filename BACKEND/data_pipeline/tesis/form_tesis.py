#!/usr/bin/env python3
"""S2 step 1: TESIS selection from raw ISIC 2019 (preparation phase, CRISP-DM).

Forms TESIS: 30 subsets x 15 images sampled from the ISIC 2019 training set
with a fixed seed from BACKEND/data_pipeline/config.json. Each subset holds
>= min_bcc_per_subset BCC images (guided sampling, not raw random). Images may
repeat across subsets. The TEST split of ISIC 2019 is never touched.

WHAT TESIS IS AND WHY IT EXISTS
-------------------------------
TESIS is the demo/evidence pack for the thesis: 30 small, self-contained
subsets of 15 dermoscopic images each (450 picked images, images may repeat
across subsets). It exists so the final app can be demonstrated and scored on
images that are pinned, hashed and excluded from training — otherwise any
"demo accuracy" number would be meaningless because the model may have
memorized those images during fine-tuning.

Three consequences of that purpose drive every design decision here:

  1. Reproducibility: the subset selection is fully determined by the fixed
     project seed (config.json -> "seed": 42). Re-running this script on the
     same dataset yields the identical manifest. 42 is the project-wide seed
     (audits, splits, training) so one constant explains the whole pipeline
     in the thesis. Any change to sampling order, list contents or RNG calls
     changes subsets — which is why the code is append-only stable and must
     never be "cleaned up" cosmetically.
  2. Exclusion from training: every picked id is listed in
     `tesis_exclude.csv`. Downstream split formation (70/20/10, grouped by
     lesion_id, stratified) MUST exclude these ids so TESIS stays a clean,
     never-trained probe set. The same fixed seed is what lets any third
     party regenerate this exact exclusion list and verify it (seed ->
     subsets -> exclusion list is a pure function of the raw dataset).
  3. Guided sampling, not raw random: BCC is ~13.1% of the training set
     (3,323 / 25,331). A raw uniform 15-image subset would carry a BCC
     count distributed around ~2 per subset, frequently 0-1 — useless for a
     demo whose purpose is BCC screening. Instead each subset is composed
     of exactly `min_bcc_per_subset` BCC images plus the remainder drawn
     from the non-BCC pool, i.e. we sample the composition, not the
     fairness. This is deliberate oversampling of the positive class in the
     demo pack ONLY; it has no effect on training data or metrics.

This script reads ONLY the training zip and training GT/metadata CSVs. The
ISIC 2019 TEST split is never opened, listed, hashed or copied — it has no
lesion_id, ~24.9% UNK labels and is reserved exclusively for final
evaluation. Do not add any code path that references the test files here.

OUTPUTS (under --out-dir, default resultados/s2-tesis/)
-------------------------------------------------------
  tesis_manifest.csv  one row per (subset_id, image_id) pick with label,
                      sha256 of the copied file and original zip entry path
  tesis_exclude.csv   deduplicated image_id list to exclude from training
                      splits (columns image_id, label_full)
  summary.json        run receipt: seed, config, counts, min BCC per subset,
                      class distribution of picks, artifact provenance
  README-corrida.md   (written by a follow-up step, not by this script)

The copied images land in TESIS/<subset_id>/<isic_id>.jpg (under the workspace
root, outside the repo), never in .git — see .gitignore rules for TESIS/.

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

# Repository layout: APP_DERMICA/ is the workspace container; dermica-bcc/ is
# the git repo inside it and this file lives at BACKEND/data_pipeline/tesis/.
# parents[0]=tesis/, parents[1]=data_pipeline/, parents[2]=BACKEND/,
# parents[3]=dermica-bcc/, parents[4]=APP_DERMICA/ (holds DATA_CRUDA/,
# resultados/, TESIS/). Resolved from __file__ so CWD does not matter.
APP_ROOT = Path(__file__).resolve().parents[4]  # .../APP_DERMICA

# config.json sits next to this pipeline (data_pipeline/config.json). It is
# the single source of truth for the seed and the TESIS shape; do not inline
# these values — the thesis cites config.json as the frozen contract.
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"
DEFAULT_DATA_DIR = APP_ROOT / "DATA_CRUDA"
DEFAULT_OUT_DIR = APP_ROOT / "resultados" / "s2-tesis"
DEFAULT_TESIS_DIR = APP_ROOT / "TESIS"
TRAIN_ZIP = "ISIC_2019_Training_Input.zip"
GT_TRAIN = "ISIC_2019_Training_GroundTruth.csv"
META_TRAIN = "ISIC_2019_Training_Metadata.csv"

# One-hot column order exactly as shipped by the ISIC 2019 GroundTruth CSV.
# Used for label resolution via idxmax (the GT is single-label: exactly one
# column is hot per row, enforced upstream by the S1 audit). SCC is present
# in the GT format although unused elsewhere in the pipeline.
CLASS_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]

# Top-level folder INSIDE the training zip; entries are addressed as
# f"{ZIP_PREFIX}{image_id}.jpg". The zip is never mass-extracted — each
# picked entry is streamed individually (see the copy loop in form()).
ZIP_PREFIX = "ISIC_2019_Training_Input/"


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, streamed in 1 MiB chunks.

    Parameters
    ----------
    path : Path
        File to hash. Read fully once; nothing is modified.

    Returns
    -------
    str
        Lowercase hex digest (64 chars).

    Side effects
    ------------
    None (read-only). Chunked reading keeps memory O(1 MiB); the hash is
    computed on the COPIED file on disk (not on the zip entry bytes in
    memory), so the manifest proves what was actually written to TESIS/,
    which is what --verificar re-checks.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_train_labels(data_dir: Path) -> pd.Series:
    """Load training GT and resolve the single label per image.

    Parameters
    ----------
    data_dir : Path
        Root of DATA_CRUDA containing GT_TRAIN.

    Returns
    -------
    pd.Series
        Series indexed by image id (gt["image"]) whose values are the class
        name with the hot one-hot column. idxmax works because the GT is
        single-label (one column = 1 per row); ties are impossible by that
        invariant, and the S1 audit enforces it before this runs.

    Side effects
    ------------
    Read-only CSV parse. Does NOT open the ISIC test GT.
    """
    gt = pd.read_csv(data_dir / GT_TRAIN)
    labels = gt[CLASS_COLS].idxmax(axis=1)
    labels.index = gt["image"]
    return labels


def pick_subset(rng: random.Random, bcc_ids: list, rest_ids: list, n_bcc: int, n_total: int) -> list:
    """Draw one subset: n_bcc BCC ids + (n_total - n_bcc) non-BCC ids, shuffled.

    Parameters
    ----------
    rng : random.Random
        Seeded generator (seed 42 from config.json). The CALLER owns the RNG
        so all 30 subsets are produced by one deterministic sequence; each
        pick_subset call consumes rng state in a fixed order.
    bcc_ids : list
        All training image ids whose label is BCC (~3,323). Sampling is with
        replacement ACROSS subsets (subsets are drawn independently), so the
        same BCC image may appear in several subsets — intended by
        "repeat_across_subsets": true in config.json. Within one subset,
        rng.sample guarantees no duplicate.
    rest_ids : list
        All training image ids whose label is NOT BCC.
    n_bcc : int
        BCC images to draw (config "min_bcc_per_subset", default 5). This is
        the guided part: fixed positive-class composition per subset.
    n_total : int
        Subset size (config "images_per_subset", default 15).

    Returns
    -------
    list
        n_total image ids, shuffled so the manifest row order does not encode
        "first the BCC picks, then the rest" (the shuffled order is what the
        copy loop and the manifest preserve).

    Side effects
    ------------
    Advances rng state. Touches no files.
    """
    picks = rng.sample(bcc_ids, n_bcc) + rng.sample(rest_ids, n_total - n_bcc)
    rng.shuffle(picks)
    return picks


def form(data_dir: Path, out_dir: Path, tesis_dir: Path, force: bool) -> int:
    """Form the TESIS subsets, copy the images and write all artifacts.

    Steps:
      1. Read config.json (seed + TESIS shape). Note: config values are
         trusted, not validated — a config with subsets*images_per_subset
         mismatched against available BCC ids would fail at rng.sample with a
         clear error, which is acceptable for a one-shot pipeline step.
      2. Load training labels; split ids into BCC / rest pools.
      3. Refuse to overwrite a non-empty TESIS/ unless force=True — the
         guard exists so a re-run cannot silently mix stale and fresh
         copies under different seeds.
      4. Draw the 30 subsets from ONE seeded RNG (order: subset_01..30).
      5. Stream-copy each picked entry from the training zip into
         TESIS/<subset_id>/<image_id>.jpg (1 MiB chunks, never via extractall)
         and hash the copied file on disk.
      6. Write tesis_manifest.csv, tesis_exclude.csv (deduplicated ids) and
         summary.json; print a one-line receipt.

    Parameters
    ----------
    data_dir : Path
        Root of DATA_CRUDA.
    out_dir : Path
        Destination of the manifest/exclusion/summary artifacts.
    tesis_dir : Path
        Destination of the copied images (TESIS/<subset_id>/...).
    force : bool
        If True, allow writing into an existing non-empty tesis_dir.

    Returns
    -------
    int
        0 on success; 1 if tesis_dir exists and is non-empty without force.

    Side effects
    ------------
    Creates tesis_dir and out_dir; writes ~450+ jpg copies and three
    artifacts. Never writes into data_dir. Never opens the ISIC test zip or
    test CSVs.
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
    subset_picks = {}
    for i in range(1, n_subsets + 1):
        subset_picks[f"subset_{i:02d}"] = pick_subset(rng, bcc_ids, rest_ids, min_bcc, per_subset)

    # Copy images out of the training zip one entry at a time. No mass
    # extraction: only the picked (~450, fewer unique) images are written.
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

    # Deduplicate: the manifest carries 450 rows (30 x 15), but an image
    # repeated across subsets appears under several rows. The exclusion list
    # is over ids, so it collapses to the unique set; labels are looked up
    # from the GT series (same source as the manifest — consistent by
    # construction).
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
        # Post-conditions recorded explicitly so the receipt shows they were
        # checked at generation time, not merely hoped for: every subset
        # meets min_bcc and every picked id exists in the training GT.
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
    """Re-verify a tesis_manifest.csv against GT, config and the copied files.

    Checks (every check runs; no short-circuit, so ALL failures are printed):
      1. manifest has exactly subsets*images_per_subset rows (30*15=450);
      2. subset_id set is exactly subset_01..subset_<n_subsets>;
      3. every subset has exactly images_per_subset rows;
      4. every image_id exists in the training GT (no contamination from the
         test split or outside ids);
      5. manifest label_full matches the GT label for each id;
      6. every subset holds >= min_bcc BCC images (the guided-sampling
         contract);
      7. every copied file exists at TESIS/<subset_id>/<image_id>.jpg and its
         SHA-256 matches the manifest — i.e. the copies on disk are exactly
         what was picked, not stale overwrites from an older run.

    Why StratifiedGroupKFold (not a plain random shuffle) for the downstream
    split — context for what this manifest protects: after TESIS ids are
    excluded, the training partition still contains multiple images per
    lesion_id (same lesion, different frames). A random image-level shuffle
    would scatter near-identical lesion crops across train/val/test, leaking
    lesion-specific signal and inflating every metric. The split must group
    by lesion_id (no lesion in two partitions) and keep the BCC prevalence
    (~13.1%) balanced across partitions — hence a STRATIFIED, GROUPED fold
    (StratifiedGroupKFold), never RandomShuffleSplit. TESIS exclusion is the
    first half of that hygiene; this verifier is the receipt that TESIS is a
    clean held-out probe set.

    Note: --verificar reads DEFAULT_DATA_DIR and DEFAULT_TESIS_DIR (the repo
    defaults), NOT the --data-dir/--tesis-dir flags — verification is meant
    to check the canonical locations produced by a real run.

    Parameters
    ----------
    manifest_path : Path
        Path to tesis_manifest.csv produced by form().

    Returns
    -------
    int
        0 if every check passes, 1 otherwise.

    Side effects
    ------------
    Read-only: hashes the copied files under DEFAULT_TESIS_DIR and parses
    the GT CSV. Writes nothing.
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
        # Accumulator instead of early return: report EVERY failing check so
        # a broken manifest shows its full damage in one pass.
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
    """CLI entry point: parse args, dispatch to form() or verificar().

    Side effects
    ------------
    See form()/verificar(); nothing happens beyond argument parsing and
    delegation here.
    """
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
