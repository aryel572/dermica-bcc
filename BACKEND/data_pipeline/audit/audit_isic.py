#!/usr/bin/env python3
"""S1 audit of the raw ISIC 2019 dataset (understanding phase, CRISP-DM).

Reads DATA_CRUDA in place: zips are never extracted, CSVs are parsed with
pandas. Writes every artifact to resultados/s1-audit-isic/ (outside the repo).

WHAT THIS SCRIPT PROVES (and what it deliberately does NOT do)
--------------------------------------------------------------
This is the "data understanding" gate for the thesis pipeline. Before any
split, subset (TESIS) or model is formed, we need machine-checked evidence
that the raw release on disk is complete and self-consistent:

  * The training/test zips contain exactly the official image counts.
  * Every image id in the ground truth (GT) exists in the zip and vice versa
    (strict 1:1 matching, zero duplicates on either side).
  * The one-hot class columns sum to exactly one label per row (single-label
    invariant of the ISIC 2019 GT format).
  * The metadata CSVs are parseable and their missing-value profile is known.
  * The four CSVs and every artifact we produce are pinned by SHA-256 so the
    audit is reproducible and tamper-evident (`--verificar` re-checks them).
  * CRC integrity of both zips is verified end-to-end via `unzip -t`.
    (`zipfile.namelist()` does NOT verify CRCs — it only reads the central
    directory — so a separate full pass is required.)

It NEVER extracts the archives, NEVER writes into DATA_CRUDA/ (the dataset is
read-only for this project), and NEVER touches the ISIC 2019 test set for any
purpose other than auditing it. The test split is not used for training,
splitting or subset formation anywhere in the pipeline (see thesis pipeline
docs: only the training partition, grouped by lesion_id, enters the 70/20/10
split; the official test set stays untouched until final evaluation).

OUTPUTS (all under --out-dir, default results/s1-audit-isic/)
-------------------------------------------------------------
  auditoria.json     machine-readable audit report (invariants + hashes)
  clases.csv         class distribution per split (counts and percentages)
  lesion_id.csv      images per lesion_id (training only; absent ids skipped)
  dimensiones.csv    sampled image widths/heights/modes (see SAMPLE_SEED)
  figuras/clases.png per-class bar charts
  figuras/faltantes.png missing-values bar charts
  README-corrida.md  human-readable run notes with the exact commands used

Usage:
    uv run python audit_isic.py                 # full audit
    uv run python audit_isic.py --verificar resultados/s1-audit-isic/auditoria.json
"""

import argparse
import hashlib
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

# Headless backend MUST be selected before pyplot is imported, otherwise
# matplotlib tries to load a GUI backend and crashes on machines without a
# display (Colab/CI/servers). "Agg" renders to files only.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# Repository layout: APP_DERMICA/ is the container; dermica-bcc/ is a git repo
# inside it, and this file lives at BACKEND/data_pipeline/audit/. parents[0]
# is audit/, parents[1] data_pipeline/, parents[2] BACKEND/, parents[3]
# dermica-bcc/, parents[4] APP_DERMICA/ (the workspace root that holds
# DATA_CRUDA/ and resultados/). Resolved from __file__ so the script works
# regardless of the current working directory.
APP_ROOT = Path(__file__).resolve().parents[4]  # .../APP_DERMICA
DEFAULT_DATA_DIR = APP_ROOT / "DATA_CRUDA"
DEFAULT_OUT_DIR = APP_ROOT / "resultados" / "s1-audit-isic"

# Canonical ISIC 2019 file names inside DATA_CRUDA/. These are the exact
# names shipped by the official release; the audit asserts nothing about
# their content beyond what the checks below measure.
TRAIN_ZIP = "ISIC_2019_Training_Input.zip"
TEST_ZIP = "ISIC_2019_Test_Input.zip"
GT_TRAIN = "ISIC_2019_Training_GroundTruth.csv"
GT_TEST = "ISIC_2019_Test_GroundTruth.csv"
META_TRAIN = "ISIC_2019_Training_Metadata.csv"
META_TEST = "ISIC_2019_Test_Metadata.csv"

# Official ISIC 2019 table sizes. Hard-coding them turns the audit from a
# "report generator" into an actual gate: any missing/extra image fails the
# run instead of silently propagating into splits and subsets.
EXPECTED = {"training": 25331, "test": 8238}

# One-hot column order exactly as shipped by the ISIC 2019 GroundTruth CSVs:
# MEL (melanoma), NV (nevus), BCC (basal cell carcinoma), AK (actinic
# keratosis), BKL (benign keratosis-like), DF (dermatofibroma), VASC
# (vascular lesions), SCC (squamous cell carcinoma), UNK (unknown). Note SCC
# and UNK appear in the GT even if rare/absent per split; the class column
# list is intersected with whatever is actually present (see load_gt).
CLASS_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]

# Seed for the dimension/mode sample (~600 images). 42 is the project-wide
# fixed seed (same value used for TESIS subset formation and downstream
# splits) so every stochastic choice in the pipeline is reproducible from the
# same constant and auditable in the thesis. Changing it would silently
# change the sample, so the seed itself is part of the artifact contract:
# dimensiones.csv always describes the same images for a given dataset.
SAMPLE_SEED = 42


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
    None (read-only). Chunked reading keeps memory O(1 MiB) even for large
    files, which matters if this ever runs against the zips themselves.
    1 << 20 = 1 MiB: large enough to amortize syscalls, small enough to stay
    cache-friendly.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unzip_test(zip_path: Path) -> tuple[bool, str, float]:
    """Full CRC check via `unzip -t`. Returns (ok, tail_of_output, seconds).

    Why the system `unzip` and not stdlib zipfile: listing or extracting with
    Python's zipfile verifies CRCs only when bytes are actually read (on
    extract or on `ZipFile.read`), and never for metadata-only operations.
    `unzip -t` is the canonical end-to-end integrity check for these
    multi-GB archives and requires no disk for extraction. This function
    shells out to it and measures wall-clock time.

    Parameters
    ----------
    zip_path : Path
        Archive to test. Not modified, nothing is written to disk.

    Returns
    -------
    tuple[bool, str, float]
        (True if return code is 0, last non-empty output line for the audit
        log, elapsed seconds). On failure the tuple still returns so the
        caller can decide how to report; the caller asserts on the flag.

    Side effects
    ------------
    Spawns a subprocess (`unzip -t`). No files are created or deleted; the
    archive is opened read-only.
    """
    start = time.perf_counter()
    proc = subprocess.run(["unzip", "-t", str(zip_path)], capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    out = (proc.stdout or "") + (proc.stderr or "")
    tail = out.strip().splitlines()[-1] if out.strip() else ""
    return proc.returncode == 0, tail, elapsed


def zip_entries(zip_path: Path) -> tuple[list[str], int]:
    """List image entries and total entries of a zip without extraction.

    Parameters
    ----------
    zip_path : Path
        Archive to inspect. Read-only.

    Returns
    -------
    tuple[list[str], int]
        (stems of all entries whose name ends in .jpg, case-insensitive,
        total number of entries in the archive including non-jpg/directories).
        The stem (file name without extension) is what matches the GT
        `image` column in ISIC 2019.

    Side effects
    ------------
    Reads the central directory only; no CRC verification here (that is
    unzip_test's job) and no bytes of image content are decompressed.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    return [Path(n).stem for n in names if n.lower().endswith(".jpg")], len(names)


def load_gt(csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Parse a GroundTruth CSV and resolve which class columns are present.

    Parameters
    ----------
    csv_path : Path
        Path to ISIC_2019_*_GroundTruth.csv.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        (full GT frame, CLASS_COLS intersected with columns actually present,
        preserving CLASS_COLS order). The intersection tolerates GT releases
        that omit classes (e.g. a split with zero SCC rows may drop the
        column) without failing; the invariant that matters — each row has
        exactly one hot label — is enforced by class_distribution().

    Side effects
    ------------
    None (read-only CSV parse).
    """
    gt = pd.read_csv(csv_path)
    class_cols = [c for c in CLASS_COLS if c in gt.columns]
    return gt, class_cols


def class_distribution(gt: pd.DataFrame, class_cols: list[str]) -> pd.DataFrame:
    """Compute per-class counts and percentages from the one-hot GT.

    This is also where the single-label invariant is enforced: ISIC 2019 GT
    rows carry exactly one hot class column and zeros elsewhere. If a CSV
    were multi-label or corrupted, the column sums would not equal the row
    count and the assertion below fails fast instead of silently producing a
    wrong distribution.

    Parameters
    ----------
    gt : pd.DataFrame
        Ground truth frame; must contain all `class_cols` and the `image` id
        column (not used here, but required by callers for zip matching).
    class_cols : list[str]
        One-hot columns to aggregate, typically the output of load_gt.

    Returns
    -------
    pd.DataFrame
        Columns `clase`, `conteo`, `porcentaje` (percent of GT rows).

    Side effects
    ------------
    None beyond building the small result frame.
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
    """Run the full per-split audit and return its results.

    Checks, in order:
      1. zip entry list vs GT rows (exact official counts, EXPECTED).
      2. duplicate image ids in GT and in the zip (must be 0).
      3. strict 1:1 set matching between GT ids and zip jpg stems, listing up
         to a few offending ids in the assertion message for debugging.
      4. per-class distribution (delegated to class_distribution).
      5. metadata CSV parse + per-column missing counts.

    Parameters
    ----------
    name : str
        Split name, exactly "training" or "test"; indexes EXPECTED and is
        echoed into the report.
    data_dir : Path
        Root of DATA_CRUDA (used for reporting paths; the zip/gt/meta paths
        are passed explicitly as absolute paths).
    zip_path : Path
        The ISIC 2019 *_Input.zip for this split.
    gt_path : Path
        The matching *_GroundTruth.csv.
    meta_path : Path
        The matching *_Metadata.csv.
    out_dir : Path
        Unused in this function (kept in the signature for symmetry with the
        rest of the pipeline and future artifacts); no files are written here.
    crc : tuple[bool, str, float]
        Pre-computed result of unzip_test for this zip (run once per split in
        main to avoid double work and to fail before the expensive audit).

    Returns
    -------
    dict
        Machine-readable per-split report: sizes, CRCs, duplicates, missing
        counts, plus the embedded DataFrames `dist`, `meta`, `gt` (used by
        main for figures/JSON; stripped before serialization). The `gt_zip_1a1`,
        `gt_sin_zip` and `zip_sin_gt` fields hard-code the post-assertion
        state: reaching the return implies 1:1 with zero misses.

    Side effects
    ------------
    Reads the zip (central directory only) and two CSVs. Asserts abort the
    whole run on any invariant violation; this is intentional — a failed
    invariant means the dataset on disk is not what the thesis assumes.
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

    # Hard gates. These are the properties the rest of the pipeline
    # (grouping by lesion_id, stratified splits, TESIS subsets) silently
    # relies on; failing here is cheaper than debugging a skewed model later.
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
        # After the asserts above these are invariantly True/0; they are
        # serialized so the JSON records the checked property explicitly
        # instead of implying it from the absence of errors.
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
    """Sample images per split and record width/height/mode.

    Deterministic: the RNG is seeded with SAMPLE_SEED and entries are sorted
    before sampling, so the SAME images are measured on every run (rng.sample
    on the same sorted list with the same seed is stable across machines and
    Python versions for random.Random). Sorting is what makes the sample
    independent of zip internal ordering.

    Parameters
    ----------
    splits : dict
        Output of main's `audit_split` calls per split (needs each split's
        "zip" file name).
    data_dir : Path
        Root of DATA_CRUDA.
    out_dir : Path
        Destination of dimensiones.csv.
    sample_per_split : int
        Max images to open per split (default 300 in main → ~600 total).
        If a split has fewer jpgs than requested, all are sampled.

    Returns
    -------
    pd.DataFrame
        Columns image, split, width, height, mode — one row per sampled
        image. Also written to out_dir/dimensiones.csv (index=False).

    Side effects
    ------------
    Opens and decodes (headers + pixels) the sampled images from inside the
    zips via zf.open; writes dimensiones.csv. Does NOT extract archives.
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
    """Run the full audit and write all artifacts; return process exit code.

    Exit code 0 means every invariant held and artifacts were written. Any
    failed assert or exception exits non-zero (traceback), which is the
    intended failure mode for CI/manual runs: the audit is all-or-nothing.

    Side effects
    ------------
    Creates --out-dir and --out-dir/figuras/ if missing; writes
    auditoria.json, clases.csv, lesion_id.csv, dimensiones.csv,
    figuras/*.png and README-corrida.md; prints progress to stdout.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--sample-per-split", type=int, default=300)
    ap.add_argument("--verificar", type=Path, default=None)
    args = ap.parse_args()

    if args.verificar:
        return verificar(args.verificar)

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
    # CRC is the most expensive check; if it fails we stop here instead of
    # burning time on the rest of the audit over a corrupt archive.
    assert crc_train[0] and crc_test[0], "CRC failed (unzip -t); see output above"

    print("[3/6] Auditando GT/metadata/zips...")
    splits = {
        "training": audit_split("training", data_dir, data_dir / TRAIN_ZIP, data_dir / GT_TRAIN, data_dir / META_TRAIN, out_dir, crc_train),
        "test": audit_split("test", data_dir, data_dir / TEST_ZIP, data_dir / GT_TEST, data_dir / META_TEST, out_dir, crc_test),
    }

    # lesion_id exists only in the TRAINING metadata (ISIC 2019 test metadata
    # has no lesion_id — lesions there are unknown). This block computes the
    # lesion-level statistics that justify grouping by lesion_id instead of
    # image_id in the 70/20/10 split: images from the same lesion are
    # near-duplicates, and letting them fall into different partitions would
    # leak visually identical content across train/validation.
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
        # Terracotta for training, blue for test: same palette semantics as
        # the rest of the project's figures (train/test must never be
        # confused visually).
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
        # Barh labels read top-to-bottom; inverting puts the first column
        # (alphabetical by CSV order) on top, matching the CSV for humans.
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
        # Strip the embedded DataFrames before serialization: dist/meta/gt are
        # kept in-memory only for figures and summaries, not in the JSON.
        "zips": {
            name: {k: v for k, v in s.items() if k not in ("dist", "meta", "gt")} for name, s in splits.items()
        },
        "clases_train": dict(zip(splits["training"]["dist"]["clase"], splits["training"]["dist"]["conteo"])),
        "clases_test": dict(zip(splits["test"]["dist"]["clase"], splits["test"]["dist"]["conteo"])),
        "faltantes_metadata": {name: s["meta_faltantes"] for name, s in splits.items()},
        "lesion_id_training": lesion_summary,
        "dimensiones_muestra": dims_summary,
        # Invariants are stored explicitly, not recomputed on read: the JSON
        # is the signed receipt that the dataset satisfied the thesis
        # preconditions at generation time.
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
    # Hash every artifact we promise, so --verificar can detect any drift or
    # manual edit after the fact.
    audit["artefactos"] = {p.relative_to(out_dir).as_posix(): sha256_file(p) for p in artifact_paths}
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")

    # self-check: re-read what we just wrote. Cheap integrity tripwire — if
    # the serializer mangled anything or an assert was stripped (python -O),
    # this fails before the README is written.
    check = json.loads(audit_path.read_text())
    assert check["invariantes"]["train_imagenes"] == EXPECTED["training"]
    assert check["invariantes"]["test_imagenes"] == EXPECTED["test"]
    assert check["invariantes"]["gt_zip_1a1"] is True
    assert check["invariantes"]["crc_ok"] is True

    write_readme(out_dir, audit)
    print(f"Listo en {timings['total_seg']}s -> {out_dir}")
    return 0


def verificar(audit_path: Path) -> int:
    """Re-verify a previously written auditoria.json against disk.

    Re-derives nothing from the dataset except the four input CSVs: it checks
    (a) all required keys are present, (b) the serialized invariants match
    the official counts, (c) the input CSVs still hash to the recorded
    digests, (d) every listed artifact still exists and hashes to its
    recorded digest. This is the reproducibility receipt for the audit; it
    must pass on the machine that produced the JSON (and does not depend on
    the zips' CRCs being re-run).

    Parameters
    ----------
    audit_path : Path
        Path to auditoria.json. Its parent directory is taken as out_dir and
        the `data_dir` recorded inside the JSON as the location of the input
        CSVs (falling back to DEFAULT_DATA_DIR if absent).

    Returns
    -------
    int
        0 if every check passes, 1 otherwise. Individual checks are printed
        as PASS/FAIL lines; all checks run even after a failure (no early
        exit) so the summary shows the full damage.

    Side effects
    ------------
    Read-only: hashes the CSVs and artifacts listed in the JSON.
    """
    audit_path = audit_path.resolve()
    out_dir = audit_path.parent
    data_dir = Path(json.loads(audit_path.read_text()).get("data_dir", str(DEFAULT_DATA_DIR)))
    ok = True

    def check(label: str, cond: bool) -> None:
        # nonlocal accumulator: collect every verdict, print each one, never
        # short-circuit — the summary must list ALL failures, not just the
        # first.
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

    # Re-hash inputs and artifacts; both checks compare against digests
    # recorded at generation time. A CSV or artifact edited after the audit
    # fails here even if every count still looks right.
    for name, digest in audit.get("hashes_sha256_csv", {}).items():
        p = data_dir / name
        check(f"SHA-256 {name}", p.exists() and sha256_file(p) == digest)

    for rel, digest in audit.get("artefactos", {}).items():
        p = out_dir / rel
        check(f"artefacto {rel}", p.exists() and sha256_file(p) == digest)

    print("RESULTADO:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def write_readme(out_dir: Path, audit: dict) -> None:
    """Write README-corrida.md — a human-readable receipt of this audit run.

    The README records the exact commands, environment versions, timings and
    input hashes so the audit can be reproduced or cited from the thesis
    without opening the JSON. Contents come from the audit dict (never
    recomputed), so the README always matches auditoria.json.

    Parameters
    ----------
    out_dir : Path
        Destination directory (README-corrida.md is created there).
    audit : dict
        The in-memory audit report (same object serialized to
        auditoria.json); reads hashes, timings, environment and out_dir.

    Side effects
    ------------
    Writes out_dir/README-corrida.md.
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
        f"cd {APP_ROOT / 'dermica-bcc' / 'BACKEND' / 'data_pipeline' / 'audit'}",
        "uv venv",
        "uv pip install -r requirements.txt",
        "uv run python audit_isic.py",
        f"uv run python audit_isic.py --verificar {out_dir / 'auditoria.json'}",
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
        "- `dimensiones.csv` — muestra de 600 imágenes (width/height/mode), semilla 42",
        "- `figuras/clases.png`, `figuras/faltantes.png`",
        "",
        "CC BY-NC 4.0 — datos ISIC 2019 (International Skin Imaging Collaboration). Solo auditoría; ningún dato se redistribuye.",
    ]
    (out_dir / "README-corrida.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
