#!/usr/bin/env python3
"""S1 audit of the raw ISIC 2019 dataset (understanding phase, CRISP-DM).

Reads DATA_CRUDA in place: zips are never extracted, CSVs are parsed with
pandas. Writes every artifact to resultados/s1-audit-isic/ (outside the repo).

Checks: image counts (GT vs zip entries), GT<->zip 1:1 matching, duplicate
image ids, class distribution (train/test), lesion_id coverage (train),
missing values per metadata column, image dimensions/modes on a sample of
~600 images, SHA-256 of the four CSVs. CRC is verified end to end with
`unzip -t` per zip.

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

APP_ROOT = Path(__file__).resolve().parents[4]  # .../APP_DERMICA
DEFAULT_DATA_DIR = APP_ROOT / "DATA_CRUDA"
DEFAULT_OUT_DIR = APP_ROOT / "resultados" / "s1-audit-isic"

TRAIN_ZIP = "ISIC_2019_Training_Input.zip"
TEST_ZIP = "ISIC_2019_Test_Input.zip"
GT_TRAIN = "ISIC_2019_Training_GroundTruth.csv"
GT_TEST = "ISIC_2019_Test_GroundTruth.csv"
META_TRAIN = "ISIC_2019_Training_Metadata.csv"
META_TEST = "ISIC_2019_Test_Metadata.csv"

EXPECTED = {"training": 25331, "test": 8238}
CLASS_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
SAMPLE_SEED = 42


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unzip_test(zip_path: Path) -> tuple[bool, str, float]:
    """Full CRC check via `unzip -t`. Returns (ok, tail_of_output, seconds)."""
    start = time.perf_counter()
    proc = subprocess.run(["unzip", "-t", str(zip_path)], capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    out = (proc.stdout or "") + (proc.stderr or "")
    tail = out.strip().splitlines()[-1] if out.strip() else ""
    return proc.returncode == 0, tail, elapsed


def zip_entries(zip_path: Path) -> tuple[list[str], int]:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    return [Path(n).stem for n in names if n.lower().endswith(".jpg")], len(names)


def load_gt(csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    gt = pd.read_csv(csv_path)
    class_cols = [c for c in CLASS_COLS if c in gt.columns]
    return gt, class_cols


def class_distribution(gt: pd.DataFrame, class_cols: list[str]) -> pd.DataFrame:
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
    assert crc_train[0] and crc_test[0], "CRC failed (unzip -t); see output above"

    print("[3/6] Auditando GT/metadata/zips...")
    splits = {
        "training": audit_split("training", data_dir, data_dir / TRAIN_ZIP, data_dir / GT_TRAIN, data_dir / META_TRAIN, out_dir, crc_train),
        "test": audit_split("test", data_dir, data_dir / TEST_ZIP, data_dir / GT_TEST, data_dir / META_TEST, out_dir, crc_test),
    }

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
        "zips": {
            name: {k: v for k, v in s.items() if k not in ("dist", "meta", "gt")} for name, s in splits.items()
        },
        "clases_train": dict(zip(splits["training"]["dist"]["clase"], splits["training"]["dist"]["conteo"])),
        "clases_test": dict(zip(splits["test"]["dist"]["clase"], splits["test"]["dist"]["conteo"])),
        "faltantes_metadata": {name: s["meta_faltantes"] for name, s in splits.items()},
        "lesion_id_training": lesion_summary,
        "dimensiones_muestra": dims_summary,
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
    audit["artefactos"] = {p.relative_to(out_dir).as_posix(): sha256_file(p) for p in artifact_paths}
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")

    # self-check: re-read what we just wrote
    check = json.loads(audit_path.read_text())
    assert check["invariantes"]["train_imagenes"] == EXPECTED["training"]
    assert check["invariantes"]["test_imagenes"] == EXPECTED["test"]
    assert check["invariantes"]["gt_zip_1a1"] is True
    assert check["invariantes"]["crc_ok"] is True

    write_readme(out_dir, audit)
    print(f"Listo en {timings['total_seg']}s -> {out_dir}")
    return 0


def verificar(audit_path: Path) -> int:
    audit_path = audit_path.resolve()
    out_dir = audit_path.parent
    data_dir = Path(json.loads(audit_path.read_text()).get("data_dir", str(DEFAULT_DATA_DIR)))
    ok = True

    def check(label: str, cond: bool) -> None:
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

    for name, digest in audit.get("hashes_sha256_csv", {}).items():
        p = data_dir / name
        check(f"SHA-256 {name}", p.exists() and sha256_file(p) == digest)

    for rel, digest in audit.get("artefactos", {}).items():
        p = out_dir / rel
        check(f"artefacto {rel}", p.exists() and sha256_file(p) == digest)

    print("RESULTADO:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def write_readme(out_dir: Path, audit: dict) -> None:
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
