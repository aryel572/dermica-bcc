#!/usr/bin/env python3
"""Subida/borrado de checkpoints y métricas a los repos HF privados del proyecto.

Repos: dermica-checkpoints-tmp (modelo+pesos de época `1.0.0-epN`, temporal),
dermica-metrics (JSONs `1.0.0-epN.json` y `1.0.0.json`), dermica-models
(versiones completas `1.0.0` sin sufijo). HF_TOKEN del .env se lee inline
(sin dotenv, sin imprimir).

Reglas de vida de archivos:
- subir `1.0.0-epN` borra cualquier `-epM` con M < N ya superado;
- al subir la versión completa `1.0.0`, el último `-epN` se borra también;
- el borrado se hace en el repo tmp Y en el de métricas a la vez.

Uso CLI:
    uv run python push_hf.py latest [VERSION]   # tag más nuevo en tmp (vacío si no hay)
"""

import argparse
import json
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

APP_ROOT = Path(__file__).resolve().parents[3]  # .../APP_DERMICA
ENV_PATH = APP_ROOT / "dermica-bcc" / ".env"
REPO_TMP = "dermica-checkpoints-tmp"
REPO_METRICS = "dermica-metrics"
REPO_MODELS = "dermica-models"


def _read_env() -> dict:
    env = {}
    if not ENV_PATH.exists():
        raise SystemExit(f"ERROR: falta .env en dermica-bcc ({ENV_PATH.name})")
    for raw in ENV_PATH.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _api() -> HfApi:
    env = _read_env()
    token = env.get("HF_TOKEN")
    owner = env.get("HF_OWNER")
    if not token or not owner:
        raise SystemExit("ERROR: HF_TOKEN u HF_OWNER sin definir en .env")
    return HfApi(token=token), owner


def latest_tag(version: str):
    """Devuelve el tag `VERSION-epN` con N máximo en el repo tmp; None si no hay."""
    api, owner = _api()
    files = api.list_repo_files(f"{owner}/{REPO_TMP}", repo_type="model")
    best, best_n = None, -1
    for f in files:
        m = re.match(rf"^{re.escape(version)}-ep(\d+)\.(keras|weights\.h5)$", f)
        if m and int(m.group(1)) > best_n:
            best, best_n = f"{version}-ep{m.group(1)}", int(m.group(1))
    return best


def download(filename: str, dest: Path, repo: str = "tmp") -> None:
    """Descarga un archivo del repo indicado a dest (tmp/metrics/models)."""
    api, owner = _api()
    repo_id = {"tmp": REPO_TMP, "metrics": REPO_METRICS, "models": REPO_MODELS}[repo]
    api.hf_hub_download(f"{owner}/{repo_id}", repo_type="model" if repo != "metrics" else "dataset",
                        filename=filename, local_dir=str(dest.parent))
    # hf_hub_download deja el archivo bajo local_dir; normaliza el nombre
    src = dest.parent / filename
    if src != dest and src.exists():
        src.rename(dest)


def upload_checkpoint(paths: dict, tag: str, metrics_row: dict) -> None:
    """Sube {keras, weights} como `tag.*` al repo tmp (crea commit atómico)."""
    api, owner = _api()
    repo_id = f"{owner}/{REPO_TMP}"
    operations = []
    import huggingface_hub
    from huggingface_hub import CommitOperationAdd
    for kind, path in (("keras", paths["keras"]), ("weights", paths["weights"])):
        ext = ".keras" if kind == "keras" else ".weights.h5"
        operations.append(CommitOperationAdd(path_in_repo=f"{tag}{ext}", path_or_fileobj=str(path)))
    # subir también métricas asociadas como JSON de contexto de época
    meta_json = json.dumps(
        {"tag": tag, **{k: float(v) for k, v in metrics_row.items()
                        if isinstance(v, (int, float))}}, indent=2, ensure_ascii=False)
    meta_path = Path(paths["keras"]).parent / f"{tag}.meta.json"
    meta_path.write_text(meta_json)
    operations.append(CommitOperationAdd(path_in_repo=f"{tag}.meta.json", path_or_fileobj=str(meta_path)))
    api.create_commit(repo_id=repo_id, repo_type="model", operations=operations,
                      commit_message=f"checkpoint {tag}")


def upload_metrics_json(metrics_path: Path, tag: str) -> None:
    """Sube/sobreescribe `tag.json` en dermica-metrics (dataset)."""
    api, owner = _api()
    from huggingface_hub import CommitOperationAdd
    api.create_commit(
        repo_id=f"{owner}/{REPO_METRICS}", repo_type="dataset",
        operations=[CommitOperationAdd(path_in_repo=f"{tag}.json", path_or_fileobj=str(metrics_path))],
        commit_message=f"métricas {tag}")


def delete_stale(version: str, current_epoch: int) -> None:
    """Borra en tmp y metrics los `-epN` con N inferior al actual."""
    api, owner = _api()
    for repo, rtype in ((REPO_TMP, "model"), (REPO_METRICS, "dataset")):
        files = api.list_repo_files(f"{owner}/{repo}", repo_type=rtype)
        to_delete = []
        for f in files:
            m = re.match(rf"^{re.escape(version)}-ep(\d+)\.(keras|weights\.h5|meta\.json|json)$", f)
            if m and int(m.group(1)) < current_epoch:
                to_delete.append(f)
        if not to_delete:
            continue
        from huggingface_hub import CommitOperationDelete
        api.create_commit(repo_id=f"{owner}/{repo}", repo_type=rtype,
                          operations=[CommitOperationDelete(path_in_repo=f) for f in to_delete],
                          commit_message=f"borrar superados por ep{current_epoch}")


def upload_complete(model, final_metrics: dict, version: str, out_dir: Path) -> None:
    """Sube la versión completa (sin sufijo) a models + metrics y borra el último -epN."""
    api, owner = _api()
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    complete = out_dir / f"{version}.keras"
    model.save(complete)
    meta = out_dir / f"{version}.json"
    meta.write_text(json.dumps(
        {"tag": version, "seed": 42, **{k: float(v) for k, v in final_metrics.items()
                                          if isinstance(v, (int, float))}},
        indent=2, ensure_ascii=False))
    api.create_commit(repo_id=f"{owner}/{REPO_MODELS}", repo_type="model",
                      operations=[CommitOperationAdd(path_in_repo=f"{version}.keras", path_or_fileobj=str(complete))],
                      commit_message=f"release {version}")
    api.create_commit(repo_id=f"{owner}/{REPO_METRICS}", repo_type="dataset",
                      operations=[CommitOperationAdd(path_in_repo=f"{version}.json", path_or_fileobj=str(meta))],
                      commit_message=f"métricas {version}")
    # el último -epN se borra tras completar la versión
    tmp_files = api.list_repo_files(f"{owner}/{REPO_TMP}", repo_type="model")
    stale = [f for f in tmp_files if re.match(rf"^{re.escape(version)}-ep\d+\.", f)]
    if stale:
        api.create_commit(repo_id=f"{owner}/{REPO_TMP}", repo_type="model",
                          operations=[CommitOperationDelete(path_in_repo=f) for f in stale],
                          commit_message=f"borrar checkpoints tras release {version}")
    metrics_files = api.list_repo_files(f"{owner}/{REPO_METRICS}", repo_type="dataset")
    stale_m = [f for f in metrics_files if re.match(rf"^{re.escape(version)}-ep\d+\.json$", f)]
    if stale_m:
        api.create_commit(repo_id=f"{owner}/{REPO_METRICS}", repo_type="dataset",
                          operations=[CommitOperationDelete(path_in_repo=f) for f in stale_m],
                          commit_message=f"borrar métricas -epN tras release {version}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["latest"])
    ap.add_argument("version", nargs="?", default="1.0.0")
    args = ap.parse_args()
    if args.cmd == "latest":
        print(latest_tag(args.version) or "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
