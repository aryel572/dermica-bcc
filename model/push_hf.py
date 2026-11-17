#!/usr/bin/env python3
"""Subida/borrado de checkpoints y métricas a los repos HF privados del proyecto.

Repos: dermica-checkpoints-tmp (modelo+pesos de época `1.0.0-epN`, temporal),
dermica-metrics (JSONs `1.0.0-epN.json` y `1.0.0.json`), dermica-models
(versiones completas `1.0.0` sin sufijo). HF_TOKEN y HF_OWNER del archivo de
entorno se leen inline (sin dotenv, sin imprimir). La ruta del archivo de
entorno la decide quien ejecuta vía la variable DERMICA_ENV_FILE
(por defecto `./.env` en el directorio de trabajo); el repositorio nunca
resuelve rutas fuera del repositorio ni asume layout de entornos ajenos.

Reglas de vida de archivos:
- subir `1.0.0-epN` borra cualquier `-epM` con M < N ya superado;
- al subir la versión completa `1.0.0`, el último `-epN` se borra también;
- el borrado se hace en el repo tmp Y en el de métricas a la vez.

Uso CLI:
    uv run python push_hf.py latest [VERSION]   # tag más nuevo en tmp (vacío si no hay)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

def _env_file() -> Path:
    """Ruta del archivo de entorno: variable DERMICA_ENV_FILE o (por defecto)
    `./.env` en el directorio de trabajo actual. El módulo nunca resuelve
    rutas desde fuera del repositorio ni asume un layout de trabajo ajeno.
    """
    ruta = os.environ.get('DERMICA_ENV_FILE')
    if ruta:
        return Path(ruta).expanduser()
    return Path('.env')

REPO_TMP = "dermica-checkpoints-tmp"      # repo temporal para checkpoints intermedios (borrar al completar)
REPO_METRICS = "dermica-metrics"          # repo de dataset con JSONs de métricas por versión/época
REPO_MODELS = "dermica-models"            # repo de modelo para releases completos (versión final sin -ep#)


def _read_env() -> dict:
    """Lee el archivo de entorno (DERMICA_ENV_FILE o `./.env`) sin dependencias.

    Efectos colaterales: ninguno; el dict queda en memoria y los valores se usan solo
    para construir el cliente HF (nunca se imprimen).
    """
    env_path = _env_file()
    env = {}
    if not env_path.is_file():
        raise SystemExit(f"ERROR: falta el archivo de entorno: {env_path}")
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _api() -> HfApi:
    """Construye el cliente HF con HF_TOKEN/HF_OWNER desde .env.

    Retorna: tuple (api, owner) — api autenticada y nombre/owner del org HF.
    Por qué SystemExit en vez de raise: es un fallo de entorno/configuración;
    el caller lo ve y aborta con mensaje claro, sin rastro de secreto.
    """
    env = _read_env()
    token = env.get("HF_TOKEN")
    owner = env.get("HF_OWNER")
    if not token or not owner:
        raise SystemExit("ERROR: HF_TOKEN u HF_OWNER sin definir en .env")
    return HfApi(token=token), owner


def latest_tag(version: str):
    """Devuelve el `VERSION-epN` con N máximo en el repo tmp; None si no hay.

    Parámetros: version — línea base, p.ej. "1.0.0".
    Retorna: str con `{version}-ep{N}`, o None si el repo tmp no tiene checkpoints.
    Por qué match de keras|weights: el repo puede contener archivos de contexto
    (p.ej. .meta.json) que no son checkpoints válidos para resume.
    """
    api, owner = _api()
    files = api.list_repo_files(f"{owner}/{REPO_TMP}", repo_type="model")
    best, best_n = None, -1
    for f in files:
        m = re.match(rf"^{re.escape(version)}-ep(\d+)\.(keras|weights\.h5)$", f)
        if m and int(m.group(1)) > best_n:
            best, best_n = f"{version}-ep{m.group(1)}", int(m.group(1))
    return best


def download(filename: str, dest: Path, repo: str = "tmp") -> None:
    """Descarga filename del repo indicado a dest ('tmp'|'metrics'|'models').

    Parámetros: filename — ruta relativa en el repo (p.ej. '1.0.0-ep5.weights.h5');
            dest — Path destino local; repo — repo lógico donde buscar.
    Efectos colaterales: escribe dest (tras normalizar el nombre bajo local_dir).
    Por qué hf_hub_download + rename: garantiza nombre exacto esperado por el
    resto del pipeline (latest_tag y verificaciones comparan nombres puntuales).
    """
    api, owner = _api()
    repo_id = {"tmp": REPO_TMP, "metrics": REPO_METRICS, "models": REPO_MODELS}[repo]
    api.hf_hub_download(f"{owner}/{repo_id}", repo_type="model" if repo != "metrics" else "dataset",
                        filename=filename, local_dir=str(dest.parent))
    # hf_hub_download deja el archivo bajo local_dir; normaliza el nombre
    src = dest.parent / filename
    if src != dest and src.exists():
        src.rename(dest)


def upload_checkpoint(paths: dict, tag: str, metrics_row: dict) -> None:
    """Sube {keras, weights} como `tag.*` al repo tmp (commit atómico).

    Parámetros: paths — dict con keys 'keras' y 'weights' (Path); tag — `X.Y.Z-epN`;
            metrics_row — dict logs de la época (se filtra a floats para meta.json).
    Efectos colaterales: crea commit en dermica-checkpoints-tmp; escribe `tag.meta.json`,
    `tag.keras` y `tag.weights.h5`.
    Por qué commit único: sube modelo+pesos+meta juntos, evitando repo intermedio
    inconsistente si la sesión muere a mitad de subida.
    """
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
    """Sube/sobrescribe `tag.json` en dermica-metrics (dataset).

    Parámetros: metrics_path — ruta local del JSON; tag — `X.Y.Z-epN` o `X.Y.Z`.
    Efectos colaterales: crea/sobrescribe `tag.json` en dermica-metrics.
    Por qué sobrescribir: las métricas por época se actualizan con cada checkpoint
    nuevo; el archivo representa siempre el último estado reportado del tag.
    """
    api, owner = _api()
    from huggingface_hub import CommitOperationAdd
    api.create_commit(
        repo_id=f"{owner}/{REPO_METRICS}", repo_type="dataset",
        operations=[CommitOperationAdd(path_in_repo=f"{tag}.json", path_or_fileobj=str(metrics_path))],
        commit_message=f"métricas {tag}")


def delete_stale(version: str, current_epoch: int) -> None:
    """Borra en tmp y metrics los `-epN` con N inferior al actual.

    Parámetros: version — línea base (p.ej. '1.0.0'); current_epoch — épica actual (ya subida).
    Efectos colaterales: borra archivos `-ep<N<current_epoch>.*` en dermica-checkpoints-tmp
    y dermica-metrics (regla E6: solo vive el último checkpoint de época).
    Por qué borrar también en metrics: las métricas de épocas superadas no se
    conservan; el repo metrics solo guarda la versión final completa (1.0.0.json).
    """
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
    """Sube la versión completa (sin sufijo) a models + metrics y borra el último -epN.

    Parámetros: model — tf.keras.Model entrenado; final_metrics — dict de métricas finales;
            version — línea base (p.ej. '1.0.0'); out_dir — dir escribible para artefactos.
    Efectos colaterales: crea commits en dermica-models (`1.0.0.keras`), dermica-metrics
    (`1.0.0.json`), y borra en tmp y metrics todos los archivos `-epN` restantes.
    Por qué borrar al completar: la regla del proyecto exige que al final solo quede
    la versión completa + métricas permanentes; los checkpoints intermedios sobran.
    """
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
    """CLI: `latest` imprime el tag más nuevo en tmp (vacío si no hay).

    Parámetros: cmd — 'latest'; version — línea base (default '1.0.0').
    Retorna: 0 siempre; output stdout va a capturar por quien llama (resume logic).
    Por qué CLI mínimo: otras piezas (train_bcn) invocan funciones directamente;
    esto solo expone una consulta para depuración/scripts manuales.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["latest"])
    ap.add_argument("version", nargs="?", default="1.0.0")
    args = ap.parse_args()
    if args.cmd == "latest":
        print(latest_tag(args.version) or "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
