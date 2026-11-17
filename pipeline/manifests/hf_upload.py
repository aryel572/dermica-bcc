#!/usr/bin/env python3
"""S2 paso 3: subida de manifiestos (solo texto) al repo remoto privado dermica-dataset.

Propósito
---------
Sube los manifiestos derivados de la fase de splits y de la formación de
subconjuntos, más la configuración y la tarjeta del dataset, al repositorio
git-remote privado `dermica-dataset` en Hugging Face (repo de tipo dataset).
Nunca sube imágenes ni binarios: únicamente CSV/JSON/MD.

Flujo de ejecución, de principio a fin:
1. Lee credenciales HF de las variables de entorno del proceso o de un
   archivo KEY=VALUE (`--env-file`, por defecto el .env del repo, solo
   lectura; nunca se imprime ni se copia), con un parser inline mínimo
   (sin dependencia dotenv ni subprocesos).
2. Valida con `api.whoami()` que el token autoriza contra la cuenta indicada.
3. Sube al repo git-remote: splits (train/val/test), manifiestos de
   subconjuntos y de exclusión, config.json y la tarjeta del dataset como
   README.md, vía un único `create_commit` (repo: dermica-dataset, tipo
   dataset).
4. La compuerta `--verificar` lista el repo remoto y exige el conjunto exacto
   de archivos esperados y 0 imágenes; está separada del upload para que
   cualquier miembro pueda corroborar el estado del repo en cualquier momento
   (antes/después de subir), sin riesgo de escribir nada.

Usa `create_commit` y no `upload_folder`: son pocos archivos pequeños y un
único commit atómico evita llamadas API adicionales (1 commit vs N uploads).
Jamás sube imágenes ni binarios ISIC. Nunca imprime el token ni rutas
absolutas de la máquina en la salida.

Entradas
--------
* `--env-file ~ENV` (opcional): archivo KEY=VALUE con HF_TOKEN, HF_OWNER y
  HF_REPO_DATASET; si falta un valor se toma de las variables de entorno del
  proceso. Si no existe, se ignoran (todo debe venir del entorno).
* `--splits-dir ~DIR` (obligatorio): directorio con train.csv, val.csv y
  test.csv de la fase de splits (origen externo al repo; SOLO LECTURA).
* `--tesis-dir ~DIR` (obligatorio): directorio con tesis_manifest.csv y
  tesis_exclude.csv (origen externo al repo; SOLO LECTURA).
* `--out-dir ~DIR_OUT` (obligatorio): destino del upload-log.json.

Salidas
-------
Un único commit atómico en el git-remote `dermica-dataset` (tipo dataset) y un
`upload-log.json` en `--out-dir` con el hash del commit, la URL del repo y la
lista de archivos esperados. Con `--verificar` no se escribe nada: solo se
audita el repo remoto.

Quién consume sus salidas
-------------------------
* El git-remote `dermica-dataset` → las fases posteriores del pipeline
  (modelado y evaluación) descargan de ahí los manifiestos y la
  configuración; la app móvil no lee este repositorio.
* `upload-log.json` → la evidencia de S2 paso 3 (hash del commit, URL del
  repo y lista de archivos esperados) y la auditoría manual posterior.
* `--verificar` → el equipo, antes/después de cada corrida, para corroborar
  el estado del repo remoto sin riesgo de escribir.

Uso:
    uv run python hf_upload.py --splits-dir ~DIR_SPLITS --tesis-dir ~DIR_TESIS --out-dir ~DIR_OUT
    uv run python hf_upload.py --splits-dir ~DIR_SPLITS --tesis-dir ~DIR_TESIS --out-dir ~DIR_OUT --verificar

Cubre el trabajo de la fase S2, paso 3 — publicación de manifiestos en el
git-remote del dataset (CRISP-DM).
"""

import argparse
from dermica_core.cli import ParserEspanol  # import único desde dermica_core
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.utils import HfHubHTTPError

# Raíz del repo (este archivo vive en pipeline/manifests/, dos
# niveles por debajo de la raíz). Solo se usa para el default repo-relativo de
# --env-file (el .env del repo, gitignored); nada del repo se modifica.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# La consume ENV_POR_DEFECTO (default repo-relativo de --env-file en main()).
ENV_POR_DEFECTO = REPO_ROOT / ".env"
# config.json vive junto al código de este pipeline (pipeline/) y forma
# parte del repo: se resuelve desde __file__ para que el CWD no importe.
# La consume collect_files().
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
# Tarjeta del dataset versionada en el repo (manifests/DATASET_CARD.md).
# La consume collect_files() para subirla como README.md del git-remote.
DATASET_CARD = Path(__file__).resolve().parent / "DATASET_CARD.md"
# Prefijos de carpeta DENTRO del repo git-remote (no son rutas locales):
# así se organizan los archivos en el repo dataset de Hugging Face.
# Los consumen collect_files(), EXPECTED y verificar().
SPLITS_PREFIX = "splits"
TESIS_PREFIX = "tesis"

SPLITS_FILES = ("train.csv", "val.csv", "test.csv")   # particiones 70/20/10 (las consume collect_files())
TESIS_FILES = ("tesis_manifest.csv", "tesis_exclude.csv")   # manifiestos de subconjuntos (los consume collect_files())
# Conjunto exacto esperado tras un upload correcto. Es una lista blanca:
# verificar contra EXPECTED detecta tanto faltantes como ruido en el repo.
# La consume verificar().
EXPECTED = (
    {f"{SPLITS_PREFIX}/{f}" for f in SPLITS_FILES}
    | {f"{TESIS_PREFIX}/{f}" for f in TESIS_FILES}
    | {"config.json", "README.md"}
)
# Extensiones de imagen prohibidas: ISIC nunca se redistribuye, ni siquiera por
# error. Si la compuerta detecta una, el upload estaba equivocado y hay que borrarla.
# La consume verificar().
FORBIDDEN_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}




def read_env(env_path: Path) -> dict:
    """Lee un archivo KEY=VALUE estilo dotenv (inline, sin dotenv ni subprocesos).

    Parámetros
    ----------
    env_path : Path
        Archivo KEY=VALUE de origen; si no existe, se devuelve {} (la
        resolución de credenciales cae entonces al entorno del proceso).

    Retorno
    -------
    dict
        Pares clave→valor con comillas simples/dobles desprendidas; el
        dict queda en memoria y los valores se usan solo para construir el
        cliente HF. Los valores con '#' literal no se soportan a propósito
        (la convención del proyecto es KEY=valor plano). NUNCA imprime nada.

    Efectos colaterales
    -------------------
    Ninguno (solo lectura; jamás imprime valores).

    Globales
    --------
    Ninguna (depende solo de `Path` y de texto; jamás imprime valores).
    """
    env = {}
    if not env_path.exists():
        return env
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def load_credentials(env_file: Path) -> tuple[str, str]:
    """Resuelve token + repo_id desde el entorno del proceso o el archivo --env-file.

    Prioridad: las variables del proceso ganan sobre el archivo (lo explícito
    domina al implícito). El archivo nunca se imprime ni se copia.

    Parámetros
    ----------
    env_file : Path
        Archivo KEY=VALUE consultado como segunda fuente de credenciales.

    Retorno
    -------
    tuple[str, str]
        (token, repo_id) donde repo_id es "<HF_OWNER>/<HF_REPO_DATASET>" del
        git-remote privado dermica-dataset (tipo dataset).

    Errores
    -------
    SystemExit
        Código 2 si falta HF_TOKEN, HF_OWNER o HF_REPO_DATASET en ambas
        fuentes (error de entorno/configuración).

    Globales
    --------
    Ninguna del módulo (lee el entorno del proceso `os.environ` y el archivo
    `--env-file`; el archivo nunca se imprime ni se copia).
    """
    file_env = read_env(env_file)
    merged = {**file_env, **{k: v for k, v in os.environ.items() if k in ("HF_TOKEN", "HF_OWNER", "HF_REPO_DATASET")}}
    missing = [k for k in ("HF_TOKEN", "HF_OWNER", "HF_REPO_DATASET") if not merged.get(k)]
    if missing:
        print(f"ERROR: faltan variables en el entorno o el archivo --env-file: {', '.join(missing)}")
        sys.exit(2)
    return merged["HF_TOKEN"], f"{merged['HF_OWNER']}/{merged['HF_REPO_DATASET']}"


def collect_files(splits_dir: Path, tesis_dir: Path) -> list[tuple[str, Path]]:
    """Devuelve [(ruta_remota, ruta_local)] de los manifiestos a subir.

    Parámetros
    ----------
    splits_dir : Path
        Directorio con train.csv, val.csv y test.csv de la fase de splits.
    tesis_dir : Path
        Directorio con tesis_manifest.csv y tesis_exclude.csv.

    Retorno
    -------
    list[tuple[str, Path]]
        (ruta_remota_dentro_del_git-remote, ruta_local). La ruta remota usa
        los prefijos SPLITS_PREFIX/TESIS_PREFIX dentro del repo dataset.

    Errores
    -------
    SystemExit
        Código 2 (entorno) si falta algún archivo fuente: mejor detenerse
        antes de commitear a medias que dejar el repo remoto inconsistente.

    Globales
    --------
    `SPLITS_FILES`, `TESIS_FILES`, `SPLITS_PREFIX`, `TESIS_PREFIX` (qué sube
    y bajo qué carpeta remota), `CONFIG_PATH` (config.json como contrato
    congelado) y `DATASET_CARD` (tarjeta versionada en el repo).
    """
    files = []
    for name in SPLITS_FILES:
        files.append((f"{SPLITS_PREFIX}/{name}", splits_dir / name))
    for name in TESIS_FILES:
        files.append((f"{TESIS_PREFIX}/{name}", tesis_dir / name))
    files.append(("config.json", CONFIG_PATH))
    files.append(("README.md", DATASET_CARD))
    for repo_path, local in files:
        if not local.exists():
            print(f"ERROR: falta el archivo fuente {local.name} ({repo_path})")
            sys.exit(2)
    return files


def upload(api: HfApi, repo_id: str, splits_dir: Path, tesis_dir: Path) -> dict:
    """Crea un único commit atómico con todos los manifiestos.

    Parámetros
    ----------
    api : HfApi
        Cliente autenticado contra Hugging Face (ver load_credentials()).
    repo_id : str
        "<owner>/<repo>" del git-remote privado dermica-dataset (tipo dataset).
    splits_dir : Path
        Directorio con las particiones 70/20/10 (entrada de collect_files()).
    tesis_dir : Path
        Directorio con los manifiestos de subconjuntos (entrada de collect_files()).

    Retorno
    -------
    dict
        {"commit": str, "repo": str, "url": str} del commit creado.

    Errores
    -------
    HfHubHTTPError
        Propagada desde la API del git-remote ante fallos de red, permisos o
        estado del repo (el llamador la deja ver con su mensaje).

    Efectos colaterales
    -------------------
    Escribe en el git-remote `dermica-dataset` (un único commit atómico).

    `create_commit` con 7 operaciones Add: una sola llamada API para todo el
    conjunto (upload_folder generaría más tráfico y menos control sobre qué
    entra exactamente). El mensaje va en español neutro, coherente con el repo.

    Globales
    --------
    Ninguna propia (delega en collect_files(); las operaciones Add son del
    cliente HF).
    """
    operations = [   # 7 operaciones Add (3 splits + 2 manifiestos TESIS + config + tarjeta) en un único commit atómico
        CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(local))
        for repo_path, local in collect_files(splits_dir, tesis_dir)
    ]
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message="Manifiestos ISIC 2019: splits, TESIS y tarjeta de dataset",
    )
    return {"commit": str(commit), "repo": repo_id, "url": f"https://huggingface.co/datasets/{repo_id}"}


def verificar(api: HfApi, repo_id: str) -> int:
    """Compuerta de conformidad: auth, archivos exactos, cero imágenes.

    Parámetros
    ----------
    api : HfApi
        Cliente autenticado contra Hugging Face.
    repo_id : str
        "<owner>/<repo>" del git-remote privado dermica-dataset (tipo dataset).

    Retorno
    -------
    int
        0 si el repo remoto cumple la lista blanca EXPECTED y no hay imágenes;
        1 en caso contrario (o si la autenticación falla).

    Efectos colaterales
    -------------------
    Ninguno (solo lectura contra el git-remote; nunca escribe).

    Se mantiene separado del upload a propósito: el equipo puede corroborar el
    estado del repo ANTES o DESPUÉS de subir (o tras un fallo parcial) sin
    riesgo de escribir. Falla (exit 1) si falta algún esperado o aparece una
    imagen; extra no bloquea (aviso informativo). La tarjeta lleva
    `license: cc-by-nc-4.0` en front-matter — convención de HF para metadatos
    de licencia; sin ella, hub avisa de metadata vacía aunque el contenido sea válido.

    Globales
    --------
    `EXPECTED` (lista blanca del conjunto exacto de archivos remotos) y
    `FORBIDDEN_SUFFIXES` (extensiones de imagen prohibidas en el repo).
    """
    problems = []
    try:
        who = api.whoami()
        print(f"auth OK (cuenta: {who.get('name', 'desconocida')})")
    except HfHubHTTPError as e:
        # Error de HF (token inválido / permisos), no de entorno: exit 1
        print(f"FAIL: autenticación inválida: {e}")
        return 1

    files = set(api.list_repo_files(repo_id, repo_type="dataset"))
    missing = EXPECTED - files   # lista blanca incompleta: bloquea (exit 1)
    if missing:
        problems.append(f"faltan archivos: {', '.join(sorted(missing))}")
    extra = files - EXPECTED   # ruido en el repo: aviso informativo, no bloquea
    if extra:
        print(f"AVISO: archivos extra en el repo (no bloquean): {', '.join(sorted(extra))}")
    imgs = sorted(f for f in files if Path(f).suffix.lower() in FORBIDDEN_SUFFIXES)   # cero imágenes: compuerta dura del repo del dataset
    if imgs:
        problems.append(f"se detectaron imágenes en el repo: {', '.join(imgs)}")

    for p in problems:
        print("FAIL", p)
    print("VERIFICACIÓN:", "OK" if not problems else "FALLO")
    if not problems:
        print(f"URL del repo: https://huggingface.co/datasets/{repo_id}")
    return 0 if not problems else 1


def main() -> int:
    """CLI: sube por defecto; `--verificar` solo audita (nunca escribe).

    Contrato: la única ruta que escribe en el git-remote es la llamada a
    upload(); --verificar es read-only. Código de retorno: 0 OK, 1 fallo de
    verificación o de HF, 2 error de entorno (faltan variables o fuente local).

    Efectos colaterales
    -------------------
    Sin --verificar: crea el commit en el git-remote y escribe
    <out-dir>/upload-log.json. Con --verificar: no escribe nada.

    Globales
    --------
    `ENV_POR_DEFECTO` (default repo-relativo de `--env-file`: el .env del
    repo, resuelto desde __file__; el archivo nunca se imprime ni se copia).
    """
    ap = ParserEspanol(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-dir", type=Path, required=True, metavar="~DIR_SPLITS",
                    help="directorio con train.csv, val.csv y test.csv de la fase de splits")
    ap.add_argument("--tesis-dir", type=Path, required=True, metavar="~DIR_TESIS",
                    help="directorio con tesis_manifest.csv y tesis_exclude.csv")
    ap.add_argument("--out-dir", type=Path, required=True, metavar="~DIR_OUT",
                    help="directorio donde se escribe upload-log.json")
    ap.add_argument("--env-file", type=Path, default=ENV_POR_DEFECTO, metavar="~ENV",
                    help="archivo KEY=VALUE con HF_TOKEN/HF_OWNER/HF_REPO_DATASET (default: .env de la raíz)")
    ap.add_argument("--verificar", action="store_true",
                    help="solo audita el repo remoto (sin escribir)")
    args = ap.parse_args()

    t0 = time.time()
    token, repo_id = load_credentials(args.env_file)
    api = HfApi(token=token)

    if args.verificar:
        return verificar(api, repo_id)

    print(f"subiendo manifiestos a {repo_id} (git-remote privado)...")
    result = upload(api, repo_id, args.splits_dir, args.tesis_dir)   # dict del commit; se enriquece con recibo (segundos, fecha, archivos esperados) para el upload-log.json
    result["segundos"] = round(time.time() - t0, 2)
    result["generado_utc"] = datetime.now(timezone.utc).isoformat()
    result["archivos"] = sorted(EXPECTED)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = args.out_dir / "upload-log.json"
    log.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"commit: {result['commit']}")
    print(f"repo: {result['repo']}")
    print(f"url: {result['url']}")
    print(f"log: {log.name}")
    print("Ejecuta --verificar antes de dar por bueno el paso.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
