#!/usr/bin/env python3
"""S2 paso 3: subida de manifiestos (solo texto) al repo privado dermica-dataset.

Flujo de ejecución, de principio a fin:
1. Lee credenciales HF de `dermica-bcc/.env` con un parser inline mínimo
   (sin dependencia dotenv ni subprocesos `sed -n 's/^KEY=//p' .env`: mismo
   patrón de extracción, sin imprimir jamás valores ni rutas absolutas).
2. Valida con `api.whoami()` que el token autoriza contra la cuenta indicada.
3. Sube al repo de datasets privado: `splits/{train,val,test}.csv`,
   `tesis/{tesis_manifest,tesis_exclude}.csv`, `config.json` y la tarjeta del
   dataset como `README.md` (desde DATASET_CARD.md) vía un único `create_commit`.
4. La compuerta `--verificar` lista el repo remoto y exige el conjunto exacto
   de archivos esperados y 0 imágenes; está separada del upload para que
   cualquier miembro pueda corroborar el estado del repo en cualquier momento
   (antes/después de subir), sin riesgo de escribir nada.

Usa `create_commit` y no `upload_folder`: son pocos archivos pequeños y un
único commit atómico evita llamadas API adicionales (1 commit vs N uploads).
Jamás sube imágenes ni binarios ISIC: únicamente CSV/JSON/MD. Nunca imprime el
token ni rutas absolutas de la máquina en la salida.

Uso:
    uv run python hf_upload.py             # crea el commit con los manifiestos
    uv run python hf_upload.py --verificar # comprueba contenido remoto (sin escribir)
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.utils import HfHubHTTPError

# Monorepo: este archivo vive en BACKEND/data_pipeline/manifests/;
# parents[4] resuelve a .../APP_DERMICA (raíz que contiene DATA_CRUDA/ y dermica-bcc/).
APP_ROOT = Path(__file__).resolve().parents[4]  # .../APP_DERMICA
ENV_PATH = APP_ROOT / "dermica-bcc" / ".env"
# parents[1] = BACKEND/data_pipeline: config.json vive junto al código
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"
DATASET_CARD = Path(__file__).resolve().parent / "DATASET_CARD.md"
SPLITS_DIR = APP_ROOT / "resultados" / "s2-splits"
TESIS_DIR = APP_ROOT / "resultados" / "s2-tesis"
DEFAULT_OUT_DIR = APP_ROOT / "resultados" / "s2-hf"

SPLITS_FILES = ("train.csv", "val.csv", "test.csv")
TESIS_FILES = ("tesis_manifest.csv", "tesis_exclude.csv")
# Conjunto exacto esperado tras un upload correcto. Es una lista blanca:
# verificar contra EXPECTED detecta tanto faltantes como ruido en el repo.
EXPECTED = (
    {f"splits/{f}" for f in SPLITS_FILES}
    | {f"tesis/{f}" for f in TESIS_FILES}
    | {"config.json", "README.md"}
)
# Extensiones de imagen prohibidas: ISIC nunca se redistribuye, ni siquiera por
# error. Si la compuerta detecta una, el upload estaba equivocado y hay que borrarla.
FORBIDDEN_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


def read_env(env_path: Path) -> dict:
    """Lee .env KEY=VALUE sin dependencias ni subprocesos.

    Equivalente al patrón `sed -n 's/^KEY=//p' .env` pero inline: ignora líneas
    vacías y comentarios (#), desprende comillas simples/dobles de los valores y
    JAMÁS imprime nada — el dict queda en memoria y los valores se usan solo
    para construir el cliente HF. Los valores con '#' literal no se soportan a
    propósito (la convención local del proyecto es KEY=valor plano).
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


def load_credentials() -> tuple[str, str]:
    """Resuelve token + repo_id. Falla con exit 2 si falta algo en .env.

    exit 2 = error de entorno/configuración (faltan variables); es distinto
    del exit 1 de --verificar (repo remoto en mal estado) y de los errores
    propios de HF, que se reportan como FAIL con su mensaje.
    """
    env = read_env(ENV_PATH)
    missing = [k for k in ("HF_TOKEN", "HF_OWNER", "HF_REPO_DATASET") if not env.get(k)]
    if missing:
        print(f"ERROR: faltan variables en .env: {', '.join(missing)}")
        sys.exit(2)
    return env["HF_TOKEN"], f"{env['HF_OWNER']}/{env['HF_REPO_DATASET']}"


def collect_files() -> list[tuple[str, Path, str]]:
    """Devuelve [(repo_path, local_path)] de los manifiestos a subir.

    Falla con exit 2 (entorno) si falta algún fuente: mejor detenerse antes de
    commitear a medias que dejar el repo HF inconsistente o engañoso.
    """
    files = []
    for name in SPLITS_FILES:
        files.append((f"splits/{name}", SPLITS_DIR / name))
    for name in TESIS_FILES:
        files.append((f"tesis/{name}", TESIS_DIR / name))
    files.append(("config.json", CONFIG_PATH))
    files.append(("README.md", DATASET_CARD))
    for repo_path, local in files:
        if not local.exists():
            print(f"ERROR: falta el archivo fuente {local.name} ({repo_path})")
            sys.exit(2)
    return files


def upload(api: HfApi, repo_id: str) -> dict:
    """Crea un único commit atómico con todos los manifiestos.

    `create_commit` con 7 operaciones Add: una sola llamada API para todo el
    conjunto (upload_folder generaría más tráfico y menos control sobre qué
    entra exactamente). El mensaje va en español neutro, coherente con el repo.
    """
    operations = [
        CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(local))
        for repo_path, local in collect_files()
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

    Se mantiene separado del upload a propósito: el equipo puede corroborar el
    estado del repo HF antes o después de subir (o tras un fallo parcial) sin
    riesgo de escribir. Falla (exit 1) si falta algún esperado o aparece una
    imagen; extra no bloquea (aviso informativo). La tarjeta lleva
    `license: cc-by-nc-4.0` en front-matter — convención de HF para metadatos
    de licencia; sin ella, hub avisa de metadata vacía aunque el contenido sea válido.
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
    missing = EXPECTED - files
    if missing:
        problems.append(f"faltan archivos: {', '.join(sorted(missing))}")
    extra = files - EXPECTED
    if extra:
        print(f"AVISO: archivos extra en el repo (no bloquean): {', '.join(sorted(extra))}")
    imgs = sorted(f for f in files if Path(f).suffix.lower() in FORBIDDEN_SUFFIXES)
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

    Contrato: la única ruta que escribe en HF es la llamada a upload();
    --verificar es read-only. Código de retorno: 0 OK, 1 fallo de verificación
    o de HF, 2 error de entorno (faltan variables o fuente local).
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verificar", action="store_true", help="solo audita el repo remoto")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    t0 = time.time()
    token, repo_id = load_credentials()
    api = HfApi(token=token)

    if args.verificar:
        return verificar(api, repo_id)

    print(f"subiendo manifiestos a {repo_id} (privado)...")
    result = upload(api, repo_id)
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
