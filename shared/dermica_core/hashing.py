"""Hashing SHA-256 con chunks de 1 MiB.

Qué contiene: `sha256_bytes(data)` para bytes en memoria y `sha256_file(path)`
para archivos leídos por streaming.

Por qué 1 MiB: los artefactos que se hashean (manifiestos, modelos TFLite de
~2 MB, CSVs GT de decenas de MB) caben sin problema en chunks de 1 MiB y la
memoria queda O(1 MiB) incluso sobre archivos muy grandes — es el
comportamiento que ya usaban las copias que este módulo unifica.

Consumidores (medidos): `pipeline/audit/audit_isic.py` y
`pipeline/tesis/form_tesis.py` (tenían su propio `sha256_file` idéntico) y
`pipeline/prep/export_float16.py` (mismo patrón en línea para el .tflite).

Qué recibe y qué produce: no lee flags ni escribe artefactos; expone dos
funciones de hashing importadas por los consumidores. Fase del pipeline:
soporte transversal de la auditoría S1 a la exportación S4 (los manifiestos y
verificaciones que sella cada fase dependen de digests idénticos byte a byte).

Digest: siempre hexadecimal en minúsculas (64 caracteres).
"""

import hashlib
from pathlib import Path

# Tamaño de chunk de lectura: 1 MiB (1 << 20). No exponer como knob: cada
# llamada debe comportarse idéntica para que los manifiestos sean comparables.
# Lo consume sha256_file(); cambiarlo no altera el digest (SHA-256 es inmune al
# chunking), solo el perfil de memoria/tiempo de la lectura.
_CHUNK = 1 << 20


def sha256_bytes(data: bytes) -> str:
    """Devuelve el digest SHA-256 hexadecimal de un objeto de bytes.

    Parámetros
    ----------
    data : bytes
        Contenido a hashear. Se consume completo una vez.

    Retorno
    -------
    str
        Digest hex en minúsculas (64 chars).

    Errores
    -------
    Ninguno: `hashlib.sha256` opera sobre bytes ya en memoria.

    Globales
    --------
    Ninguna (depende solo del módulo `hashlib`).
    """
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    """Devuelve el digest SHA-256 hexadecimal de un archivo, en chunks de 1 MiB.

    Parámetros
    ----------
    path : str | Path
        Archivo a hashear. Se lee completo una vez; no se modifica nada.

    Retorno
    -------
    str
        Digest hex en minúsculas (64 chars).

    Errores
    -------
    FileNotFoundError
        Si `path` no existe.
    OSError
        Ante fallos de E/S durante la lectura.

    Efectos colaterales
    -------------------
    Ninguno (solo lectura). La lectura en chunks mantiene la memoria en
    O(1 MiB); el hash se calcula sobre el archivo COPIADO en disco, que es lo
    que manifiestos y verificaciones vuelven a comprobar.

    Globales
    --------
    `_CHUNK` (constante del módulo, 1 MiB); no toca ninguna otra global.
    """
    h = hashlib.sha256()
    with open(Path(path), "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()
