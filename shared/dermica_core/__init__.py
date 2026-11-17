"""Núcleo compartido de APP_DERMICA (paquete `dermica-core`).

Qué es: la librería única con los símbolos que usan dos o más productos del
monorepo —etiquetas de ISIC 2019, hashing SHA-256, parámetros del release,
lectura validada de los CSV de splits y el ArgumentParser en español neutro
que antes estaba copiado byte a byte en los diez módulos CLI de `pipeline/`.

Por qué existe: la tesis debe poder auditar una sola fuente por símbolo; las
copias divergentes amordazan ese requisito. Este paquete es la única verdad.

Uso típico:
    from dermica_core import labels, hashing
    digest = hashing.sha256_file(ruta)

`SHORT_CIRCUIT_OK` es un centinela de humo: tras `uv sync`, el comando
`python -c "import dermica_core; print(dermica_core.SHORT_CIRCUIT_OK)"` debe
imprimir True. No tiene más significado que "el paquete importó bien".
"""

# Import único de los submódulos: los consumidores usan `dermica_core.io`,
# `dermica_core.labels`, etc., o `from dermica_core.labels import ISIC_LABELS`.
from . import cli, hashing, io, labels, params

# Centinela de humo para verificaciones de instalación (ver docstring del módulo).
SHORT_CIRCUIT_OK = True

__all__ = ["cli", "hashing", "io", "labels", "params", "SHORT_CIRCUIT_OK"]
__version__ = "1.0.0"
