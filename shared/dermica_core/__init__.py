"""Núcleo compartido de APP_DERMICA (paquete `dermica-core`).

Qué es: la librería única con los símbolos que usan dos o más productos del
monorepo —etiquetas de ISIC 2019, hashing SHA-256, parámetros del release,
lectura validada de los CSV de splits y el ArgumentParser en español neutro
que antes estaba copiado byte a byte en los diez módulos CLI de `pipeline/`.

Por qué existe: la tesis debe poder auditar una sola fuente por símbolo; las
copias divergentes amordazan ese requisito. Este paquete es la única verdad.

Qué recibe y qué produce: no lee flags ni escribe artefactos; re-exporta los
cinco submódulos y declara la versión del release. Fase del pipeline: S1–S5
(soporte transversal; todo el monorepo importa de aquí tras la fase de
extracción del core).

Uso típico:
    from dermica_core import labels, hashing
    digest = hashing.sha256_file(ruta)

`SHORT_CIRCUIT_OK` es un centinela de humo: tras `uv sync`, el comando
`python -c "import dermica_core; print(dermica_core.SHORT_CIRCUIT_OK)"` debe
imprimir True. No tiene más significado que "el paquete importó bien".
"""

# Import único de los submódulos: los consumidores usan `dermica_core.io`,
# `dermica_core.labels`, etc., o `from dermica_core.labels import ISIC_LABELS`.
# Cualquier símbolo nuevo del core se agrega aquí y en __all__ de forma atómica.
from . import cli, hashing, io, labels, params

# Centinela de humo para verificaciones de instalación (ver docstring del módulo).
# Consumidores: checks de instalación post-`uv sync`; nadie más lo lee.
SHORT_CIRCUIT_OK = True

# Contrato público del paquete: exactamente estos submódulos y el centinela.
# Agregar o retirar un símbolo aquí cambia la API del core — editar con
# intención, no por comodidad.
__all__ = ["cli", "hashing", "io", "labels", "params", "SHORT_CIRCUIT_OK"]

# Versión del release del monorepo (1.0.0); aún no se consume programáticamente
# (el release vive en el tag git), pero se declara para que cualquier logger o
# reporte pueda citar qué versión del core produjo una corrida.
__version__ = "1.0.0"
