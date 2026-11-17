"""ArgumentParser con mensajes de error en español neutro.

Qué contiene: `ParserEspanol`, el único parser CLI del proyecto. argparse
redacta por defecto en inglés; el pipeline de la tesis es en español, así que
los fallos de invocación se traducen aquí (solo el texto mostrado, nunca el
comportamiento ni los códigos de salida: sigue siendo exit 2 ante argumentos
inválidos o faltantes).

Por qué vive en el core: medido con checksum sobre los 17 módulos Python del
árbol, esta clase estaba copiada byte a byte en los diez módulos CLI de
`pipeline/` (audit, manifests, prep ×5, splits, tesis ×2). Una divergencia
futura en cualquiera de las copias no la detectaría nadie; una sola definición
sí.

Consumidores (medidos): `pipeline/audit/audit_isic.py`,
`pipeline/manifests/hf_upload.py`, `pipeline/prep/export_float16.py`,
`pipeline/prep/export_tflite.py`, `pipeline/prep/resize_224.py`,
`pipeline/prep/resize_224_multi.py`, `pipeline/prep/resize_224_test.py`,
`pipeline/splits/make_splits.py`, `pipeline/tesis/export_ground_truth.py`,
`pipeline/tesis/form_tesis.py`.

Qué recibe y qué produce: no lee archivos ni produce artefactos; provee la
clase base que cada módulo CLI instancia dentro de su `main()` para parsear sus
flags `--*`. Fase del pipeline: soporte transversal desde la auditoría S1 hasta
la exportación S4 (todos los módulos invocables con `--help` de la tesis).
"""

import argparse
import sys


class ParserEspanol(argparse.ArgumentParser):
    """ArgumentParser con los mensajes de error en español neutro.

    argparse redacta por defecto en inglés; el pipeline de la tesis es en
    español, así que los fallos de invocación se traducen aquí (solo el
    texto mostrado, nunca el comportamiento ni los códigos de salida:
    sigue siendo exit 2 ante argumentos inválidos o faltantes).

    Parámetros
    ----------
    Los mismos de argparse.ArgumentParser (prog, description, epilog, ...):
    no agrega argumentos propios ni cambia ningún default; es un drop-in
    transparente del parser estándar. Los módulos CLI del pipeline la
    instancian con `description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter`
    para que `--help` muestre la docstring del módulo tal cual.

    Errores
    -------
    No define una ruta de error propia: argparse invoca `error()` ante
    cualquier fallo de parseo (flag desconocido, valor inválido, faltante
    obligatorio), y ese método siempre termina con exit 2.

    Globales
    --------
    Tránsito por `sys.stderr` dentro de `error()`; ninguna otra global.
    """

    def error(self, message: str) -> None:
        """Imprime el uso y un error en español; sale con código 2.

        Parámetros
        ----------
        message : str
            Detalle redactado por argparse (se conserva el dato útil, p. ej.
            los nombres de los flags faltantes, pero se enmarca en español).

        Retorno
        -------
        No retorna: termina el proceso con exit 2.

        Errores
        -------
        No lanza excepciones controladas: argparse llama a este método ante
        cualquier invocación inválida y `self.exit(2, ...)` termina el proceso.

        Globales
        --------
        `sys` (escribe en sys.stderr y llama a sys.exit vía argparse).
        """
        self.print_usage(sys.stderr)
        if "the following arguments are required" in message:
            faltan = message.split(":", 1)[1].strip()
            self.exit(2, f"ERROR: faltan los flags obligatorios: {faltan}\n"
                         f"Consulta --help para la ayuda completa en español.\n")
        self.exit(2, f"ERROR: {message}\nConsulta --help para la ayuda completa en español.\n")
