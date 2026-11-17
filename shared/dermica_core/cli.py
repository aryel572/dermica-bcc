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
"""

import argparse
import sys


class ParserEspanol(argparse.ArgumentParser):
    """ArgumentParser con los mensajes de error en español neutro.

    argparse redacta por defecto en inglés; el pipeline de la tesis es en
    español, así que los fallos de invocación se traducen aquí (solo el
    texto mostrado, nunca el comportamiento ni los códigos de salida:
    sigue siendo exit 2 ante argumentos inválidos o faltantes).
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
        """
        self.print_usage(sys.stderr)
        if "the following arguments are required" in message:
            faltan = message.split(":", 1)[1].strip()
            self.exit(2, f"ERROR: faltan los flags obligatorios: {faltan}\n"
                         f"Consulta --help para la ayuda completa en español.\n")
        self.exit(2, f"ERROR: {message}\nConsulta --help para la ayuda completa en español.\n")
