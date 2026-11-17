#!/usr/bin/env python3
"""Exporta los JSON de diagnóstico de referencia de los subconjuntos de demostración.

Propósito
---------
Genera un `ground_truth.json` por subconjunto: cada carpeta de subconjunto
(`<out-dir>/<subset_id>/`) recibe la referencia de diagnóstico que la
aplicación móvil usa para calcular TP/TN/FP/FN al procesar el subconjunto
completo. El JSON resultante es deliberadamente mínimo: solo
`{nombre_de_archivo: diagnóstico_de_referencia}` (etiqueta ISIC completa, por
ejemplo "BCC", "NV", "MEL"...). El público común no usa esta funcionalidad;
existe únicamente para la demostración de los indicadores de la tesis.

Los JSON se escriben con el mismo script en dos ubicaciones posibles:
  1. Dentro de cada carpeta de subconjunto (junto a las imágenes; la app las lee
     desde la carpeta elegida en el modo "Subir subconjunto"), con `--como-ground-truth`.
  2. En un directorio plano del repo (p. ej. el registro versionado del
     frontend), un `<subset_id>.json` por subconjunto.

Nada de imágenes se copia ni se commitea; este script solo produce texto.

Entradas
--------
* `--manifest ~CSV` (obligatorio): tesis_manifest.csv producido por la fase
  de formación de subconjuntos (origen externo al repo; solo lectura).
* `--out-dir ~DIR_OUT` (obligatorio): directorio destino de los JSON.

Salidas
-------
Un JSON por subconjunto, con dos layouts posibles:
  * plano (default): `<out-dir>/<subset_id>.json`
  * anidado (`--como-ground-truth`): `<out-dir>/<subset_id>/ground_truth.json`

Cubre el trabajo de la fase S2, paso de exportación de referencias (CRISP-DM);
provee el insumo de evaluación automática de los subconjuntos de demo.
"""

import argparse
import csv
import json
from pathlib import Path


class _ParserEspanol(argparse.ArgumentParser):
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


def cargar_manifest(ruta_manifest: Path) -> dict[str, dict[str, str]]:
    """Agrupa el manifiesto por subconjunto: {subset_id: {nombre_archivo: etiqueta}}.

    Parámetros
    ----------
    ruta_manifest : Path
        Ruta al tesis_manifest.csv (consultado por --manifest). Columnas
        esperadas: subset_id, image_id, label_full, sha256, original_path.

    Retorno
    -------
    dict[str, dict[str, str]]
        Por subconjunto, el mapa {nombre de archivo: etiqueta ISIC completa}.
        Devuelve los mismos datos verificados en la fase de verificación
        (450 filas, 15 por subconjunto, etiquetas ISIC completas).

    Errores
    -------
    ValueError
        Si una fila no trae etiqueta en `label_full` (manifiesto corrupto o
        editado a mano); el mensaje incluye la fila ofensiva.
    FileNotFoundError
        Si `ruta_manifest` no existe.

    Efectos colaterales
    -------------------
    Ninguno (solo lectura).
    """
    subconjuntos: dict[str, dict[str, str]] = {}
    with ruta_manifest.open(newline='', encoding='utf-8') as f:
        for fila in csv.DictReader(f):
            etiqueta = (fila.get('label_full') or '').strip()
            if not etiqueta:
                raise ValueError(f"Fila sin diagnóstico de referencia: {fila}")
            nombre = Path(fila['original_path']).name
            subconjuntos.setdefault(fila['subset_id'], {})[nombre] = etiqueta
    return subconjuntos


def main() -> None:
    """Punto de entrada CLI: parsea args, agrupa el manifiesto y escribe los JSON.

    Efectos colaterales
    -------------------
    Crea <out-dir> (y sus subcarpetas de subconjunto en modo anidado);
    escribe un JSON por subconjunto e imprime una línea por archivo escrito.

    Errores
    -------
    ValueError
        Propagado desde cargar_manifest si el manifiesto trae filas sin etiqueta.
    FileNotFoundError
        Si el manifiesto no existe.

    No retorna nada (exit 0 implícito tras escribir los JSON).
    """
    parser = _ParserEspanol(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, required=True, metavar='~CSV',
                        help='ruta al tesis_manifest.csv (fuera del repo, con los picks verificados)')
    parser.add_argument('--out-dir', type=Path, required=True, metavar='~DIR_OUT',
                        help='directorio donde escribir los JSON (ver --como-ground-truth para el layout)')
    parser.add_argument('--como-ground-truth', action='store_true', default=False,
                        help='escribe un único ground_truth.json dentro de cada subcarpeta <DIR_OUT>/<subset_id>/')
    args = parser.parse_args()

    subconjuntos = cargar_manifest(args.manifest)

    for subset_id, mapa in sorted(subconjuntos.items()):
        if args.como_ground_truth:
            # Un JSON dentro de la carpeta del subconjunto (modo demo de la app)
            destino = args.out_dir / subset_id / 'ground_truth.json'
        else:
            # Un JSON por subconjunto en un único directorio (registro versionado)
            destino = args.out_dir / f'{subset_id}.json'
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(mapa, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f'{destino}  ({len(mapa)} entradas)')


if __name__ == '__main__':
    main()
