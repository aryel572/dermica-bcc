#!/usr/bin/env python3
"""
Genera los archivos `ground_truth.json` de los subconjuntos de demostración de tesis.

Cada subconjunto (carpetas `TESIS/subset_XX/`) necesita su referencia de diagnóstico
para que la aplicación móvil pueda calcular TP/TN/FP/FN al procesar el subconjunto
completo. El JSON resultante es deliberadamente mínimo: solo
`{nombre_de_archivo: diagnóstico_de_referencia}` (etiqueta ISIC completa, por
ejemplo "BCC", "NV", "MEL"...). El público común no usará esta funcionalidad;
existe únicamente para la demostración de los indicadores de la tesis.

Los JSON se escriben dos veces con el mismo script:
  1. Dentro de cada carpeta de subconjunto (junto a las imágenes; la app las lee
     desde la carpeta elegida en el modo "Subir subconjunto").
  2. En el repositorio (FRONTEND/ground-truth/), como registro versionado.

Ningún archivo de imagen ISIC se copia ni se commitea; solo texto.
"""

import argparse
import csv
import json
from pathlib import Path


def cargar_manifest(ruta_manifest: Path) -> dict[str, dict[str, str]]:
    """Agrupa el manifest por subconjunto: {subset_id: {nombre_archivo: etiqueta}}.

    Devuelve los mismos datos verificados en S2 (450 filas, 15 por subconjunto,
    etiquetas ISIC completas). Falla rápido si una fila no trae etiqueta.
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True,
                        help='Ruta al tesis_manifest.csv (fuera del repo, con datos locales).')
    parser.add_argument('--out-dir', type=Path, required=True,
                        help='Directorio donde escribir los <subset_id>.json (o <carpeta>/ground_truth.json).')
    parser.add_argument('--como-ground-truth', action='store_true', default=False,
                        help='Escribe un único ground_truth.json dentro de cada subcarpeta de <out-dir>.')
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
