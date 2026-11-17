"""Lectura validada de los CSV de splits de la fase S2.

Qué contiene: `leer_csv_csv_splits(path)` — lee un CSV de partición
(train/val/test) generado por `pipeline/splits/make_splits.py` y valida sus
encabezados antes de entregar el DataFrame.

Contrato del archivo (lo escribe write_outputs en make_splits.py y lo esperan
los consumidores): columnas exactamente `image, split, label_full,
label_bcc, lesion_id` — en cualquier orden, como conjunto; de más de una fila
aceptada no se exige orden porque el escritor ya las persiste en orden fijo,
pero los consumidores operan por nombre de columna.

Por qué validar aquí y no en cada llamador: hoy los consumidores
(`pipeline/prep/resize_224.py`, `resize_224_multi.py`, `resize_224_test.py`)
hacían `pd.read_csv` directo y cualquier drift de columnas explotaba después
como KeyError opaco en medio de la preparación. Un error claro y temprano en la
lectura es la frontera correcta.

Errores
-------
FileNotFoundError
    Si `path` no existe (p. ej. splits no generados aún).
pd.errors.ParserError / UnicodeDecodeError
    Si el archivo no es CSV parseable.
ValueError
    Si los encabezados no son exactamente los esperados; el mensaje lista
    las columnas esperadas y las que faltan/sobran.

Efectos colaterales
-------------------
Ninguno (solo lectura). pandas se importa de forma perezosa dentro de la
función: el core sigue siendo instalable con cero dependencias externas (los
productos que llaman a esta función ya declaran pandas).
"""

from pathlib import Path

# Encabezados exactos del contrato de splits (ver docstring del módulo).
# Conjunto, no lista: el orden no forma parte del contrato perceptible.
ENCABEZADOS_SPLITS = frozenset({"image", "split", "label_full", "label_bcc", "lesion_id"})


def leer_csv_csv_splits(path):
    """Lee un CSV de splits y devuelve su contenido validado como DataFrame.

    Parámetros
    ----------
    path : str | Path
        Ruta al CSV de partición (p. ej. `<splits-dir>/train.csv`). Archivo en
        solo lectura; no se escribe nada.

    Retorno
    -------
    pd.DataFrame
        DataFrame con las columnas validadas `image, split, label_full,
        label_bcc, lesion_id` (mismas filas que el archivo).

    Errores
    -------
    ValueError
        Si los encabezados del archivo no coinciden exactamente con el
        contrato; el mensaje detalla faltantes y sobrantes.

    Efectos colaterales
    -------------------
    Ninguno (solo lectura).
    """
    import pandas as pd  # import perezoso: mantiene el core sin dependencias

    path = Path(path)
    df = pd.read_csv(path)
    presentes = set(df.columns)
    faltan = ENCABEZADOS_SPLITS - presentes
    sobran = presentes - ENCABEZADOS_SPLITS
    if faltan or sobran:
        detalle = []
        if faltan:
            detalle.append(f"faltan columnas: {sorted(faltan)}")
        if sobran:
            detalle.append(f"columnas inesperadas: {sorted(sobran)}")
        raise ValueError(
            f"{path}: encabezados inválidos para un CSV de splits "
            f"({'; '.join(detalle)}). Se esperaban exactamente: "
            f"{sorted(ENCABEZADOS_SPLITS)}"
        )
    return df
