"""Etiquetas de ISIC 2019 y criterio binario BCC vs no-BCC.

Qué contiene: el orden exacto de las columnas one-hot del archivo GroundTruth
de ISIC 2019 (`ISIC_LABELS`), la constante de la clase positiva del proyecto
(`BCC_LABEL`) y el predicado `is_bcc(name)`.

Por qué este orden importa: el GroundTruth es mono-etiqueta (una columna en 1
por fila) e `idxmax` resuelve la etiqueta a partir de la posición; reordenar la
lista cambia silenciosamente las etiquetas resueltas. Por eso la lista es una
constante congelada y no se deriva de ningún lado.

Consumidores (medidos): `pipeline/splits/make_splits.py`,
`pipeline/tesis/form_tesis.py`, `pipeline/audit/audit_isic.py` — los tres la
llamaban `CLASS_COLS`.
SCC está presente en el formato GT aunque el proyecto no lo use en ninguna otra
lógica: se conserva porque el orden del GT es el contrato.
"""

# Orden de columnas one-hot exactamente como lo entrega el CSV GroundTruth de
# ISIC 2019. La consume load_train_labels()/equivalentes vía idxmax.
ISIC_LABELS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]

# Clase positiva del problema de tamizaje (carcinoma basocelular).
BCC_LABEL = "BCC"


def is_bcc(name: str) -> bool:
    """Indica si una etiqueta one-hot completa corresponde a BCC.

    Parámetros
    ----------
    name : str
        Etiqueta resuelta desde el GT (p. ej. "BCC", "MEL", "UNK").

    Retorno
    -------
    bool
        True si `name` es exactamente "BCC". Comparación estricta: las
        variantes de mayúsculas/minúsculas del GT no existen en ISIC 2019 y
        tratarlas como BCC sería un positivo falso silencioso.
    """
    return name == BCC_LABEL
