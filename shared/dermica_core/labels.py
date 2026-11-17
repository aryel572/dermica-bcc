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
llamaban `CLASS_COLS`. `CLASS_COLS` es el nombre histórico previo a la
extracción del core (fase de reshuffle); no existe alias de compatibilidad:
el nombre actual es la única verdad y los tres llamadores se migraron a la
constante única.
SCC está presente en el formato GT aunque el proyecto no lo use en ninguna otra
lógica: se conserva porque el orden del GT es el contrato.

Qué recibe y qué produce: no lee flags ni escribe artefactos; expone las
etiquetas de clase y el predicado binario. Fase del pipeline: S1 (auditoría),
S2 (splits y subconjuntos TESIS) y S3/S4 (preparación y medición); es la base
léxica del problema BCC vs no-BCC.
"""

# Orden de columnas one-hot exactamente como lo entrega el CSV GroundTruth de
# ISIC 2019. Lo consumen load_train_labels() de make_splits.py, la formación de
# subconjuntos en form_tesis.py y load_gt() de audit_isic.py vía idxmax.
ISIC_LABELS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]

# Clase positiva del problema de tamizaje (carcinoma basocelular). La consume
# is_bcc() y, por extensión, toda métrica/etiquetado binario del proyecto.
BCC_LABEL = "BCC"


def is_bcc(name: str) -> bool:
    """Indica si una etiqueta one-hot completa corresponde a BCC.

    Parámetros
    ----------
    name : str
        Etiqueta resuelta desde el GT (p. ej. "BCC", "MEL", "UNK"). En ISIC
        2019, "BCC" = carcinoma basocelular, la clase positiva de la tesis;
        el resto de etiquetas se tratan como negativas para el problema binario.

    Retorno
    -------
    bool
        True si `name` es exactamente "BCC". Comparación estricta: las
        variantes de mayúsculas/minúsculas del GT no existen en ISIC 2019 y
        tratarlas como BCC sería un positivo falso silencioso.

    Errores
    -------
    Ninguno (comparación de strings en memoria).

    Globales
    --------
    `BCC_LABEL` (constante del módulo; única fuente del string "BCC").
    """
    return name == BCC_LABEL
