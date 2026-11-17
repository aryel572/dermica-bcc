"""Datasets tf.data para el entrenamiento binario BCC vs no-BCC.

Propósito
---------
Construye los `tf.data.Dataset` de entrenamiento y validación que consume el
fine-tuning de MobileNetV3-Small (S3, CRISP-DM Modeling): decodificación de
JPEG, batching y prefetch. Nada más: el augment vive en el modelo.

Qué recibe
----------
`cargar_datasets(base_dir)`: `base_dir` es la raíz local del directorio
preparado `224/` con subdirectorios `train/` y `val/`, cada uno con
subcarpetas `BCC/` y `NO_BCC/`. El origen típico es la variable de entorno
`PREPARED_DIR`, que lee el llamador (`train_bcn.py`); este módulo no lee
entorno ni flags: todo llega por parámetro.

Qué produce
-----------
`(train_ds, val_ds)` prefetchados: batches de `BATCH_SIZE`, imágenes RGB
224×224 en `float32[0,1]`; shuffle determinista con `SEED` solo en train,
val queda en orden fijo para una evaluación reproducible.

Quién consume
-------------
`model/train_bcn.py` (importa `cargar_datasets`, `IMG_SIZE`, `BATCH_SIZE`,
`SEED`). El runner de experimentos `variant_train.py` no usa este módulo:
construye sus propios datasets con la misma receta.

Fase: S3 (CRISP-DM Modeling).
"""

import tensorflow as tf

# Origen: contrato de entrada del MobileNetV3-Small con pesos ImageNet (el
# pipeline preparó las imágenes a 224 px). Consumidor: cargar_datasets() y,
# vía reexport, train_bcn.py para construir la cabeza del modelo.
IMG_SIZE = 224
# Batch de GPU T4 para MobileNetV3-Small: cabe en 16 GB con margen; más grande no mejora aquí.
# Origen: ajuste empírico del fine-tuning. Consumidor: cargar_datasets() y train_bcn.py.
BATCH_SIZE = 32
# Semilla única del proyecto: reproducibilidad del shuffle de train.
# Consumidor: cargar_datasets(shuffle=True, seed=SEED) y el shuffle/augment de todo el pipeline.
SEED = 42


def _parse(path, label):
    """Lee y normaliza una imagen para el pipeline manual de tf.data.

    Nota de uso: `cargar_datasets` NO la invoca; usa
    `tf.keras.utils.image_dataset_from_directory`, que decodifica por dentro.
    Queda como referencia explícita del pipeline (y por si se reemplaza el
    loader por `from_tensor_slices` a futuro).

    Parámetros:
        path  — tensor/string con la ruta de la imagen (típicamente jpg).
        label — etiqueta asociada (0 = NO_BCC, 1 = BCC; orden fijo del proyecto).
    Retorna:
        tuple (img, label) con img float32 [0..1] shape (H, W, 3).
    Lanza: error de tf.io si el archivo no existe o no es un JPEG decodificable.
    Globales: ninguna (toda la configuración llega por parámetro o constante de módulo).
    Por qué decode_jpeg + /255: pipeline de tf.data trabaja con tensores float32
    ya normalizados; "224 px" es el input contract del modelo.
    """
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)          # RGB forzado (3 canales) aunque vengan en otro modo
    img = tf.cast(img, tf.float32) / 255.0            # [0,255] -> [0,1]
    return img, label


def cargar_datasets(base_dir):
    """Devuelve (train_ds, val_ds) en tf.data con shuffle determinista.

    Parámetros:
        base_dir — str/Path con la raíz que contiene `train/` y `val/`
                   (subcarpetas `BCC/` y `NO_BCC/` de imágenes 224×224).
    Retorna:
        tuple (train_ds, val_ds). train viene shuffled (seed fija) y val sin shuffle
        (orden determinista para evaluación reproducible).
    Lanza: error de tf.keras si `{base_dir}/train` o `{base_dir}/val` no
                   existe o no contiene las clases esperadas.
    Globales: IMG_SIZE, BATCH_SIZE, SEED (constantes de módulo).
    Por qué dos datasets distintos: el augment de train vive en el modelo
    (Random*, solo activas en modo entrenamiento), no aquí; este pipeline solo
    toca decodificación y batching para no duplicar trabajo de CPU.
    prefetch(AUTOTUNE) solapa decodificación con el cómputo del GPU para no
    pagar dataloading durante el paso de entrenamiento.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{base_dir}/train",
        labels="inferred",                  # clase inferida de subcarpeta
        label_mode="binary",
        class_names=["NO_BCC", "BCC"],      # orden fijo: 0 = NO_BCC, 1 = BCC (debe coincidir con class_weight)
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{base_dir}/val",
        labels="inferred",
        label_mode="binary",
        class_names=["NO_BCC", "BCC"],      # mismo orden fijo que train (0 = NO_BCC, 1 = BCC)
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,                      # validación determinista, sin augment
    )
    # prefetch para solapar decodificación con entrenamiento (oculta latencia de CPU)
    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)
