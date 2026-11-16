"""Datasets tf.data para el entrenamiento binario BCC vs no-BCC.

Carga imágenes 224×224 RGB desde prepared/224/{train,val}/{BCC,NO_BCC}/<image>.jpg
con semilla fija (42) para shuffle reproducible. El augment vive en el modelo
(capas Random*, solo activas al entrenar); aquí solo decodificación y batching.
"""

import tensorflow as tf

IMG_SIZE = 224          # entrada del modelo MobileNetV3-Small (input 224, estándar del checkpoint imagenet)
BATCH_SIZE = 32         # batch de GPU T4 para MobileNetV3-Small: cabe en 16 GB con margen; más grande no mejora aquí
SEED = 42               # semilla única del proyecto: reproducibilidad del shuffle de train


def _parse(path, label):
    """Lee y normaliza una imagen para el pipeline.

    Parámetros:
        path  — ruta tf/string de la imagen (típicamente jpg).
        label — etiqueta asociada (0 = NO_BCC, 1 = BCC).
    Retorna:
        tuple (img, label) con img float32 [0..1] shape (H, W, 3).
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
        base_dir — directorio que contiene train/ y val/ con subcarpetas BCC/NO_BCC.
    Retorna:
        tuple (train_ds, val_ds). train viene shuffled (seed fija) y val sin shuffle
        (orden determinista para evaluación reproducible).
    Por qué dos cargar_datasets distintos: el augment de train vive en el modelo
    (Random*), no aquí; este pipeline solo toca decodificación y batching para no
    duplicar trabajo CPU. prefetch(AUTOTUNE) solapa decodificación con el cómputo
    del GPU para no pagar dataloading durante el paso de entrenamiento.
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
        class_names=["NO_BCC", "BCC"],
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,                      # validación determinista, sin augment
    )
    # prefetch para solapar decodificación con entrenamiento (oculta latencia de CPU)
    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)
