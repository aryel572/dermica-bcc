"""Datasets tf.data para el entrenamiento binario BCC vs no-BCC.

Carga imágenes 224×224 RGB desde prepared/224/{train,val}/{BCC,NO_BCC}/<image>.jpg
con semilla fija (42) para shuffle reproducible. El augment vive en el modelo
(capas Random*, solo activas al entrenar); aquí solo decodificación y batching.
"""

import tensorflow as tf

IMG_SIZE = 224  # input del modelo
BATCH_SIZE = 32
SEED = 42


def _parse(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def cargar_datasets(base_dir):
    """Devuelve (train_ds, val_ds) en tf.data. shuffle determinista por seed."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{base_dir}/train",
        labels="inferred",
        label_mode="binary",
        class_names=["NO_BCC", "BCC"],  # 0 = NO_BCC, 1 = BCC
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
        shuffle=False,
    )
    # prefetch para solapar decodificación con entrenamiento
    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)
