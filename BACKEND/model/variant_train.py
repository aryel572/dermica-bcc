#!/usr/bin/env python3
"""S3 experimento: comparación de arquitecturas/pérdidas a presupuesto fijo.

Mismo dataset (valid/cross-valid y val preparados), mismo seed 42, 40 épocas
con EarlyStopping (patience=10 sobre val_auprc, restore_best_weights=True),
misma receta; cambia SOLO la arquitectura o la pérdida:

  A) MobileNetV3-Small binario + BCE + class_weight {0: 1, 1: 6.87}   (reproducible?)
  B) MobileNetV3-Large binario + misma receta
  C) EfficientNet-Lite0 binario + misma receta
  D) MobileNetV3-Small 8-way (MEL,NV,BCC,AK,BKL,DF,VASC,SCC) CE ponderada;
     BCC-score = softmax[BCC] para métricas binarias (umbral 0.5)
  E) MobileNetV3-Small binario + FocalLoss(alpha=0.75, gamma=2.0)

SIN push a HF en los runs comparativos (volúmenes y ruido): solo artefactos
locales en --out-dir/<cfg>/ (log-epoch.csv, metrics.json). El ganador decide
si se implementa 1.1.0 según el gate de AUPRC/recall acordado.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 40
BIN_DIR = os.environ.get("PREPARED_BIN", "/content/prepared/224")
MULTI_DIR = os.environ.get("PREPARED_MULTI", "/content/prepared-multi/224")
CLASSES_8 = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]  # orden alfabético de carpetas
BCC_IDX = CLASSES_8.index("BCC")

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


def _build(arch: str):
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(10 / 360, seed=SEED),
        tf.keras.layers.RandomTranslation(0.1, 0.1, seed=SEED),
        tf.keras.layers.RandomZoom(0.1, seed=SEED),
        tf.keras.layers.RandomBrightness(0.2, seed=SEED),
    ], name="augment")
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augment(inputs)
    kw = dict(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False,
              weights="imagenet", pooling="avg")
    # lazy: solo se resuelve la clase elegida (p.ej. EfficientNetLite0 no existe en todos los TF)
    # EfficientNetLite0 no está en keras.applications de este TF; se usa
    # EfficientNetB0 (misma escala de red, pesos imagenet, recipe idéntica)
    base_cls = {
        "small": "MobileNetV3Small",
        "large": "MobileNetV3Large",
        "efnetlite0": "EfficientNetB0",
    }[arch]
    base = getattr(tf.keras.applications, base_cls)(**kw)
    x = base(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name=f"{arch}_bin")


def _build_multi():
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(10 / 360, seed=SEED),
        tf.keras.layers.RandomTranslation(0.1, 0.1, seed=SEED),
        tf.keras.layers.RandomZoom(0.1, seed=SEED),
        tf.keras.layers.RandomBrightness(0.2, seed=SEED),
    ], name="augment")
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augment(inputs)
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False,
        weights="imagenet", pooling="avg")
    x = base(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(8, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="small_multi")


class FocalLoss(tf.keras.losses.Loss):
    """Focal loss binario: FL(p) = -alpha*(1-p)^gamma*log(p) para positivas."""

    def __init__(self, alpha=0.75, gamma=2.0, name="focal_loss"):
        super().__init__(name=name)
        self.alpha, self.gamma = alpha, gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1, 1]), tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        return tf.reduce_mean(-alpha_t * tf.pow(1 - p_t, self.gamma) * tf.math.log(tf.clip_by_value(p_t, 1e-7, 1.0)))


def _lr(epoch):
    peak, min_lr, warm = 1e-3, 1e-5, 3
    if epoch < warm:
        return peak * (epoch + 1) / warm
    p = (epoch - warm) / max(1, EPOCHS - warm)
    return min_lr + 0.5 * (peak - min_lr) * (1 + math.cos(math.pi * p))


class EvalBinary(tf.keras.callbacks.Callback):
    """Eval binario por época: AUPRC/recall/specificity/BA/F2 sobre val."""

    def __init__(self, val_ds, out_csv):
        super().__init__()
        self.val_ds, self.out_csv = val_ds, Path(out_csv)
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)

    def _score(self, x):
        return self.model.predict(x, verbose=0).reshape(-1)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_true, y_score = [], []
        for x, y in self.val_ds:
            y_score.extend(self._score(x).tolist())
            y_true.extend(y.numpy().reshape(-1).astype(int).tolist())
        rep = metrics.reporte(y_true, y_score)
        for k, v in rep.items():
            if isinstance(v, float):
                logs[k] = v
        logs["val_auprc"] = rep["auprc"]
        logs["val_recall"] = rep["recall"]
        row = {"epoch": epoch + 1,
               **{k: round(float(rep[k]), 6) for k in ("auprc", "recall", "especificidad", "balanced_acc", "f2", "precision")}}
        new = not self.out_csv.exists()
        with open(self.out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"\n  época {row['epoch']}: val_auprc={rep['auprc']:.4f} recall={rep['recall']:.4f} ba={rep['balanced_acc']:.4f} f2={rep['f2']:.4f}")


class EvalMulti(EvalBinary):
    """Eval para 8-way: BCC-score = softmax[BCC]; métricas binarias desde él."""

    def _score(self, x):
        return self.model.predict(x, verbose=0)[:, BCC_IDX]


def _compute_class_weights_multi(train_dir):
    counts = {c: len(list(Path(train_dir, c).glob("*.jpg"))) for c in CLASSES_8}
    n = sum(counts.values())
    k = len(counts)
    weights = {i: n / (k * counts[c]) for i, c in enumerate(CLASSES_8)}
    print("class_weight multi:", {CLASSES_8[i]: round(w, 3) for i, w in weights.items()})
    return weights


def run(cfg: str, out_dir: Path) -> int:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg == "D":
        train_ds = tf.keras.utils.image_dataset_from_directory(
            f"{MULTI_DIR}/train", labels="inferred", label_mode="categorical",
            class_names=CLASSES_8, image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
        val_ds = tf.keras.utils.image_dataset_from_directory(
            f"{MULTI_DIR}/val", labels="inferred", label_mode="categorical",
            class_names=CLASSES_8, image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE, shuffle=False)
        model = _build_multi()
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")])
        class_weight = _compute_class_weights_multi(f"{MULTI_DIR}/train")
        cb = [EvalMulti(val_ds, out_dir / "log-epoch.csv"),
              tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda e, l: l.update({"val_auprc": l.get("val_auprc", 0.0)})),
              tf.keras.callbacks.EarlyStopping(monitor="val_auprc", mode="max", patience=10,
                                               restore_best_weights=True, verbose=1),
              tf.keras.callbacks.LearningRateScheduler(_lr, verbose=0)]
        model.fit(train_ds.prefetch(tf.data.AUTOTUNE), epochs=EPOCHS,
                  validation_data=val_ds.prefetch(tf.data.AUTOTUNE),
                  class_weight=class_weight, callbacks=cb, verbose=1)
        final_score = []
        for x, _ in val_ds:
            final_score.extend(model.predict(x, verbose=0)[:, BCC_IDX].tolist())
        y_true = []
        for _, y in val_ds:
            y_true.extend(np.argmax(y.numpy(), axis=1).tolist())
        binary_y = [1 if c == BCC_IDX else 0 for c in y_true]
        return _finish(cfg, t0, out_dir, metrics.reporte(binary_y, final_score))

    # configs binarias A, B, C, E
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{BIN_DIR}/train", labels="inferred", label_mode="binary",
        class_names=["NO_BCC", "BCC"], image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{BIN_DIR}/val", labels="inferred", label_mode="binary",
        class_names=["NO_BCC", "BCC"], image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=False)

    if cfg in ("A", "E"):
        model = _build("small")
    elif cfg == "B":
        model = _build("large")
    else:  # C
        model = _build("efnetlite0")

    if cfg == "E":
        loss = FocalLoss(alpha=0.75, gamma=2.0)
    else:
        loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss,
                  metrics=[tf.keras.metrics.BinaryAccuracy(name="acc")])

    class_weight = {0: 1.0, 1: 6.87}
    ev = EvalBinary(val_ds, out_dir / "log-epoch.csv")
    # EarlyStopping necesita val_auprc ya inyectado por EvalBinary en el mismo slot de logs
    cb = [ev,
          tf.keras.callbacks.EarlyStopping(monitor="val_auprc", mode="max", patience=10,
                                           restore_best_weights=True, verbose=1),
          tf.keras.callbacks.LearningRateScheduler(_lr, verbose=0)]
    model.fit(train_ds.prefetch(tf.data.AUTOTUNE), epochs=EPOCHS,
              validation_data=val_ds.prefetch(tf.data.AUTOTUNE),
              class_weight=class_weight, callbacks=cb, verbose=1)
    y_true, y_score = [], []
    for x, y in val_ds:
        y_score.extend(model.predict(x, verbose=0).reshape(-1).tolist())
        y_true.extend(y.numpy().reshape(-1).astype(int).tolist())
    return _finish(cfg, t0, out_dir, metrics.reporte(y_true, y_score))


def _finish(cfg, t0, out_dir, rep):
    payload = {"config": cfg, **rep, "segundos": round(time.time() - t0, 2)}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print("\n== RESULTADO", cfg, "==")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, choices=["A", "B", "C", "D", "E"])
    ap.add_argument("--out-dir", type=Path, default=Path(os.environ.get("OUT_DIR", "/content/out")))
    args = ap.parse_args()
    return run(args.config, args.out_dir / f"exp-{args.config}")


if __name__ == "__main__":
    sys.exit(main())
