#!/usr/bin/env python3
"""S3 (CRISP-DM Modeling): fine-tune de MobileNetV3-Small para BCC vs no-BCC.

Resumen del pipeline:
- Datos: prepared/224/{train,val}/{BCC,NO_BCC}/<image>.jpg (224 px, seed 42).
- Modelo: MobileNetV3-Small (imagenet) + GlobalAveragePooling→Dropout(0.2)→Dense(1, sigmoid).
- Optimizador: Adam(1e-3) con warmup lineal de 3 épocas + cosine decay.
- Augment (solo train, capas Random* activas en modo entrenamiento): flip H/V,
  rotación 10°, traslación/zoom 10%, brillo ±20%.
- class_weight {0: 1.0, 1: 6.87} por el desbalance (BCC ≈ 12.76%).
- Checkpoints cada 5 épocas → dermica-checkpoints-tmp con tag `1.0.0-epN`,
  borrando el tag anterior; métricas por época en dermica-metrics/1.0.0-epN.json.
- EarlyStopping patience=10 sobre val_auprc (restaura mejores pesos).
- Resume: detecta el último `1.0.0-epN` en el repo tmp, descarga pesos y
  continúa desde epN+1 (tolerante a cortes de cuota).

Uso:
    uv run python train_bcn.py --dry-run       # 2 batches, sanity check
    uv run python train_bcn.py --verificar     # verificación rápida de métricas
    uv run python train_bcn.py                 # 40 épocas (EarlyStopping puede cortar)
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_tf import cargar_datasets, IMG_SIZE, BATCH_SIZE, SEED
import metrics
import push_hf

VERSION = "1.0.0"   # línea de versión del modelo (tag completo `1.0.0`; en desarrollo `1.0.0-epN`)
BASE_DIR = os.environ.get("PREPARED_DIR", "/content/prepared/224")  # path al dataset en el VM de Colab (local vía env var)
OUT_DIR = Path(os.environ.get("OUT_DIR", "/content/out"))  # OUT_DIR escribible en Colab; local se ajusta con env var
EPOCHS = 40                     # tope de épocas: EarlyStopping puede cortar antes (patience=10 sobre val_auprc)
# class_weight 0:1.0 / 1:6.87 → compensa el desbalance (BCC ≈ 1/7,83 del train, ratio inversa ~6.87);
# el modelo pagaría el doble de gradiente por cada BCC, priorizando recall BCC
CLASS_WEIGHT = {0: 1.0, 1: 6.87}

# Semillas de todos los RNG antes de cualquier azar (reproducibilidad)
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

METRIC_KEYS = ("val_auprc", "val_recall", "val_specificity", "val_balanced_acc", "val_f2", "val_loss")


def build_model():
    """MobileNetV3-Small + cabeza binaria. Augment solo en modo train."""
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(10 / 360, seed=SEED),
        tf.keras.layers.RandomTranslation(0.1, 0.1, seed=SEED),
        tf.keras.layers.RandomZoom(0.1, seed=SEED),
        tf.keras.layers.RandomBrightness(0.2, seed=SEED),
    ], name="augment")
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False,
        weights="imagenet", pooling="avg",
    )
    x = augment(inputs := tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
    x = base(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="mobile_bcc")


def lr_schedule(epoch):
    """Warmup lineal 3 épocas → cosine decay hasta 1e-5 al final.

    Parámetros: epoch — época 0-based.
    Retorna: float con la tasa de aprendizaje de esa época.
    Por qué warmup+cosine: régimen estándar para fine-tuning con pesos imagenet;
    evita destruir los pesos pre-entrenados en las primeras épocas y decae suave
    hacia el mínimo para convergencia fina sobre el desbalance BCC.
    """
    peak, min_lr, warm = 1e-3, 1e-5, 3
    if epoch < warm:
        return peak * (epoch + 1) / warm
    p = (epoch - warm) / max(1, EPOCHS - warm)
    return min_lr + 0.5 * (peak - min_lr) * (1 + math.cos(math.pi * p))


class EvalMetrics(tf.keras.callbacks.Callback):
    """Evalúa val al cerrar cada época; inyecta val_* en logs y persiste CSV.

    Parámetros:
        val_ds   — dataset de validación (sin augment, sin shuffle).
        out_csv  — ruta para log-epoch.csv (append por época).
    Efectos colaterales: escribe log-epoch.csv y muta los logs del entrenamiento con
    val_auprc/val_recall/val_specificity/val_balanced_acc/val_f2 para que
    EarlyStopping pueda monitorear val_auprc directamente.
    Por qué no keras.metrics: AUPRC no se puede agregar por batch; se computa
    sobre todas las predicciones de val al cerrar la época.
    """

    def __init__(self, val_ds, out_csv):
        super().__init__()
        self.val_ds = val_ds
        self.out_csv = Path(out_csv)
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_true, y_score = [], []
        for x, y in self.val_ds:
            y_score.extend(self.model.predict(x, verbose=0).reshape(-1).tolist())
            y_true.extend(y.numpy().reshape(-1).astype(int).tolist())
        rep = metrics.reporte(y_true, y_score)
        logs["val_auprc"] = rep["auprc"]
        logs["val_recall"] = rep["recall"]
        logs["val_specificity"] = rep["especificidad"]
        logs["val_balanced_acc"] = rep["balanced_acc"]
        logs["val_f2"] = rep["f2"]
        row = {"epoch": epoch + 1, "loss": round(float(logs.get("loss", 0)), 6),
               "val_loss": round(float(logs.get("val_loss", 0)), 6),
               **{k: round(float(logs[k]), 6) for k in METRIC_KEYS if k != "val_loss"}}
        new_file = not self.out_csv.exists()
        with open(self.out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new_file:
                w.writeheader()
            w.writerow(row)
        print(f"\n  época {row['epoch']}: val_auprc={rep['auprc']:.4f} recall={rep['recall']:.4f} "
              f"ba={rep['balanced_acc']:.4f} f2={rep['f2']:.4f}")


class PushCheckpoint(tf.keras.callbacks.Callback):
    """Push del checkpoint cada 5 épocas (modelo completo + pesos + métricas).

    Parámetros: out_dir — directorio donde se escribe `1.0.0-epN.keras`,
    `.weights.h5`, `.json` (métricas de la época) y se sube a los repos HF.
    Efectos colaterales: escribe archivos en OUT_DIR; sube a dermica-checkpoints-tmp
    (tag `1.0.0-epN`) y dermica-metrics (JSON `1.0.0-epN.json`); borra en ambos
    los tags anteriores al actual (reglas E6). Solo actúa cada 5 épocas.
    """

    def __init__(self, out_dir):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        n = epoch + 1
        if n % 5:
            return
        tag = f"{VERSION}-ep{n}"
        keras_path = self.out_dir / f"{tag}.keras"
        weights_path = self.out_dir / f"{tag}.weights.h5"
        self.model.save(keras_path)
        self.model.save_weights(weights_path)
        push_hf.upload_checkpoint({"keras": keras_path, "weights": weights_path}, tag=tag,
                                  metrics_row=logs)
        # métricas de la época en el repo de métricas
        metrics_json = self.out_dir / f"{tag}.json"
        metrics_json.write_text(json.dumps(
            {"tag": tag, "epoch": n, "seed": SEED,
             **{k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}},
            indent=2, ensure_ascii=False))
        push_hf.upload_metrics_json(metrics_json, tag)
        push_hf.delete_stale(VERSION, n)
        print(f"\n  [hf] checkpoint {tag} subido; superados borrados")


def restore_if_possible(model):
    """Descarga el último 1.0.0-epN y restaura pesos. Devuelve época inicial.

    Parámetros: model — modelo compilado ya instanciado.
    Retorna: int, época inicial para model.fit (0 si no hay resume posible).
    Efectos colaterales: lee/descarga del repo tmp en HF; muta los pesos del modelo en sitio.
    Por qué fallo silencioso a 0: perder un resume no debe abortar el
    entrenamiento, pero el corte de cuota entre sesiones sí puede pausar; HF
    queda como estado persistente entre VMs (resume honesto).
    """
    try:
        latest = push_hf.latest_tag(VERSION)
    except Exception as e:
        print(f"resume: sin checkpoints en HF ({type(e).__name__}); empezando desde cero")
        return 0
    if not latest:
        return 0
    import re
    m = re.search(r"ep(\d+)$", latest)
    if not m:
        return 0
    start = int(m.group(1))
    weights = OUT_DIR / f"{latest}.weights.h5"
    try:
        push_hf.download(f"{latest}.weights.h5", weights, repo="tmp")
    except Exception as e:
        print(f"resume: no se pudo descargar {latest} ({e}); empezando desde cero")
        return 0
    model.load_weights(str(weights))
    print(f"resume: {latest} restaurado; continúo desde época {start + 1}")
    return start


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="1 batch, sanity check")
    ap.add_argument("--verificar", action="store_true", help="verificación rápida de métricas")
    args = ap.parse_args()

    if args.verificar:
        y_true = [1, 0, 1, 0] * 50
        y_score = [0.9, 0.1, 0.6, 0.4] * 50
        rep = metrics.reporte(y_true, y_score)
        assert rep["auprc"] > 0.5 and rep["recall"] == 1.0
        print("VERIFICACIÓN: OK — métricas coherentes")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds = cargar_datasets(BASE_DIR)

    if args.dry_run:
        batch = next(iter(train_ds))
        print(f"dry-run: batch {batch[0].shape} labels mean {batch[1].numpy().mean():.3f}")
        return 0

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="acc")],
    )
    start_epoch = restore_if_possible(model)
    callbacks = [
        EvalMetrics(val_ds, OUT_DIR / "log-epoch.csv"),   # 1º: inyecta val_* en logs
        PushCheckpoint(OUT_DIR),                          # 2º: usa logs ya completos
        tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0),
        tf.keras.callbacks.EarlyStopping(monitor="val_auprc", mode="max", patience=10,
                                         restore_best_weights=True, verbose=1),
    ]
    model.fit(
        train_ds,
        epochs=EPOCHS,
        initial_epoch=start_epoch,
        validation_data=val_ds,
        class_weight=CLASS_WEIGHT,
        callbacks=callbacks,
        verbose=1,
    )
    # Versión completa 1.0.0 → dermica-models; métricas finales permanentes en dermica-metrics
    final_metrics = metrics.reporte(*eval_val(model, val_ds))
    push_hf.upload_complete(model, final_metrics, VERSION, OUT_DIR)
    print("entrenamiento completado")
    return 0


def eval_val(model, val_ds):
    """Predicciones sobre val: devuelve (y_true, y_score).

    Parámetros: model — modelo entrenado; val_ds — dataset de validación.
    Retorna: tuple (y_true, y_score) con listas binarias/[0..1].
    Por qué separado: la versión completa final no requiere restaurar best weights,
    usa el modelo ya post-EarlyStopping (que restauró el mejor epoch solo).
    """
    y_true, y_score = [], []
    for x, y in val_ds:
        y_score.extend(model.predict(x, verbose=0).reshape(-1).tolist())
        y_true.extend(y.numpy().reshape(-1).astype(int).tolist())
    return y_true, y_score


if __name__ == "__main__":
    sys.exit(main())
