#!/usr/bin/env python3
"""S3 (CRISP-DM Modeling): comparación de arquitecturas/pérdidas a presupuesto fijo.

Propósito
---------
Experimento de elección de arquitectura/pérdida para el proyecto, bajo idéntico
presupuesto (dataset, seed, épocas, receta): solo cambian la arquitectura o la
pérdida. El ganador (si pasa el gate AUPRC/recall acordado) decide la variante
del release siguiente.

Qué recibe
----------
Flags de CLI (origen: línea de comandos):
    --config A..E  Experimento a correr (obligatorio).
    --out-dir ~DIR Raíz de escritura; los artefactos caen en `<out-dir>/exp-<cfg>/`.
Entorno (nunca secretos; HF no interviene en este runner):
    PREPARED_BIN    Raíz del dataset binario 224/ (train/ y val/ con BCC/NO_BCC/).
    PREPARED_MULTI  Raíz del dataset multiclase 224/ (8 clases).
    OUT_DIR         Fallback de --out-dir.
Si falta PREPARED_BIN/PREPARED_MULTI, el valor por defecto del módulo solo es
válido en el runtime de entrenamiento.

Qué produce
-----------
En `<out-dir>/exp-<cfg>/`: `log-epoch.csv` (una fila por época con las métricas)
y `metrics.json` (config + métricas finales + segundos de corrida). Ningún
artefacto se sube a HF: los runs comparativos son locales.

Quién consume
-------------
Nadie importa este módulo: lo ejecuta el operador. El `exp-<cfg>/metrics.json`
de cada corrida alimenta la comparación y la decisión de 1.1.0 (ver gate en el
cuerpo del módulo); la documentación de S3 recoge la tabla resultante.

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

Fase: S3 (CRISP-DM Modeling).
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

# Origen: duplicadas deliberadamente de train_bcn.py — este runner es autocontenido
# (no importa train_bcn) para comparar a idéntica receta sin acoplar el baseline.
# Consumidores: todo este módulo (datasets, augment, scheduler, fit).
SEED = 42          # semilla única del proyecto: shuffle/augments reproducibles
IMG_SIZE = 224     # input del modelo binario/multiclase (checkpoint imagenet)
BATCH_SIZE = 32    # batch GPU T4: cabe sin swapear con MobileNetV3
EPOCHS = 40        # tope; EarlyStopping puede cortar (patience=10 sobre val_auprc)
BIN_DIR = os.environ.get("PREPARED_BIN", "/content/prepared/224")      # dataset binario BCC/NO_BCC en el VM
MULTI_DIR = os.environ.get("PREPARED_MULTI", "/content/prepared-multi/224")  # dataset multi-clase (8 clases) en el VM
# Orden alfabético de las subcarpetas tal como lo infiere image_dataset_from_directory
# (labels=inferred); sin UNK porque no tiene imágenes en train.
# Consumidores: _build_multi, EvalMulti, _compute_class_weights_multi, run (rama D).
CLASSES_8 = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
# Índice de la clase BCC dentro del softmax 8-way; de ahí sale el BCC-score binario en D.
# Consumidores: EvalMulti.on_epoch_end, run (rama D).
BCC_IDX = CLASSES_8.index("BCC")

# Semillas antes de cualquier azar (datasets shuffle, capas Random*, inicialización).
# Consumidores: todo el proceso de este runner, igual que en train_bcn.py.
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


def _build(arch: str):
    """Construye el modelo binario para la arquitectura pedida (small|large|efnetlite0).

    Parámetros: arch — clave de arquitectura: A/E→"small", B→"large",
                C→"efnetlite0" (mapeada a EfficientNetB0, ver abajo).
    Retorna: tf.keras.Model "<arch>_bin" sin compilar (cabeza sigmoid sobre
             GlobalAveragePooling con Dropout(0.2)).
    Lanza: KeyError si `arch` no está en el mapa (el llamador interno solo pasa claves válidas);
           error de descarga de pesos si ImageNet no es accesible.
    Globales: SEED (Random*), IMG_SIZE (forma de entrada).
    Por qué lazy getattr: keras.applications no garantiza el símbolo en todos los TF
    (p.ej. EfficientNetLite0 ausente en 2.21); esto falla solo si se pide esa clase.
    """
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
    """Construye MobileNetV3-Small 8-way (CE multiclase) como referencia alternativa.

    Parámetros: ninguno.
    Retorna: tf.keras.Model "small_multi" sin compilar, cabeza softmax 8 clases
             (AK, BCC, BKL, DF, MEL, NV, SCC, VASC — orden alfabético fijo).
    Lanza: error de descarga de pesos si ImageNet no es accesible en el primer build.
    Globales: SEED (Random*), IMG_SIZE (forma de entrada).
    Por qué CE multiclase ponderada sobre baseline binario: comparación directa de
    infraestructura multiclass sobre el mismo pipeline; el BCC-score = softmax[BCC].
    """
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
    """Focal loss binario: FL(p) = -alpha*(1-p)^gamma*log(p) para positivas.

    Parámetros (constructor):
        alpha — ponderación de la clase positiva (0.75: BCC es la clase rara);
        gamma — exponente que enfoca ejemplos difíciles (2.0 estándar RetinaNet);
        name  — nombre del loss (cosmético, aparece en tablas de keras).
    Por qué sobre desbalance: BCE trata igual todas las positivas; focal premia
    los BCC difíciles/confundidos con NV, alineando con la prioridad clínica
    del proyecto sobre recall de BCC.

    Globales: ninguna (la pérdida es autocontenida; estilo keras.losses).
    Nota de `call`: y_true se redimensiona a 2D [-1,1] para el broadcast con y_pred [batch, 1];
    p_t es la prob. asignada al lado correcto y alpha_t su ponderación por clase.
    """

    def __init__(self, alpha=0.75, gamma=2.0, name="focal_loss"):
        super().__init__(name=name)
        self.alpha, self.gamma = alpha, gamma

    def call(self, y_true, y_pred):
        """Calcula FL medio sobre el batch (ver fórmula arriba).

        Parámetros: y_true — etiquetas 0/1 (shape arbitraria); y_pred — probas sigmoide.
        Retorna: escalar tf.float32 con la media del focal loss.
        Lanza: nada específico; el `clip_by_value` acota log(p) y evita NaN por bordes exactos.
        """
        y_true = tf.cast(tf.reshape(y_true, [-1, 1]), tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        return tf.reduce_mean(-alpha_t * tf.pow(1 - p_t, self.gamma) * tf.math.log(tf.clip_by_value(p_t, 1e-7, 1.0)))


def _lr(epoch):
    """Warmup lineal 3 épocas → cosine decay hasta 1e-5 (misma receta que baseline S3).

    Parámetros: epoch — época 0-based.
    Retorna: float con la tasa de aprendizaje de esa época.
    Globales: EPOCHS (tope del cosine decay).
    Notas: idéntica en comportamiento a `lr_schedule` de train_bcn.py; se duplica
    aquí para que el runner sea autocontenido (no importa train_bcn).
    """
    peak, min_lr, warm = 1e-3, 1e-5, 3  # pico post-warmup, piso final del cosine y nº de épocas de calentamiento
    if epoch < warm:
        return peak * (epoch + 1) / warm
    p = (epoch - warm) / max(1, EPOCHS - warm)
    return min_lr + 0.5 * (peak - min_lr) * (1 + math.cos(math.pi * p))


class EvalBinary(tf.keras.callbacks.Callback):
    """Eval binario por época: AUPRC/recall/specificity/BA/F2 sobre val.

    Parámetros (constructor): val_ds — dataset de validación; out_csv — str/Path
               de log-epoch.csv (append por época; el padre se crea si falta).
    Efectos colaterales: escribe log-epoch.csv (append; crea con header si no existe)
    y muta los logs con val_auprc/val_recall (clave para EarlyStopping, que monitorea val_auprc).
    Lanza: errores de escritura de CSV o de predict propagan al fit.
    Globales: ninguna fuera de sus parámetros (métricas.reporte hace el cálculo).
    Por qué VAR /CSV: métricas de cada época permiten comparar curva baseline vs
    variantes a idéntico presupuesto (mismo #épocas, mismo seed, mismos datos).
    """

    def __init__(self, val_ds, out_csv):
        super().__init__()
        self.val_ds, self.out_csv = val_ds, Path(out_csv)
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)

    def _score(self, x):
        """Puntuación BCC del batch x (sigmoide ya en [0,1]), como vector plano.

        Parámetros: x — batch de imágenes (H, W, 3) float32.
        Retorna: np.ndarray 1D con una probabilidad BCC por ejemplo.
        """
        return self.model.predict(x, verbose=0).reshape(-1)

    def on_epoch_end(self, epoch, logs=None):
        """Calcula métricas de val al cerrar la época y las deja en logs y en el CSV.

        Parámetros: epoch — época 0-based; logs — logs de keras (se muta in-place).
        Retorna: None.
        """
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


class EvalMulti(tf.keras.callbacks.Callback):
    """Eval para 8-way: BCC-score = softmax[BCC]; métricas binarias desde él.

    Parámetros (constructor): val_ds — dataset de validación multiclase (one-hot);
               out_csv — str/Path de log-epoch.csv (append por época).
    Efectos colaterales: escribe log-epoch.csv y muta logs con val_auprc/val_recall.
    Lanza: errores de escritura o de predict propagan al fit.
    Globales: BCC_IDX (índice de BCC en el softmax), vía derivación del score.
    Por qué NO heredar de EvalBinary: la base binaria asume `y_score = sigmoid(x)`
    (un solo factor = probabilidad de BCC). Aquí el score de BCC es softmax[BCC]
    y el label_true debe derivarse con `argmax` de one-hot. Compartir clase
    llevaría a condicionales dentro de cada método (anti-patrón); la duplicación es deliberada
    aunque el cálculo métrico final sea idéntico (metrics.reporte).
    """

    def __init__(self, val_ds, out_csv):
        super().__init__()
        self.val_ds, self.out_csv = val_ds, Path(out_csv)
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        """Calcula métricas binarias (BCC vs resto) desde el softmax 8-way.

        Parámetros: epoch — época 0-based; logs — logs de keras (se muta in-place).
        Retorna: None.
        Nota: binary_y deriva 1 iff argmax==BCC_IDX; y_true conserva el índice para
              comparar contra softmax[BCC_IDX] como score del positivo.
        """
        logs = logs or {}
        y_true, y_score = [], []
        for x, y in self.val_ds:
            y_score.extend(self.model.predict(x, verbose=0)[:, BCC_IDX].tolist())
            y_true.extend(np.argmax(y.numpy(), axis=1).tolist())
        binary_y = [1 if c == BCC_IDX else 0 for c in y_true]
        rep = metrics.reporte(binary_y, y_score)
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


def _compute_class_weights_multi(train_dir):
    """Calcula class_weight inversamente proporcional a la frecuencia de clase.

    Parámetros: train_dir — str/Path del directorio train con subcarpetas por clase
               (nombres = CLASSES_8; se cuentan *.jpg).
    Retorna: dict {índice_clase: peso}; con 8 clases, n/(k*count_c) donde n =
             total de imágenes y k = nº de clases.
    Lanza: nada propio; una subcarpeta ausente da count 0 → ZeroDivisionError (falla fuerte, no silenciosa).
    Globales: CLASSES_8 (orden de índices).
    Por qué pesos inversamente proporcionales: mismo principio que el binario
    (compensar desbalance), pero a 8 clases con frecuencias muy dispersas.
    """
    counts = {c: len(list(Path(train_dir, c).glob("*.jpg"))) for c in CLASSES_8}
    n = sum(counts.values())
    k = len(counts)
    weights = {i: n / (k * counts[c]) for i, c in enumerate(CLASSES_8)}
    print("class_weight multi:", {CLASSES_8[i]: round(w, 3) for i, w in weights.items()})
    return weights


def run(cfg: str, out_dir: Path) -> int:
    """Ejecuta la corrida completa del experimento cfg (A..E) sin tocar HF.

    Parámetros: cfg — 'A'..'E' (decisión de arquitectura/pérdida); out_dir —
                Path escribible (se crea si falta) para log-epoch.csv y metrics.json.
    Retorna: 0 al terminar (escribe artefactos locales); no publica nada en HF.
    Lanza: propaga errores de build/compile/fit (descarga de pesos, OOM); para
           cfg inesperado cae en la rama binaria (else → C), comportamiento actual.
    Efectos colaterales: escribe en out_dir: log-epoch.csv y metrics.json.
    Globales: BIN_DIR, MULTI_DIR (raíces de datasets), CLASSES_8, BCC_IDX,
    EPOCHS, BATCH_SIZE, IMG_SIZE, SEED (receta compartida con la línea base).
    Por qué sin HF: comparación pura; sólo el GANADOR (si pasa el gate) se publica.
    """
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
        # LambdaCallback garantiza que val_auprc exista en logs (default 0.0) para EarlyStopping,
        # incluso si EvalMulti no llegó a inyectarlo (p.ej. primera época con fallo de eval)
        cb = [EvalMulti(val_ds, out_dir / "log-epoch.csv"),
              tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda e, l: l.update({"val_auprc": l.get("val_auprc", 0.0)})),
              tf.keras.callbacks.EarlyStopping(monitor="val_auprc", mode="max", patience=10,
                                               restore_best_weights=True, verbose=1),
              tf.keras.callbacks.LearningRateScheduler(_lr, verbose=0)]
        model.fit(train_ds.prefetch(tf.data.AUTOTUNE), epochs=EPOCHS,
                  validation_data=val_ds.prefetch(tf.data.AUTOTUNE),
                  class_weight=class_weight, callbacks=cb, verbose=1)
        # BCC-score: softmax[BCC_IDX] por ejemplo — el único canal que el gate binario necesita
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

    # Misma proporción inversa del desbalance BCC del baseline (S2): idéntica receta para comparar honestamente
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
    """Escribe metrics.json final y lo imprime (config + métricas + tiempo).

    Parámetros: cfg — 'A'..'E'; t0 — time.time() de inicio de la corrida;
                out_dir — Path ya creado por el llamador; rep — dict de
                metrics.reporte ya calculado sobre el val final.
    Retorna: 0 siempre (es el código de salida de run()/main()).
    Lanza: errores de escritura del JSON propagan (out_dir debe ser escribible).
    Globales: ninguna.
    Nota: `segundos` redondea el tiempo total de run() a 2 decimales (referencia
    de costo por experimento, no una métrica clínica).
    """
    payload = {"config": cfg, **rep, "segundos": round(time.time() - t0, 2)}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print("\n== RESULTADO", cfg, "==")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    """CLI de experimento: --config A..E; artefactos locales en --out-dir/exp-X/.

    Parámetros: ninguno (los flags llegan por línea de comandos vía argparse).
    Retorna: código de salida de run() (0). argparse sale con 2 ante --config inválido.
    Lanza: nada propio (run() propaga sus errores al proceso).
    Globales: OUT_DIR (fallback de --out-dir).
    Nota: el directorio de artefactos es siempre `<out-dir>/exp-<cfg>/` — un nivel
    extra, para que varias corridas A..E compartan la misma raíz sin pisarse.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, choices=["A", "B", "C", "D", "E"])
    ap.add_argument("--out-dir", type=Path, default=Path(os.environ.get("OUT_DIR", "/content/out")))
    args = ap.parse_args()
    return run(args.config, args.out_dir / f"exp-{args.config}")


if __name__ == "__main__":
    sys.exit(main())
