#!/usr/bin/env python3
"""Métricas del modelo (binario BCC vs no-BCC).

Cálculo de métricas sobre predicciones sigmoid: AUPRC (área bajo la curva
precisión-recall), que es la métrica PRIMARIA del proyecto para desbalanceo de
clases (BCC ≈ 12.76 % del train), recall (sensibilidad), especificidad,
balanced accuracy y F2. Todas partiendo de las salidas del modelo en [0, 1].

Decisiones de definición:
- AUPRC se usa como métrica primaria porque, con ~13 % de prevalencia, la curva
  ROC luce optimista y el desbalance sesga el accuracy binario; PR curve y su
  área son más informativas clínicamente.
- F2 (no F1): recall pesa el doble que precisión, alineado con la prioridad
  clínica de no omitir BCC (error de omisión inaceptable).
"""

import numpy as np


def auprc(y_true, y_score):
    """AUPRC (precisión promedio) vía scikit-learn.

    Params:
        y_true  — etiquetas binarias 0/1 (iterable de len n).
        y_score — puntuaciones/probabilidad de la clase positiva BCC en [0, 1].
    Returns:
        float con el average_precision (área bajo la curva precisión-recall).
    Por qué sklearn: implementa AP nativamente con mismos def que en S1/S2.
    """
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y_true, y_score))


def confusion_at(y_true, y_score, threshold=0.5):
    """Divide las predicciones en TP/FP/TN/FN según el umbral.

    Params:
        y_true    — etiquetas binarias 0/1.
        y_score   — puntuaciones/probabilidad en [0, 1].
        threshold — corte de decisión (por defecto 0.5, neutral para sigmoide).
    Returns:
        tuple (tp, fp, tn, fn) con conteos enteros.
    Por qué umbral 0.5: la sigmoide del modelo y la F2 no penalizan el umbral;
    si se requiere calibración clínica, ajustar acá (no en el cálculo).
    """
    preds = (np.asarray(y_score) >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    return tp, fp, tn, fn


def reporte(y_true, y_score, threshold=0.5):
    """Devuelve el dict completo de métricas sobre el umbral dado.

    Params:
        y_true    — etiquetas binarias 0/1.
        y_score   — puntuaciones/probabilidad en [0, 1].
        threshold — umbral de decisión (por defecto 0.5).
    Returns:
        dict con auprc, recall/sensibilidad, especificidad, balanced_acc, f2,
        precision, y los conteos crudos tp/fp/tn/fn.
    Por qué no fallar con divisiones por cero: se devuelve 0.0 y se sigue
    (métricas documentales, no deben tirar abajo el entrenamiento).
    """
    tp, fp, tn, fn = confusion_at(y_true, y_score, threshold)
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_acc = (recall + specificity) / 2.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    beta2 = 4.0  # F2: recall pesa el doble que precision (β=2 → β²=4)
    f2 = (1 + beta2) * precision * recall / ((beta2 * precision) + recall) if (precision + recall) else 0.0
    return {
        "auprc": auprc(y_true, y_score),
        "recall": recall,
        "sensibilidad": recall,          # alias clínico de recall
        "especificidad": specificity,
        "balanced_acc": balanced_acc,
        "f2": f2,
        "precision": precision,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
