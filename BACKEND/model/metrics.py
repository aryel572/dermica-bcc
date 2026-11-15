#!/usr/bin/env python3
"""Métricas del modelo (binario BCC vs no-BCC).

Cálculo de métricas sobre predicciones sigmoid: promedio de precisión (AUPRC,
métrica primaria del proyecto), recall (sensibilidad), especificidad,
balanced accuracy y F2. Todas partiendo de las salidas del modelo en [0, 1].
"""

import numpy as np


def auprc(y_true, y_score):
    """AUPRC vía scikit-learn (average_precision)."""
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y_true, y_score))


def confusion_at(y_true, y_score, threshold=0.5):
    """Divide las predicciones en TP/FP/TN/FN según el umbral."""
    preds = (np.asarray(y_score) >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    return tp, fp, tn, fn


def reporte(y_true, y_score, threshold=0.5):
    """Devuelve el dict completo de métricas."""
    tp, fp, tn, fn = confusion_at(y_true, y_score, threshold)
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_acc = (recall + specificity) / 2.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    beta2 = 4.0  # F2: recall pesa el doble que precision
    f2 = (1 + beta2) * precision * recall / ((beta2 * precision) + recall) if (precision + recall) else 0.0
    return {
        "auprc": auprc(y_true, y_score),
        "recall": recall,
        "sensibilidad": recall,
        "especificidad": specificity,
        "balanced_acc": balanced_acc,
        "f2": f2,
        "precision": precision,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
