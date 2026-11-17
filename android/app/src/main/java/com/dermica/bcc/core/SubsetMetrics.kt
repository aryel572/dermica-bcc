package com.dermica.bcc.core

/**
 * Totales de la matriz de confusión del modo subconjunto (S4).
 *
 * Lugar en el flujo: núcleo puro (sin Android, sin Room). Lo construyen
 * ui/subset/SubsetViewModel (al final del lote) y HistoryRepository (al persistir
 * el resultado); lo consume [compute], que deriva TA/NFS/PEO. Cubierto por
 * SubsetMetricsTest en JVM.
 *
 * Qué significa cada campo:
 * - tp: imágenes CON carcinoma basocelular detectadas como carcinoma basocelular.
 * - tn: imágenes SIN carcinoma basocelular detectadas como no carcinoma.
 * - fp: imágenes SIN carcinoma basocelular detectadas como carcinoma (falsas sospechas).
 * - fn: imágenes CON carcinoma basocelular detectadas como no carcinoma (omisiones).
 *
 * Decisiones de borde (S4, especificación del usuario):
 * - N = 0 (subconjunto vacío o sin etiquetados): TA = 0 y PEO = 0.
 * - TP + FN = 0 (no hay positivos reales): PEO = 0 para no dividir entre cero.
 * - NFS es un conteo entero (= FP); se reporta como número exacto, no como porcentaje.
 *
 * Las fórmulas viven SOLO aquí (un único punto de verdad, testeable en JVM);
 * ninguna pantalla las muestra: la UI renderiza únicamente los resultados.
 */
data class Totals(
    val tp: Int,
    val tn: Int,
    val fp: Int,
    val fn: Int,
) {
    /**
     * Calcula TA (%), NFS (entero) y PEO (%) a partir de los totales.
     *
     * @return SubsetMetrics: ta y peo en porcentaje [0,100] (0 si la población es 0),
     *         nfs como conteo exacto de falsas sospechas.
     */
    fun compute(): SubsetMetrics {
        val total = tp + tn + fp + fn
        val ta = if (total == 0) 0.0 else (tp + tn).toDouble() / total * 100.0
        val peo = if (tp + fn == 0) 0.0 else fn.toDouble() / (tp + fn) * 100.0
        return SubsetMetrics(ta = ta, nfs = fp, peo = peo)
    }
}

/**
 * Resultado de las métricas del subconjunto.
 *
 * Lugar en el flujo: núcleo puro; lo produce [Totals.compute] y lo consumen
 * ui/subset/SubsetViewModel (pantalla de resultado), HistoryRepository (persiste
 * en history_events) y HistoryItem.Subset (lo renderiza HistoryScreen). Cubierto
 * por SubsetMetricsTest en JVM.
 *
 * @param ta — Double: tasa de aciertos en porcentaje [0,100] ((tp+tn)/N·100).
 * @param nfs — Int: número de falsas sospechas (equivale a FP), conteo exacto.
 * @param peo — Double: porcentaje de errores de omisión en [0,100] (fn/(tp+fn)·100).
 */
data class SubsetMetrics(
    val ta: Double,
    val nfs: Int,
    val peo: Double,
)
