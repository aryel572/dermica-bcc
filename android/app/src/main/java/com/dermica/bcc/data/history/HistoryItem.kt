package com.dermica.bcc.data.history

/**
 * Modelo de dominio del historial (lo que consume la UI, S6).
 *
 * Lugar en el flujo: datos (data/history), frontera dominio↔UI. Lo produce
 * HistoryMappers (toItemOrNull) y lo consume ui/history/HistoryScreen para
 * renderizar las cards; también lo usa SubsetViewModel/HistoryRepository por
 * tránsito del addSubsetResult. Salida visible: qué card y qué campos pinta la
 * lista del historial.
 *
 * Sellada por tipo de evento: la card de imagen única y la card de subconjunto
 * renderizan campos distintos; el `when` exhaustivo en la UI garantiza que
 * ninguno quede sin estilo. Los mapeos se hacen en `HistoryMappers.kt`.
 */
sealed interface HistoryItem {
    val id: Long
    val createdAt: Long

    /**
     * Card de predicción individual: título = nombre del archivo.
     *
     * @param id — Long: clave primaria en history_events (identifica la fila).
     * @param createdAt — Long: epoch millis UTC (la UI formatea con DateTimes).
     * @param imageName — String?: nombre del archivo; null si la predicción se hizo
     *        sin nombre de origen conocido.
     * @param verdict — String: "BCC" o "NO_BCC" (ver HistoryMappers.VERDICT_*).
     * @param confidence — Float: probabilidad de la clase predicha [0,1].
     */
    data class Image(
        override val id: Long,
        override val createdAt: Long,
        val imageName: String?,
        val verdict: String,
        val confidence: Float,
    ) : HistoryItem

    /**
     * Card de subconjunto: título = carpeta; TA/NFS/PEO siempre visibles.
     *
     * @param id — Long: clave primaria en history_events.
     * @param createdAt — Long: epoch millis UTC.
     * @param folderLabel — String: etiqueta de la carpeta procesada.
     * @param imageCount — Int: imágenes evaluadas (excluye las sin etiqueta).
     * @param excludedCount — Int: imágenes sin entrada en el ground truth, fuera de
     *        la matriz (S3); 0 para filas persistidas antes de la migración v2.
     * @param ta — Double: tasa de aciertos [0,100].
     * @param nfs — Int: falsas sospechas (equivale a FP).
     * @param peo — Double: errores de omisión [0,100].
     * @param tp/tn/fp/fn — Int: totales de la matriz de confusión.
     */
    data class Subset(
        override val id: Long,
        override val createdAt: Long,
        val folderLabel: String,
        val imageCount: Int,
        val excludedCount: Int = 0,
        val ta: Double,
        val nfs: Int,
        val peo: Double,
        val tp: Int,
        val tn: Int,
        val fp: Int,
        val fn: Int,
    ) : HistoryItem
}
