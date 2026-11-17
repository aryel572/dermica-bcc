package com.dermica.bcc.data.history

/**
 * Modelo de dominio del historial (lo que consume la UI, S6).
 *
 * Sellada por tipo de evento: la card de imagen única y la card de subconjunto
 * renderizan campos distintos; el `when` exhaustivo en la UI garantiza que
 * ninguno quede sin estilo. Los mapeos se hacen en `HistoryMappers.kt`.
 */
sealed interface HistoryItem {
    val id: Long
    val createdAt: Long

    /** Card de predicción individual: título = nombre del archivo. */
    data class Image(
        override val id: Long,
        override val createdAt: Long,
        val imageName: String?,
        val verdict: String,
        val confidence: Float,
    ) : HistoryItem

    /** Card de subconjunto: título = carpeta; TA/NFS/PEO siempre visibles. */
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
