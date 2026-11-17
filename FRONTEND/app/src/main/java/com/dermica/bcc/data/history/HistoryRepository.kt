package com.dermica.bcc.data.history

import com.dermica.bcc.core.SubsetMetrics
import com.dermica.bcc.core.Totals
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

/**
 * Repositorio del historial (S6/S9): único punto de escritura/lectura.
 *
 * El historial es offline-first: las predicciones se persisten localmente al
 * calcularse y la red nunca participa. La UI consume [items] como Flow y no
 * toca el DAO directamente.
 *
 * @param createdAt epochMillis opcional para tests deterministas; en producción
 *                  usa System.currentTimeMillis().
 */
class HistoryRepository(private val dao: HistoryDao) {

    /** Stream reactivo del historial ordenado por fecha descendente. */
    val items: Flow<List<HistoryItem>> =
        dao.observeAll().map { events -> events.mapNotNull { it.toItemOrNull() } }

    /** Registra una predicción de imagen única (se llama al predecir, S6). */
    suspend fun addImagePrediction(
        imageName: String?,
        isBcc: Boolean,
        confidence: Float,
        createdAt: Long = System.currentTimeMillis(),
    ) {
        dao.insert(
            HistoryEvent(
                kind = HistoryKind.IMAGE,
                createdAt = createdAt,
                imageName = imageName,
                verdict = if (isBcc) VERDICT_BCC else VERDICT_NO_BCC,
                confidence = confidence,
            ),
        )
    }

    /** Registra el resultado del modo subconjunto (lo usa T4, S6). */
    suspend fun addSubsetResult(
        folderLabel: String,
        imageCount: Int,
        totals: Totals,
        createdAt: Long = System.currentTimeMillis(),
    ) {
        val metrics: SubsetMetrics = totals.compute()
        dao.insert(
            HistoryEvent(
                kind = HistoryKind.SUBSET,
                createdAt = createdAt,
                folderLabel = folderLabel,
                imageCount = imageCount,
                tp = totals.tp,
                tn = totals.tn,
                fp = totals.fp,
                fn = totals.fn,
                ta = metrics.ta,
                nfs = metrics.nfs.toDouble(),
                peo = metrics.peo,
            ),
        )
    }
}
