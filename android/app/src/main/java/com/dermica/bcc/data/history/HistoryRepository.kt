package com.dermica.bcc.data.history

import com.dermica.bcc.core.SubsetMetrics
import com.dermica.bcc.core.Totals
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

/**
 * Repositorio del historial (S6/S9): único punto de escritura/lectura.
 *
 * Lugar en el flujo: datos (data/history), fachada sobre HistoryDao. Lo consume
 * AppRoot (predict → addImagePrediction tras cada inferencia), SubsetViewModel
 * (addSubsetResult al terminar el lote) y HistoryScreen (colecta items). Salida
 * visible: el Flow<List<HistoryItem>> que pinta las cards del historial.
 *
 * El historial es offline-first: las predicciones se persisten localmente al
 * calcularse y la red nunca participa. La UI consume [items] como Flow y no
 * toca el DAO directamente.
 *
 * @param dao — HistoryDao: acceso Room a history_events (Room lo inyecta vía
 *        HistoryDatabase.getInstance(...).historyDao()).
 * @param createdAt en los métodos de escritura — Long: epoch millis UTC; opcional
 *        para tests deterministas; en producción usa System.currentTimeMillis().
 */
class HistoryRepository(private val dao: HistoryDao) {

    /**
     * Stream reactivo del historial ordenado por fecha descendente.
     *
     * @return Flow<List<HistoryItem>>: reemite la lista completa en cada cambio;
     *         las filas inválidas se descartan en el mapeo (toItemOrNull).
     */
    val items: Flow<List<HistoryItem>> =
        dao.observeAll().map { events -> events.mapNotNull { it.toItemOrNull() } }

    /**
     * Registra una predicción de imagen única (se llama al predecir, S6).
     *
     * @param imageName — String?: nombre del archivo origen; null si se desconoce.
     * @param isBcc — Boolean: true si la inferencia decodificó BCC (umbral 0.5).
     * @param confidence — Float: probabilidad de la clase predicha [0,1].
     * @param createdAt — Long: epoch millis UTC; por defecto System.currentTimeMillis().
     *
     * Efectos colaterales: inserta la fila kind=IMAGE en history_events.
     */
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

    /**
     * Registra el resultado del modo subconjunto (S6). Las métricas se derivan de
     * [totals] aquí (S4: las fórmulas viven solo en SubsetMetrics); `excluded`
     * son las imágenes sin entrada en el ground truth, fuera de la matriz (S3).
     *
     * @param folderLabel — String: etiqueta visible de la carpeta procesada.
     * @param imageCount — Int: imágenes evaluadas (excluye las sin etiqueta).
     * @param totals — Totals: matriz de confusión acumulada por SubsetViewModel.
     * @param excluded — Int: imágenes sin etiqueta, fuera de la matriz (S3).
     * @param createdAt — Long: epoch millis UTC; por defecto System.currentTimeMillis().
     *
     * Efectos colaterales: inserta la fila kind=SUBSET en history_events con TA/NFS/PEO.
     */
    suspend fun addSubsetResult(
        folderLabel: String,
        imageCount: Int,
        totals: Totals,
        excluded: Int = 0,
        createdAt: Long = System.currentTimeMillis(),
    ) {
        val metrics: SubsetMetrics = totals.compute()
        dao.insert(
            HistoryEvent(
                kind = HistoryKind.SUBSET,
                createdAt = createdAt,
                folderLabel = folderLabel,
                imageCount = imageCount,
                excludedCount = excluded,
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
