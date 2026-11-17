package com.dermica.bcc.data.history

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Clases de objeto de historial guardadas en `history_events.kind` (S6).
 *
 * Lugar en el flujo: datos (data/history). Lo consumen HistoryMappers
 * (toItemOrNull discrimina por kind), HistoryRepository (escribe kind en cada
 * inserción) y la UI vía HistoryItem (sealed). Cubierto indirectamente por
 * HistoryMappersTest.
 */
object HistoryKind {
    /** Predicción sobre una sola imagen dermatoscópica. */
    const val IMAGE = "image"

    /** Procesamiento del modo subconjunto (lo alimenta T4). */
    const val SUBSET = "subset"
}

/**
 * Fila persistente del historial (tabla `history_events`, S6/S9).
 *
 * Lugar en el flujo: datos (data/history), entidad Room. La escribe
 * HistoryRepository (única vía de escritura) y la lee HistoryDao.observeAll;
 * HistoryMappers la traduce a HistoryItem para la UI. Salida visible: el
 * contenido de las cards del historial tras el mapeo.
 *
 * Una sola tabla para ambos tipos de evento: los campos que no aplican a un
 * kind quedan NULL (imagen única no tiene totales de matriz; subconjunto no
 * tiene nombre de archivo). Solo id/kind/createdAt son obligatorios.
 *
 * @param id — Long: clave primaria autogenerada (0 = aún no persistido).
 * @param kind — String: discriminador IMAGE|SUBSET (ver [HistoryKind]).
 * @param createdAt — Long: epoch millis UTC de creación (la UI formatea con la zona local).
 *
 * Decisiones:
 * - createdAt como epochMillis (UTC implícito): la UI formatea con la zona local.
 * - verdict como texto ("BCC"/"NO_BCC") para que la tabla sea legible por SQL
 *   sin depender de serialización de enums.
 * - Métricas (ta/nfs/peo) en REAL: NFS es un conteo entero persistido como
 *   REAL por uniformidad de la familia numérica; el dominio lo expone como Int.
 * - Las fórmulas de S4 NUNCA se calculan aquí: solo se persisten resultados.
 */
@Entity(tableName = "history_events")
data class HistoryEvent(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val kind: String,
    val createdAt: Long,
    // Evento imagen única
    val imageName: String? = null,
    val verdict: String? = null,
    val confidence: Float? = null,
    // Evento subconjunto
    val folderLabel: String? = null,
    val imageCount: Int? = null,
    val tp: Int? = null,
    val tn: Int? = null,
    val fp: Int? = null,
    val fn: Int? = null,
    val ta: Double? = null,
    val nfs: Double? = null,
    val peo: Double? = null,
    // Evento subconjunto (v2): imágenes de la carpeta sin entrada en el ground truth,
    // excluidas de la matriz de confusión (S3). Nullable: las filas v1 quedan NULL → 0 al mapear.
    val excludedCount: Int? = null,
)
