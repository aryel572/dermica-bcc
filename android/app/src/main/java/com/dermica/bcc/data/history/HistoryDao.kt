package com.dermica.bcc.data.history

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

/**
 * Acceso a `history_events` (S6).
 *
 * Lugar en el flujo: datos (data/history), Room DAO. Lo consume
 * HistoryRepository (única vía de lectura/escritura; la UI no toca el DAO
 * directamente). Cubierto indirectamente por HistoryMappersTest/Room en
 * instrumentación; en JVM se prueba el mapper, no el DAO.
 *
 * observeAll emite una nueva lista en cada inserción/borrado, ordenada por
 * fecha descendente (lo más reciente primero): la UI solo colecta el Flow.
 */
@Dao
interface HistoryDao {

    /**
     * Todos los eventos ordenados por createdAt DESC (lo más reciente primero).
     *
     * @return Flow<List<HistoryEvent>>: se reevalúa en cada cambio de tabla
     *         (Room invalida y reemite la lista completa); nunca emite listas
     *         parciales por kind, el filtrado vive en HistoryMappers.
     */
    @Query("SELECT * FROM history_events ORDER BY createdAt DESC")
    fun observeAll(): Flow<List<HistoryEvent>>

    /**
     * Inserta un evento en la tabla.
     *
     * @param event — HistoryEvent: fila a persistir; el id se autogenera
     *        (se ignora cualquier id precargado).
     *
     * Efectos colaterales: escribe en history_events y dispara la reemisión del
     * Flow de [observeAll].
     */
    @Insert
    suspend fun insert(event: HistoryEvent)
}
