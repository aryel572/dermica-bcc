package com.dermica.bcc.data.history

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

/**
 * Acceso a `history_events` (S6).
 *
 * observeAll emite una nueva lista en cada inserción/borrado, ordenada por
 * fecha descendente (lo más reciente primero): la UI solo colecta el Flow.
 */
@Dao
interface HistoryDao {

    @Query("SELECT * FROM history_events ORDER BY createdAt DESC")
    fun observeAll(): Flow<List<HistoryEvent>>

    @Insert
    suspend fun insert(event: HistoryEvent)
}
