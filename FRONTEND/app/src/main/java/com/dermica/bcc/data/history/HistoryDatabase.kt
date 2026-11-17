package com.dermica.bcc.data.history

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

/**
 * Base de datos Room del historial (S6/S9). Versión 1; sin migraciones todavía.
 * Si el esquema cambia antes del release 1.0.0, bumpear `version` y agregar
 * la migración correspondiente (exportSchema queda activo vía ksp arg).
 */
@Database(entities = [HistoryEvent::class], version = 1, exportSchema = true)
abstract class HistoryDatabase : RoomDatabase() {

    abstract fun historyDao(): HistoryDao

    companion object {
        private const val DB_NAME = "dermica_history.db"

        @Volatile
        private var instance: HistoryDatabase? = null

        /** Singleton de proceso; Room.databaseBuilder es seguro desde cualquier hilo. */
        fun getInstance(context: Context): HistoryDatabase =
            instance ?: synchronized(this) {
                instance ?: Room.databaseBuilder(
                    context.applicationContext,
                    HistoryDatabase::class.java,
                    DB_NAME,
                ).build().also { instance = it }
            }
    }
}
