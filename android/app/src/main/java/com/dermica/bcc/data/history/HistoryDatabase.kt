package com.dermica.bcc.data.history

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

/**
 * Base de datos Room del historial (S6/S9). Versión 2; migración v1 → v2 exportada.
 * Las migraciones son aditivas y conservan el historial: nunca se borra información
 * del usuario por un bumpeo de esquema.
 */
@Database(entities = [HistoryEvent::class], version = 2, exportSchema = true)
abstract class HistoryDatabase : RoomDatabase() {

    abstract fun historyDao(): HistoryDao

    companion object {
        private const val DB_NAME = "dermica_history.db"

        /** v1 → v2 (T4): columna `excludedCount` (imágenes sin etiqueta, S3). Nullable, sin default SQL. */
        private val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE history_events ADD COLUMN excludedCount INTEGER")
            }
        }

        @Volatile
        private var instance: HistoryDatabase? = null

        /** Singleton de proceso; Room.databaseBuilder es seguro desde cualquier hilo. */
        fun getInstance(context: Context): HistoryDatabase =
            instance ?: synchronized(this) {
                instance ?: Room.databaseBuilder(
                    context.applicationContext,
                    HistoryDatabase::class.java,
                    DB_NAME,
                ).addMigrations(MIGRATION_1_2).build().also { instance = it }
            }
    }
}
