package com.dermica.bcc.data.history

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

/**
 * Base de datos Room del historial (S6/S9). Versión 2; migración v1 → v2 exportada.
 *
 * Lugar en el flujo: datos (data/history), @Database Room. La consume AppRoot
 * (construye el singleton una vez por proceso) y la usa HistoryRepository como
 * backend. Salida visible: el Flow de observeAll a través del DAO.
 *
 * Las migraciones son aditivas y conservan el historial: nunca se borra información
 * del usuario por un bumpeo de esquema.
 */
@Database(entities = [HistoryEvent::class], version = 2, exportSchema = true)
abstract class HistoryDatabase : RoomDatabase() {

    /** DAO del historial. @return HistoryDao listo para HistoryRepository. */
    abstract fun historyDao(): HistoryDao

    companion object {
        // Nombre del archivo Room en su sandbox; no es configurable ni legible por otros procesos.
        private const val DB_NAME = "dermica_history.db"

        // Migración aditiva v1→v2 (T4): imágenes sin etiqueta del ground truth (S3).
        // Nullable y SIN default SQL: las filas v1 quedan NULL y el mapper las traduce a 0.
        private val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE history_events ADD COLUMN excludedCount INTEGER")
            }
        }

        // Singleton de proceso; @Volatile garantiza visibilidad entre hilos del doble chequeo.
        @Volatile
        private var instance: HistoryDatabase? = null

        /**
         * Singleton de proceso; Room.databaseBuilder es seguro desde cualquier hilo.
         *
         * @param context — Context: se usa applicationContext (no fugas de Activity).
         * @return HistoryDatabase: instancia única por proceso, con MIGRATION_1_2 aplicada.
         *
         * Efectos colaterales: la primera llamada crea el archivo de base de datos
         * y aplica la migración si venía de v1.
         */
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
