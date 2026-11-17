package com.dermica.bcc.data.model

import android.util.Log
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File

// DataStore dedicado del modelo: flag de borrado por el usuario (S1/S7).
private val android.content.Context.modelDataStore by preferencesDataStore(name = "model_prefs")
private val KEY_DELETED_BY_USER = booleanPreferencesKey("model_deleted_by_user")

private const val TAG = "ModelManager"
private const val MODEL_FILE_NAME = "model.tflite"
private const val ASSET_NAME = "model-float16.tflite"

/**
 * Gestiona el ciclo de vida del modelo base empaquetado en assets (S1):
 * copia asset → filesDir/model.tflite en el primer arranque, verifica SHA-256
 * y publica un estado observable para la UI.
 *
 * Reglas del ciclo de vida:
 * - Si existe filesDir/model.tflite, se verifica siempre; nunca se asume íntegro.
 * - Si el usuario lo borró (flag model_deleted_by_user=true) y no hay modelo,
 *   NO se re-copia el asset: la app queda realmente sin modelo hasta que el
 *   usuario restaure o descargue (S1/S7).
 * - Si la verificación falla, se borra la copia interna y se publica FAILED:
 *   la app no puede ejecutar inferencia con un modelo corrupto.
 */
class ModelManager(
    private val context: android.content.Context,
    private val scope: CoroutineScope,
) {
    private val filesDir: File = context.filesDir
    private val modelFile = File(filesDir, MODEL_FILE_NAME)
    private val mutex = Mutex()

    private val _state = MutableStateFlow(ModelState.NO_MODEL)

    /** Estado observable del modelo interno (Flow para colectores Compose). */
    val state: StateFlow<ModelState> = _state.asStateFlow()

    /** Archivo del modelo activo, o null si no hay copia interna íntegra disponible. */
    fun modelFile(): File? = if (modelFile.exists()) modelFile else null

    /**
     * Asegura el modelo en el primer arranque. Idempotente: puede llamarse en cada
     * creación de Activity; solo actúa cuando falta la copia interna o hay que
     * verificar la existente. Nunca re-copia si el flag de borrado está activo.
     */
    suspend fun ensureModel() {
        mutex.withLock {
            if (modelFile.exists()) {
                verifyAndPublish()
                return@withLock
            }
            if (deletedByUser()) {
                _state.value = ModelState.NO_MODEL
                return@withLock
            }
            copyFromAssetAndVerify()
        }
    }

    /** Borra la copia interna (incluidos temporales) y persiste el flag de borrado (S7). */
    suspend fun deleteModel() {
        mutex.withLock {
            filesDir.listFiles { file -> file.name.startsWith(MODEL_FILE_NAME) }
                ?.forEach { it.delete() }
            context.modelDataStore.edit { prefs -> prefs[KEY_DELETED_BY_USER] = true }
            _state.value = ModelState.NO_MODEL
        }
    }

    /** Revierte el borrado: limpia el flag y restaura desde el asset empaquetado (S1). */
    suspend fun restoreFromAsset() {
        mutex.withLock {
            context.modelDataStore.edit { prefs -> prefs[KEY_DELETED_BY_USER] = false }
            copyFromAssetAndVerify()
        }
    }

    /** SHA-256 esperado del asset base (release 1.0.0, float16). Constante, no secreto. */
    companion object {
        const val EXPECTED_SHA256 = "d62c401c5f66a7eccd5f752b2d7aa73964fe6f75a92acde3cae951b0d911cc0a"
    }

    private suspend fun deletedByUser(): Boolean =
        context.modelDataStore.data.first()[KEY_DELETED_BY_USER] ?: false

    /** Copia asset → filesDir/model.tflite vía temporal y renombrado, luego verifica (S1). */
    private suspend fun copyFromAssetAndVerify() {
        _state.value = ModelState.COPYING
        val tmp = File(filesDir, "$MODEL_FILE_NAME.tmp")
        try {
            context.assets.open(ASSET_NAME).use { input ->
                tmp.outputStream().use { output -> input.copyTo(output) }
            }
            // Renombrado atómico dentro del mismo filesystem: nunca queda un modelo a medias.
            if (!tmp.renameTo(modelFile)) {
                tmp.delete()
                check(false) { "no se pudo renombrar ${tmp.name} tras la copia" }
            }
            verifyAndPublish()
        } catch (e: Exception) {
            // Sin detalles de contenido: solo el mensaje; el modelo corrupto nunca se usa.
            Log.e(TAG, "copia/verificación del modelo fallida: ${e.message}")
            tmp.delete()
            modelFile.delete()
            _state.value = ModelState.FAILED
        }
    }

    /** Verifica SHA-256 del archivo interno y publica READY o FAILED (borrando el corrupto). */
    private suspend fun verifyAndPublish() {
        _state.value = ModelState.COPYING // verificación breve durante el arranque
        val integrityOk = runCatching { Sha256.ofFile(modelFile) == EXPECTED_SHA256 }
            .getOrDefault(false)
        if (integrityOk) {
            _state.value = ModelState.READY
        } else {
            Log.e(TAG, "SHA-256 del modelo interno no coincide; se elimina la copia corrupta")
            modelFile.delete()
            _state.value = ModelState.FAILED
        }
    }
}
