package com.dermica.bcc.data.model

import android.util.Log
import androidx.datastore.core.handlers.ReplaceFileCorruptionHandler
import androidx.datastore.preferences.core.emptyPreferences
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
// Fix reviewer-corroborated R4-002: si model_prefs se corrompe o no se puede leer
// (IOException/CorruptionException), el handler reemplaza el archivo por
// Preferencias vacías. Semántica correcta: prefs ilegibles => NO borrado por el
// usuario => ensureModel() re-copia el asset como en un primer arranque
// (fallo seguro: la app nunca queda sin modelo por un archivo dañado).
// Nombre de archivo «model_prefs» elegido por compatibilidad: ya estaba en uso;
// cambiarlo huerfanaría el flag persistido por los borrados previos del usuario.
private val android.content.Context.modelDataStore by preferencesDataStore(
    name = "model_prefs",
    corruptionHandler = ReplaceFileCorruptionHandler { emptyPreferences() },
)

// Flag persistente «el usuario borró el modelo» (S7): gobierna que ensureModel()
// NO re-copie el asset tras un borrado explícito. Consumidor: ensureModel,
// deleteModel, restoreFromAsset, installDownloadedModel.
private val KEY_DELETED_BY_USER = booleanPreferencesKey("model_deleted_by_user")

private const val TAG = "ModelManager"

// Nombre del archivo interno del modelo en filesDir (clave de toda la vida del artefacto).
private const val MODEL_FILE_NAME = "model.tflite"

// Asset empaquetado en el APK (float16, release 1.0.0); fuente de la copia inicial.
private const val ASSET_NAME = "model-float16.tflite"

/**
 * Gestiona el ciclo de vida del modelo base empaquetado en assets (S1):
 * copia asset → filesDir/model.tflite en el primer arranque, verifica SHA-256
 * y publica un estado observable para la UI.
 *
 * Lugar en el flujo: datos (data/model) — el único escritor del artefacto
 * interno. Lo consume ui/AppRoot (ensureModel en el arranque; gobernar
 * BccPredictor y el CTA «Imagen única» según ModelState) y
 * ui/settings/SettingsViewModel (deleteModel/restoreFromAsset/
 * installDownloadedModel). Salida visible: el StateFlow<ModelState> que expone
 * [state] y que pintan HomeScreen y SettingsScreen.
 *
 * Reglas del ciclo de vida:
 * - Si existe filesDir/model.tflite, se verifica siempre; nunca se asume íntegro.
 * - Si el usuario lo borró (flag model_deleted_by_user=true) y no hay modelo,
 *   NO se re-copia el asset: la app queda realmente sin modelo hasta que el
 *   usuario restaure o descargue (S1/S7).
 * - Si model_prefs es ilegible, se trata como NO borrado por el usuario
 *   (fix reviewer-corroborated R4-002): se re-copia el asset; nunca crash en arranque.
 * - Si la verificación falla, se borra la copia interna y se publica FAILED:
 *   la app no puede ejecutar inferencia con un modelo corrupto.
 * - Toda la sección crítica (copia/verificación/borrado/instalación) va bajo
 *   [mutex]: una sola operación a la vez, sin estados intermedios visibles.
 *
 * @param context — Context de aplicación: fuente del assets y de filesDir.
 * @param scope — CoroutineScope de la app (AppRoot): reservado para operaciones
 *        de ciclo de vida; hoy el cuerpo no lo usa (todas las funciones suspenden
 *        en el caller), se conserva por compatibilidad de la firma pública.
 */
class ModelManager(
    private val context: android.content.Context,
    private val scope: CoroutineScope,
) {
    private val filesDir: File = context.filesDir
    private val modelFile = File(filesDir, MODEL_FILE_NAME)

    // Candado único de la copia/verificación/borrado/instalación: garantiza que
    // state nunca publique un estado intermedio de una operación a medias.
    private val mutex = Mutex()

    // Estado interno mutable; solo se muta bajo mutex (o en ensureModel). Lo que
    // se expone hacia afuera es la vista asStateFlow() de solo lectura.
    private val _state = MutableStateFlow(ModelState.NO_MODEL)

    /**
     * Estado observable del modelo interno (Flow para colectores Compose).
     *
     * @return StateFlow<ModelState>: siempre con el último estado publicado
     *         (NO_MODEL inicial); los colectores reciben el valor vigente al suscribirse.
     */
    val state: StateFlow<ModelState> = _state.asStateFlow()

    /**
     * Archivo del modelo activo.
     *
     * @return File? con filesDir/model.tflite si existe (sin verificar integridad:
     *         la verificación vive en ensureModel/verifyAndPublish), o null si no
     *         hay copia interna. AppRoot lo usa para construir BccPredictor.
     */
    fun modelFile(): File? = if (modelFile.exists()) modelFile else null

    /**
     * Asegura el modelo en el primer arranque. Idempotente: puede llamarse en cada
     * creación de Activity; solo actúa cuando falta la copia interna o hay que
     * verificar la existente. Nunca re-copia si el flag de borrado está activo.
     *
     * Efectos colaterales: bajo [mutex], puede copiar/verificar/borrar el archivo
     * interno y publicar COPYING/READY/FAILED/NO_MODEL en [state].
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

    /**
     * Borra la copia interna (incluidos temporales) y persiste el flag de borrado (S7).
     *
     * Efectos colaterales: bajo [mutex], borra filesDir/model.tflite* y publica
     * NO_MODEL; persiste model_deleted_by_user=true para que el siguiente
     * ensureModel NO re-copie el asset. Lo dispara SettingsViewModel tras la
     * confirmación del diálogo de borrado.
     */
    suspend fun deleteModel() {
        mutex.withLock {
            filesDir.listFiles { file -> file.name.startsWith(MODEL_FILE_NAME) }
                ?.forEach { it.delete() }
            context.modelDataStore.edit { prefs -> prefs[KEY_DELETED_BY_USER] = true }
            _state.value = ModelState.NO_MODEL
        }
    }

    /**
     * Revierte el borrado: limpia el flag y restaura desde el asset empaquetado (S1).
     *
     * Efectos colaterales: bajo [mutex], borra el flag model_deleted_by_user,
     * copia el asset a filesDir/model.tflite con renombrado atómico, verifica el
     * SHA-256 y publica COPYING→READY|FAILED. Acción expuesta en Settings.
     */
    suspend fun restoreFromAsset() {
        mutex.withLock {
            context.modelDataStore.edit { prefs -> prefs[KEY_DELETED_BY_USER] = false }
            copyFromAssetAndVerify()
        }
    }

    /**
     * Publica el modelo descargado del servidor (S8): verifica el temporal [tmpFile]
     * contra [expectedSha256] y, solo si íntegro, lo instala atómicamente como
     * filesDir/model.tflite (rename; mismo directorio). El modelo actual NUNCA se
     * borra antes de la verificación: si el temporal no pasa, se descarta solo él.
     *
     * Efectos: limpia el flag model_deleted_by_user (el modelo vuelve a existir)
     * y publica READY o FAILED.
     *
     * @param tmpFile — File: temporal producido por [downloadTemporaryFile] en el
     *        mismo filesDir (rename atómico intra-directorio).
     * @param expectedSha256 — String: SHA-256 esperado (del release o, como
     *        fallback del asset base, [EXPECTED_SHA256]).
     * @return Boolean: true si quedó instalado; false si la verificación falló
     *         (nada cambió: el modelo previo sigue intacto).
     */
    suspend fun installDownloadedModel(tmpFile: File, expectedSha256: String): Boolean {
        mutex.withLock {
            val integrityOk = runCatching { Sha256.ofFile(tmpFile) == expectedSha256 }.getOrDefault(false)
            if (!integrityOk) {
                Log.e(TAG, "el modelo descargado no superó la verificación; se descarta")
                tmpFile.delete()
                return false
            }
            // renameTo en el mismo directorio es atómico. Si fallara pese a ello
            // (I/O transitorio), se borra el anterior y se reintenta una vez;
            // nunca queda model.tflite a medias porque el rename es point-in-time.
            if (!tmpFile.renameTo(modelFile)) {
                modelFile.delete()
                if (!tmpFile.renameTo(modelFile)) {
                    Log.e(TAG, "no se pudo renombrar el modelo descargado")
                    tmpFile.delete()
                    return false
                }
            }
            context.modelDataStore.edit { prefs -> prefs[KEY_DELETED_BY_USER] = false }
            _state.value = ModelState.READY
            return true
        }
    }

    companion object {
        /**
         * SHA-256 del artefacto float16 del release 1.0.0 publicado en dermica-models
         * (1 945 356 bytes). Origen: publicación del release 1.0.0 (verificado contra
         * HF al publicar); lo sirve /model/latest como float16_sha256 y lo replica
         * aquí para (a) verificar el asset empaquetado en el primer arranque y
         * (b) servir de fallback de verificación cuando el release no trae SHA.
         * Consumidores: verifyAndPublish, installDownloadedModel y
         * SettingsViewModel (fallback de expectedSha256).
         * Nota: si el release publicara otra cuantización/versión, este valor deja
         * de ser válido para el asset nuevo; antes de cambiarlo, actualizar también
         * ASSET_NAME y el valor del servidor.
         */
        const val EXPECTED_SHA256 = "d62c401c5f66a7eccd5f752b2d7aa73964fe6f75a92acde3cae951b0d911cc0a"

        // Nombre del temporal de descarga en filesDir (mismo filesystem que el destino:
        // el rename final es atómico). Consumido por downloadTemporaryFile.
        private const val MODEL_DOWNLOAD_TMP_NAME = "$MODEL_FILE_NAME.download"
    }

    /**
     * Temporal para la descarga del release (mismo filesystem: el rename posterior es atómico).
     *
     * @return File: filesDir/model.tflite.download; lo consume SettingsViewModel
     *         como destFile de ModelRelease.downloadModel y lo instala
     *         installDownloadedModel si la verificación es íntegra.
     */
    fun downloadTemporaryFile(): File = File(filesDir, MODEL_DOWNLOAD_TMP_NAME)

    // Fix reviewer-corroborated R4-002: cualquier fallo al leer las prefs se
    // interpreta como "no borrado por el usuario" (fallo seguro: re-copia el asset).
    // El corruptionHandler cubre el archivo dañado; este guard cubre I/O transitoria.
    /**
     * Lee el flag «borrado por el usuario» con fallo seguro.
     *
     * @return Boolean: true solo si model_deleted_by_user está en true; cualquier
     *         error de lectura (archivo ausente, prefs dañadas, I/O transitoria)
     *         se interpreta como false (fallo seguro: se re-copia el asset).
     */
    private suspend fun deletedByUser(): Boolean = runCatching {
        context.modelDataStore.data.first()[KEY_DELETED_BY_USER] ?: false
    }.getOrElse { error ->
        Log.w(TAG, "prefs ilegibles; se asume modelo no borrado: ${error.message}")
        false
    }

    /**
     * Copia asset → filesDir/model.tflite vía temporal y renombrado, luego verifica (S1).
     *
     * Efectos colaterales: publica COPYING al iniciar; borra el temporal y el
     * destino si cualquier paso falla y publica FAILED en ese caso; en éxito
     * delega en verifyAndPublish (READY o FAILED).
     */
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

    /**
     * Verifica SHA-256 del archivo interno y publica READY o FAILED.
     *
     * Efectos colaterales: publica COPYING brevemente durante la verificación;
     * si el digest no coincide con [EXPECTED_SHA256], borra el archivo interno y
     * publica FAILED (la inferencia queda deshabilitada hasta restaurar/descargar).
     */
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
