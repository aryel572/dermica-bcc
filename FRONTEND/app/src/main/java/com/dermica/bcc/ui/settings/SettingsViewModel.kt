package com.dermica.bcc.ui.settings

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.dermica.bcc.BuildConfig
import com.dermica.bcc.data.model.ModelManager
import com.dermica.bcc.data.net.LOCAL_MODEL_VERSION
import com.dermica.bcc.data.net.ModelRelease
import com.dermica.bcc.data.net.ReleaseException
import com.dermica.bcc.data.net.UpdateDecision
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Estado del flujo «Buscar actualización» / «Descargar modelo» (S8).
 * Una sola fuente de verdad para la UI; nunca hay estados paralelos.
 *
 * - [Checking] cubre el ping público y /model/latest; en Render free el arranque
 *   en frío puede tardar 30–60 s (la UI muestra un aviso honesto).
 * - [Downloading] reporta bytes leídos y el total si el servidor lo declara.
 * - Estados finales: [Success], [UpToDate], [Error].
 */
sealed interface ReleaseUiState {
    data object Idle : ReleaseUiState
    data object Checking : ReleaseUiState
    data class Downloading(val bytesRead: Long, val totalBytes: Long?) : ReleaseUiState
    data class Success(val version: String) : ReleaseUiState
    data object UpToDate : ReleaseUiState
    data class Error(val message: String) : ReleaseUiState
}

/** Resultado intermedio del trabajo en Dispatchers.IO (no sale del VM). */
private sealed interface DownloadOutcome
private data object UpToDate : DownloadOutcome
private data class Downloaded(val version: String, val expectedSha256: String, val tmpFile: File) : DownloadOutcome

/**
 * Orquestador de la sección Configuración (S7/S8).
 *
 * Responsabilidades:
 * - Disparar [ModelManager.deleteModel] tras la confirmación del diálogo (S7).
 * - Ejecutar el flujo S8: ping → metadata → decisión → descarga con verificación
 *   → instalación atómica vía [ModelManager.installDownloadedModel].
 * - Traducir los fallos a mensajes legibles en español sin exponer la APP_API_KEY.
 */
class SettingsViewModel(
    private val modelManager: ModelManager,
    private val release: ModelRelease = ModelRelease(),
    apiKey: String = BuildConfig.APP_API_KEY,
) : ViewModel() {

    /** S10: sin APP_API_KEY los botones de red se deshabilitan con texto honesto (nunca se loguea). */
    val isApiKeyConfigured: Boolean = apiKey.isNotBlank()
    private val appApiKey = apiKey

    private val _releaseState = MutableStateFlow<ReleaseUiState>(ReleaseUiState.Idle)
    val releaseState: StateFlow<ReleaseUiState> = _releaseState.asStateFlow()

    /** Borra la copia interna del modelo y persiste el flag de borrado (S7). */
    fun deleteModel() {
        viewModelScope.launch { modelManager.deleteModel() }
    }

    /**
     * Flujo S8 completo (ping → metadata → decisión → descarga → instalación).
     * Idempotente: ignora re-entradas mientras hay un trabajo en curso.
     */
    fun checkForUpdateOrDownload() {
        val current = _releaseState.value
        if (current is ReleaseUiState.Checking || current is ReleaseUiState.Downloading) return
        if (!isApiKeyConfigured) {
            _releaseState.value = ReleaseUiState.Error(
                "Configura APP_API_KEY en FRONTEND/local.properties para descargar.",
            )
            return
        }
        viewModelScope.launch {
            _releaseState.value = ReleaseUiState.Checking
            try {
                val outcome = withContext(Dispatchers.IO) {
                    release.checkServer()
                    val info = release.fetchLatest(appApiKey)
                    val modelPresent = modelManager.modelFile() != null
                    if (ModelRelease.decide(info.version, LOCAL_MODEL_VERSION, modelPresent) == UpdateDecision.UP_TO_DATE) {
                        return@withContext UpToDate
                    }
                    // SHA publicado en /model/latest; si el release no lo trae, se usa
                    // la constante del asset base (el asset ES el release 1.0.0).
                    val expectedSha256 = info.float16Sha256 ?: ModelManager.EXPECTED_SHA256
                    val tmpFile = modelManager.downloadTemporaryFile()
                    release.downloadModel(appApiKey, info, expectedSha256, tmpFile) { bytesRead, totalBytes ->
                        _releaseState.value = ReleaseUiState.Downloading(bytesRead, totalBytes)
                    }
                    Downloaded(info.version, expectedSha256, tmpFile)
                }
                when (outcome) {
                    UpToDate -> _releaseState.value = ReleaseUiState.UpToDate
                    is Downloaded -> {
                        val installed = modelManager.installDownloadedModel(outcome.tmpFile, outcome.expectedSha256)
                        _releaseState.value = if (installed) {
                            ReleaseUiState.Success(outcome.version)
                        } else {
                            ReleaseUiState.Error("La descarga no superó la verificación de integridad; no se instaló nada.")
                        }
                    }
                }
            } catch (error: ReleaseException) {
                // Sin la clave ni trazas internas: solo el tipo de fallo en el log.
                Log.w(TAG, "flujo de descarga falló: ${error.kind}")
                _releaseState.value = ReleaseUiState.Error(error.message)
            } catch (error: Exception) {
                // CancellationException jamás se traga (cancela la corrida de verdad).
                if (error is CancellationException) throw error
                Log.e(TAG, "error inesperado en la descarga: ${error.message}")
                _releaseState.value = ReleaseUiState.Error("Error inesperado al descargar el modelo.")
            }
        }
    }
}

private const val TAG = "SettingsViewModel"
