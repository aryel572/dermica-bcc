package com.dermica.bcc.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.widthIn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.dermica.bcc.data.model.ModelState
import java.util.Locale

// Datos estáticos del release 1.0.0 (asset base + metrics JSON del release);
// 1.0.0 comparte versión con la app hoy. AUPRC = en validación (val), no en test.
private const val MODEL_VERSION_TEXT = "1.0.0"
private const val MODEL_DETAILS_TEXT = "v1.0.0 · float16 · 1.95 MB · AUPRC 0.7786 en validación"
private const val MODEL_QUANTIZATION_TEXT = "float16"

/**
 * Sección Configuración (S1/S7/S8): estado del modelo, borrado con confirmación
 * y flujo de descarga/actualización contra el backend.
 *
 * State hoisting: la pantalla recibe estado + callbacks; el trabajo vive en
 * [SettingsViewModel]; la confirmación del borrado es estado local de UI.
 */
@Composable
fun SettingsScreen(
    modelState: ModelState,
    releaseState: ReleaseUiState,
    /** S10: sin APP_API_KEY los botones de red quedan deshabilitados con texto honesto. */
    isApiKeyConfigured: Boolean,
    onDeleteModel: () -> Unit,
    onCheckDownload: () -> Unit,
    modifier: Modifier = Modifier,
) {
    var confirmDelete by remember { mutableStateOf(false) }
    val modelReady = modelState == ModelState.READY
    val working = releaseState is ReleaseUiState.Checking || releaseState is ReleaseUiState.Downloading

    Surface(modifier = modifier.fillMaxSize(), color = MaterialTheme.colorScheme.surface) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(20.dp)
                .widthIn(max = 520.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Text(
                text = "Configuración",
                style = MaterialTheme.typography.headlineMedium,
                color = MaterialTheme.colorScheme.primary,
            )

            Text(text = "Modelo", style = MaterialTheme.typography.titleMedium)
            ModelCard(modelState)

            // Filas estáticas del release instalado; solo tienen sentido con modelo.
            if (modelReady) {
                InfoRow(label = "Versión instalada", value = MODEL_VERSION_TEXT)
                InfoRow(label = "Cuantización", value = MODEL_QUANTIZATION_TEXT)
                // Truncada (primeros 8 hex): verificación rápida visual, sin loguear ni copiar todo.
                InfoRow(label = "SHA-256", value = MODEL_SHA_TRUNCATED)
            }

            // «Eliminar modelo»: visible solo si hay modelo instalado (S7).
            if (modelReady) {
                OutlinedButton(
                    onClick = { confirmDelete = true },
                    colors = ButtonDefaults.outlinedButtonColors(
                        contentColor = MaterialTheme.colorScheme.error,
                    ),
                ) {
                    Icon(Icons.Filled.Delete, contentDescription = null, modifier = Modifier.padding(end = 8.dp))
                    Text("Eliminar modelo")
                }
            }

            Text(text = "Actualización", style = MaterialTheme.typography.titleMedium)

            // Botón siempre visible: descarga si falta el modelo o busca versión nueva (S8).
            Button(
                onClick = onCheckDownload,
                enabled = isApiKeyConfigured && !working,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(if (working) "Buscando…" else if (modelReady) "Buscar actualización" else "Descargar modelo")
            }
            if (!isApiKeyConfigured) {
                // Honesto y sin exponer nada: falta la clave en local.properties (gitignored).
                Text(
                    text = "Configura APP_API_KEY en FRONTEND/local.properties para descargar.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                )
            }

            DownloadProgress(releaseState)
        }
    }

    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            icon = { Icon(Icons.Filled.Warning, contentDescription = null, tint = MaterialTheme.colorScheme.error) },
            title = { Text("Eliminar modelo") },
            text = {
                Text(
                    "Se eliminará el modelo de este dispositivo. La aplicación quedará " +
                        "sin modelo hasta que lo descargues de nuevo desde el servidor.",
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        confirmDelete = false
                        onDeleteModel()
                    },
                    colors = ButtonDefaults.textButtonColors(contentColor = MaterialTheme.colorScheme.error),
                ) { Text("Eliminar") }
            },
            dismissButton = {
                TextButton(onClick = { confirmDelete = false }) { Text("Cancelar") }
            },
        )
    }
}

/** Estado del modelo always-visible y coherente con ModelState (S1/S7). */
@Composable
private fun ModelCard(modelState: ModelState) {
    ElevatedCard(modifier = Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier.padding(16.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            when (modelState) {
                ModelState.READY -> {
                    Icon(
                        Icons.Filled.CheckCircle,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(28.dp),
                    )
                    Column {
                        Text(text = "Modelo listo", style = MaterialTheme.typography.titleSmall)
                        Text(
                            text = MODEL_DETAILS_TEXT,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
                ModelState.COPYING -> {
                    CircularProgressIndicator(modifier = Modifier.size(24.dp), strokeWidth = 2.dp)
                    Text(text = "Preparando modelo…", style = MaterialTheme.typography.titleSmall)
                }
                ModelState.NO_MODEL -> {
                    Icon(
                        Icons.Filled.Warning,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.error,
                        modifier = Modifier.size(28.dp),
                    )
                    Column {
                        Text(
                            text = "Sin modelo",
                            style = MaterialTheme.typography.titleSmall,
                            color = MaterialTheme.colorScheme.error,
                        )
                        Text(
                            text = "La aplicación no puede evaluar imágenes hasta que descargues el modelo desde el servidor.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
                ModelState.FAILED -> {
                    Icon(
                        Icons.Filled.Warning,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.error,
                        modifier = Modifier.size(28.dp),
                    )
                    Column {
                        Text(
                            text = "Error de integridad del modelo",
                            style = MaterialTheme.typography.titleSmall,
                            color = MaterialTheme.colorScheme.error,
                        )
                        Text(
                            text = "No se pudo verificar el modelo. Descárgalo de nuevo para restaurarlo.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }
        }
    }
}

/** Fila etiqueta/valor para los datos estáticos del release (S7). */
@Composable
private fun InfoRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(text = value, style = MaterialTheme.typography.bodyMedium)
    }
}

/**
 * Progreso/resultados del flujo de descarga (S8). Durante el ping se avisa del
 * arranque en frío de Render free (30–60 s); durante la descarga se muestran
 * los bytes (porcentaje si el servidor declara Content-Length, o bytes leídos
 * si viene chunked).
 */
@Composable
private fun DownloadProgress(state: ReleaseUiState) {
    when (state) {
        ReleaseUiState.Idle -> Unit
        ReleaseUiState.Checking -> Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            CircularProgressIndicator(modifier = Modifier.size(22.dp), strokeWidth = 2.dp)
            Text(
                text = "Conectando con el servidor… El plan gratuito puede tardar hasta un minuto en arrancar (arranque en frío).",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        is ReleaseUiState.Downloading -> Column {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                CircularProgressIndicator(modifier = Modifier.size(22.dp), strokeWidth = 2.dp)
                Text(text = downloadingLabel(state), style = MaterialTheme.typography.bodySmall)
            }
            Text(
                text = "La descarga también puede tardar; el arranque en frío del servidor es lo más lento.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        is ReleaseUiState.Success -> Text(
            text = "Modelo v${state.version} instalado y verificado.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.primary,
        )
        ReleaseUiState.UpToDate -> Text(
            text = "Ya tienes la última versión ($MODEL_VERSION_TEXT).",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.primary,
        )
        is ReleaseUiState.Error -> Text(
            text = state.message,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.error,
        )
    }
}

/** Etiqueta de la descarga: MB leídos (y total/porcentaje si el servidor lo declara). */
private fun downloadingLabel(state: ReleaseUiState.Downloading): String {
    val readMb = state.bytesRead / 1_000_000.0
    val total = state.totalBytes
    return if (total != null && total > 0) {
        val totalMb = total / 1_000_000.0
        val percent = (state.bytesRead * 100 / total).coerceIn(0, 100)
        "Descargando… %.1f de %.1f MB (%d%%)".format(Locale.US, readMb, totalMb, percent)
    } else {
        "Descargando… %.1f MB leídos".format(Locale.US, readMb)
    }
}

// SHA-256 del asset base 1.0.0, truncado a 8 hex para lectura visual (el valor
// completo se verifica programáticamente contra ModelManager.EXPECTED_SHA256).
private const val MODEL_SHA_TRUNCATED = "d62c401c…"
