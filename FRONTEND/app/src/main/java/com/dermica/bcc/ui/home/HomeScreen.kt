package com.dermica.bcc.ui.home

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.rememberCoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Folder
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.dermica.bcc.data.model.ModelState

/**
 * Pantalla de inicio (S2): cabecera de marca, aviso fijo de tamizaje y las dos
 * acciones principales. Estado del modelo siempre visible y coherente con el
 * StateFlow de ModelManager (nunca se inventa un estado intermedio).
 *
 * Diseño state hoisting: la pantalla recibe el estado del modelo y los callbacks;
 * la corrida de la predicción la decide el owner (AppRoot), no la UI.
 */
@Composable
fun HomeScreen(
    modelState: ModelState,
    predicting: Boolean,
    predictionError: String?,
    /** Fix review-corroborated R3-001: fallo de construcción del motor; deshabilita la CTA. */
    modelLoadError: String? = null,
    onImageBytesChosen: (ByteArray, String?) -> Unit,
    onOpenSubset: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    // Fix review-corroborated R4-PICKER-MAIN-IO: antes la lectura ocurría en el hilo
    // principal, sin try/catch; un stream roto crasheaba el Activity y un stream nulo
    // producía el no-op silencioso "elegí imagen y no pasa nada".
    var pickerError by mutableStateOf<String?>(null)
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri ?: return@rememberLauncherForActivityResult
        pickerError = null
        scope.launch {
            // Lectura y metadatos fuera del hilo principal (R4-PICKER-MAIN-IO).
            val loaded = withContext(Dispatchers.IO) {
                runCatching {
                    context.contentResolver.openInputStream(uri)?.use { it.readBytes() } to displayNameOf(context, uri)
                }.getOrElse { error ->
                    Log.w(TAG, "fallo leyendo la imagen del picker: ${error.message}")
                    null to null
                }
            }
            val bytes = loaded.first
            if (bytes == null) {
                // Stream nulo o proveedor roto: mensaje visible, nunca crash ni silencio.
                pickerError = "No se pudo leer la imagen. Intenta con otra."
                return@launch
            }
            onImageBytesChosen(bytes, loaded.second)
        }
    }
    val modelReady = modelState == ModelState.READY

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
                text = "APP_DERMICA",
                style = MaterialTheme.typography.headlineMedium,
                color = MaterialTheme.colorScheme.primary,
            )
            Text(
                text = "Apoyo al tamizaje de carcinoma basocelular",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            MedicalDisclaimer()
            ModelStatusRow(modelState)

            // Acción principal 1: imagen única → resultado.
            // Fix review-corroborated R3-001: con modelLoadError la CTA queda deshabilitada
            // y muestra el motivo real (nunca el engañoso «todavía no está listo»).
            ActionCard(
                title = "Imagen única",
                description = "Elige una imagen dermatoscópica y obtén el resultado del tamizaje.",
                buttonText = if (predicting) "Procesando…" else "Elegir imagen",
                buttonEnabled = modelReady && !predicting && modelLoadError == null,
                onAction = { picker.launch(arrayOf("image/*")) },
                leading = { Icon(Icons.Filled.Image, contentDescription = null) },
                supportingText = modelLoadError ?: (if (modelReady) null else modeloNoDisponibleTexto(modelState)),
            )

            // Acción principal 2: subconjunto (T4 lo implementa; el VM falla honesto si LiteRT no inicializa).
            ActionCard(
                title = "Subir subconjunto",
                description = "Carpeta con las imágenes dermatoscópicas del subconjunto y su referencia.",
                buttonText = "Elegir carpeta",
                buttonEnabled = true,
                onAction = onOpenSubset,
                leading = { Icon(Icons.Filled.Folder, contentDescription = null) },
                supportingText = null,
            )

            if (predicting) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    CircularProgressIndicator(modifier = Modifier.padding(2.dp), strokeWidth = 2.dp)
                    Text(text = "Procesando imagen…", style = MaterialTheme.typography.bodyMedium)
                }
            }
            predictionError?.let { errorMessage ->
                Text(
                    text = errorMessage,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                )
            }
            pickerError?.let { errorMessage ->
                Text(
                    text = errorMessage,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                )
            }
        }
    }
}

/** Aviso fijo de tamizaje (S1/S7): siempre visible, nunca dismissible. */
@Composable
private fun MedicalDisclaimer() {
    Surface(
        color = MaterialTheme.colorScheme.secondaryContainer,
        shape = MaterialTheme.shapes.medium,
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                Icons.Filled.Info,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSecondaryContainer,
            )
            Text(
                text = "Esta aplicación apoya la decisión médica; no sustituye el diagnóstico profesional.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSecondaryContainer,
            )
        }
    }
}

/** Estado del modelo siempre visible y coherente (StateFlow de ModelManager, S1). */
@Composable
private fun ModelStatusRow(modelState: ModelState) {
    val (label, tint) = when (modelState) {
        ModelState.READY -> "Modelo listo" to MaterialTheme.colorScheme.primary
        ModelState.COPYING -> "Preparando modelo…" to MaterialTheme.colorScheme.secondary
        ModelState.NO_MODEL -> "Modelo no disponible" to MaterialTheme.colorScheme.error
        ModelState.FAILED -> "Error de integridad del modelo" to MaterialTheme.colorScheme.error
    }
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Box(
            modifier = Modifier
                .size(10.dp)
                .background(tint, shape = CircleShape),
        )
        Text(text = label, style = MaterialTheme.typography.labelLarge, color = tint)
    }
}

/** Texto honesto cuando la inferencia no es posible todavía (descarga real llega en T5). */
private fun modeloNoDisponibleTexto(state: ModelState): String = when (state) {
    ModelState.NO_MODEL -> "Modelo no disponible: la descarga se habilitará en la próxima entrega."
    ModelState.COPYING -> "Preparando el modelo, un momento…"
    ModelState.FAILED -> "No se pudo verificar el modelo. La restauración llegará en la próxima entrega."
    ModelState.READY -> "" // no se usa: la CTA está habilitada
}

/** Card de acción reutilizable por las dos vías de subida (componente interno, no API pública). */
@Composable
private fun ActionCard(
    title: String,
    description: String,
    buttonText: String,
    buttonEnabled: Boolean,
    onAction: () -> Unit,
    leading: @Composable () -> Unit,
    supportingText: String?,
    modifier: Modifier = Modifier,
) {
    ElevatedCard(modifier = modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(18.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Row(
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                leading()
                Text(text = title, style = MaterialTheme.typography.titleMedium)
            }
            Text(text = description, style = MaterialTheme.typography.bodyMedium)
            supportingText?.let {
                Text(
                    text = it,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                )
            }
            Button(onClick = onAction, enabled = buttonEnabled) {
                Text(text = buttonText)
            }
        }
    }
}

/** Nombre visible del documento elegido (p.ej. ISIC_0012345.jpg); null si el proveedor no lo da. */
private fun displayNameOf(context: Context, uri: Uri): String? =
    context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
        if (cursor.moveToFirst()) cursor.getString(0)?.takeIf { it.isNotBlank() } else null
    } ?: uri.lastPathSegment

private const val TAG = "HomeScreen"
