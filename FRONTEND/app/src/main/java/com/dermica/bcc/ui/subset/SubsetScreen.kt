package com.dermica.bcc.ui.subset

import android.content.Context
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ErrorOutline
import androidx.compose.material.icons.filled.Folder
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.documentfile.provider.DocumentFile
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.dermica.bcc.data.model.ModelState
import java.util.Locale

private const val GROUND_TRUTH_FILE_NAME = "ground_truth.json"
private val IMAGE_EXTENSIONS = listOf(".jpg", ".jpeg", ".png")

/**
 * Modo subconjunto (S2/S3/S4/S6): selección de carpeta vía SAF
 * (ACTION_OPEN_DOCUMENT_TREE, sin permisos de almacenamiento), lectura de
 * ground_truth.json de la carpeta, procesado con progreso honesto y cancelación,
 * y navegación al resultado. Las fórmulas NO se muestran jamás (S4).
 *
 * State hoisting: la pantalla recibe [modelState] y callbacks; la corrida vive en
 * [SubsetViewModel] (viewModelScope), que sobrevive a rotaciones de pantalla.
 * La app NO adivina el subconjunto por nombre de carpeta: la referencia viaja en
 * ground_truth.json dentro del árbol elegido.
 */
@Composable
fun SubsetScreen(
    modelState: ModelState,
    viewModel: SubsetViewModel,
    onBack: () -> Unit,
    onViewHistory: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val runState by viewModel.state.collectAsStateWithLifecycle()

    val folderPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { treeUri ->
        treeUri ?: return@rememberLauncherForActivityResult
        // Permisos SAF de sesión: no se persisten entre sesiones (regla de la casa).
        prepareAndRun(context = context, treeUri = treeUri, viewModel = viewModel)
    }

    Box(modifier = modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        when (val state = runState) {
            SubsetRunState.Idle -> IdleContent(
                modelReady = modelState == ModelState.READY,
                onPickFolder = { folderPicker.launch(null) },
                onBack = onBack,
            )

            is SubsetRunState.Running -> RunningContent(
                state = state,
                onCancel = viewModel::cancel,
            )

            is SubsetRunState.Error -> ErrorContent(
                message = state.message,
                onRetry = viewModel::reset,
                onBack = onBack,
            )

            is SubsetRunState.Cancelled -> CancelledContent(
                savedSubsets = state.savedSubsets,
                onPickAgain = viewModel::reset,
                onBack = onBack,
            )

            is SubsetRunState.Done -> SubsetResultScreen(
                summary = state.summary,
                onAnotherFolder = viewModel::reset,
                onViewHistory = onViewHistory,
            )
        }
    }
}

/**
 * Resolución de la carpeta elegida: listado de imágenes (orden por nombre),
 * búsqueda de ground_truth.json en la RAÍZ del árbol y lectura a texto.
 * Cualquier fallo → Error state con explicación (nunca crash, S3).
 */
private fun prepareAndRun(context: Context, treeUri: Uri, viewModel: SubsetViewModel) {
    val root = DocumentFile.fromTreeUri(context, treeUri)
    if (root == null || !root.isDirectory) {
        viewModel.fail("No se pudo abrir la carpeta elegida.")
        return
    }
    val folderLabel = root.name?.takeIf { it.isNotBlank() } ?: "subconjunto"
    val children = root.listFiles()
    val images = children
        .filter { it.isFile }
        .mapNotNull { document ->
            val name = document.name ?: return@mapNotNull null
            if (IMAGE_EXTENSIONS.any { extension -> name.lowercase(Locale.ROOT).endsWith(extension) }) {
                SubsetImage(name = name, uri = document.uri)
            } else {
                null
            }
        }
        .sortedBy { it.name.lowercase(Locale.ROOT) }
    if (images.isEmpty()) {
        viewModel.fail("La carpeta no contiene imágenes .jpg, .jpeg ni .png.")
        return
    }

    // ground_truth.json SOLO en la raíz del árbol (S3): la referencia viaja con la carpeta.
    val groundTruthDocument = children.firstOrNull { it.isFile && it.name == GROUND_TRUTH_FILE_NAME }
    if (groundTruthDocument == null) {
        viewModel.fail(
            "La carpeta no contiene ground_truth.json. La referencia " +
                "(nombre de imagen → diagnóstico) viaja en ese archivo dentro de la carpeta.",
        )
        return
    }
    val jsonText = try {
        context.contentResolver.openInputStream(groundTruthDocument.uri)
            ?.use { it.readBytes().toString(Charsets.UTF_8) }
    } catch (error: Exception) {
        null
    }
    if (jsonText == null) {
        viewModel.fail("No se pudo leer ground_truth.json desde la carpeta.")
        return
    }
    viewModel.startRun(
        folderLabel = folderLabel,
        images = images,
        groundTruthJson = jsonText,
        contentResolver = context.contentResolver,
    )
}

/** Selector de carpeta (CTA) o estado vacío cuando el modelo no está disponible (patrón T3). */
@Composable
private fun IdleContent(
    modelReady: Boolean,
    onPickFolder: () -> Unit,
    onBack: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(24.dp)
            .widthIn(max = 480.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Icon(
            Icons.Filled.Folder,
            contentDescription = null,
            modifier = Modifier.padding(top = 32.dp),
            tint = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(text = "Subir subconjunto", style = MaterialTheme.typography.headlineSmall)
        Text(
            text = "Elige la carpeta del subconjunto: debe contener las imágenes dermatoscópicas y un archivo ground_truth.json con la referencia de cada imagen.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
        Surface(
            color = MaterialTheme.colorScheme.secondaryContainer,
            shape = MaterialTheme.shapes.medium,
        ) {
            Text(
                text = "Apoyo al tamizaje; no sustituye el diagnóstico profesional.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSecondaryContainer,
                modifier = Modifier.padding(12.dp),
            )
        }
        if (modelReady) {
            Button(onClick = onPickFolder, modifier = Modifier.fillMaxWidth()) {
                Text(text = "Elegir carpeta")
            }
        } else {
            Text(
                text = "El modelo todavía no está disponible.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.error,
                textAlign = TextAlign.Center,
            )
        }
        TextButton(onClick = onBack) {
            Text(text = "Volver")
        }
    }
}

/** Progreso honesto en curso: qué imagen va, cuánto lleva y Cancelar siempre visible (S2). */
@Composable
private fun RunningContent(
    state: SubsetRunState.Running,
    onCancel: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp)
            .widthIn(max = 480.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Surface(
            color = MaterialTheme.colorScheme.surfaceVariant,
            shape = MaterialTheme.shapes.medium,
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 32.dp),
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Text(text = "Procesando subconjunto…", style = MaterialTheme.typography.titleMedium)
                Text(
                    text = state.currentName ?: "Preparando…",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                )
                LinearProgressIndicator(
                    progress = { if (state.total > 0) state.done.toFloat() / state.total else 0f },
                    modifier = Modifier.fillMaxWidth(),
                )
                Text(
                    text = "${state.done} de ${state.total} imágenes",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
        OutlinedButton(onClick = onCancel, modifier = Modifier.fillMaxWidth()) {
            Text(text = "Cancelar")
        }
    }
}

/** Error previo o durante el procesado: explicación + Reintentar/Volver (S3). */
@Composable
private fun ErrorContent(
    message: String,
    onRetry: () -> Unit,
    onBack: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp)
            .widthIn(max = 480.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Icon(
            Icons.Filled.ErrorOutline,
            contentDescription = null,
            modifier = Modifier.padding(top = 32.dp),
            tint = MaterialTheme.colorScheme.error,
        )
        Text(text = "No se pudo procesar", style = MaterialTheme.typography.titleLarge)
        Text(
            text = message,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
        Button(onClick = onRetry, modifier = Modifier.fillMaxWidth()) {
            Text(text = "Reintentar")
        }
        TextButton(onClick = onBack) {
            Text(text = "Volver")
        }
    }
}

/** Cancelación pedida por el usuario: estado final honesto. Las subcarpetas ya persistidas se informan. */
@Composable
private fun CancelledContent(
    savedSubsets: Int,
    onPickAgain: () -> Unit,
    onBack: () -> Unit,
) {
    val body = if (savedSubsets > 0) {
        // Corrida multi: lo ya persistido NO se pierde (S5-multi); la carpeta en curso se descarta.
        "Cancelaste el procesamiento. Los $savedSubsets subconjuntos ya procesados quedaron guardados en el historial."
    } else {
        "No se guardó ningún resultado en el historial."
    }
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp)
            .widthIn(max = 480.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Icon(
            Icons.Filled.Info,
            contentDescription = null,
            modifier = Modifier.padding(top = 32.dp),
            tint = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(text = "Procesamiento cancelado", style = MaterialTheme.typography.titleLarge)
        Text(
            text = body,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
        Button(onClick = onPickAgain, modifier = Modifier.fillMaxWidth()) {
            Text(text = "Elegir otra carpeta")
        }
        TextButton(onClick = onBack) {
            Text(text = "Volver")
        }
    }
}
