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
import androidx.compose.material.icons.filled.CheckCircle
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
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.dermica.bcc.data.model.ModelState

/**
 * Modo subconjunto (S2/S3/S4/S6 · S5-multi): selección de carpeta vía SAF
 * (ACTION_OPEN_DOCUMENT_TREE, sin permisos de almacenamiento). Una carpeta por
 * intent: si el árbol tiene ground_truth.json en la raíz corre el flujo T4;
 * si es un PADRE, la app enumera las subcarpetas calificadas y las procesa
 * todas en una corrida (S5-multi); cualquier otra cosa → error honesto.
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

            is SubsetRunState.MultiDone -> MultiDoneContent(
                summary = state.summary,
                onAnotherFolder = viewModel::reset,
                onViewHistory = onViewHistory,
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
 * Resolución del árbol elegido (S5-multi): delega en [resolveTree] (adapter Android).
 * - RAÍZ con ground_truth.json → flujo T4 (una sola carpeta, como hoy).
 * - PADRE con subcarpetas calificadas → lote multi (todas en una corrida).
 * - Nada elegible → Error state con botón Reintentar/Volver. Nunca crash (S3).
 */
private fun prepareAndRun(context: Context, treeUri: Uri, viewModel: SubsetViewModel) {
    when (val resolution = resolveTree(context, treeUri)) {
        is TreeResolution.Fail -> viewModel.fail(resolution.message)
        is TreeResolution.SingleSubset -> viewModel.startRun(
            folderLabel = resolution.folderLabel,
            images = resolution.images,
            groundTruthJson = resolution.groundTruthJson,
            contentResolver = context.contentResolver,
        )
        is TreeResolution.MultiSubset -> viewModel.startMultiRun(
            folders = resolution.inputs,
            contentResolver = context.contentResolver,
        )
    }
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
            text = "Elige la carpeta del subconjunto (imágenes + ground_truth.json), o una carpeta que contenga varios subconjuntos: la app procesará cada subcarpeta con su ground_truth.json.",
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

/**
 * Progreso honesto en curso (S2): en carpeta única, sólo el avance de la carpeta;
 * en multi (currentFolder != null), posición en el lote + barra general + barra por imagen.
 */
@Composable
private fun RunningContent(
    state: SubsetRunState.Running,
    onCancel: () -> Unit,
) {
    val isMulti = state.currentFolder != null
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
                if (isMulti) {
                    // Encabezado de lote: qué subcarpeta va de cuántas (S5-multi).
                    Text(
                        text = "Subconjunto ${state.folderIndex} de ${state.folderCount} — ${state.currentFolder}",
                        style = MaterialTheme.typography.titleMedium,
                    )
                    Text(
                        text = "Progreso del lote",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    LinearProgressIndicator(
                        progress = { batchProgress(state) },
                        modifier = Modifier.fillMaxWidth(),
                    )
                    Text(
                        text = "${state.foldersDone} de ${state.folderCount} subconjuntos procesados",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                } else {
                    Text(text = "Procesando subconjunto…", style = MaterialTheme.typography.titleMedium)
                }
                Text(
                    text = state.currentName ?: "Preparando…",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                )
                if (isMulti) {
                    Text(
                        text = "Imágenes del subconjunto actual",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
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

/** Avance general del lote (0..1): subcarpetas cerradas + fracción de la carpeta actual. */
private fun batchProgress(state: SubsetRunState.Running): Float {
    if (state.folderCount <= 0) return 0f
    val perImage = if (state.total > 0) state.done.toFloat() / state.total else 0f
    return ((state.foldersDone + perImage) / state.folderCount).coerceIn(0f, 1f)
}

/**
 * Lote multi terminado (S5-multi): «Listo: procesé N subconjuntos» + lista corta
 * de fallos. El historial ya tiene una card por subcarpeta; aquí solo el resumen.
 */
@Composable
private fun MultiDoneContent(
    summary: SubsetBatchSummary,
    onAnotherFolder: () -> Unit,
    onViewHistory: () -> Unit,
) {
    val headline = if (summary.completed == 0) {
        "No se procesó ningún subconjunto"
    } else {
        val noun = if (summary.completed == 1) "subconjunto" else "subconjuntos"
        "Listo: procesé ${summary.completed} $noun"
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
            Icons.Filled.CheckCircle,
            contentDescription = null,
            modifier = Modifier.padding(top = 32.dp),
            tint = MaterialTheme.colorScheme.primary,
        )
        Text(text = headline, style = MaterialTheme.typography.titleLarge, textAlign = TextAlign.Center)
        if (summary.failed.isNotEmpty()) {
            Text(
                text = "${summary.failed.size} fallaron:",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.error,
            )
            // Lista corta: los 3 primeros con motivo; el resto, contados.
            summary.failed.take(3).forEach { failure ->
                Text(
                    text = "• ${failure.folderLabel}: ${failure.reason}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                )
            }
            val extra = summary.failed.size - 3
            if (extra > 0) {
                Text(
                    text = "…y $extra más",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
        Text(
            text = "Cada subconjunto quedó guardado en el historial con sus métricas.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
        Button(onClick = onViewHistory, modifier = Modifier.fillMaxWidth()) {
            Text(text = "Ver historial")
        }
        OutlinedButton(onClick = onAnotherFolder, modifier = Modifier.fillMaxWidth()) {
            Text(text = "Otra carpeta")
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
