package com.dermica.bcc.ui.subset

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp

/**
 * Placeholder del modo subconjunto (S2).
 *
 * T4 reemplaza este archivo entero por el flujo real: SAF (ACTION_OPEN_DOCUMENT_TREE),
 * lectura de ground_truth.json de la carpeta, 15 imágenes, progreso, exclusión de
 * no-etiquetados, TA/NFS/PEO + TP/TN/FP/FN difuminado y card en el historial.
 * Aquí solo se promete honestamente lo que la app NO hace todavía.
 */
@Composable
fun SubsetPlaceholderScreen(
    onBack: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Surface(modifier = modifier.fillMaxSize(), color = MaterialTheme.colorScheme.surface) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp)
                .widthIn(max = 480.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Icon(
                Icons.Filled.FolderOpen,
                contentDescription = null,
                modifier = Modifier.padding(top = 40.dp),
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(text = "Subir subconjunto", style = MaterialTheme.typography.headlineSmall)
            Text(
                text = "Disponible en la próxima entrega.",
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
            )
            Button(onClick = onBack) {
                Text(text = "Volver")
            }
        }
    }
}
