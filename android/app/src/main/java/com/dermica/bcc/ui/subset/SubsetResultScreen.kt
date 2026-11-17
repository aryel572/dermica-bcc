package com.dermica.bcc.ui.subset

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import java.util.Locale

/**
 * Resultado del modo subconjunto (S4/S5): nombre de la carpeta, cobertura
 * (procesadas / sin etiqueta excluidas) y TA · NFS · PEO SIEMPRE visibles como
 * números grandes. Los totales TP/TN/FP/FN NO se repiten aquí: viven difuminados
 * en la card del historial con el ojo (flujo idéntico al de la imagen única, S5/S6).
 *
 * Las fórmulas NO se muestran jamás (S4): solo los resultados.
 */
@Composable
fun SubsetResultScreen(
    summary: SubsetSummary,
    onAnotherFolder: () -> Unit,
    onViewHistory: () -> Unit,
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
            Text(
                text = "SUBCONJUNTO PROCESADO",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                text = summary.folderLabel,
                style = MaterialTheme.typography.headlineSmall,
                textAlign = TextAlign.Center,
            )
            Text(
                text = coverageText(summary),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
            )

            // Fila SIEMPRE VISIBLE: TA · NFS · PEO en grande (S5). Sin fórmulas (S4).
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                BigStatCell(label = "TA", value = "${formatNumber(summary.metrics.ta)} %", modifier = Modifier.weight(1f))
                BigStatCell(label = "NFS", value = summary.metrics.nfs.toString(), modifier = Modifier.weight(1f))
                BigStatCell(label = "PEO", value = "${formatNumber(summary.metrics.peo)} %", modifier = Modifier.weight(1f))
            }

            // Aviso breve fijo de tamizaje (S1/S7): esta app no diagnostica.
            Text(
                text = "Resultado de tamizaje; no sustituye el diagnóstico profesional.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
            )

            // Fallo honesto de persistencia: los números siguen en pantalla, se avisa.
            if (!summary.historySaved) {
                Text(
                    text = "El resultado no se pudo guardar en el historial.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                    textAlign = TextAlign.Center,
                )
            }

            Button(onClick = onViewHistory, modifier = Modifier.fillMaxWidth()) {
                Text(text = "Ver historial")
            }
            OutlinedButton(onClick = onAnotherFolder, modifier = Modifier.fillMaxWidth()) {
                Text(text = "Otra carpeta")
            }
        }
    }
}

/** Celda de métrica grande: etiqueta pequeña + valor protagonista (solo resultado, sin fórmula). */
@Composable
private fun BigStatCell(label: String, value: String, modifier: Modifier = Modifier) {
    Surface(
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = RoundedCornerShape(12.dp),
        modifier = modifier,
    ) {
        Column(
            modifier = Modifier.padding(vertical = 16.dp, horizontal = 4.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(text = label, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text(text = value, style = MaterialTheme.typography.headlineMedium)
        }
    }
}

/** Línea de cobertura honesta: procesadas, excluidas y (si hubo) no decodificables. */
private fun coverageText(summary: SubsetSummary): String {
    val base = "${summary.processed} de ${summary.totalImages} imágenes procesadas"
    val excluded = if (summary.excluded > 0) " · ${summary.excluded} sin etiqueta excluidas" else ""
    val failed = if (summary.failed > 0) " · ${summary.failed} no decodificables" else ""
    return base + excluded + failed
}

/** Formato numérico con punto decimal y un decimal (Locale.US, estable entre dispositivos). */
private fun formatNumber(value: Double): String = String.format(Locale.US, "%.1f", value)
