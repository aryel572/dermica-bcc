package com.dermica.bcc.ui.history

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Folder
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material.icons.filled.VisibilityOff
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.FilledTonalIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.hideFromAccessibility
import androidx.compose.ui.unit.dp
import com.dermica.bcc.core.DateTimes
import com.dermica.bcc.data.history.HistoryItem
import java.util.Locale

/**
 * Historial de predicciones (S6): lista vertical de cards mixtas.
 *
 * - Card de imagen única: título = nombre del archivo, fecha-hora corta local,
 *   chip de veredicto (BCC / No-BCC) y confianza.
 * - Card de subconjunto: título = carpeta, fecha-hora, n/15 imágenes, fila
 *   SIEMPRE VISIBLE TA · NFS · PEO (números, sin fórmulas, S4/S5) y un ojo que
 *   alterna el desglose TP/TN/FP/FN difuminado por defecto (S4/S5).
 * - Empty state honesto cuando no hay predicciones.
 *
 * La pantalla recibe `items` ya mapeados (state hoisting); la conexión con el
 * Flow del repositorio vive en AppRoot. Las fórmulas de las métricas NO se
 * muestran jamás (S4).
 *
 * Invocada por AppRoot en AppRoute.HISTORY. Flujo de datos completo:
 * HistoryDatabase → HistoryRepository.items (Flow<List<HistoryItem>>, ya
 * mapeado desde HistoryEvent con HistoryMappers) → AppRoot
 * (collectAsStateWithLifecycle) → aquí como `items`. Esta pantalla solo
 * pinta lo que recibe: no consulta Room, no lanza coroutines y no conoce
 * los eventos.
 *
 * Variantes de HistoryItem (sealed): [HistoryItem.Image] → ImageCard;
 * [HistoryItem.Subset] → SubsetCard. El `key = { it.id }` de la LazyColumn
 * mantiene el estado de scroll/foco por tarjeta entre recomposiciones.
 *
 * @param items historial ya mapeado a HistoryItem, ordenado como lo entrega el
 *        repositorio (más reciente arriba).
 * @param modifier slot de layout para la LazyColumn.
 */
@Composable
fun HistoryScreen(
    items: List<HistoryItem>,
    modifier: Modifier = Modifier,
) {
    if (items.isEmpty()) {
        EmptyHistory(modifier = modifier)
        return
    }
    LazyColumn(
        modifier = modifier.fillMaxSize(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        items(items = items, key = { it.id }) { item ->
            when (item) {
                is HistoryItem.Image -> ImageCard(item)
                is HistoryItem.Subset -> SubsetCard(item)
            }
        }
    }
}

/** Card de predicción de imagen única. Invocada por el items(...) de HistoryScreen. */
@Composable
private fun ImageCard(item: HistoryItem.Image, modifier: Modifier = Modifier) {
    val isBcc = item.verdict == "BCC"
    ElevatedCard(modifier = modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier.padding(14.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(Icons.Filled.Image, contentDescription = null)
            Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text(
                    text = item.imageName ?: "Imagen",
                    style = MaterialTheme.typography.titleSmall,
                    maxLines = 1,
                )
                Text(
                    text = DateTimes.formatDateTime(item.createdAt),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Text(
                    text = "Confianza: ${formatPercent(item.confidence * 100.0)} %",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            VerdictChip(isBcc = isBcc)
        }
    }
}

/**
 * Cobertura honesta: procesadas; si hubo sin etiqueta, se informa también.
 *
 * @param item card de subconjunto con contadores ya calculados por el repositorio.
 * @return texto listo para pintar, p.ej. «38 imágenes procesadas · 7 sin etiqueta».
 */
private fun coverageText(item: HistoryItem.Subset): String =
    if (item.excludedCount > 0) {
        "${item.imageCount} imágenes procesadas · ${item.excludedCount} sin etiqueta"
    } else {
        "${item.imageCount} imágenes procesadas"
    }

/**
 * Chip pequeño de veredicto: BCC en error/contenedor y No-BCC en contenedor
 * terciario. Invocado por ImageCard. Contrato de accesibilidad: el chip se
 * anuncia como un único nodo "Veredicto: …" (clearAndSetSemantics) para que
 * TalkBack no lea la palabra suelta sin contexto.
 */
@Composable
private fun VerdictChip(isBcc: Boolean, modifier: Modifier = Modifier) {
    val (text, bg, fg) = if (isBcc) {
        Triple("BCC", MaterialTheme.colorScheme.errorContainer, MaterialTheme.colorScheme.onErrorContainer)
    } else {
        Triple("No-BCC", MaterialTheme.colorScheme.tertiaryContainer, MaterialTheme.colorScheme.onTertiaryContainer)
    }
    Surface(
        color = bg,
        shape = RoundedCornerShape(8.dp),
        // clearAndSetSemantics: el chip anuncia "Veredicto: …" como un solo nodo;
        // sin esto TalkBack leería solo "BCC"/"No-BCC", sin el prefijo de contexto.
        modifier = modifier.clearAndSetSemantics {
            contentDescription = if (isBcc) "Veredicto: BCC" else "Veredicto: No carcinoma basocelular"
        },
    ) {
        Text(
            text = text,
            style = MaterialTheme.typography.labelMedium,
            color = fg,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
        )
    }
}

/**
 * Card de subconjunto procesado: TA·NFS·PEO siempre; TP/TN/FP/FN difuminados
 * con ojo (S4/S5). Invocada por el items(...) de HistoryScreen.
 *
 * Estado local: `truthVisible` gobierna SOLO el panel de desglose; la fila TA·NFS·PEO
 * es siempre visible. El ojo (ConfusionPanel.onToggle) alterna ese flag.
 */
@Composable
private fun SubsetCard(item: HistoryItem.Subset, modifier: Modifier = Modifier) {
    // Por defecto difuminado; el ojo alterna visible/oculto (S4: «un icono de ojo en la esquina»).
    var truthVisible by remember { mutableStateOf(false) }

    ElevatedCard(modifier = modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Filled.Folder, contentDescription = null)
                Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text(
                        text = item.folderLabel,
                        style = MaterialTheme.typography.titleSmall,
                        maxLines = 1,
                    )
                    Text(
                        text = DateTimes.formatDateTime(item.createdAt),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Text(
                        text = coverageText(item),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }

            // Fila SIEMPRE VISIBLE: TA · NFS · PEO (números, fórmulas jamás, S4/S5).
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                StatCell(label = "TA", value = "${formatNumber(item.ta)} %", modifier = Modifier.weight(1f))
                StatCell(label = "NFS", value = item.nfs.toString(), modifier = Modifier.weight(1f))
                StatCell(label = "PEO", value = "${formatNumber(item.peo)} %", modifier = Modifier.weight(1f))
            }

            // Panel opcional TP/TN/FP/FN: difuminado por defecto; ojo alterna visible/oculto.
            ConfusionPanel(
                tp = item.tp,
                tn = item.tn,
                fp = item.fp,
                fn = item.fn,
                visible = truthVisible,
                onToggle = { truthVisible = !truthVisible },
            )
        }
    }
}

/** Celda de métrica: etiqueta pequeña + valor grande (sin fórmula visible, S4). Invocada por SubsetCard y ConfusionPanel. */
@Composable
private fun StatCell(label: String, value: String, modifier: Modifier = Modifier) {
    Surface(
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = RoundedCornerShape(10.dp),
        modifier = modifier,
    ) {
        Column(
            modifier = Modifier.padding(vertical = 10.dp, horizontal = 4.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(text = label, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text(text = value, style = MaterialTheme.typography.titleMedium)
        }
    }
}

/**
 * Desglose de matriz de confusión con eye toggle (contrato UI: `visible` +
 * `onToggle` par, igual que el SubsetCard que lo invoca).
 *
 * Difuminado por defecto: alpha bajo sobre el contenido + scrim semitransparente
 * (~0.35) por encima. minSdk 26: sin gaussianBlur obligatorio, el scrim cumple
 * el contrato visual (S4/S5). Solo números, sin fórmulas.
 *
 * @param visible true = desglose revelado (alpha 1, sin scrim, accesible); false =
 *        velado (alpha 0.12 + scrim y ocultación auditiva).
 * @param onToggle callback del ojo; invierte `visible` en el caller. El botón
 *        mide ≥ 48 dp (T6) y anuncia «Mostrar/Ocultar totales TP, TN, FP y FN».
 */
@Composable
private fun ConfusionPanel(
    tp: Int,
    tn: Int,
    fp: Int,
    fn: Int,
    visible: Boolean,
    onToggle: () -> Unit,
    modifier: Modifier = Modifier,
) {
    // Scrim 0.35: surfaceVariant semitransparente; se intuye la estructura de la
    // tabla sin filtrar los dígitos (contrato visual S4/S5).
    val scrimColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.35f)
    Row(
        modifier = modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text(
            text = "Desglose",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.weight(1f),
        )
        // 48 dp táctiles explícitos (T6): el default visual (40 dp) cumple el mínimo
        // de sistema pero el requisito del proyecto exige objetivo medible ≥ 48 dp.
        FilledTonalIconButton(onClick = onToggle, modifier = Modifier.size(48.dp)) {
            Icon(
                if (visible) Icons.Filled.VisibilityOff else Icons.Filled.Visibility,
                contentDescription = if (visible) "Ocultar totales TP, TN, FP y FN" else "Mostrar totales TP, TN, FP y FN",
            )
        }
    }

    Box(modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp))) {
        // Contenido: legible solo cuando visible == true. hideFromAccessibility
        // cuando está velado: el difuminado es un resguardo visual y también auditivo;
        // TalkBack salta los totales TP/TN/FP/FN hasta que el usuario los revele con el ojo.
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .alpha(if (visible) 1f else 0.12f)
                .then(if (visible) Modifier else Modifier.clearAndSetSemantics { hideFromAccessibility() }),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            StatCell(label = "TP", value = tp.toString(), modifier = Modifier.weight(1f))
            StatCell(label = "TN", value = tn.toString(), modifier = Modifier.weight(1f))
            StatCell(label = "FP", value = fp.toString(), modifier = Modifier.weight(1f))
            StatCell(label = "FN", value = fn.toString(), modifier = Modifier.weight(1f))
        }
        // Scrim ~0.35 por encima cuando está oculto: se intuye la estructura, no los números.
        if (!visible) {
            Box(
                modifier = Modifier
                    .matchParentSize()
                    .background(scrimColor),
            )
        }
    }
}

/** Empty state: icono + mensaje honesto (nada de listas vacías mudas). Invocada por HistoryScreen cuando `items` está vacío. */
@Composable
private fun EmptyHistory(modifier: Modifier = Modifier) {
    Box(modifier = modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Icon(
                Icons.Filled.History,
                contentDescription = null,
                modifier = Modifier.size(56.dp),
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                text = "Todavía no hay predicciones",
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

/**
 * Formato numérico con punto decimal y un decimal (Locale.US, estable entre
 * dispositivos). Consumido por la fila TA·NFS·PEO y PEO/TA de la card.
 *
 * @param value valor en la unidad ya convertida (p.ej. porcentaje 0–100).
 * @return cadena con un decimal, p.ej. "74.5".
 */
private fun formatNumber(value: Double): String = String.format(Locale.US, "%.1f", value)

/**
 * Confianza expresada en % con un decimal. Consumida por la card de imagen única.
 *
 * @param value porcentaje ya multiplicado por 100.
 * @return cadena con un decimal, p.ej. "93.2".
 */
private fun formatPercent(value: Double): String = String.format(Locale.US, "%.1f", value)
