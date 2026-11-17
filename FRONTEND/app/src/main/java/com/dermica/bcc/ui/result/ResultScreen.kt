package com.dermica.bcc.ui.result

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.dermica.bcc.ui.PredictionUi
import java.util.Locale

/**
 * Pantalla de resultado de imagen única (S2/S6).
 *
 * - Veredicto grande + confianza en % (la clase predicha, no la sigmoid cruda).
 * - Disclaimer fijo de tamizaje (S1/S7).
 * - La predicción ya se guardó en el historial al calcularla (AppRoot llama a
 *   HistoryRepository.addImagePrediction): esta pantalla no escribe nada.
 * - Thumbnail decodificado downsampled (bounds → inSampleSize) para no inflar
 *   memoria con fotos de alta resolución.
 */
@Composable
fun ResultScreen(
    result: PredictionUi,
    onAnotherImage: () -> Unit,
    onViewHistory: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val thumbnail = remember(result.imageBytes) { decodeSampled(result.imageBytes, targetPx = 768) }
    val verdict = if (result.isBcc) "Carcinoma basocelular" else "No carcinoma basocelular"
    val verdictColor = if (result.isBcc) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary
    val confidencePct = String.format(Locale.US, "%.1f", result.confidence * 100f)

    Surface(modifier = modifier.fillMaxSize(), color = MaterialTheme.colorScheme.surface) {
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center,
        ) {
        Column(
            modifier = Modifier
                .widthIn(max = 520.dp)
                .verticalScroll(rememberScrollState())
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text(
                text = result.imageName ?: "Imagen seleccionada",
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            thumbnail?.let { bitmap ->
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = "Imagen analizada",
                    modifier = Modifier
                        .size(240.dp)
                        .clip(RoundedCornerShape(16.dp)),
                    contentScale = ContentScale.Crop,
                )
            }

            Text(
                text = verdict,
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
                color = verdictColor,
                textAlign = TextAlign.Center,
            )
            Text(
                text = "Confianza: $confidencePct %",
                style = MaterialTheme.typography.titleLarge,
            )

            ElevatedCard {
                Text(
                    text = "Resultado de tamizaje: apoya la decisión médica y no sustituye el diagnóstico del profesional de salud.",
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.padding(14.dp),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            Spacer(modifier = Modifier.height(4.dp))
            Button(onClick = onAnotherImage, modifier = Modifier.fillMaxWidth()) {
                Text(text = "Otra imagen")
            }
            OutlinedButton(onClick = onViewHistory, modifier = Modifier.fillMaxWidth()) {
                Text(text = "Ver historial")
            }
        }
        }
    }
}

/**
 * Decodifica downsampled con bounds → inSampleSize (potencia de 2). Si la
 * imagen no decodifica (bytes corruptos), devuelve null: el resultado textual
 * ya es válido y el thumbnail es prescindible.
 */
private fun decodeSampled(bytes: ByteArray, targetPx: Int): Bitmap? {
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
    val largestSide = maxOf(bounds.outWidth, bounds.outHeight)
    if (largestSide <= 0) return null
    var sample = 1
    while (largestSide / sample > targetPx) sample *= 2
    return BitmapFactory.decodeByteArray(
        bytes,
        0,
        bytes.size,
        BitmapFactory.Options().apply { inSampleSize = sample },
    )
}
