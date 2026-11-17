package com.dermica.bcc

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.dermica.bcc.data.model.ModelManager
import com.dermica.bcc.data.model.ModelState
import com.dermica.bcc.ui.theme.DermicaTheme

/**
 * Actividad única (single-activity, S9). En T1/T2 solo cablea el estado del modelo
 * para validar el flujo end-to-end; las pantallas de subida y configuración llegan en T3/T5.
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            DermicaTheme {
                ModelStatusPlaceholder()
            }
        }
    }
}

/** Placeholder mínimo: muestra el estado observable del ModelManager (S1). */
@Composable
private fun ModelStatusPlaceholder() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val manager = remember { ModelManager(context.applicationContext, scope) }
    val state by manager.state.collectAsStateWithLifecycle()

    // Primer arranque: copia el asset a filesDir y verifica SHA-256 (S1).
    LaunchedEffect(Unit) { manager.ensureModel() }

    Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.surface) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp)
                .widthIn(max = 480.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp, Alignment.CenterVertically),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(text = "Derma BCC", style = MaterialTheme.typography.headlineMedium)
            Text(text = estadoTexto(state), style = MaterialTheme.typography.bodyLarge)
            Text(
                text = "Tamizaje, no diagnóstico. La app no sustituye el criterio del médico.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
            )
        }
    }
}

/** Texto plano del estado del modelo para el placeholder (T3 lo reemplazará por UI real). */
private fun estadoTexto(state: ModelState): String = when (state) {
    ModelState.NO_MODEL -> "Estado del modelo: no disponible"
    ModelState.COPYING -> "Estado del modelo: preparando…"
    ModelState.READY -> "Estado del modelo: listo"
    ModelState.FAILED -> "Estado del modelo: error de integridad"
}
