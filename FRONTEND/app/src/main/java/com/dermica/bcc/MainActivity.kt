package com.dermica.bcc

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.dermica.bcc.ui.AppRoot
import com.dermica.bcc.ui.theme.DermicaTheme

/**
 * Actividad única (single-activity, S9).
 *
 * Toda la UI vive en Compose bajo ui/: AppRoot cablea ModelManager, Room y el
 * enrutado (Inicio/Historial/Resultado/Subconjunto-placeholder). T5 agrega
 * Configuración como tercer destino de la bottom navigation.
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            DermicaTheme {
                AppRoot()
            }
        }
    }
}
