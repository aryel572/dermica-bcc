package com.dermica.bcc

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.dermica.bcc.ui.AppRoot
import com.dermica.bcc.ui.theme.DermicaTheme

/**
 * Actividad única (single-activity, S9). La invoca el sistema Android; monta la
 * UI completa en Compose bajo ui/ y nada más: no contiene estado ni maneja
 * Intents de negocio.
 *
 * Toda la UI vive en Compose bajo ui/: AppRoot cablea ModelManager, Room y el
 * enrutado (Inicio/Historial/Resultado/Subconjunto-placeholder). T5 agrega
 * Configuración como tercer destino de la bottom navigation.
 *
 * Flujo de datos: MainActivity → DermicaTheme → AppRoot (owner de estado) →
 * pantallas (HomeScreen, HistoryScreen, ResultScreen, SubsetScreen,
 * SettingsScreen). Los datos del dispositivo (modelo en filesDir, historial en
 * Room) los poseen AppRoot y los ViewModels; esta actividad solo los expone
 * en pantalla.
 */
class MainActivity : ComponentActivity() {
    /**
     * Punto de entrada estándar de Android. Habilita edge-to-edge y monta
     * [DermicaTheme] + [AppRoot] como contenido. Sin extras ni savedInstanceState.
     */
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
