package com.dermica.bcc.ui

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.foundation.layout.Box
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import android.util.Log
import com.dermica.bcc.data.history.HistoryDatabase
import com.dermica.bcc.data.history.HistoryRepository
import com.dermica.bcc.data.model.ModelManager
import com.dermica.bcc.data.model.ModelState
import com.dermica.bcc.ml.BccLabel
import com.dermica.bcc.ml.BccPredictor
import com.dermica.bcc.ui.history.HistoryScreen
import com.dermica.bcc.ui.home.HomeScreen
import com.dermica.bcc.ui.result.ResultScreen
import com.dermica.bcc.ui.settings.SettingsScreen
import com.dermica.bcc.ui.settings.SettingsViewModel
import com.dermica.bcc.ui.subset.SubsetScreen
import com.dermica.bcc.ui.subset.SubsetViewModel
import kotlinx.coroutines.launch

/**
 * Rutas internas de la app (single activity, S9). Consumidor único: AppRoot
 * (when(route)). RESULT y SUBSET no muestran bottom navigation porque ocupan
 * pantalla completa y tienen su propia salida (volver / ver historial).
 */
private enum class AppRoute {
    HOME,
    HISTORY,
    RESULT,
    SUBSET,
    SETTINGS,
}

/**
 * Raíz de la aplicación (single activity). La monta MainActivity (setContent).
 *
 * Responsabilidades del owner (state hoisting):
 * - Construye singletons de proceso: ModelManager, HistoryDatabase/Repository.
 * - Garantiza el modelo en el primer arranque (LaunchedEffect + ensureModel, S1).
 * - Crea/libera el BccPredictor según ModelState (se cierra al salir de la app).
 * - Ejecuta la predicción de imagen única y guarda el resultado en el historial
 *   ANTES de navegar a RESULT (S6: el historial refleja cada predicción).
 * - Aloja la ruta SUBSET: crea el SubsetViewModel acotado a la pantalla y navega
 *   a HISTORY al terminar (flujo completo en SubsetScreen/SubsetViewModel).
 * - Aloja la ruta SETTINGS (Configuración, S7/S8): crea el SettingsViewModel
 *   acotado a la pantalla; el borrado/descarga reaccionan aquí vía ModelManager.
 *
 * Contrato con las pantallas (state hoisting): ninguna pantalla conoce el
 * Repository ni el Predictor; todas reciben estado por parámetros y emiten
 * eventos por callbacks. Los ViewModels de SUBSET y SETTINGS se crean aquí con
 * `viewModel()` y fábrica explícita, acotados a su pantalla: sobreviven a
 * rotaciones pero no a la navegación (se recrean al entrar de nuevo).
 *
 * Flujos de datos principales:
 * - Historial: HistoryDatabase → HistoryRepository.items (Flow) → aquí vía
 *   collectAsStateWithLifecycle → HistoryScreen(items). Las pantallas solo pintan.
 * - Modelo: ModelManager.state (StateFlow<ModelState>) → aquí → HomeScreen,
 *   SettingsScreen y SubsetScreen (para habilitar/deshabilitar CTAs).
 *
 * Las pantallas no conocen el Repository ni el Predictor: reciben estado + callbacks.
 */
@Composable
fun AppRoot() {
    val context = LocalContext.current
    // Ámbito de composición: quien cancela las escrituras al historial y las
    // predicciones al abandonar la pantalla es el ciclo de vida de la composición.
    val appScope = rememberCoroutineScope()
    val modelManager = remember { ModelManager(context.applicationContext, appScope) }
    val repository = remember { HistoryRepository(HistoryDatabase.getInstance(context.applicationContext).historyDao()) }

    val modelState by modelManager.state.collectAsStateWithLifecycle()
    val historyItems by repository.items.collectAsStateWithLifecycle(initialValue = emptyList())

    // Ruta actual de la navegación manual (sin Navigation Compose): HOME es el
    // arranque; RESULT/SUBSET son destinos de pantalla completa (sin bottom bar).
    var route by remember { mutableStateOf(AppRoute.HOME) }
    // Predicción de imagen única en vuelo: la usa HomeScreen (CTA «Procesando…»).
    var predicting by remember { mutableStateOf(false) }
    // Error de cara al usuario; null = sin error. Detalle técnico solo por Log.
    var predictionError by remember { mutableStateOf<String?>(null) }
    // Payload del resultado que RESULT pinta; null = no hay nada que mostrar.
    var result by remember { mutableStateOf<PredictionUi?>(null) }
    // Fix review-corroborated R3-001: si LiteRT no puede inicializar, modelState
    // queda en READY con predictor null; sin esto cada tap en «Imagen única» mostraba
    // «El modelo todavía no está listo» (mensaje engañoso) sin log ni salida.
    // Ahora se loguea el fallo y se deshabilita la CTA con un mensaje claro.
    var modelError by remember { mutableStateOf<String?>(null) }
    // Motor de inferencia perezoso: se crea cuando ModelState pasa a READY y se
    // invalida (close) si el modelo deja de estar disponible (borrado/fallo).
    val predictorBox = remember { mutableStateOf<BccPredictor?>(null) }

    // Libera el Interpreter cuando el owner (actividad) sale de composición.
    DisposableEffect(Unit) {
        onDispose { predictorBox.value?.close() }
    }

    // Garantiza el modelo en el primer arranque: descarga/copia/verificación; el
    // resultado llega por ModelState (StateFlow), no por callback.
    LaunchedEffect(Unit) { modelManager.ensureModel() }

    // Predictor se crea perezosamente cuando el modelo está listo y se invalida
    // si el modelo deja de estarlo (borrado/fallo). Fix review-corroborated R3-001:
    // el fallo de construcción se loguea y publica modelError (CTA deshabilitada);
    // nunca se queda un estado READY engañoso sin salida visible.
    LaunchedEffect(modelState) {
        if (modelState == ModelState.READY && predictorBox.value == null) {
            val file = modelManager.modelFile()
            val built = file?.let { modelFile ->
                runCatching { BccPredictor(modelFile) }
                    .onFailure { error -> Log.e(TAG, "BccPredictor no pudo inicializar: ${error.message}", error) }
                    .getOrNull()
            }
            predictorBox.value = built
            modelError = if (built == null) {
                "No se pudo cargar el modelo; reinicia la aplicación."
            } else {
                null
            }
        } else if (modelState != ModelState.READY && predictorBox.value != null) {
            predictorBox.value?.close()
            predictorBox.value = null
            modelError = null
        }
    }

    /**
     * Callback de HomeScreen: recibe los bytes leídos por su picker y persiste y
     * navega. Llamado desde el launcher del picker (fuera de composición) y desde
     * el botón «Elegir imagen»; corre en [appScope] (ámbito de composición).
     *
     * Orden intencional: primero el historial (addImagePrediction), después el
     * estado `result` y el route=RESULT. Si la escritura del historial fallara,
     * salta al catch y el usuario ve el error sin navegar (S6 estricto).
     */
    fun onImageChosen(bytes: ByteArray, imageName: String?) {
        val predictor = predictorBox.value
        if (predictor == null) {
            predictionError = "El modelo todavía no está listo."
            return
        }
        predicting = true
        predictionError = null
        appScope.launch {
            try {
                val prediction = predictor.predict(bytes)
                val isBcc = prediction.label == BccLabel.BCC
                repository.addImagePrediction(
                    imageName = imageName,
                    isBcc = isBcc,
                    confidence = prediction.confidence,
                )
                result = PredictionUi(
                    isBcc = isBcc,
                    confidence = prediction.confidence,
                    imageBytes = bytes,
                    imageName = imageName,
                )
                route = AppRoute.RESULT
            } catch (error: Exception) {
                // Sin detalles internos: mensaje corto, sin stack traces al usuario.
                predictionError = "No se pudo procesar la imagen. Intenta con otra."
            } finally {
                predicting = false
            }
        }
    }

    // Bottom navigation solo en destinos de nivel superior; RESULT y SUBSET son
    // hijas a pantalla completa con su propia salida y no la muestran.
    val showNavBar = route == AppRoute.HOME || route == AppRoute.HISTORY || route == AppRoute.SETTINGS

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        containerColor = MaterialTheme.colorScheme.surface,
        bottomBar = {
            if (showNavBar) {
                NavigationBar {
                    NavigationBarItem(
                        selected = route == AppRoute.HOME,
                        onClick = { route = AppRoute.HOME },
                        icon = { Icon(Icons.Filled.Home, contentDescription = "Inicio") },
                        label = { Text("Inicio") },
                    )
                    NavigationBarItem(
                        selected = route == AppRoute.HISTORY,
                        onClick = { route = AppRoute.HISTORY },
                        icon = { Icon(Icons.Filled.History, contentDescription = "Historial") },
                        label = { Text("Historial") },
                    )
                    NavigationBarItem(
                        selected = route == AppRoute.SETTINGS,
                        onClick = { route = AppRoute.SETTINGS },
                        icon = { Icon(Icons.Filled.Settings, contentDescription = "Configuración") },
                        label = { Text("Configuración") },
                    )
                }
            }
        },
    ) { padding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
            contentAlignment = Alignment.Center,
        ) {
            when (route) {
                AppRoute.HOME -> HomeScreen(
                    modelState = modelState,
                    predicting = predicting,
                    predictionError = predictionError,
                    modelLoadError = modelError,
                    onImageBytesChosen = ::onImageChosen,
                    onOpenSubset = { route = AppRoute.SUBSET },
                    onOpenSettings = { route = AppRoute.SETTINGS },
                )

                AppRoute.HISTORY -> HistoryScreen(items = historyItems)

                AppRoute.RESULT -> {
                    val current = result
                    if (current != null) {
                        ResultScreen(
                            result = current,
                            onAnotherImage = { route = AppRoute.HOME },
                            onViewHistory = { route = AppRoute.HISTORY },
                        )
                    } else {
                        // Defensa: RESULT sin payload (p.ej. cambio de configuración raro).
                        HomeScreen(
                            modelState = modelState,
                            predicting = predicting,
                            predictionError = predictionError,
                            modelLoadError = modelError,
                            onImageBytesChosen = ::onImageChosen,
                            onOpenSubset = { route = AppRoute.SUBSET },
                            onOpenSettings = { route = AppRoute.SETTINGS },
                        )
                    }
                }

                AppRoute.SETTINGS -> {
                    // viewModel acotado a la pantalla; sobrevive a rotaciones.
                    val settingsViewModel: SettingsViewModel = viewModel(
                        factory = viewModelFactory {
                            initializer {
                                SettingsViewModel(modelManager = modelManager)
                            }
                        },
                    )
                    val releaseState by settingsViewModel.releaseState.collectAsStateWithLifecycle()
                    SettingsScreen(
                        modelState = modelState,
                        releaseState = releaseState,
                        isApiKeyConfigured = settingsViewModel.isApiKeyConfigured,
                        onDeleteModel = settingsViewModel::deleteModel,
                        onCheckDownload = settingsViewModel::checkForUpdateOrDownload,
                    )
                }

                AppRoute.SUBSET -> {
                    // viewModel acotado a esta pantalla: sobrevive a rotaciones; la corrida
                    // y su cancelación viven en viewModelScope (S2). El predictor del flujo
                    // subconjunto lo crea el VM perezosamente desde el archivo verificado.
                    val subsetViewModel: SubsetViewModel = viewModel(
                        factory = viewModelFactory {
                            initializer {
                                SubsetViewModel(
                                    modelFileProvider = { modelManager.modelFile() },
                                    repository = repository,
                                )
                            }
                        },
                    )
                    SubsetScreen(
                        modelState = modelState,
                        viewModel = subsetViewModel,
                        onBack = { route = AppRoute.HOME },
                        onViewHistory = { route = AppRoute.HISTORY },
                    )
                }
            }
        }
    }
}

// Etiqueta de Logcat de este archivo; solo se loguea la causa del fallo del
// motor (nunca contenido de imágenes ni claves). Consumida por Log.e del
// LaunchedEffect(modelState) (R3-001).
private const val TAG = "AppRoot"
