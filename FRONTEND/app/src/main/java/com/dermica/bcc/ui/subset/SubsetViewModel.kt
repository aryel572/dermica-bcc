package com.dermica.bcc.ui.subset

import android.content.ContentResolver
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.dermica.bcc.core.GroundTruth
import com.dermica.bcc.core.GroundTruthFormatException
import com.dermica.bcc.core.SubsetMetrics
import com.dermica.bcc.core.Totals
import com.dermica.bcc.data.history.HistoryRepository
import com.dermica.bcc.ml.BccLabel
import com.dermica.bcc.ml.BccPredictor
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import java.io.File

/** Imagen candidata dentro de la carpeta SAF, ya resuelta a su URI (S2). */
data class SubsetImage(
    val name: String,
    val uri: Uri,
)

/**
 * Resumen final del modo subconjunto (S4/S5): lo consume SubsetResultScreen y
 * alimenta la card del historial (S6).
 *
 * @param totalImages imágenes imagen (*.jpg/.jpeg/.png) encontradas en la carpeta.
 * @param processed imágenes con etiqueta e inferencia completada (entran a la matriz).
 * @param excluded imágenes sin entrada en el ground truth (excluidas y contadas, S3).
 * @param failed imágenes con etiqueta que no se pudieron leer/decodificar (se omiten
 *               de la matriz; jamás se inventa un resultado para ellas).
 * @param historySaved false si el resultado no se pudo persistir en el historial;
 *               la UI lo avisa en una línea sin descartar los números en pantalla.
 */
data class SubsetSummary(
    val folderLabel: String,
    val totalImages: Int,
    val processed: Int,
    val excluded: Int,
    val failed: Int,
    val totals: Totals,
    val metrics: SubsetMetrics,
    val historySaved: Boolean,
)

/**
 * Estados del flujo subconjunto (S2/S3/S4). Cola de trabajo explícita para la UI:
 * una sola fuente de verdad, sin observables paralelos.
 *
 * - [Running] lleva el nombre de la imagen en curso y el avance (S2: progreso honesto).
 * - Estados finales: [Done], [Error] y [Cancelled]; nunca hay trabajo huérfano visible.
 */
sealed interface SubsetRunState {
    data object Idle : SubsetRunState
    data class Running(val done: Int, val total: Int, val currentName: String?) : SubsetRunState
    data class Done(val summary: SubsetSummary) : SubsetRunState
    data class Error(val message: String) : SubsetRunState
    data object Cancelled : SubsetRunState
}

/**
 * Orquestador del modo subconjunto (S2–S6): parsea ground_truth.json, infiere con
 * el mismo motor que la imagen única y acumula la matriz de confusión.
 *
 * Dueño de la corrida: viewModelScope. [cancel] coopera entre imágenes (cada
 * iteración empieza con ensureActive()); el pipeline de una imagen en curso termina
 * —coste acotado por el timeout de 10 s de [BccPredictor]— y el usuario vuelve a ver
 * progreso honesto hasta entonces.
 *
 * El predictor se crea por corrida a partir del archivo del modelo y se cierra al
 * final: decoupled del BccPredictor compartido de la imagen única (que puede
 * cerrarse/recrearse en otra pantalla con el ciclo de vida del modelo) y el coste
 * es una carga por corrida (~2 MB, despreciable frente a la inferencia).
 */
class SubsetViewModel(
    private val modelFileProvider: () -> File?,
    private val repository: HistoryRepository,
) : ViewModel() {

    private val _state = MutableStateFlow<SubsetRunState>(SubsetRunState.Idle)
    val state: StateFlow<SubsetRunState> = _state.asStateFlow()

    private var runJob: Job? = null

    /** Publica un error previo al procesado (p.ej. SAF no devolvió la carpeta/JSON legible). */
    fun fail(message: String) {
        _state.value = SubsetRunState.Error(message)
    }

    /** Vuelve al selector de carpeta («Otra carpeta» / «Reintentar»). */
    fun reset() {
        _state.value = SubsetRunState.Idle
    }

    /** Cancelación cooperativa: el bucle corta en la siguiente imagen (S2). */
    fun cancel() {
        runJob?.cancel()
    }

    /**
     * Lanza el procesado completo de la carpeta.
     *
     * Precondiciones verificadas aquí (la UI ya las filtra, aquí es defensa en profundidad):
     * ground_truth.json parsea; si no, se publica Error y no se procesa nada (S3).
     */
    fun startRun(
        folderLabel: String,
        images: List<SubsetImage>,
        groundTruthJson: String,
        contentResolver: ContentResolver,
    ) {
        // Un toque duplicado en «Elegir carpeta» no debe lanzar dos corridas en paralelo.
        if (runJob?.isActive == true) return

        val groundTruth = try {
            GroundTruth.parse(groundTruthJson)
        } catch (error: GroundTruthFormatException) {
            _state.value = SubsetRunState.Error(error.message ?: "ground_truth.json no es válido.")
            return
        }

        val modelFile = modelFileProvider()
        if (modelFile == null) {
            _state.value = SubsetRunState.Error("El modelo todavía no está listo.")
            return
        }

        runJob = viewModelScope.launch(Dispatchers.Default) {
            var predictor: BccPredictor? = null
            try {
                predictor = try {
                    BccPredictor(modelFile)
                } catch (error: Exception) {
                    _state.value = SubsetRunState.Error("El modelo no se pudo cargar.")
                    return@launch
                }
                var tp = 0
                var tn = 0
                var fp = 0
                var fn = 0
                var processed = 0
                var excluded = 0
                var failed = 0

                for ((index, image) in images.withIndex()) {
                    ensureActive() // punto de corte de la cancelación, entre imágenes
                    _state.value = SubsetRunState.Running(
                        done = index,
                        total = images.size,
                        currentName = image.name,
                    )
                    try {
                        val label = groundTruth.labelFor(image.name)
                        if (label == null) {
                            // Sin referencia: se EXCLUYE de la matriz y se reporta aparte (S3).
                            excluded++
                        } else {
                            val bytes = contentResolver.openInputStream(image.uri)?.use { it.readBytes() }
                            if (bytes == null) {
                                failed++
                            } else {
                                // Mismo motor y contrato que la imagen única (S9): no hay otro pipeline.
                                val prediction = predictor.predict(bytes)
                                val expectedBcc = groundTruth.isBcc(label)
                                val predictedBcc = prediction.label == BccLabel.BCC
                                when {
                                    expectedBcc && predictedBcc -> tp++
                                    !expectedBcc && !predictedBcc -> tn++
                                    !expectedBcc && predictedBcc -> fp++
                                    else -> fn++
                                }
                                processed++
                            }
                        }
                    } catch (cancel: CancellationException) {
                        throw cancel // nunca se cuenta como imagen fallida
                    } catch (error: Exception) {
                        // Imagen corrupta o timeout de inferencia: se omite y se reporta.
                        failed++
                    }
                    _state.value = SubsetRunState.Running(index + 1, images.size, image.name)
                }

                val totals = Totals(tp = tp, tn = tn, fp = fp, fn = fn)
                val metrics = totals.compute()
                // Historial al final del procesado (S6). Fallo de escritura no descarta
                // el resultado: se avisa en pantalla (historySaved=false) en vez de mentir.
                val historySaved = try {
                    repository.addSubsetResult(
                        folderLabel = folderLabel,
                        imageCount = processed,
                        totals = totals,
                        excluded = excluded,
                    )
                    true
                } catch (cancel: CancellationException) {
                    throw cancel
                } catch (error: Exception) {
                    false
                }
                _state.value = SubsetRunState.Done(
                    SubsetSummary(
                        folderLabel = folderLabel,
                        totalImages = images.size,
                        processed = processed,
                        excluded = excluded,
                        failed = failed,
                        totals = totals,
                        metrics = metrics,
                        historySaved = historySaved,
                    ),
                )
            } catch (cancel: CancellationException) {
                // Cancelación pedida: estado final SIN resultados parciales; una matriz
                // incompleta no se persiste ni se presenta como válida (S4).
                _state.value = SubsetRunState.Cancelled
                throw cancel
            } catch (error: Exception) {
                _state.value = SubsetRunState.Error("No se pudo procesar la carpeta. Intenta de nuevo.")
            } finally {
                predictor?.close()
            }
        }
    }
}
