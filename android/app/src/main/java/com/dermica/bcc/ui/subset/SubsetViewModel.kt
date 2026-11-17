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
import kotlin.coroutines.coroutineContext
import java.io.File

/**
 * Imagen candidata dentro de la carpeta SAF, ya resuelta a su URI (S2).
 * Consumidor: SubsetViewModel (abre el stream por contentResolver; nunca copia
 * el archivo) y SubsetScreen (solo para el contador done/total).
 */
data class SubsetImage(
    val name: String,
    val uri: Uri,
)

/**
 * Entrada de una subcarpeta en la corrida multi (S5-multi): ya inspeccionada por
 * el adapter Android (imágenes compatibles + texto del ground truth).
 *
 * Consumidor: [SubsetViewModel.startMultiRun]; la construye el adapter vía
 * [SubsetFolderInput] para cada subcarpeta que SubsetScanner calificó.
 *
 * @param folderLabel nombre de la subcarpeta; queda etiquetado en cada card del historial.
 * @param groundTruthJson texto del ground_truth.json de la subcarpeta; null = no se pudo
 *        leer (la subcarpeta se reporta fallida en el resumen y el lote continúa).
 */
data class SubsetFolderInput(
    val folderLabel: String,
    val images: List<SubsetImage>,
    val groundTruthJson: String?,
)

/**
 * Resumen final del modo subconjunto (S4/S5): lo consume SubsetResultScreen y
 * alimenta la card del historial (S6).
 *
 * Lo produce processFolder al cerrar una carpeta; en modo single navega a
 * [SubsetRunState.Done] con él; en multi el lote lo agrega por subcarpeta.
 *
 * @param totalImages imágenes (*.jpg/.jpeg/.png) encontradas en la carpeta.
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
 * Fallo acotado de UNA subcarpeta dentro de la corrida multi (S5-multi): nombre
 * + motivo corto. Nunca aborta el lote; se acumula y se muestra al final.
 * Consumidor: [SubsetBatchSummary.failed] → MultiDoneContent (lista de 3 + «…y N más»).
 */
data class FolderFailure(
    val folderLabel: String,
    val reason: String,
)

/**
 * Resumen de la corrida multi (S5-multi): lo consume la pantalla final
 * (MultiDoneContent en SubsetScreen). Lo publica el VM en
 * [SubsetRunState.MultiDone].
 *
 * @param completed subcarpetas procesadas y persistidas en el historial.
 * @param failed subcarpetas omitidas o no persistidas, con motivo corto.
 */
data class SubsetBatchSummary(
    val completed: Int,
    val failed: List<FolderFailure>,
)

/**
 * Estados del flujo subconjunto (S2/S3/S4 · S5-multi). Cola de trabajo explícita
 * para la UI: una sola fuente de verdad, sin observables paralelos.
 *
 * Contrato de estados: [Idle] (sin trabajo) → [Running] → estados finales
 * [Done] | [MultiDone] | [Error] | [Cancelled].
 *
 * - [Running] lleva el avance (imágenes de la carpeta actual) y, en multi, la
 *   subcarpeta en curso ([Running.currentFolder] no null) + posición en el lote.
 * - Estados finales: [Done] (carpeta única, T4), [MultiDone] (lote, S5-multi),
 *   [Error] y [Cancelled]; nunca hay trabajo huérfano visible.
 *
 * Quién lo consume: SubsetScreen (collectAsStateWithLifecycle + when(state)).
 * El VM sobrescribe `_state` de forma monótona; la UI pinta solo el último.
 */
sealed interface SubsetRunState {
    data object Idle : SubsetRunState

    /**
     * @param done imágenes procesadas dentro de la carpeta actual.
     * @param total imágenes de la carpeta actual.
     * @param currentName imagen en curso (null = preparando).
     * @param currentFolder null = carpeta única (T4); nombre = caso multi (S5-multi).
     * @param folderIndex subcarpeta actual (1-based); 1 en carpeta única.
     * @param folderCount subcarpetas totales de la corrida; 1 en carpeta única.
     * @param foldersDone subcarpetas ya completadas y persistidas en esta corrida.
     */
    data class Running(
        val done: Int,
        val total: Int,
        val currentName: String?,
        val currentFolder: String? = null,
        val folderIndex: Int = 1,
        val folderCount: Int = 1,
        val foldersDone: Int = 0,
    ) : SubsetRunState

    data class Done(val summary: SubsetSummary) : SubsetRunState

    /** Lote multi terminado (S5-multi): N subconjuntos persistidos + fallos acotados. Estado final del flujo multi; lo pinta MultiDoneContent. */
    data class MultiDone(val summary: SubsetBatchSummary) : SubsetRunState

    /** Fallo terminal previo o durante la corrida; mensaje ya redactado para la UI (ErrorContent). */
    data class Error(val message: String) : SubsetRunState

    /**
     * Cancelación pedida (S2/S5-multi).
     * @param savedSubsets subconjuntos ya procesados y persistidos antes del corte
     *        (0 en carpeta única: la carpeta en curso nunca se persiste a medias).
     */
    data class Cancelled(val savedSubsets: Int = 0) : SubsetRunState
}

/**
 * Orquestador del modo subconjunto (S2–S6 · S5-multi): parsea ground_truth.json,
 * infiere con el mismo motor que la imagen única y acumula la matriz de confusión.
 *
 * Dueño de la corrida: viewModelScope. [cancel] coopera entre imágenes (cada
 * iteración empieza con ensureActive()) Y entre subcarpetas (un ensureActive()
 * por carpeta); el pipeline de una imagen en curso termina —coste acotado por el
 * timeout de 10 s de [BccPredictor]— y el usuario vuelve a ver progreso honesto
 * hasta entonces.
 *
 * En multi, cada subcarpeta se persiste EN CUANTO termina (atómico vía
 * [HistoryRepository.addSubsetResult]): si el usuario cancela a mitad de lote o
 * una subcarpeta falla (JSON corrupto, sin imágenes, no persiste), las ya
 * cerradas NO se pierden; la fallida se reporta en [SubsetBatchSummary.failed]
 * y el lote continúa.
 *
 * El predictor se crea por corrida a partir del archivo del modelo y se cierra al
 * final: decoupled del BccPredictor compartido de la imagen única (que puede
 * cerrarse/recrearse en otra pantalla con el ciclo de vida del modelo) y el coste
 * es una carga por corrida (~2 MB, despreciable frente a la inferencia).
 *
 * Quién lo consume: SubsetScreen (state + callbacks). Quien lo crea: AppRoot
 * (AppRoute.SUBSET) con factory (modelFileProvider desde ModelManager y el
 * HistoryRepository compartido). Persistencia: HistoryRepository.addSubsetResult
 * por carpeta; lecturas en Dispatchers.Default dentro de [viewModelScope].
 *
 * @param modelFileProvider proveedor perezoso del archivo verificado del modelo;
 *        null = el VM publica Error honesto (nunca lanza).
 * @param repository historial compartido (Room) al que se persiste cada carpeta.
 */
class SubsetViewModel(
    private val modelFileProvider: () -> File?,
    private val repository: HistoryRepository,
) : ViewModel() {

    /** Estado privado mutable de [state]; la UI solo ve la versión asStateFlow (sin setters). */
    private val _state = MutableStateFlow<SubsetRunState>(SubsetRunState.Idle)

    /** Salida pública hacia SubsetScreen: fase actual del flujo. Única fuente de verdad de la cola. */
    val state: StateFlow<SubsetRunState> = _state.asStateFlow()

    /** Corrida en vuelo; permite cancelar y detectar doble-toque (startRun/startMultiRun). */
    private var runJob: Job? = null

    /**
     * Plan interno de una carpeta: ground truth ya parseado (single) o pendiente
     * (multi). `groundTruth == null` ⇒ se parsea dentro del bucle, con `groundTruthJson`
     * como entrada; así un JSON corrupto de una subcarpeta no tumba el lote.
     */
    private data class FolderPlan(
        val label: String,
        val images: List<SubsetImage>,
        val groundTruth: GroundTruth?,
        val groundTruthJson: String? = null,
    )

    /**
     * Publica un error previo al procesado (p.ej. SAF no devolvió la carpeta/JSON legible).
     * Llamado por prepareAndRun cuando el adapter devuelve TreeResolution.Fail.
     */
    fun fail(message: String) {
        _state.value = SubsetRunState.Error(message)
    }

    /**
     * Vuelve al selector de carpeta («Otra carpeta» / «Reintentar»).
     * Solo cambia la fase visible (Idle); no cancela nada: la corrida ya terminó
     * cuando aparecen esos botones.
     */
    fun reset() {
        _state.value = SubsetRunState.Idle
    }

    /**
     * Cancelación cooperativa: el bucle corta en la siguiente imagen o subcarpeta (S2/S5-multi).
     * Llamado por el botón Cancelar (RunningContent) vía viewModel::cancel. No inventa
     * un resultado: la carpeta en curso se descarta y se publica Cancelled.
     */
    fun cancel() {
        runJob?.cancel()
    }

    /**
     * Lanza el procesado de UNA carpeta (flujo T4 intacto).
     *
     * @param folderLabel nombre visible de la carpeta (queda en el historial y en el resultado).
     * @param images imágenes candidatas en orden estable (adapter ya las inspeccionó).
     * @param groundTruthJson texto del ground_truth.json de la raíz.
     * @param contentResolver resolver SAF del llamador (la VM lo recibe y delega en él la apertura de streams).
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

        startInternal(
            plans = listOf(FolderPlan(label = folderLabel, images = images, groundTruth = groundTruth)),
            multi = false,
            modelFile = modelFile,
            contentResolver = contentResolver,
        )
    }

    /**
     * Lanza el procesado del LOTE multi (S5-multi): cada subcarpeta corre el pipeline
     * exacto de T4 con su ground_truth.json, se persiste al terminar (folderLabel =
     * nombre de la subcarpeta) y los fallos acotados no tumban el lote.
     *
     * @param folders subcarpetas calificadas por el scanner (adapter ya las inspeccionó).
     * @param contentResolver resolver SAF del llamador.
     */
    fun startMultiRun(
        folders: List<SubsetFolderInput>,
        contentResolver: ContentResolver,
    ) {
        if (runJob?.isActive == true) return
        if (folders.isEmpty()) {
            // Defensa en profundidad: el adapter no llama con la lista vacía.
            _state.value = SubsetRunState.Error(
                "No se encontraron subconjuntos elegibles aquí; elige otra carpeta o una subcarpeta con ground_truth.json.",
            )
            return
        }

        val modelFile = modelFileProvider()
        if (modelFile == null) {
            _state.value = SubsetRunState.Error("El modelo todavía no está listo.")
            return
        }

        startInternal(
            plans = folders.map { input ->
                FolderPlan(
                    label = input.folderLabel,
                    images = input.images,
                    groundTruth = null, // se parsea dentro del bucle: un JSON corrupto no tumba el lote
                    groundTruthJson = input.groundTruthJson,
                )
            },
            multi = true,
            modelFile = modelFile,
            contentResolver = contentResolver,
        )
    }

    /**
     * Motor de la corrida (single y multi comparten pipeline). [multi] decide el
     * estado final: Done(carpeta única) o MultiDone(lote); Cancelled lleva el conteo
     * de subconjuntos ya persistidos en multi.
     *
     * @param plans carpetas a procesar (single = 1 plan con groundTruth ya parseado).
     * @param multi true = lote S5-multi; false = T4 (una carpeta).
     * @param modelFile archivo del modelo verificado (ya resuelto por el caller).
     * @param contentResolver resolver SAF para abrir los streams de las imágenes.
     *
     * Corre en [viewModelScope] con Dispatchers.Default; la cancelación corta entre
     * subcarpetas y entre imágenes vía ensureActive().
     */
    private fun startInternal(
        plans: List<FolderPlan>,
        multi: Boolean,
        modelFile: File,
        contentResolver: ContentResolver,
    ) {
        runJob = viewModelScope.launch(Dispatchers.Default) {
            var predictor: BccPredictor? = null
            var foldersDone = 0
            val failures = mutableListOf<FolderFailure>()
            try {
                predictor = try {
                    BccPredictor(modelFile)
                } catch (error: Exception) {
                    _state.value = SubsetRunState.Error("El modelo no se pudo cargar.")
                    return@launch
                }

                for ((folderIndex, plan) in plans.withIndex()) {
                    ensureActive() // punto de corte de la cancelación ENTRE subcarpetas (S5-multi)
                    if (plan.images.isEmpty()) {
                        // El adapter ya filtra vacías; defensa en profundidad sin tumbar el lote.
                        failures += FolderFailure(plan.label, "Sin imágenes .jpg, .jpeg ni .png.")
                        continue
                    }
                    val groundTruth = plan.groundTruth ?: run {
                        val text = plan.groundTruthJson
                        try {
                            GroundTruth.parse(
                                text ?: throw GroundTruthFormatException("No se pudo leer ground_truth.json."),
                            )
                        } catch (error: GroundTruthFormatException) {
                            if (!multi) {
                                _state.value = SubsetRunState.Error(
                                    error.message ?: "ground_truth.json no válido.",
                                )
                                return@launch
                            }
                            // JSON corrupto o ilegible: se reporta y el lote CONTINÚA.
                            failures += FolderFailure(plan.label, error.message ?: "ground_truth.json no válido.")
                            null
                        }
                    } ?: continue

                    val summary = processFolder(
                        plan = plan,
                        groundTruth = groundTruth,
                        predictor = predictor,
                        contentResolver = contentResolver,
                        multi = multi,
                        folderIndex = folderIndex + 1,
                        folderCount = plans.size,
                        foldersDone = foldersDone,
                    )
                    if (multi) {
                        if (summary.historySaved) {
                            foldersDone++
                        } else {
                            // Persistencia fallida: la subcarpeta no cuenta como completada.
                            failures += FolderFailure(plan.label, "No se pudo guardar en el historial.")
                        }
                    } else {
                        foldersDone++ // carpeta única: Done ya quedó publicado por processFolder
                    }
                }

                if (multi) {
                    _state.value = SubsetRunState.MultiDone(
                        SubsetBatchSummary(completed = foldersDone, failed = failures.toList()),
                    )
                }
            } catch (cancel: CancellationException) {
                // Cancelación pedida: las subcarpetas ya cerradas se persistieron al
                // terminar; la carpeta en curso se descarta (matriz incompleta, no se
                // presenta como válida). En carpeta única: savedSubsets = 0.
                _state.value = SubsetRunState.Cancelled(savedSubsets = if (multi) foldersDone else 0)
                throw cancel
            } catch (error: Exception) {
                _state.value = SubsetRunState.Error("No se pudo procesar la carpeta. Intenta de nuevo.")
            } finally {
                predictor?.close()
            }
        }
    }

    /**
     * Procesa UNA carpeta completa: matcher + inferencia + matriz + persistencia.
     * Publica [SubsetRunState.Done] sólo en modo single (contrato T4 intacto);
     * en multi devuelve el resumen y el autor del lote decide el estado final.
     *
     * @param plan carpeta a procesar con sus imágenes y ground truth (o JSON pendiente).
     * @param groundTruth ground truth ya parseado de esta carpeta.
     * @param predictor motor ya cargado por startInternal (compartido por la corrida).
     * @param contentResolver resolver SAF para abrir cada imagen.
     * @param multi true = lote (Done no se publica; el autor del lote decide el final).
     * @param folderIndex posición 1-based de la carpeta en el plan (1 en T4).
     * @param folderCount total de carpetas de la corrida (1 en T4).
     * @param foldersDone subcarpetas ya cerradas en esta corrida (0 en T4).
     * @return resumen de la carpeta (lo consume startInternal para el estado final).
     *
     * @throws CancellationException si el usuario cancela (la carpeta en curso no se
     *         persiste; las subcarpetas cerradas sí).
     */
    private suspend fun processFolder(
        plan: FolderPlan,
        groundTruth: GroundTruth,
        predictor: BccPredictor,
        contentResolver: ContentResolver,
        multi: Boolean,
        folderIndex: Int,
        folderCount: Int,
        foldersDone: Int,
    ): SubsetSummary {
        val images = plan.images
        var tp = 0
        var tn = 0
        var fp = 0
        var fn = 0
        var processed = 0
        var excluded = 0
        var failed = 0

        for ((index, image) in images.withIndex()) {
            coroutineContext.ensureActive() // punto de corte de la cancelación, entre imágenes
            _state.value = SubsetRunState.Running(
                done = index,
                total = images.size,
                currentName = image.name,
                currentFolder = if (multi) plan.label else null,
                folderIndex = folderIndex,
                folderCount = folderCount,
                foldersDone = foldersDone,
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
            _state.value = SubsetRunState.Running(
                done = index + 1,
                total = images.size,
                currentName = image.name,
                currentFolder = if (multi) plan.label else null,
                folderIndex = folderIndex,
                folderCount = folderCount,
                foldersDone = foldersDone,
            )
        }

        val totals = Totals(tp = tp, tn = tn, fp = fp, fn = fn)
        val metrics = totals.compute()
        // Historial al final del procesado de la carpeta (S6). Fallo de escritura no
        // descarta el resultado: se avisa en pantalla (historySaved=false) en vez de
        // mentir; en multi además la subcarpeta se reporta fallida y el lote sigue.
        val historySaved = try {
            repository.addSubsetResult(
                folderLabel = plan.label,
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
        val summary = SubsetSummary(
            folderLabel = plan.label,
            totalImages = images.size,
            processed = processed,
            excluded = excluded,
            failed = failed,
            totals = totals,
            metrics = metrics,
            historySaved = historySaved,
        )
        if (!multi) {
            _state.value = SubsetRunState.Done(summary)
        }
        return summary
    }
}
