package com.dermica.bcc.ml

// LiteRT 2.x: el artefacto oficial es com.google.ai.edge.litert:litert, pero la API Java
// conserva el paquete org.tensorflow.lite (Renombrado de artefacto, no de paquete).
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Motor de inferencia on-device (S9): carga filesDir/model.tflite con LiteRT y
 * produce una predicción por imagen. 100% local: no toca la red.
 *
 * Lugar en el flujo: motor ML (ml). Lo consume ui/AppRoot (lo crea cuando
 * ModelState pasa a READY y lo libera al salir o al pasar a otro estado). En
 * JVM la inferencia real no se ejecuta: OutputDecoderTest y TensorBufferTest
 * cubren las piezas testeables sin artefacto .tflite. Salida visible: la card
 * de resultado con etiqueta, confianza y aviso de que no sustituye criterio médico.
 *
 * Fidelidad con la evaluación (S4): entrada float32 [1,224,224,3] sobre [0,255]
 * sin normalización; salida sigmoid 1 unidad decodificada con umbral 0.5.
 *
 * Thread-safety: LiteRT no permite invoke concurrente sobre el mismo Interpreter;
 * se serializa con [lock]. La inferencia corre en Dispatchers.Default y con
 * [TIMEOUT_MS] defensivo para no colgar la UI si el runtime se degrada.
 *
 * @param modelFile — File: artefacto interno a cargar; debe existir (require) y ser
 *        un .tflite float16 compatible con el contrato de tensores verificado al init.
 * @throws IllegalStateException si LiteRT no puede cargar el modelo (artefacto
 *         dañado o incompatible con el runtime).
 */
class BccPredictor(modelFile: File) : AutoCloseable {

    // Mutex lógico del Interpreter (LiteRT: un invoke a la vez). Lo toma
    // predictBlocking; close() no lo toma (el caller debe cerrar en exclusión).
    private val lock = Any()

    private val interpreter: Interpreter

    // Salida preasignada [1,1] float: evita asignar un array por inferencia;
    // se reescribe en cada interpreter.run bajo [lock].
    private val outputArray = Array(1) { FloatArray(1) }

    init {
        require(modelFile.exists()) { "No existe el modelo en ${modelFile.path}" }
        val modelBytes = modelFile.readBytes()
        val buffer = ByteBuffer.allocateDirect(modelBytes.size).order(ByteOrder.nativeOrder())
        buffer.put(modelBytes)
        buffer.rewind()
        val options = Interpreter.Options().apply { setNumThreads(4) } // evaluate.py usa num_threads=4
        interpreter = try {
            Interpreter(buffer, options)
        } catch (e: Exception) {
            throw IllegalStateException("LiteRT no pudo cargar el modelo: ${e.message}", e)
        }
        // Contrato de tensores verificado al cargar: mejor fallar aquí que en la primera predicción.
        val inputTensor = interpreter.getInputTensor(0)
        require(inputTensor.shape().contentEquals(intArrayOf(1, TensorBuffer.IMAGE_SIZE, TensorBuffer.IMAGE_SIZE, TensorBuffer.CHANNELS))) {
            "Entrada inesperada del modelo: shape=${inputTensor.shape().joinToString()} (se esperaba [1,224,224,3])"
        }
        require(inputTensor.dataType() == DataType.FLOAT32) {
            "Entrada inesperada del modelo: dtype=${inputTensor.dataType()} (se esperaba FLOAT32)"
        }
        val outputTensor = interpreter.getOutputTensor(0)
        require(outputTensor.shape().contentEquals(intArrayOf(1, 1))) {
            "Salida inesperada del modelo: shape=${outputTensor.shape().joinToString()} (se esperaba [1,1] sigmoid)"
        }
    }

    /**
     * Predice BCC/NO_BCC sobre los bytes de una imagen.
     *
     * @param imageBytes — ByteArray: bytes JPEG/PNG de la imagen dermatoscópica.
     * @return Prediction con etiqueta, confianza y sigmoid crudo.
     * @throws IllegalArgumentException si la imagen no decodifica (Preprocess.decodeTo224).
     * @throws kotlinx.coroutines.TimeoutCancellationException si supera [TIMEOUT_MS].
     * @throws IllegalStateException si el runtime falla al ejecutar la inferencia.
     *
     * Efectos colaterales: carga el bitmap en memoria y lo recicla al terminar;
     * no escribe disco ni red.
     */
    suspend fun predict(imageBytes: ByteArray): Prediction = withContext(Dispatchers.Default) {
        withTimeout(TIMEOUT_MS) { predictBlocking(imageBytes) }
    }

    // Decode → píxeles → tensor → invoke → decodificar. Sin lock externo: quien
    // llama desde predict() ya está en Dispatchers.Default; el lock interno
    // serializa contra otros invocadores y contra close().
    private fun predictBlocking(imageBytes: ByteArray): Prediction {
        val bitmap = Preprocess.decodeTo224(imageBytes)
        try {
            val pixels = Preprocess.toArgbPixels(bitmap)
            val input = TensorBuffer.fromArgb(pixels)
            synchronized(lock) { // LiteRT: un invoke a la vez por Interpreter
                interpreter.run(input, outputArray)
            }
        } finally {
            bitmap.recycle()
        }
        return OutputDecoder.decode(outputArray[0])
    }

    /**
     * Libera el Interpreter; la instancia no debe usarse después de close().
     *
     * Efectos colaterales: cierra el Interpreter nativo (libera memoria del modelo).
     * Idempotente en la práctica: LiteRT tolera doble close.
     */
    override fun close() = interpreter.close()

    private companion object {
        // TIMEOUT_MS: margen amplio sobre el presupuesto de MobileNetV3-Small en
        // frío (1–2 s medidos en gama media); corta una inferencia degradada sin
        // cortar una válida. Consumido por predict(); no es configurable.
        const val TIMEOUT_MS = 10_000L
    }
}
