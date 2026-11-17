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
 * Fidelidad con la evaluación (S4): entrada float32 [1,224,224,3] sobre [0,255]
 * sin normalización; salida sigmoid 1 unidad decodificada con umbral 0.5.
 *
 * Thread-safety: LiteRT no permite invoke concurrente sobre el mismo Interpreter;
 * se serializa con un lock. La inferencia corre en Dispatchers.Default y con
 * timeout defensivo para no colgar la UI si el runtime se degrada.
 */
class BccPredictor(modelFile: File) : AutoCloseable {

    private val lock = Any()
    private val interpreter: Interpreter
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
     * Parámetros:
     *   imageBytes — bytes JPEG/PNG de la imagen dermatoscópica.
     * Retorna: Prediction con etiqueta, confianza y sigmoid crudo.
     * Lanza: IllegalArgumentException si la imagen no decodifica;
     *         kotlinx.coroutines.TimeoutCancellationException si supera [TIMEOUT_MS];
     *         IllegalStateException si el runtime falla.
     */
    suspend fun predict(imageBytes: ByteArray): Prediction = withContext(Dispatchers.Default) {
        withTimeout(TIMEOUT_MS) { predictBlocking(imageBytes) }
    }

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

    /** Libera el Interpreter; la instancia no debe usarse después de close(). */
    override fun close() = interpreter.close()

    private companion object {
        const val TIMEOUT_MS = 10_000L // margen amplio: MobileNetV3-Small en frío ronda 1-2 s en gama media
    }
}
