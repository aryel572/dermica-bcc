package com.dermica.bcc.ml

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Construye el tensor de entrada [1, 224, 224, 3] float32 sin normalización:
 * píxeles en rango crudo [0,255], RGB por píxel, en orden de fila.
 *
 * Fidelidad de entrenamiento/evaluación (S3/S4): evaluate.py (`_parse`) hace
 * decode_jpeg → cast float32 sobre [0,255] SIN /255 ni preprocess_input, porque
 * image_dataset_from_directory no normaliza en entrenamiento. Se replica tal cual.
 *
 * Puro JVM: la parte testeable (forma del tensor y rango de valores) vive aquí;
 * el decode/redimensionado de bytes a píxeles es la capa Android (Preprocess).
 */
object TensorBuffer {

    const val IMAGE_SIZE = 224
    const val CHANNELS = 3
    const val PIXEL_COUNT = IMAGE_SIZE * IMAGE_SIZE
    const val INPUT_FLOATS = PIXEL_COUNT * CHANNELS
    const val INPUT_BYTES = INPUT_FLOATS * Float.SIZE_BYTES

    /** Buffer directo nativo [1,224,224,3] float32; `rewind`ed y listo para el Interpreter. */
    fun allocateInput(): ByteBuffer =
        ByteBuffer.allocateDirect(INPUT_BYTES).order(ByteOrder.nativeOrder())

    /**
     * Vuelca píxeles ARGB (0xAARRGGBB) al buffer de entrada, sin normalizar.
     *
     * Parámetros:
     *   argbPixels — 224×224 píxeles ARGB en orden de fila.
     *   out — buffer que se rellena (por defecto, uno nuevo directo).
     * Retorna: el mismo buffer, rebobinado a la posición 0 para el Interpreter.
     * Lanza IllegalArgumentException si el conteo de píxeles no es exactamente 224×224.
     */
    fun fromArgb(argbPixels: IntArray, out: ByteBuffer = allocateInput()): ByteBuffer {
        require(argbPixels.size == PIXEL_COUNT) {
            "Se esperaban $PIXEL_COUNT píxeles (${IMAGE_SIZE}×$IMAGE_SIZE), recibidos ${argbPixels.size}"
        }
        val floats = out.asFloatBuffer()
        for (argb in argbPixels) {
            floats.put(((argb shr 16) and 0xFF).toFloat()) // R
            floats.put(((argb shr 8) and 0xFF).toFloat())  // G
            floats.put((argb and 0xFF).toFloat())          // B
        }
        out.rewind()
        return out
    }
}
