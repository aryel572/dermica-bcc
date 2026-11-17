package com.dermica.bcc.ml

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Construye el tensor de entrada [1, 224, 224, 3] float32 sin normalización:
 * píxeles en rango crudo [0,255], RGB por píxel, en orden de fila.
 *
 * Lugar en el flujo: motor ML (ml), puro JVM. Lo consumen BccPredictor
 * (allocateInput indirecto vía fromArgb) y TensorBufferTest en JVM; el contrato
 * de tensores lo completa Preprocess (decode Android específico de plataforma).
 * Salida visible: el ByteBuffer directo que recibe interpreter.run como entrada.
 *
 * Fidelidad de entrenamiento/evaluación (S3/S4): evaluate.py (`_parse`) hace
 * decode_jpeg → cast float32 sobre [0,255] SIN /255 ni preprocess_input, porque
 * image_dataset_from_directory no normaliza en entrenamiento. Se replica tal cual.
 */
object TensorBuffer {

    // 224 px por lado: resolución de entrada del modelo fine-tuneado (S3). No cambiar
    // sin re-entrenar. Consumidores: Preprocess, BccPredictor (verificación de tensores).
    const val IMAGE_SIZE = 224

    // RGB, sin canal alpha: la entrada del modelo es [1,224,224,3].
    const val CHANNELS = 3

    // PIXEL_COUNT = IMAGE_SIZE² : píxeles por tensor [1,224,224,3].
    const val PIXEL_COUNT = IMAGE_SIZE * IMAGE_SIZE

    // INPUT_FLOATS = PIXEL_COUNT × CHANNELS : elementos float32 de la entrada.
    const val INPUT_FLOATS = PIXEL_COUNT * CHANNELS

    // INPUT_BYTES = INPUT_FLOATS × 4 : tamaño del ByteBuffer directo que LiteRT exige.
    const val INPUT_BYTES = INPUT_FLOATS * Float.SIZE_BYTES

    /**
     * Buffer directo nativo [1,224,224,3] float32; posición 0 y orden nativo.
     *
     * @return ByteBuffer directo (allocateDirect) en orden nativo: posición 0,
     *         límite = capacidad = INPUT_BYTES. LiteRT exige direct buffers.
     */
    fun allocateInput(): ByteBuffer =
        ByteBuffer.allocateDirect(INPUT_BYTES).order(ByteOrder.nativeOrder())

    /**
     * Vuelca píxeles ARGB (0xAARRGGBB) al buffer de entrada, sin normalizar.
     *
     * @param argbPixels — IntArray: 224×224 píxeles ARGB en orden de fila
     *        (lo produce Preprocess.toArgbPixels).
     * @param out — ByteBuffer: buffer que se rellena; por defecto uno nuevo directo
     *        (allocateInput). Recibe RGB por píxel como float32 en [0,255].
     * @return ByteBuffer: el mismo [out], rebobinado a la posición 0 para el Interpreter.
     * @throws IllegalArgumentException si el conteo de píxeles no es exactamente 224×224.
     *
     * Efectos: escribe [out] in-place; no aloca más memoria que el buffer pasado o uno nuevo.
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
