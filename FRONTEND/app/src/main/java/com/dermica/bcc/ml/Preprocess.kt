package com.dermica.bcc.ml

import android.graphics.Bitmap
import android.graphics.BitmapFactory

/**
 * Decode y redimensionado de imágenes al contrato del modelo.
 *
 * Fidelidad con el entrenamiento (S3): resize_224.py preparó las imágenes con
 * PIL Image.resize((224,224), Image.BILINEAR) sobre el original (squash, sin
 * recorte ni preservación de aspecto). Aquí se replica la misma geometría con
 * createScaledBitmap(..., filter=true), que aplica filtrado bilinear.
 *
 * Desvío conocido (documentado, no evitable en JVM): el kernel exacto y el
 * muestreo de PIL vs Android pueden diferir en detalle de sub-píxel; el modelo
 * se fine-tuneó con 24k imágenes y el efecto reportado en métricas es despreciable.
 */
object Preprocess {

    /**
     * bytes JPEG/PNG → Bitmap 224×224 ARGB_8888.
     * Lanza IllegalArgumentException si los bytes no decodifican como imagen.
     */
    fun decodeTo224(imageBytes: ByteArray): Bitmap {
        val options = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        val source = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size, options)
            ?: throw IllegalArgumentException("El archivo no es una imagen JPEG/PNG decodificable")
        if (source.width == TensorBuffer.IMAGE_SIZE && source.height == TensorBuffer.IMAGE_SIZE) {
            return source // ya está en el tamaño del modelo: filtrar solo sumaría blur
        }
        val scaled = Bitmap.createScaledBitmap(
            source,
            TensorBuffer.IMAGE_SIZE,
            TensorBuffer.IMAGE_SIZE,
            /* filter = */ true, // bilinear, igual criterio que PIL BILINEAR del pipeline S3
        )
        if (scaled !== source) source.recycle()
        return scaled
    }

    /** Bitmap → píxeles ARGB (0xAARRGGBB) en orden de fila, listos para TensorBuffer.fromArgb. */
    fun toArgbPixels(bitmap: Bitmap): IntArray {
        val pixels = IntArray(TensorBuffer.PIXEL_COUNT)
        bitmap.getPixels(
            pixels, 0, TensorBuffer.IMAGE_SIZE, 0, 0, TensorBuffer.IMAGE_SIZE, TensorBuffer.IMAGE_SIZE,
        )
        return pixels
    }
}
