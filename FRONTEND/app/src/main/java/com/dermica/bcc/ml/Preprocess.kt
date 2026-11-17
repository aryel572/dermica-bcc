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

    // Fix R4-009: umbral de memoria en el lado mayor. Las dermatocopias grandes se
    // submuestrean (inSampleSize, potencia de 2) antes del squash final a 224 px;
    // por debajo el decode es idéntico al original píxel a píxel (fidelidad S3 intacta).
    private const val MEMORY_LIMIT_SIDE_PX = 2048

    /**
     * bytes JPEG/PNG → Bitmap 224×224 ARGB_8888.
     * Lanza IllegalArgumentException si los bytes no decodifican como imagen.
     *
     * Memoria (fix R4-009): si el lado mayor de la imagen supera [MEMORY_LIMIT_SIDE_PX],
     * se acota primero con inJustDecodeBounds + inSampleSize (potencia de 2) a ≤2048 px
     * antes del squash bilinear final; debajo del umbral no cambia ni un píxel.
     */
    fun decodeTo224(imageBytes: ByteArray): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size, bounds)
        val sourceWidth = bounds.outWidth
        val sourceHeight = bounds.outHeight
        if (sourceWidth <= 0 || sourceHeight <= 0) {
            // Bytes no decodificables: el submuestreo no tiene sentido, se falla igual que antes.
            throw IllegalArgumentException("El archivo no es una imagen JPEG/PNG decodificable")
        }
        val options = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        val maxSide = maxOf(sourceWidth, sourceHeight)
        if (maxSide > MEMORY_LIMIT_SIDE_PX) {
            options.inSampleSize = inSampleSizeFor(maxSide)
        }
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

    /**
     * Mayor potencia de 2 por la que submuestrear para dejar el lado mayor ≤ umbral.
     * Busca la primera potencia de 2 que satisface el límite; nunca devuelve 1 para
     * imágenes sobre el umbral.
     */
    private fun inSampleSizeFor(maxSide: Int): Int {
        var sample = 1
        while (maxSide / sample > MEMORY_LIMIT_SIDE_PX) sample *= 2
        return sample
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
