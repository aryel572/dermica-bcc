package com.dermica.bcc

import com.dermica.bcc.ml.TensorBuffer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteOrder

/**
 * Verifica el contrato del tensor de entrada [1,224,224,3] float32 sin normalización:
 * forma, capacidad y rango [0,255], con el orden RGB por píxel de entrenamiento.
 */
class TensorBufferTest {

    private fun argb(r: Int, g: Int, b: Int): Int = (0xFF shl 24) or (r shl 16) or (g shl 8) or b

    @Test
    fun `tensor de entrada tiene forma y capacidad 1x224x224x3 float32`() {
        val pixels = IntArray(TensorBuffer.PIXEL_COUNT) { argb(10, 20, 30) }
        val buffer = TensorBuffer.fromArgb(pixels)

        assertEquals("capacidad en bytes inconsistente", TensorBuffer.INPUT_BYTES, buffer.capacity())
        assertEquals("floats esperados", TensorBuffer.INPUT_FLOATS, buffer.asFloatBuffer().capacity())
        assertEquals("el buffer queda rebobinado para el Interpreter", 0, buffer.position())
        assertEquals("orden nativo requerido por LiteRT", ByteOrder.nativeOrder(), buffer.order())
    }

    @Test
    fun `valores crudos se vuelcan sin normalizar con orden RGB`() {
        val pixels = IntArray(TensorBuffer.PIXEL_COUNT)
        pixels[0] = argb(255, 0, 128)  // primer píxel: conocido
        pixels[1] = argb(1, 2, 3)      // segundo píxel: verifica zancada de 3 floats
        val buffer = TensorBuffer.fromArgb(pixels).asFloatBuffer()

        assertEquals(255f, buffer.get(0), 0f)
        assertEquals(0f, buffer.get(1), 0f)
        assertEquals(128f, buffer.get(2), 0f)
        assertEquals(1f, buffer.get(3), 0f)
        assertEquals(2f, buffer.get(4), 0f)
        assertEquals(3f, buffer.get(5), 0f)
    }

    @Test
    fun `todos los valores permanecen en rango 0 a 255`() {
        val pixels = IntArray(TensorBuffer.PIXEL_COUNT) { index ->
            // Ciclo determinista por todo el rango: 0 y 255 incluidos.
            argb(index % 256, (index * 7) % 256, (index * 13) % 256)
        }
        val buffer = TensorBuffer.fromArgb(pixels).asFloatBuffer()
        var min = Float.MAX_VALUE
        var max = -Float.MAX_VALUE
        for (i in 0 until buffer.capacity()) {
            val v = buffer.get(i)
            if (v < min) min = v
            if (v > max) max = v
        }
        assertTrue("min fuera de rango: $min", min >= 0f)
        assertTrue("max fuera de rango: $max", max <= 255f)
        assertEquals(0f, min, 0f)     // el ciclo toca 0
        assertEquals(255f, max, 0f)   // y 255
    }

    @Test
    fun `conteo de píxeles incorrecto lanza excepción`() {
        assertThrows(IllegalArgumentException::class.java) {
            TensorBuffer.fromArgb(IntArray(100))
        }
    }
}
