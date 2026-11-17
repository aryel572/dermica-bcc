package com.dermica.bcc

import com.dermica.bcc.ml.BccLabel
import com.dermica.bcc.ml.OutputDecoder
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Semántica de la salida sigmoid usada en la evaluación (umbral 0.5, clase positiva BCC):
 * paridad con metrics.confusion_at (preds = score >= 0.5) de S4.
 */
class OutputDecoderTest {

    @Test
    fun `sigmoid alto predice BCC con confianza igual a la probabilidad`() {
        val prediction = OutputDecoder.decode(floatArrayOf(0.87f))
        assertEquals(BccLabel.BCC, prediction.label)
        assertEquals(0.87f, prediction.confidence, 1e-4f)
        assertEquals(0.87f, prediction.rawProbability, 1e-4f)
    }

    @Test
    fun `sigmoid bajo predice NO_BCC con confianza complementaria`() {
        val prediction = OutputDecoder.decode(floatArrayOf(0.13f))
        assertEquals(BccLabel.NO_BCC, prediction.label)
        assertEquals(0.87f, prediction.confidence, 1e-4f)
        assertEquals(0.13f, prediction.rawProbability, 1e-4f)
    }

    @Test
    fun `umbral 0_5 es inclusivo hacia BCC igual que en la evaluación`() {
        // metrics.confusion_at: preds = (y_score >= threshold) -> el empate 0.5 clasifica BCC.
        val prediction = OutputDecoder.decode(floatArrayOf(OutputDecoder.THRESHOLD))
        assertEquals(BccLabel.BCC, prediction.label)
    }

    @Test
    fun `justo por debajo del umbral predice NO_BCC`() {
        val prediction = OutputDecoder.decode(floatArrayOf(0.4999f))
        assertEquals(BccLabel.NO_BCC, prediction.label)
    }

    @Test
    fun `probabilidad fuera de rango se acota defensivamente`() {
        assertEquals(BccLabel.BCC, OutputDecoder.decode(floatArrayOf(1.7f)).label)
        assertEquals(BccLabel.NO_BCC, OutputDecoder.decode(floatArrayOf(-0.2f)).label)
    }

    @Test
    fun `salida con número inesperado de unidades lanza excepción`() {
        assertThrows(IllegalArgumentException::class.java) {
            OutputDecoder.decode(floatArrayOf(0.1f, 0.9f))
        }
    }

    @Test
    fun `confianza siempre queda en 0_5 a 1_0`() {
        for (i in 0..100) {
            val p = i / 100f
            val prediction = OutputDecoder.decode(floatArrayOf(p))
            assertTrue("confianza fuera de [0.5, 1]: ${prediction.confidence}", prediction.confidence in 0.5f..1.0f)
        }
    }
}
