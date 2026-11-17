package com.dermica.bcc

import com.dermica.bcc.core.Totals
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Tests JVM de las métricas del subconjunto (S4).
 * Las fórmulas NO se muestran en la UI; estos tests fijan su comportamiento numérico.
 */
class SubsetMetricsTest {

    @Test
    fun `precision perfecta da TA 100 PEO 0`() {
        val metrics = Totals(tp = 10, tn = 5, fp = 0, fn = 0).compute()
        assertEquals(100.0, metrics.ta, 1e-9)
        assertEquals(0, metrics.nfs)
        assertEquals(0.0, metrics.peo, 1e-9)
    }

    @Test
    fun `subconjunto vacio da TA 0 y PEO 0`() {
        val metrics = Totals(tp = 0, tn = 0, fp = 0, fn = 0).compute()
        assertEquals(0.0, metrics.ta, 1e-9)
        assertEquals(0, metrics.nfs)
        assertEquals(0.0, metrics.peo, 1e-9)
    }

    @Test
    fun `sin positivos reales PEO es 0 aunque haya falsas sospechas`() {
        val metrics = Totals(tp = 0, tn = 12, fp = 3, fn = 0).compute()
        assertEquals((12.0 / 15.0) * 100.0, metrics.ta, 1e-9)
        assertEquals(3, metrics.nfs)
        assertEquals(0.0, metrics.peo, 1e-9) // TP + FN = 0: no se puede omitir lo que no existe
    }

    @Test
    fun `mezcla real de la demo TESIS da cifras coherentes`() {
        // 15 imágenes: 2 errores (1 falsa sospecha + 1 omisión)
        val metrics = Totals(tp = 3, tn = 10, fp = 1, fn = 1).compute()
        assertEquals((13.0 / 15.0) * 100.0, metrics.ta, 1e-9)
        assertEquals(1, metrics.nfs)
        assertEquals((1.0 / 4.0) * 100.0, metrics.peo, 1e-9)
    }

    @Test
    fun `NFS equivale exactamente a FP`() {
        val metrics = Totals(tp = 1, tn = 1, fp = 7, fn = 1).compute()
        assertEquals(7, metrics.nfs)
    }
}
