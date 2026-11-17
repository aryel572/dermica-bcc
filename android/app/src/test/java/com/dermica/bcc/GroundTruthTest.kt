package com.dermica.bcc

import com.dermica.bcc.core.GroundTruth
import com.dermica.bcc.core.GroundTruthFormatException
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/** Tests JVM del matcher de ground truth (S3): parseo, tolerancia y exclusión. */
class GroundTruthTest {

    @Test
    fun `json plano valido parsea etiquetas y cero descartadas`() {
        val groundTruth = GroundTruth.parse(
            """{"ISIC_0026350.jpg":"BCC","ISIC_0010445.jpg":"NV"}""",
        )
        assertEquals(2, groundTruth.labels.size)
        assertEquals(0, groundTruth.skippedEntries)
        assertEquals("BCC", groundTruth.labelFor("ISIC_0026350.jpg"))
        assertEquals("NV", groundTruth.labelFor("ISIC_0010445.jpg"))
    }

    @Test
    fun `load devuelve el mismo mapa que parse`() {
        val map = GroundTruth.load("""{"a.jpg":"BCC","b.jpg":"MEL"}""")
        assertEquals(mapOf("a.jpg" to "BCC", "b.jpg" to "MEL"), map)
    }

    @Test
    fun `entradas malformadas se omiten y se cuentan`() {
        // Número, booleano, null, objeto anidado y etiqueta en blanco: todas fuera;
        // las dos válidas sobreviven.
        val groundTruth = GroundTruth.parse(
            """{"ok1.jpg":"BCC","num.jpg":7,"bool.jpg":true,"null.jpg":null,
                "obj.jpg":{"x":1},"blank.jpg":"   ","ok2.jpg":"NV"}""",
        )
        assertEquals(mapOf("ok1.jpg" to "BCC", "ok2.jpg" to "NV"), groundTruth.labels)
        assertEquals(5, groundTruth.skippedEntries)
    }

    @Test
    fun `json corrupto lanza GroundTruthFormatException`() {
        for (broken in listOf("no-json", "", "{sin cerrar", "[1,2,3]", "42")) {
            assertThrows(GroundTruthFormatException::class.java) {
                GroundTruth.parse(broken)
            }
        }
    }

    @Test
    fun `objeto sin ninguna etiqueta usable lanza GroundTruthFormatException`() {
        assertThrows(GroundTruthFormatException::class.java) {
            GroundTruth.parse("{}")
        }
        assertThrows(GroundTruthFormatException::class.java) {
            GroundTruth.parse("""{"a.jpg":1,"b.jpg":null}""")
        }
    }

    @Test
    fun `labelFor hace match por basename exacto`() {
        val groundTruth = GroundTruth.parse("""{"ISIC_1.jpg":"BCC"}""")
        assertEquals("BCC", groundTruth.labelFor("ISIC_1.jpg"))
        assertEquals("BCC", groundTruth.labelFor("/tree/document/ISIC_1.jpg"))
        // Sin entrada → null (el caller excluye la imagen; nunca se inventa etiqueta).
        assertNull(groundTruth.labelFor("otra/ISIC_2.jpg"))
    }

    @Test
    fun `isBcc es exacto al literal BCC`() {
        val groundTruth = GroundTruth.parse("""{"x.jpg":"BCC"}""")
        assertTrue(groundTruth.isBcc("BCC"))
        assertFalse(groundTruth.isBcc("bcc"))
        assertFalse(groundTruth.isBcc("Bcc"))
        assertFalse(groundTruth.isBcc("BCCI"))
        assertFalse(groundTruth.isBcc(""))
        assertFalse(groundTruth.isBcc("NV"))
    }

    @Test
    fun `las etiquetas se recortan al parsear pero la comparacion sigue siendo exacta`() {
        val groundTruth = GroundTruth.parse("""{"x.jpg":"  BCC  "}""")
        assertEquals("BCC", groundTruth.labelFor("x.jpg"))
        assertTrue(groundTruth.isBcc(groundTruth.labelFor("x.jpg") ?: ""))
    }
}
