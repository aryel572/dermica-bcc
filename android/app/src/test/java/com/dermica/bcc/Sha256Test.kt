package com.dermica.bcc

import com.dermica.bcc.data.model.Sha256
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Test
import java.io.File

/** Verificación SHA-256 usada en la copia y restauración del modelo (S1). */
class Sha256Test {

    @Test
    fun `vector conocido sha256 de abc`() {
        // Vector de referencia público (FIPS 180-4 / RFC 6234).
        assertEquals(
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            Sha256.ofBytes("abc".toByteArray(Charsets.UTF_8)),
        )
    }

    @Test
    fun `archivo y bytes producen el mismo digest`() {
        val contenido = ByteArray(4096) { (it * 31 % 251).toByte() }
        val tmp = File.createTempFile("sha256-test", ".bin")
        try {
            tmp.writeBytes(contenido)
            assertEquals(Sha256.ofBytes(contenido), Sha256.ofFile(tmp))
        } finally {
            tmp.delete()
        }
    }

    @Test
    fun `contenido distinto produce digest distinto`() {
        assertNotEquals(Sha256.ofBytes(byteArrayOf(1, 2, 3)), Sha256.ofBytes(byteArrayOf(1, 2, 4)))
    }
}
