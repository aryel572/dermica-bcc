package com.dermica.bcc.data.model

import java.io.File
import java.security.MessageDigest

/**
 * SHA-256 en hexadecimal minúsculas.
 *
 * Puro JVM (sin dependencias Android) para que las pruebas unitarias lo cubran
 * y ModelManager lo use para verificar la copia interna del modelo (S1).
 */
object Sha256 {

    /** SHA-256 de un archivo leyéndolo en streaming (1 MiB por bloque). */
    fun ofFile(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(1 shl 20)
            while (true) {
                val read = input.read(buffer)
                if (read == -1) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().toHex()
    }

    /** SHA-256 de bytes crudos ya en memoria. */
    fun ofBytes(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).toHex()

    private fun ByteArray.toHex(): String = joinToString("") { "%02x".format(it) }
}
