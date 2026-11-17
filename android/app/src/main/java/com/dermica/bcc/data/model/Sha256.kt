package com.dermica.bcc.data.model

import java.io.File
import java.security.MessageDigest

/**
 * SHA-256 en hexadecimal minúsculas.
 *
 * Lugar en el flujo: datos (data/model), puro JVM (sin dependencias Android)
 * para que las pruebas unitarias lo cubran directo. Lo consumen ModelManager
 * (verifica la copia interna del modelo en el arranque y al instalar uno
 * descargado), ModelRelease (verifica el temporal descargado antes de instalar)
 * y lo cubre Sha256Test en JVM.
 *
 * Salida visible: 64 caracteres hexadecimales en minúsculas, comparables
 * directamente con ModelManager.EXPECTED_SHA256.
 */
object Sha256 {

    /**
     * SHA-256 de un archivo leyéndolo en streaming (1 MiB por bloque).
     *
     * @param file — File: archivo existente a digerir (típicamente model.tflite).
     * @return String: digest en hexadecimal minúsculas (64 caracteres).
     * @throws java.io.FileNotFoundException si [file] no existe (el caller decide
     *         cómo mapearlo; ModelManager/ModelRelease lo absorben con runCatching).
     */
    fun ofFile(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            // 1 MiB por bloque: equilibra lecturas y memoria; el modelo ronda 2 MB.
            val buffer = ByteArray(1 shl 20)
            while (true) {
                val read = input.read(buffer)
                if (read == -1) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().toHex()
    }

    /**
     * SHA-256 de bytes crudos ya en memoria.
     *
     * @param bytes — ByteArray: contenido completo a digerir.
     * @return String: digest en hexadecimal minúsculas (64 caracteres).
     */
    fun ofBytes(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).toHex()

    /** Digest → hexadecimal minúsculas sin separadores (formato"%02x" por byte). */
    private fun ByteArray.toHex(): String = joinToString("") { "%02x".format(it) }
}
