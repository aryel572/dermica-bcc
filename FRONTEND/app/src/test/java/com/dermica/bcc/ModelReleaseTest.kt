package com.dermica.bcc

import com.dermica.bcc.data.model.Sha256
import com.dermica.bcc.data.net.GatewayResponse
import com.dermica.bcc.data.net.HttpGateway
import com.dermica.bcc.data.net.LOCAL_MODEL_VERSION
import com.dermica.bcc.data.net.ModelRelease
import com.dermica.bcc.data.net.ReleaseErrorKind
import com.dermica.bcc.data.net.ReleaseException
import com.dermica.bcc.data.net.UpdateDecision
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.io.File
import java.io.IOException

/**
 * Pruebas JVM del flujo de release (S8): parseo, decisión, verificación y
 * autorización. Sin red real: [FakeGateway] devuelve respuestas guionizadas.
 */
class ModelReleaseTest {

    /** Gateway guionizado: registra las peticiones y vuelca bytes fijos en descargas. */
    private class FakeGateway(
        private val responses: Map<String, GatewayResponse> = emptyMap(),
        private val downloadBytes: ByteArray? = null,
        private val downloadCode: Int = 200,
        private val downloadException: IOException? = null,
    ) : HttpGateway {
        val requests = mutableListOf<Pair<String, String?>>()

        override fun get(url: String, bearer: String?): GatewayResponse {
            requests += url to bearer
            return responses[url] ?: GatewayResponse(404, "")
        }

        override fun downloadToFile(
            url: String,
            bearer: String?,
            destFile: File,
            onProgress: (bytesRead: Long, totalBytes: Long?) -> Unit,
        ): Int {
            requests += url to bearer
            if (downloadException != null) throw downloadException
            if (downloadCode != 200) {
                destFile.delete()
                return downloadCode
            }
            val bytes = downloadBytes ?: ByteArray(0)
            destFile.writeBytes(bytes)
            onProgress(bytes.size.toLong(), bytes.size.toLong())
            return 200
        }
    }

    private fun tempFile(): File = File.createTempFile("model-download-test", ".tflite")

    // ── (a) parseo de /model/latest ─────────────────────────────────────────

    @Test
    fun `latest parseado OK extrae version auprc y campos float16`() {
        // Forma documentada por S8 (versión futura del metrics JSON).
        val json = """{"version":"1.1.0","quantization":"float16","auprc":0.78,
            "float16_size_bytes":1945356,"float16_sha256":"$EXPECTED_SHA"}"""
        val info = ModelRelease.parseReleaseInfo(json)
        assertEquals("1.1.0", info.version)
        assertEquals("float16", info.quantization)
        assertEquals(0.78, info.auprc!!, 1e-9)
        assertEquals(1945356L, info.float16SizeBytes)
        assertEquals(EXPECTED_SHA, info.float16Sha256)
    }

    @Test
    fun `latest con tag legado (backend actual) tambien parsea`() {
        // Forma real publicada hoy en dermica-metrics: {tag, seed, auprc, ...} sin float16_*.
        val json = """{"tag":"1.0.0","seed":42,"auprc":0.77880643334541,"recall":0.7767}"""
        val info = ModelRelease.parseReleaseInfo(json)
        assertEquals("1.0.0", info.version)
        assertEquals(0.77880643334541, info.auprc!!, 1e-9)
        assertNull(info.float16SizeBytes)
        assertNull(info.float16Sha256)
    }

    // ── (b) JSON corrupto ───────────────────────────────────────────────────

    @Test
    fun `json corrupto produce error BAD_RESPONSE honesto`() {
        try {
            ModelRelease.parseReleaseInfo("{esto no es json")
            fail("debe lanzar ReleaseException")
        } catch (error: ReleaseException) {
            assertEquals(ReleaseErrorKind.BAD_RESPONSE, error.kind)
        }
    }

    @Test
    fun `json valido sin version ni tag produce error BAD_RESPONSE`() {
        try {
            ModelRelease.parseReleaseInfo("""{"auprc":0.7}""")
            fail("debe lanzar ReleaseException")
        } catch (error: ReleaseException) {
            assertEquals(ReleaseErrorKind.BAD_RESPONSE, error.kind)
        }
    }

    // ── (c) decisión de descarga (matriz) ───────────────────────────────────

    @Test
    fun `matriz de decision de descarga`() {
        // Sin modelo instalado: siempre se descarga, aunque la versión sea igual.
        assertEquals(UpdateDecision.DOWNLOAD, ModelRelease.decide("1.0.0", LOCAL_MODEL_VERSION, modelPresent = false))
        // Versión remota mayor: actualizar.
        assertEquals(UpdateDecision.DOWNLOAD, ModelRelease.decide("1.1.0", LOCAL_MODEL_VERSION, modelPresent = true))
        assertEquals(UpdateDecision.DOWNLOAD, ModelRelease.decide("2.0.0", LOCAL_MODEL_VERSION, modelPresent = true))
        // Versión remota igual o menor: no se toca nada.
        assertEquals(UpdateDecision.UP_TO_DATE, ModelRelease.decide("1.0.0", LOCAL_MODEL_VERSION, modelPresent = true))
        assertEquals(UpdateDecision.UP_TO_DATE, ModelRelease.decide("0.9.0", LOCAL_MODEL_VERSION, modelPresent = true))
        // Semver real: 1.10.0 > 1.9.0 (comparación numérica, no de strings).
        assertEquals(UpdateDecision.DOWNLOAD, ModelRelease.decide("1.10.0", "1.9.0", modelPresent = true))
    }

    @Test
    fun `version remota malformada produce BAD_RESPONSE`() {
        try {
            ModelRelease.decide("no-es-semver", LOCAL_MODEL_VERSION, modelPresent = true)
            fail("debe lanzar ReleaseException")
        } catch (error: ReleaseException) {
            assertEquals(ReleaseErrorKind.BAD_RESPONSE, error.kind)
        }
    }

    // ── (d) SHA mismatch aborta sin tocar el modelo actual ──────────────────

    @Test
    fun `sha mismatch borra el temporal y aborta`() {
        val basura = "esto no es el modelo".toByteArray()
        val tmp = tempFile()
        try {
            val gateway = FakeGateway(downloadBytes = basura)
            val release = ModelRelease(gateway)
            val info = ModelRelease.parseReleaseInfo("""{"version":"1.0.0"}""")
            try {
                release.downloadModel("clave-de-prueba", info, EXPECTED_SHA, tmp) { _, _ -> }
                fail("debe lanzar ReleaseException VERIFICATION_FAILED")
            } catch (error: ReleaseException) {
                assertEquals(ReleaseErrorKind.VERIFICATION_FAILED, error.kind)
            }
            // El temporal quedó borrado y nunca se instaló nada.
            assertEquals(false, tmp.exists())
        } finally {
            tmp.delete()
        }
    }

    @Test
    fun `tamano declarado que no coincide aborta y borra el temporal`() {
        val bytes = "contenido cualquiera".toByteArray()
        val tmp = tempFile()
        try {
            val gateway = FakeGateway(downloadBytes = bytes)
            val release = ModelRelease(gateway)
            // El release declara un tamaño distinto al descargado.
            val info = com.dermica.bcc.data.net.ReleaseInfo(
                version = "1.0.0", quantization = "float16", auprc = null,
                float16SizeBytes = bytes.size + 1L, float16Sha256 = null,
            )
            try {
                release.downloadModel("clave-de-prueba", info, EXPECTED_SHA, tmp) { _, _ -> }
                fail("debe lanzar ReleaseException VERIFICATION_FAILED")
            } catch (error: ReleaseException) {
                assertEquals(ReleaseErrorKind.VERIFICATION_FAILED, error.kind)
            }
            assertEquals(false, tmp.exists())
        } finally {
            tmp.delete()
        }
    }

    @Test
    fun `descarga integra espera y devuelve el temporal intacto`() {
        val bytes = "modelo íntegro de prueba".toByteArray()
        val sha = Sha256.ofBytes(bytes)
        val tmp = tempFile()
        try {
            val gateway = FakeGateway(downloadBytes = bytes)
            val release = ModelRelease(gateway)
            val info = com.dermica.bcc.data.net.ReleaseInfo(
                version = "1.0.0", quantization = "float16", auprc = null,
                float16SizeBytes = bytes.size.toLong(), float16Sha256 = sha,
            )
            var progressCalls = 0
            release.downloadModel("clave-de-prueba", info, sha, tmp) { _, _ -> progressCalls++ }
            assertArrayEquals(bytes, tmp.readBytes())
            assertTrue(progressCalls >= 1)
        } finally {
            tmp.delete()
        }
    }

    // ── (e) 401/403 → error de autorización legible (sin el token) ──────────

    @Test
    fun `latest con 401 produce UNAUTHORIZED sin exponer la clave`() {
        val clave = "super-secreta-de-prueba"
        val gateway = FakeGateway(
            responses = mapOf(
                com.dermica.bcc.data.net.RELEASE_BASE_URL + "/model/latest" to GatewayResponse(401, "unauthorized"),
            ),
        )
        val release = ModelRelease(gateway)
        try {
            release.fetchLatest(clave)
            fail("debe lanzar ReleaseException")
        } catch (error: ReleaseException) {
            assertEquals(ReleaseErrorKind.UNAUTHORIZED, error.kind)
            assertTrue(!error.message.contains(clave))
        }
    }

    @Test
    fun `latest con 403 produce UNAUTHORIZED`() {
        val gateway = FakeGateway(
            responses = mapOf(
                com.dermica.bcc.data.net.RELEASE_BASE_URL + "/model/latest" to GatewayResponse(403, "forbidden"),
            ),
        )
        try {
            ModelRelease(gateway).fetchLatest("clave-de-prueba")
            fail("debe lanzar ReleaseException")
        } catch (error: ReleaseException) {
            assertEquals(ReleaseErrorKind.UNAUTHORIZED, error.kind)
        }
    }

    @Test
    fun `download con 401 produce UNAUTHORIZED y borra el temporal`() {
        val tmp = tempFile()
        try {
            val info = com.dermica.bcc.data.net.ReleaseInfo(
                version = "1.0.0", quantization = null, auprc = null,
                float16SizeBytes = null, float16Sha256 = null,
            )
            val release = ModelRelease(FakeGateway(downloadCode = 401))
            try {
                release.downloadModel("clave-de-prueba", info, EXPECTED_SHA, tmp) { _, _ -> }
                fail("debe lanzar ReleaseException")
            } catch (error: ReleaseException) {
                assertEquals(ReleaseErrorKind.UNAUTHORIZED, error.kind)
            }
            assertEquals(false, tmp.exists())
        } finally {
            tmp.delete()
        }
    }

    @Test
    fun `fallo de red en latest produce NO_CONNECTION`() {
        val gateway = object : HttpGateway {
            override fun get(url: String, bearer: String?): GatewayResponse =
                throw IOException("timeout simulado")

            override fun downloadToFile(
                url: String,
                bearer: String?,
                destFile: File,
                onProgress: (bytesRead: Long, totalBytes: Long?) -> Unit,
            ): Int = 200
        }
        try {
            ModelRelease(gateway).fetchLatest("clave-de-prueba")
            fail("debe lanzar ReleaseException")
        } catch (error: ReleaseException) {
            assertEquals(ReleaseErrorKind.NO_CONNECTION, error.kind)
        }
    }

    private companion object {
        // SHA-256 conocido del release float16 1.0.0 (mismo valor que EXPECTED_SHA256).
        const val EXPECTED_SHA = "d62c401c5f66a7eccd5f752b2d7aa73964fe6f75a92acde3cae951b0d911cc0a"
    }
}
