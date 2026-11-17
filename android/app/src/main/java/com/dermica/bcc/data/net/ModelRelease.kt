package com.dermica.bcc.data.net

import com.dermica.bcc.data.model.Sha256
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/**
 * Endpoint base del backend FastAPI (Render free, S8). HTTPS siempre (S10).
 *
 * Origen: URL pública del servicio Render del proyecto; la sirve SettingsViewModel
 * a ModelRelease. Nunca lleva credenciales: el Bearer viaja por cabecera por petición.
 */
const val RELEASE_BASE_URL = "https://dermica-api.onrender.com"

/**
 * Versión del modelo publicado en assets y en el último release (1.0.0).
 *
 * Origen: contrato de release 1.0.0 (float16); la usa SettingsViewModel como
 * `localVersion` en ModelRelease.decide para decidir si hay algo que descargar.
 */
const val LOCAL_MODEL_VERSION = "1.0.0"

/**
 * Resultado mínimo de una petición GET: código HTTP y cuerpo como texto.
 *
 * @param code — Int: código HTTP devuelto por el servidor (p.ej. 200, 401).
 * @param body — String: cuerpo en UTF-8, posiblemente vacío; nunca null.
 *
 * Lo produce HttpGateway.get (UrlHttpGateway en producción, fake en tests).
 */
data class GatewayResponse(val code: Int, val body: String)

/**
 * Puerta HTTP inyectable para que el flujo de release sea testeable en JVM sin red.
 *
 * Lugar en el flujo: red (data/net). La consume ModelRelease (única cliente del
 * contrato) y la implementan UrlHttpGateway (producción) y los fakes de
 * ModelReleaseTest (JVM).
 */
interface HttpGateway {

    /**
     * GET sencillo; devuelve código y cuerpo. Lanza IOException ante fallo de red/timeout.
     *
     * @param url — String: URL completa (sin credenciales embebidas).
     * @param bearer — String?: token Bearer opcional (null en /model/version público).
     * @return GatewayResponse con código y cuerpo; el llamador decide qué códigos acepta.
     * @throws java.io.IOException ante fallo de red/timeout/DNS (ModelRelease lo traduce a ReleaseException).
     */
    fun get(url: String, bearer: String?): GatewayResponse

    /**
     * GET con volcado a archivo (stream). [onProgress] recibe bytes leídos y el
     * total declarado por el servidor (null si no lo envía). Devuelve el código HTTP;
     * con código != 200 no debe quedar archivo útil (el temporal se borra).
     *
     * @param url — String: URL del artefacto (típicamente /model/download).
     * @param bearer — String?: token Bearer obligatorio (el endpoint exige auth).
     * @param destFile — File: destino temporal (mismo filesystem que el modelo interno).
     * @param onProgress — (bytesRead: Long, totalBytes: Long?) -> Unit: progreso
     *        para la UI; totalBytes null cuando el servidor responde chunked.
     * @return Int: código HTTP de la respuesta (200 en éxito).
     * @throws java.io.IOException ante fallo de red a mitad del stream (el temporal queda parcial; el llamador lo borra).
     */
    fun downloadToFile(
        url: String,
        bearer: String?,
        destFile: File,
        onProgress: (bytesRead: Long, totalBytes: Long?) -> Unit,
    ): Int
}

/** Ejecuta [block] traduciendo IOException a ReleaseException NO_CONNECTION (S8: error claro). */
private inline fun <T> ioUnlessNoConnection(block: () -> T): T = try {
    block()
} catch (error: IOException) {
    // Sin detalles internos al usuario: mensaje corto y accionable en español.
    throw ReleaseException(ReleaseErrorKind.NO_CONNECTION, "Sin conexión con el servidor. Revisa tu red e inténtalo de nuevo.")
}

/**
 * Cliente real: HttpURLConnection pelado (sin OkHttp/Retrofit, S8 mínimo).
 *
 * Lugar en el flujo: red (data/net), implementación de [HttpGateway] para
 * producción. ModelRelease solo conoce la interfaz; las pruebas inyectan un fake.
 */
class UrlHttpGateway : HttpGateway {

    // Timeout de conexión 10 s: margen amplio para DNS+TCP+TLS sin colgar la UI.
    private val connectTimeoutMs = 10_000

    // Timeout de lectura 60 s: Render free tarda 30–60 s en el arranque en frío
    // y acortar la lectura cortaría descargas/respuestas válidas.
    private val readTimeoutMs = 60_000

    // El cuerpo de /model/latest es un JSON pequeño; se acota la lectura por defensa.
    private val maxBodyBytes = 1 shl 20

    override fun get(url: String, bearer: String?): GatewayResponse {
        val connection = open(url, bearer)
        try {
            val code = connection.responseCode
            val body = runCatching {
                val stream = if (code in 200..299) connection.inputStream else connection.errorStream
                stream?.use { it.readNBytes(maxBodyBytes).toString(Charsets.UTF_8) }.orEmpty()
            }.getOrDefault("")
            return GatewayResponse(code, body)
        } finally {
            connection.disconnect()
        }
    }

    override fun downloadToFile(
        url: String,
        bearer: String?,
        destFile: File,
        onProgress: (bytesRead: Long, totalBytes: Long?) -> Unit,
    ): Int {
        val connection = open(url, bearer)
        try {
            val code = connection.responseCode
            if (code != HttpURLConnection.HTTP_OK) {
                destFile.delete()
                return code
            }
            // Content-Length puede venir en -1 (stream chunked); entonces el progreso
            // es indeterminado y solo se reportan bytes leídos.
            val totalBytes = connection.contentLengthLong.takeIf { it >= 0 }
            connection.inputStream.use { input ->
                destFile.outputStream().use { output ->
                    val buffer = ByteArray(1 shl 16)
                    var bytesRead = 0L
                    while (true) {
                        val read = input.read(buffer)
                        if (read == -1) break
                        output.write(buffer, 0, read)
                        bytesRead += read
                        onProgress(bytesRead, totalBytes)
                    }
                }
            }
            return code
        } finally {
            connection.disconnect()
        }
    }

    private fun open(url: String, bearer: String?): HttpURLConnection {
        val connection = URL(url).openConnection() as HttpURLConnection
        connection.connectTimeout = connectTimeoutMs
        connection.readTimeout = readTimeoutMs
        connection.requestMethod = "GET"
        if (bearer != null) connection.setRequestProperty("Authorization", "Bearer $bearer")
        return connection
    }
}

/**
 * Metadatos del release publicados en /model/latest (campos del metrics JSON).
 *
 * Lugar en el flujo: red (data/net). Lo produce ModelRelease.parseReleaseInfo y
 * lo consume SettingsViewModel (decide descarga y verificación) y
 * ModelRelease.downloadModel (tamaño/SHA esperados). Los campos float16* son
 * opcionales: el backend real publica `tag` + métricas sin ellos.
 *
 * @param version — String: versión del release (o `tag` del backend), normalizada en el parseo.
 * @param quantization — String?: p.ej. "float16"; null si el backend no la publica.
 * @param auprc — Double?: AUPRC reportada en validación; null si no está publicada.
 * @param float16SizeBytes — Long?: tamaño exacto del artefacto float16; null si no se publica.
 * @param float16Sha256 — String?: SHA-256 del artefacto float16; null si no se publica.
 */
data class ReleaseInfo(
    val version: String,
    val quantization: String?,
    val auprc: Double?,
    val float16SizeBytes: Long?,
    val float16Sha256: String?,
)

/**
 * Tipos de fallo del flujo de release; la UI los mapea a mensajes legibles.
 *
 * Lugar en el flujo: red (data/net). Los produce ModelRelease y los consume
 * SettingsViewModel/SettingsScreen para pintar el error correspondiente.
 */
enum class ReleaseErrorKind {
    /** Red/timeout/DNS: no se pudo hablar con el servidor. */
    NO_CONNECTION,

    /** 401/403: el servidor rechazó la APP_API_KEY (nunca se muestra ni se loguea). */
    UNAUTHORIZED,

    /** Respuesta no parseable o sin los campos mínimos (versión). */
    BAD_RESPONSE,

    /** El servidor respondió con otro código no soportado. */
    SERVER_ERROR,

    /** El archivo descargado no superó SHA-256 o tamaño; se descartó intacto el modelo actual. */
    VERIFICATION_FAILED,
}

/**
 * Error del flujo de release con un mensaje listo para mostrar (español, sin
 * secretos ni trazas internas) y el código HTTP cuando aplica.
 *
 * @param kind — ReleaseErrorKind: categoría del fallo (la UI mapea a texto/acción).
 * @param message — String: mensaje accionable en español; nunca contiene el Bearer.
 * @param httpCode — Int?: código HTTP si el fallo vino de una respuesta HTTP.
 */
class ReleaseException(
    val kind: ReleaseErrorKind,
    override val message: String,
    val httpCode: Int? = null,
) : Exception(message)

/**
 * Decisión del paso 3 (S8): descargar o no.
 *
 * @property DOWNLOAD hay un release más nuevo (o no hay modelo instalado).
 * @property UP_TO_DATE el modelo local ya está al día; no se toca nada.
 */
enum class UpdateDecision {
    DOWNLOAD,
    UP_TO_DATE,
}

/**
 * Flujo de descarga/actualización del modelo contra el backend (S8).
 *
 * Lugar en el flujo: red (data/net). Lo consume ui/settings/SettingsViewModel
 * (checkForUpdateOrDownload) y lo cubre ModelReleaseTest en JVM con un
 * HttpGateway guionizado. Extraído de la UI para ser testeable en JVM: toda la
 * lógica (parseo, decisión, verificación) vive aquí y [HttpGateway] se inyecta.
 *
 * La instalación atómica (tmp → filesDir/model.tflite) y el estado del
 * ModelManager son responsabilidad de `ModelManager.installDownloadedModel`.
 *
 * @param gateway — HttpGateway: puerta HTTP; por defecto UrlHttpGateway (producción);
 *        las pruebas inyectan un fake con respuestas guionizadas.
 */
class ModelRelease(private val gateway: HttpGateway = UrlHttpGateway()) {

    /**
     * Paso 1 (S8): ping público a /model/version.
     *
     * @throws ReleaseException NO_CONNECTION si la red falla/timeout; SERVER_ERROR
     *         si el servidor no responde 200.
     *
     * Efectos: ninguno en la app; solo constata que el servicio está vivo.
     */
    fun checkServer() {
        val response = ioUnlessNoConnection { gateway.get("$RELEASE_BASE_URL/model/version", bearer = null) }
        when (response.code) {
            200 -> Unit
            else -> throw ReleaseException(
                ReleaseErrorKind.SERVER_ERROR,
                "El servidor respondió con un error (código ${response.code}). Inténtalo más tarde.",
                response.code,
            )
        }
    }

    /**
     * Paso 2 (S8): GET /model/latest con Bearer. Devuelve los metadatos del release.
     * El cuerpo real del backend trae `tag` + métricas (sin campos float16); el
     * parser acepta `version` o `tag` y los campos float16 como opcionales.
     *
     * @param apiKey — String: APP_API_KEY configurada; viaja solo como cabecera
     *        Authorization Bearer, nunca se registra ni se incluye en mensajes.
     * @return ReleaseInfo con los metadatos del release ya parseados.
     * @throws ReleaseException NO_CONNECTION (red/timeout), UNAUTHORIZED (401/403),
     *         SERVER_ERROR (otro código) o BAD_RESPONSE (cuerpo no parseable/sin versión).
     */
    fun fetchLatest(apiKey: String): ReleaseInfo {
        val response = ioUnlessNoConnection { gateway.get("$RELEASE_BASE_URL/model/latest", apiKey) }
        when (response.code) {
            200 -> return parseReleaseInfo(response.body)
            401, 403 -> throw ReleaseException(
                ReleaseErrorKind.UNAUTHORIZED,
                "El servidor rechazó la autorización. Revisa la APP_API_KEY configurada.",
                response.code,
            )
            else -> throw ReleaseException(
                ReleaseErrorKind.SERVER_ERROR,
                "El servidor respondió con un error (código ${response.code}). Inténtalo más tarde.",
                response.code,
            )
        }
    }

    /**
     * Paso 4 (S8): descarga a [destFile] (temporal), con progreso y verificación
     * de tamaño y SHA-256. Si la verificación falla borra el temporal y lanza
     * [ReleaseException] tipo VERIFICATION_FAILED: el modelo actual no se toca.
     *
     * @param apiKey — String: APP_API_KEY configurada; viaja solo como cabecera.
     * @param info — ReleaseInfo: metadatos de fetchLatest; se usan tamaño y SHA esperados.
     * @param expectedSha256 — String: SHA publicado en /model/latest; si el release no lo
     *   trae, el llamador pasa la constante EXPECTED_SHA256 del asset base.
     * @param destFile — File: destino temporal (normalmente ModelManager.downloadTemporaryFile()).
     * @param onProgress — (bytesRead: Long, totalBytes: Long?) -> Unit: progreso para la UI.
     * @throws ReleaseException NO_CONNECTION, UNAUTHORIZED, SERVER_ERROR o
     *         VERIFICATION_FAILED. Si la verificación falla borra el temporal: el
     *         modelo actual no se toca.
     *
     * Efectos colaterales: escribe destFile y lo borra en cualquier fallo; nunca
     * instala ni toca el modelo interno (eso es trabajo de ModelManager).
     */
    fun downloadModel(
        apiKey: String,
        info: ReleaseInfo,
        expectedSha256: String,
        destFile: File,
        onProgress: (bytesRead: Long, totalBytes: Long?) -> Unit,
    ) {
        // Fallo de red en mitad del stream: se borra el temporal parcial (nunca se instala nada a medias).
        val code = try {
            gateway.downloadToFile("$RELEASE_BASE_URL/model/download", apiKey, destFile, onProgress)
        } catch (error: IOException) {
            destFile.delete()
            throw ReleaseException(
                ReleaseErrorKind.NO_CONNECTION,
                "Sin conexión con el servidor. Revisa tu red e inténtalo de nuevo.",
            )
        }
        if (code != 200) {
            destFile.delete()
            throw when (code) {
                401, 403 -> ReleaseException(
                    ReleaseErrorKind.UNAUTHORIZED,
                    "El servidor rechazó la autorización. Revisa la APP_API_KEY configurada.",
                    code,
                )
                else -> ReleaseException(
                    ReleaseErrorKind.SERVER_ERROR,
                    "El servidor respondió con un error (código $code). Inténtalo más tarde.",
                    code,
                )
            }
        }
        // Verificación de tamaño exacto cuando el release lo publica.
        if (info.float16SizeBytes != null && destFile.length() != info.float16SizeBytes) {
            destFile.delete()
            throw ReleaseException(
                ReleaseErrorKind.VERIFICATION_FAILED,
                "La descarga no superó la verificación de integridad; no se instaló nada.",
            )
        }
        // SHA-256 reutilizando la implementación canónica (data/model/Sha256.kt).
        val actualSha = runCatching { Sha256.ofFile(destFile) }
            .getOrElse {
                destFile.delete()
                throw ReleaseException(
                    ReleaseErrorKind.VERIFICATION_FAILED,
                    "La descarga no se pudo verificar; no se instaló nada.",
                )
            }
        if (actualSha != expectedSha256) {
            destFile.delete()
            throw ReleaseException(
                ReleaseErrorKind.VERIFICATION_FAILED,
                "La descarga no superó la verificación de integridad; no se instaló nada.",
            )
        }
    }

    companion object {

        /**
         * Paso 2 (parseo): extrae los metadatos del JSON de /model/latest.
         *
         * @param json — String: cuerpo íntegro de la respuesta 200 de /model/latest.
         * @return ReleaseInfo: metadatos normalizados (versión nunca vacía).
         * @throws ReleaseException BAD_RESPONSE si el JSON no es un objeto, no trae
         *         versión/tag usable, o la versión no es parseable.
         */
        fun parseReleaseInfo(json: String): ReleaseInfo {
            val obj = try {
                JSONObject(json)
            } catch (error: org.json.JSONException) {
                throw ReleaseException(ReleaseErrorKind.BAD_RESPONSE, "La respuesta del servidor no es válida.")
            }
            // El backend real publica `tag`; los releases futuros pueden traer `version`.
            val version = obj.optString("version").ifBlank { obj.optString("tag") }.ifBlank {
                throw ReleaseException(ReleaseErrorKind.BAD_RESPONSE, "La respuesta del servidor no incluye la versión.")
            }
            val quantization = obj.optString("quantization").takeIf { it.isNotBlank() }
            val auprc = if (obj.has("auprc") && !obj.isNull("auprc")) {
                obj.optDouble("auprc").takeIf { !it.isNaN() }
            } else {
                null
            }
            val sizeBytes = if (obj.has("float16_size_bytes") && !obj.isNull("float16_size_bytes")) {
                obj.optLong("float16_size_bytes").takeIf { it > 0 }
            } else {
                null
            }
            val sha256 = obj.optString("float16_sha256").takeIf { it.isNotBlank() }
            return ReleaseInfo(version, quantization, auprc, sizeBytes, sha256)
        }

        /**
         * Paso 3 (S8): decide si descargar. Descarga si no hay modelo instalado
         * o si la versión remota es estrictamente mayor que la local; en cualquier
         * otro caso no toca nada ("Ya tienes la última versión").
         *
         * @param remoteVersion — String: versión publicada por /model/latest.
         * @param localVersion — String: LOCAL_MODEL_VERSION o la instalada.
         * @param modelPresent — Boolean: true si hay un modelo interno disponible.
         * @return UpdateDecision: DOWNLOAD o UP_TO_DATE.
         * @throws ReleaseException BAD_RESPONSE si la versión remota no es X.Y.Z
         *   con partes numéricas (no se improvisa una comparación de strings).
         */
        fun decide(remoteVersion: String, localVersion: String, modelPresent: Boolean): UpdateDecision {
            if (!modelPresent) return UpdateDecision.DOWNLOAD
            return if (compareVersions(remoteVersion, localVersion) > 0) {
                UpdateDecision.DOWNLOAD
            } else {
                UpdateDecision.UP_TO_DATE
            }
        }

        /**
         * Comparación semver estricta sobre partes numéricas (1.10.0 > 1.9.0).
         *
         * @param a — String: versión candidata, partes numéricas separadas por '.'.
         * @param b — String: versión contra la que comparar.
         * @return Int: >0 si a>b, 0 si son equivalentes, <0 si a<b.
         * @throws ReleaseException BAD_RESPONSE si alguna parte no es numérica.
         */
        fun compareVersions(a: String, b: String): Int {
            val partsA = a.split('.').map { it.toLongOrNull() }
            val partsB = b.split('.').map { it.toLongOrNull() }
            if (partsA.any { it == null } || partsB.any { it == null }) {
                throw ReleaseException(
                    ReleaseErrorKind.BAD_RESPONSE,
                    "La versión publicada por el servidor no es válida.",
                )
            }
            val len = maxOf(partsA.size, partsB.size)
            for (i in 0 until len) {
                val pa = partsA.getOrNull(i) ?: 0L
                val pb = partsB.getOrNull(i) ?: 0L
                if (pa != pb) return pa.compareTo(pb)
            }
            return 0
        }
    }
}
