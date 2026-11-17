package com.dermica.bcc.core

import org.json.JSONException
import org.json.JSONObject

/**
 * Fallo irrecuperable de la referencia de un subconjunto (S3): el texto no es un
 * objeto JSON válido, o no contiene ninguna etiqueta usable.
 *
 * Lugar en el flujo: la lanzan [GroundTruth.parse] y [GroundTruth.load] (núcleo
 * puro, sin dependencias Android); la consume ui/subset/SubsetViewModel, que la
 * captura y la traduce a un estado de error con reintento.
 *
 * Nunca se propaga como crash: la UI la captura y muestra una pantalla de error
 * con explicación y botón para reintentar.
 */
class GroundTruthFormatException(message: String) : IllegalArgumentException(message)

/**
 * Matcher puro del ground truth de un subconjunto (S3): JSON plano
 * `{filename: etiqueta}` versionado junto a la carpeta de imágenes del TESIS.
 *
 * Lugar en el flujo: núcleo puro (sin Android, sin Room, sin red). Lo consume
 * ui/subset/SubsetViewModel para etiquetar cada imagen del lote y excluye del
 * cálculo las que devuelven null; la cubre GroundTruthTest en JVM. Su salida
 * visible en la UI es el rótulo de la exclusión («N imágenes sin etiqueta») y
 * el conteo que se persiste en history_events.excludedCount.
 *
 * Reglas del contrato (S3):
 * - El match es por basename exacto: `labelFor("subset_01/ISIC_1.jpg")` equivale a
 *   `labelFor("ISIC_1.jpg")`. Sin normalización de mayúsculas ni de separadores:
 *   los nombres del TESIS son únicos por carpeta y estables entre corridas.
 * - BCC = positivo; cualquier otra etiqueta es negativo. `isBcc("bcc")` es falso:
 *   la referencia ISIC escribe "BCC" en mayúsculas y se respeta tal cual.
 * - Las etiquetas se recortan (`trim`) al parsear (espacios accidentales de un
 *   JSON editado a mano); la comparación posterior es exacta.
 * - Imagen sin entrada → `labelFor` devuelve null → el caller la EXCLUYE del
 *   cálculo y la cuenta aparte. Jamás se inventa una etiqueta.
 * - JSON estructuralmente corrupto (no es un objeto / sintaxis inválida) →
 *   [GroundTruthFormatException]. Las entradas individuales malformadas se omiten
 *   y se cuentan en [skippedEntries]; el resto del archivo parsea normalmente.
 */
class GroundTruth private constructor(
    /** Entradas válidas: nombre de archivo (tal cual, como llave del JSON) → etiqueta. */
    val labels: Map<String, String>,
    /** Entradas descartadas por malformadas (0 en un archivo sano). */
    val skippedEntries: Int,
) {

    /**
     * Etiqueta de referencia para [file] (basename exacto tras el último '/').
     *
     * @param file — String: nombre de archivo o «ruta» arbitraria; solo se usa el
     *        segmento tras el último '/', recortado con [String.trim].
     * @return String? con la etiqueta tal cual del JSON (sin normalizar), o null
     *         si el nombre no tiene entrada (el caller lo excluye y lo cuenta aparte).
     *
     * Efectos: ninguno; es una consulta pura sobre el mapa cargado.
     */
    fun labelFor(file: String): String? = labels[basename(file)]

    /**
     * Positivo de referencia (S3): solo la etiqueta exacta "BCC".
     *
     * @param label — String: etiqueta devuelta por [labelFor] (ya recortada en parse).
     * @return Boolean: true únicamente para "BCC" en mayúsculas; cualquier otra
     *         escritura ("bcc", "Bcc") es negativo, igual que en la referencia ISIC.
     */
    fun isBcc(label: String): Boolean = label == "BCC"

    companion object {
        /**
         * Carga tolerante: devuelve el mapa filename→label (entradas rotas omitidas).
         *
         * @param json — String: contenido íntegro del ground_truth.json del subconjunto.
         * @return Map<String, String>: filename (tal cual del JSON) → etiqueta recortada.
         * @throws GroundTruthFormatException si el JSON es inválido o no trae ninguna
         *         etiqueta usable (delega en [parse]).
         */
        @Throws(GroundTruthFormatException::class)
        fun load(json: String): Map<String, String> = parse(json).labels

        /**
         * Parsea el JSON plano del subconjunto.
         *
         * @param json — String: contenido íntegro del ground_truth.json; se espera un
         *        objeto JSON plano `{"NOMBRE.jpg": "ETIQUETA"}`.
         * @return GroundTruth: instancia con las entradas válidas y el conteo de
         *         descartadas en [GroundTruth.skippedEntries].
         * @throws GroundTruthFormatException si el texto no es un objeto JSON válido o
         *         no contiene ninguna etiqueta usable (p.ej. se apuntó al archivo
         *         equivocado: una matriz de ceros no sería un resultado honesto).
         */
        @Throws(GroundTruthFormatException::class)
        fun parse(json: String): GroundTruth {
            val root = try {
                JSONObject(json)
            } catch (error: JSONException) {
                throw GroundTruthFormatException(
                    "ground_truth.json no es un objeto JSON válido (${error.message})",
                )
            }
            val labels = HashMap<String, String>(root.length())
            var skipped = 0
            val keys = root.keys()
            while (keys.hasNext()) {
                val name = keys.next()
                val raw = root.opt(name)
                val label = (raw as? String)?.trim()
                if (label.isNullOrEmpty()) {
                    // Valor no textual (número, booleano, null, objeto/array anidado)
                    // o etiqueta en blanco: se omite sin tumbar el resto del archivo.
                    skipped++
                    continue
                }
                labels[name] = label
            }
            if (labels.isEmpty()) {
                throw GroundTruthFormatException(
                    "ground_truth.json no contiene ninguna etiqueta válida ($skipped entradas malformadas)",
                )
            }
            return GroundTruth(labels = labels, skippedEntries = skipped)
        }

        /**
         * Nombre base tras el último '/', recortado; tolera tanto rutas como nombres planos.
         *
         * @param file — String original del caller (puede traer separadores y espacios).
         * @return String con el segmento tras el último '/', sin espacios en los extremos.
         */
        private fun basename(file: String): String = file.trim().substringAfterLast('/')
    }
}
