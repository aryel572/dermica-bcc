package com.dermica.bcc.core

import org.json.JSONException
import org.json.JSONObject

/**
 * Fallo irrecuperable de la referencia de un subconjunto (S3): el texto no es un
 * objeto JSON válido, o no contiene ninguna etiqueta usable.
 *
 * Nunca se propaga como crash: la UI la captura y muestra una pantalla de error
 * con explicación y botón para reintentar.
 */
class GroundTruthFormatException(message: String) : IllegalArgumentException(message)

/**
 * Matcher puro del ground truth de un subconjunto (S3): JSON plano
 * `{filename: etiqueta}` versionado junto a la carpeta de imágenes del TESIS.
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

    /** Etiqueta de referencia para `file` (basename exacto), o null si no tiene entrada. */
    fun labelFor(file: String): String? = labels[basename(file)]

    /** Positivo de referencia (S3): solo la etiqueta exacta "BCC". */
    fun isBcc(label: String): Boolean = label == "BCC"

    companion object {
        /** Carga tolerante: devuelve el mapa filename→label (entradas rotas omitidas). */
        fun load(json: String): Map<String, String> = parse(json).labels

        /**
         * Parsea el JSON plano del subconjunto.
         *
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

        /** Nombre base tras el último '/', recortado; tolera tanto rutas como nombres planos. */
        private fun basename(file: String): String = file.trim().substringAfterLast('/')
    }
}
