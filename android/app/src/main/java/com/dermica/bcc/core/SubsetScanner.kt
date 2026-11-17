package com.dermica.bcc.core

import java.util.Locale

/**
 * Scanner puro del árbol elegido (S5-multi): decide si el árbol es una carpeta
 * de subconjunto (`ground_truth.json` en la raíz, flujo T4), un PADRE con
 * subcarpetas calificables (flujo multi) o nada elegible.
 *
 * La parte Android (DocumentFile / lectura de bytes) vive aparte, en la capa de
 * UI; aquí sólo entran y salen datos puros, testeable en JVM. El adapter
 * pre-resuelve cada hijo del árbol a un [TreeChild] (nombre + si es directorio
 * + flags de contenido directo) y este objeto decide.
 *
 * Reglas del contrato (S5-multi, verbatim del requerimiento):
 * - RAÍZ con ground_truth.json → [TreeScanResult.SingleSubset], aunque también
 *   haya subcarpetas válidas: el flujo T4 queda intacto.
 * - SIN ground_truth.json en raíz y ≥1 subcarpeta con ground_truth.json + ≥1
 *   imagen compatible → [TreeScanResult.MultiSubset] con las calificadas,
 *   ordenadas por nombre (mismo orden case-insensitive que las imágenes).
 * - Cualquier otra cosa → [TreeScanResult.NotEligible] (error honesto, jamás
 *   un fallo silencioso ni un lote inventado).
 * - Nombres null o en blanco se descartan: un DocumentFile sin nombre no puede
 *   recibir etiqueta en el historial.
 *
 * Consumidores: el adapter Android (`resolveTree` en ui/subset) lo invoca tras
 * pre-resolver los hijos del árbol elegido; los tests JVM lo ejercitan directo con
 * [TreeChild] falsos. La UI jamás decide nada aquí: el flujo de datos es
 * árbol SAF → adapter → [decide] → [TreeScanResult] → SubsetViewModel.
 */
object SubsetScanner {

    /**
     * Nombre exacto del ground truth dentro de una carpeta de subconjunto (S3).
     * Origen: contrato del dataset del proyecto; consumen el adapter (raíz y
     * subcarpetas) y [decide] para distinguir carpeta T4 de carpeta padre S5-multi.
     */
    const val GROUND_TRUTH_FILE_NAME = "ground_truth.json"

    // Extensiones aceptadas por toda la app (imágenes dermatoscópicas comunes).
    // Única fuente: el pipeline prepara el mismo set; consumen isImageName, el
    // adapter (imagenChildren), el filtro de la RAÍZ y los mensajes de error.
    private val IMAGE_EXTENSIONS = listOf(".jpg", ".jpeg", ".png")

    /**
     * Determina si [name] termina en una extensión de imagen compatible.
     *
     * @param name nombre tal cual lo reporta SAF (DocumentFile.name); null no llega aquí.
     * @return true si es .jpg/.jpeg/.png en cualquier capitalización.
     */
    fun isImageName(name: String): Boolean =
        IMAGE_EXTENSIONS.any { name.lowercase(Locale.ROOT).endsWith(it) }

    /**
     * Decide el flujo del árbol a partir de sus hijos de primer nivel.
     *
     * Orden del resultado: las subcarpetas elegibles salen ordenadas por nombre
     * (`lowercase(Locale.ROOT)`) para que el lote y el historial sigan el mismo
     * orden estable en cualquier dispositivo, coincidiendo con el orden de las
     * imágenes del adapter.
     *
     * @param children hijos directos del árbol elegido, ya inspeccionados por el adapter.
     * @return [TreeScanResult.SingleSubset] si la raíz trae ground_truth.json;
     *         [TreeScanResult.MultiSubset] si alguna subcarpeta califica;
     *         [TreeScanResult.NotEligible] en cualquier otro caso.
     */
    fun decide(children: List<TreeChild>): TreeScanResult {
        // ground_truth.json en la RAÍZ manda sobre cualquier subcarpeta (T4 intacto).
        val hasRootJson = children.any { !it.isDirectory && it.name == GROUND_TRUTH_FILE_NAME }
        if (hasRootJson) return TreeScanResult.SingleSubset

        val subfolders = children
            .filter { it.isDirectory }
            .mapNotNull { child ->
                val name = child.name?.takeIf { it.isNotBlank() } ?: return@mapNotNull null
                SubfolderCandidate(
                    name = name,
                    hasGroundTruth = child.hasGroundTruth,
                    hasImages = child.hasImages,
                )
            }
            .sortedBy { it.name.lowercase(Locale.ROOT) }

        val eligible = subfolders.filter { it.eligible }
        return if (eligible.isEmpty()) {
            TreeScanResult.NotEligible
        } else {
            TreeScanResult.MultiSubset(eligible)
        }
    }
}

/**
 * Hijo de un directorio SAF, pre-resuelto a datos puros por el adapter Android
 * (o por un fake en tests). Los flags de contenido directo sólo aplican a
 * directorios; para archivos quedan en false.
 *
 * Es el contrato entre el adapter (que sí toca DocumentFile) y el scanner (que
 * no puede, para seguir siendo testeable en JVM).
 *
 * @param name nombre tal cual lo da SAF; null o en blanco equivale a descartable.
 * @param isDirectory true si el hijo es carpeta.
 * @param hasGroundTruth sólo directorios: contiene ground_truth.json directamente en su interior.
 * @param hasImages sólo directorios: contiene al menos una imagen .jpg/.jpeg/.png.
 */
data class TreeChild(
    val name: String?,
    val isDirectory: Boolean,
    /** Sólo directorios: contiene ground_truth.json directamente en su interior. */
    val hasGroundTruth: Boolean = false,
    /** Sólo directorios: contiene al menos una imagen .jpg/.jpeg/.png. */
    val hasImages: Boolean = false,
)

/**
 * Subcarpeta hija del árbol elegido, con el estado de su contenido directo.
 * `eligible` = referencia + al menos una imagen compatible (S5-multi).
 */
data class SubfolderCandidate(
    val name: String,
    val hasGroundTruth: Boolean,
    val hasImages: Boolean,
) {
    /** Calificada = ground_truth.json + ≥1 imagen; es la que entra al lote. */
    val eligible: Boolean get() = hasGroundTruth && hasImages
}

/**
 * Resultado del escaneo del árbol (S5-multi). Lo consume `prepareAndRun`
 * (ui/subset/SubsetScreen.kt) para derivar a flujo T4, lote multi o error honesto.
 */
sealed interface TreeScanResult {
    /** ground_truth.json en la raíz → flujo T4 (una sola carpeta, como hoy). */
    data object SingleSubset : TreeScanResult

    /** ≥1 subcarpeta calificada → flujo multi; ordenadas por nombre. */
    data class MultiSubset(val subfolders: List<SubfolderCandidate>) : TreeScanResult

    /** Nada calificable ni en raíz ni en hijos: la UI muestra el error honesto. */
    data object NotEligible : TreeScanResult
}
