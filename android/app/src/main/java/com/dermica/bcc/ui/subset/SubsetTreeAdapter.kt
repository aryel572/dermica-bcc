package com.dermica.bcc.ui.subset

import android.content.Context
import android.net.Uri
import androidx.documentfile.provider.DocumentFile
import com.dermica.bcc.core.SubsetScanner
import com.dermica.bcc.core.TreeChild
import com.dermica.bcc.core.TreeScanResult
import java.util.Locale

/**
 * Adapter Android del escaneo (S5-multi): camina el árbol SAF y lo pre-resuelve
 * a datos puros para [SubsetScanner]. La decisión single/multi/inelegible vive en
 * core (testeada en JVM); aquí sólo queda lo no portable a JVM (DocumentFile,
 * lectura de bytes por contentResolver).
 *
 * Restricción de plataforma: SAF (ACTION_OPEN_DOCUMENT_TREE) permite elegir UNA
 * carpeta por intent, por eso el caso "padre con N subcarpetas" se resuelve con
 * este escaneo en vez de un multi-picker.
 *
 * Quién lo consume: `prepareAndRun` (ui/subset/SubsetScreen.kt), único caller;
 * éste deriva el TreeResolution al ViewModel (startRun / startMultiRun / fail).
 * Alcance: `internal` de :app — la decisión vive en core (JVM-testeable), aquí
 * sólo DocumentFile y contentResolver. Las imágenes viajan como URI; jamás se
 * copian a disco fuera del árbol.
 */
internal sealed interface TreeResolution {
    /** Raíz CON ground_truth.json → flujo T4 intacto (una sola carpeta). */
    data class SingleSubset(
        val folderLabel: String,
        val images: List<SubsetImage>,
        val groundTruthJson: String,
    ) : TreeResolution

    /** SIN json en raíz pero ≥1 subcarpeta calificada → flujo multi (S5-multi). */
    data class MultiSubset(val inputs: List<SubsetFolderInput>) : TreeResolution

    /** Nada calificable → la UI muestra el error honesto con Reintentar/Volver. */
    data class Fail(val message: String) : TreeResolution
}

/**
 * Inspecciona el árbol elegido una sola vez y devuelve el camino a seguir.
 *
 * - RAÍZ con ground_truth.json → [TreeResolution.SingleSubset] (prioridad T4).
 * - Subcarpetas calificadas → [TreeResolution.MultiSubset] con imágenes y JSON
 *   ya leídos (JSON ilegible viaja como null: esa subcarpeta se reporta fallida
 *   y el lote continúa).
 * - Cualquier otra cosa → [TreeResolution.Fail] con mensaje honesto (S5-multi).
 *
 * Trabajo síncrono y barato (lista y lecturas pequeñas, nunca copia imágenes):
 * por eso se llama en el hilo del picker sin scope propio.
 *
 * @param context contexto de la Activity (DocumentFile + contentResolver).
 * @param treeUri URI del árbol devuelto por OpenDocumentTree.
 * @return plan tipado para el caller: qué flujo corre, con qué imágenes/JSON ya leídos.
 */
internal fun resolveTree(context: Context, treeUri: Uri): TreeResolution {
    val root = DocumentFile.fromTreeUri(context, treeUri)
    if (root == null || !root.isDirectory) {
        return TreeResolution.Fail("No se pudo abrir la carpeta elegida.")
    }
    val folderLabel = root.name?.takeIf { it.isNotBlank() } ?: "subconjunto"
    val children = root.listFiles()

    // 1) ground_truth.json en la RAÍZ → flujo T4 (manda sobre cualquier subcarpeta).
    val rootJson = children.firstOrNull { it.isFile && it.name == SubsetScanner.GROUND_TRUTH_FILE_NAME }
    if (rootJson != null) {
        val images = imageChildren(children)
        if (images.isEmpty()) {
            return TreeResolution.Fail("La carpeta no contiene imágenes .jpg, .jpeg ni .png.")
        }
        val jsonText = readText(context, rootJson.uri)
            ?: return TreeResolution.Fail("No se pudo leer ground_truth.json desde la carpeta.")
        return TreeResolution.SingleSubset(folderLabel, images, jsonText)
    }

    // 2) SIN json en raíz: pre-resolver los hijos a datos puros y decidir (S5-multi).
    val treeChildren = children.map { child ->
        if (child.isDirectory) {
            val inner = child.listFiles()
            TreeChild(
                name = child.name,
                isDirectory = true,
                hasGroundTruth = inner.any {
                    it.isFile && it.name == SubsetScanner.GROUND_TRUTH_FILE_NAME
                },
                hasImages = inner.any { it.isFile && it.name?.let(SubsetScanner::isImageName) == true },
            )
        } else {
            TreeChild(name = child.name, isDirectory = false)
        }
    }

    return when (val scan = SubsetScanner.decide(treeChildren)) {
        TreeScanResult.NotEligible -> TreeResolution.Fail(
            "No se encontraron subconjuntos elegibles aquí; elige otra carpeta o una subcarpeta con ground_truth.json.",
        )
        is TreeScanResult.SingleSubset -> TreeResolution.Fail(
            // Inalcanzable: la raíz ya se comprobó sin json; defensa honesta, nunca crash.
            "No se encontraron subconjuntos elegibles aquí; elige otra carpeta o una subcarpeta con ground_truth.json.",
        )
        is TreeScanResult.MultiSubset -> {
            val inputs = scan.subfolders.map { candidate ->
                val dir = children.first { it.isDirectory && it.name == candidate.name }
                val inner = dir.listFiles()
                val images = imageChildren(inner)
                val jsonText = inner.firstOrNull {
                    it.isFile && it.name == SubsetScanner.GROUND_TRUTH_FILE_NAME
                }?.let { readText(context, it.uri) } // null = ilegible: el lote la reporta y sigue
                SubsetFolderInput(folderLabel = candidate.name, images = images, groundTruthJson = jsonText)
            }
            TreeResolution.MultiSubset(inputs)
        }
    }
}

/**
 * Imágenes compatibles de un listado de hijos, ordenadas por nombre
 * (orden estable, mismo criterio case-insensitive que SubsetScanner).
 *
 * @param children hijos directos (RAÍZ o subcarpeta) ya resueltos por DocumentFile.
 * @return lista ordenada; vacía si no hay .jpg/.jpeg/.png legibles por nombre.
 */
private fun imageChildren(children: Array<DocumentFile>): List<SubsetImage> =
    children
        .filter { it.isFile }
        .mapNotNull { document ->
            val name = document.name ?: return@mapNotNull null
            if (SubsetScanner.isImageName(name)) SubsetImage(name = name, uri = document.uri) else null
        }
        .sortedBy { it.name.lowercase(Locale.ROOT) }

/**
 * Lee el contenido de un documento como texto UTF-8; null ante cualquier fallo
 * (S3 tolerante). Nunca lanza: el caller decide reportar la subcarpeta fallida
 * o abortar la carpeta única.
 *
 * @param context contexto de la Activity (contentResolver).
 * @param uri URI del documento (ground_truth.json) elegido con SAF.
 * @return texto UTF-8 o null si falla la lectura/decodificación.
 */
private fun readText(context: Context, uri: Uri): String? = try {
    context.contentResolver.openInputStream(uri)?.use { it.readBytes().toString(Charsets.UTF_8) }
} catch (error: Exception) {
    null
}
