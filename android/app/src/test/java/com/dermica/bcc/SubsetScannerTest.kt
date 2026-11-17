package com.dermica.bcc

import com.dermica.bcc.core.SubfolderCandidate
import com.dermica.bcc.core.SubsetScanner
import com.dermica.bcc.core.TreeChild
import com.dermica.bcc.core.TreeScanResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** Tests JVM del scanner del árbol (S5-multi): decisión single/multi/inelegible con fakes. */
class SubsetScannerTest {

    private fun dir(name: String?, hasGroundTruth: Boolean = false, hasImages: Boolean = false) =
        TreeChild(name = name, isDirectory = true, hasGroundTruth = hasGroundTruth, hasImages = hasImages)

    private fun file(name: String?) = TreeChild(name = name, isDirectory = false)

    @Test
    fun `arbol vacio no es elegible`() {
        assertEquals(TreeScanResult.NotEligible, SubsetScanner.decide(emptyList()))
    }

    @Test
    fun `padre solo con imagenes en la raiz no es elegible`() {
        val children = listOf(file("ISIC_001.jpg"), file("ISIC_002.png"), file("notas.txt"))
        assertEquals(TreeScanResult.NotEligible, SubsetScanner.decide(children))
    }

    @Test
    fun `raiz con ground_truth_json es flujo single aunque haya subcarpetas validas`() {
        val children = listOf(
            file("ground_truth.json"),
            file("ISIC_001.jpg"),
            dir("subset_01", hasGroundTruth = true, hasImages = true),
        )
        assertEquals(TreeScanResult.SingleSubset, SubsetScanner.decide(children))
    }

    @Test
    fun `subcarpetas validas producen multi ordenadas por nombre`() {
        val children = listOf(
            dir("subset_10", hasGroundTruth = true, hasImages = true),
            dir("subset_02", hasGroundTruth = true, hasImages = true),
            dir("subset_01", hasGroundTruth = true, hasImages = true),
        )
        val result = SubsetScanner.decide(children)
        assertTrue(result is TreeScanResult.MultiSubset)
        assertEquals(
            listOf("subset_01", "subset_02", "subset_10"),
            (result as TreeScanResult.MultiSubset).subfolders.map { it.name },
        )
    }

    @Test
    fun `subcarpeta sin ground_truth_json queda fuera del lote`() {
        val children = listOf(dir("subset_sin_json", hasGroundTruth = false, hasImages = true))
        assertEquals(TreeScanResult.NotEligible, SubsetScanner.decide(children))
    }

    @Test
    fun `subcarpeta con json pero sin imagenes queda fuera del lote`() {
        val children = listOf(dir("subset_vacia", hasGroundTruth = true, hasImages = false))
        assertEquals(TreeScanResult.NotEligible, SubsetScanner.decide(children))
    }

    @Test
    fun `mezcla validas e invalidas conserva solo las calificadas`() {
        val children = listOf(
            dir("valida_b", hasGroundTruth = true, hasImages = true),
            dir("sin_json", hasGroundTruth = false, hasImages = true),
            dir("sin_imagenes", hasGroundTruth = true, hasImages = false),
            dir("valida_a", hasGroundTruth = true, hasImages = true),
            file("suelta.jpg"),
        )
        val result = SubsetScanner.decide(children)
        assertTrue(result is TreeScanResult.MultiSubset)
        val multi = result as TreeScanResult.MultiSubset
        assertEquals(listOf("valida_a", "valida_b"), multi.subfolders.map { it.name })
        assertTrue(multi.subfolders.all { it.eligible })
    }

    @Test
    fun `nombres null o en blanco se descartan`() {
        val children = listOf(
            dir(null, hasGroundTruth = true, hasImages = true),
            dir("   ", hasGroundTruth = true, hasImages = true),
            dir("valida", hasGroundTruth = true, hasImages = true),
        )
        val result = SubsetScanner.decide(children)
        assertTrue(result is TreeScanResult.MultiSubset)
        assertEquals(
            listOf("valida"),
            (result as TreeScanResult.MultiSubset).subfolders.map { it.name },
        )
    }

    @Test
    fun `isImageName acepta jpg jpeg y png sin importar mayusculas`() {
        assertTrue(SubsetScanner.isImageName("a.jpg"))
        assertTrue(SubsetScanner.isImageName("a.JPEG"))
        assertTrue(SubsetScanner.isImageName("a.Png"))
        assertFalse(SubsetScanner.isImageName("a.gif"))
        assertFalse(SubsetScanner.isImageName("a.json"))
        assertFalse(SubsetScanner.isImageName("jpg"))
    }

    @Test
    fun `candidate eligible exige json e imagenes`() {
        assertTrue(SubfolderCandidate("x", hasGroundTruth = true, hasImages = true).eligible)
        assertFalse(SubfolderCandidate("x", hasGroundTruth = true, hasImages = false).eligible)
        assertFalse(SubfolderCandidate("x", hasGroundTruth = false, hasImages = true).eligible)
        assertFalse(SubfolderCandidate("x", hasGroundTruth = false, hasImages = false).eligible)
    }
}
