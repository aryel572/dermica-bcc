package com.dermica.bcc

import com.dermica.bcc.data.history.HistoryEvent
import com.dermica.bcc.data.history.HistoryKind
import com.dermica.bcc.data.history.toItemOrNull
import com.dermica.bcc.data.history.HistoryItem
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Tests JVM del mapeo entidad Room → dominio del historial (S6). */
class HistoryMappersTest {

    @Test
    fun `evento imagen completo mapea a item Image`() {
        val event = HistoryEvent(
            id = 7,
            kind = HistoryKind.IMAGE,
            createdAt = 1_000L,
            imageName = "ISIC_0000001.jpg",
            verdict = "BCC",
            confidence = 0.91f,
        )
        val item = event.toItemOrNull() as? HistoryItem.Image
        requireNotNull(item)
        assertEquals(7L, item.id)
        assertEquals(1_000L, item.createdAt)
        assertEquals("ISIC_0000001.jpg", item.imageName)
        assertEquals("BCC", item.verdict)
        assertEquals(0.91f, item.confidence)
    }

    @Test
    fun `evento subconjunto completo mapea a item Subset con NFS como entero`() {
        val event = HistoryEvent(
            id = 9,
            kind = HistoryKind.SUBSET,
            createdAt = 2_000L,
            folderLabel = "subset_01",
            imageCount = 15,
            tp = 4,
            tn = 9,
            fp = 1,
            fn = 1,
            ta = 86.66666666666667,
            nfs = 1.0,
            peo = 20.0,
        )
        val item = event.toItemOrNull() as? HistoryItem.Subset
        requireNotNull(item)
        assertEquals("subset_01", item.folderLabel)
        assertEquals(15, item.imageCount)
        assertEquals(1, item.nfs) // persistido como REAL, expuesto como Int
        assertEquals(4, item.tp)
        assertEquals(1, item.fn)
    }

    @Test
    fun `evento imagen sin verdict se omite del historial`() {
        val event = HistoryEvent(kind = HistoryKind.IMAGE, createdAt = 1_000L, confidence = 0.5f)
        assertNull(event.toItemOrNull())
    }

    @Test
    fun `evento subconjunto incompleto se omite del historial`() {
        val event = HistoryEvent(kind = HistoryKind.SUBSET, createdAt = 1_000L, folderLabel = "subset_02", imageCount = 15)
        assertNull(event.toItemOrNull())
    }

    @Test
    fun `kind desconocido se omite del historial`() {
        val event = HistoryEvent(kind = "audio", createdAt = 1_000L)
        assertNull(event.toItemOrNull())
    }

    @Test
    fun `confidencia nula en imagen se omite del historial`() {
        val event = HistoryEvent(kind = HistoryKind.IMAGE, createdAt = 1_000L, verdict = "NO_BCC")
        assertNull(event.toItemOrNull())
    }
}
