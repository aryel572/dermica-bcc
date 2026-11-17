package com.dermica.bcc

import com.dermica.bcc.core.DateTimes
import org.junit.Assert.assertEquals
import org.junit.Test
import java.time.ZoneId

/** Tests JVM del formato corto de fecha-hora de las cards (S6). */
class DateTimesTest {

    @Test
    fun `formatea dd-MM-yyyy HH-mm en 24h con zona UTC`() {
        // 2026-11-16T12:34:56Z
        assertEquals("16/11/2026 12:34", DateTimes.formatDateTime(1_794_832_496_000L, ZoneId.of("UTC")))
    }

    @Test
    fun `medianoche se formatea con ceros a la izquierda`() {
        // 2026-01-02T00:05:00Z
        assertEquals("02/01/2026 00:05", DateTimes.formatDateTime(1_767_312_300_000L, ZoneId.of("UTC")))
    }

    @Test
    fun `hora de Lima resta 5 horas a UTC`() {
        // 2026-11-16T12:34:56Z → 07:34 en America/Lima
        assertEquals("16/11/2026 07:34", DateTimes.formatDateTime(1_794_832_496_000L, ZoneId.of("America/Lima")))
    }
}
