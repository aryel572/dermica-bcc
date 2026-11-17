package com.dermica.bcc.core

import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter

/**
 * Fecha-hora corta para las cards del historial (S6).
 *
 * Patrón fijo dd/MM/yyyy HH:mm en 24 h (convención peruana), independiente del
 * idioma del dispositivo: el historial es un registro clínico y el formato debe
 * ser estable entre sesiones y capturas de pantalla de la tesis.
 */
object DateTimes {

    private val SHORT: DateTimeFormatter = DateTimeFormatter.ofPattern("dd/MM/yyyy HH:mm")

    /** Formatea [epochMillis] en la zona horaria indicada (por defecto, la del dispositivo). */
    fun formatDateTime(epochMillis: Long, zone: ZoneId = ZoneId.systemDefault()): String =
        SHORT.withZone(zone).format(Instant.ofEpochMilli(epochMillis))
}
